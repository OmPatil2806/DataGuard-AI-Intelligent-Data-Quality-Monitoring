"""Write registered source files to Bronze, unchanged, with lineage columns added.

Layout (one folder per file version, so Bronze never overwrites or deletes data):

    data/bronze/<dataset>/period=2023-01/version=<sha256[:12]>/part-0.parquet

Every source column keeps its name, type, values and row order. Four columns are added:

    bronze_row_id         1-based position of the row in the source file
    source_file           source file name
    ingestion_timestamp   when the row was written to Bronze (UTC)
    pipeline_run_id       run that wrote it

(source_file, bronze_row_id) identifies a row; the manifest maps source_file and
version to the SHA-256 of the exact content that was loaded.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb

from common.config import REPO_ROOT, Dataset
from common.runs import utc_now
from ingestion import manifest
from ingestion.manifest import ManifestEntry

LINEAGE_COLUMNS = ("bronze_row_id", "source_file", "ingestion_timestamp", "pipeline_run_id")
PART_FILE = "part-0.parquet"


class BronzeError(RuntimeError):
    """Raised when a file cannot be written to Bronze correctly."""


@dataclass(frozen=True)
class BronzeResult:
    path: Path
    rows: int
    status: str  # "loaded" or "skipped" (this version is already in Bronze)


def _sql_path(path: Path) -> str:
    return "'" + str(path).replace("\\", "/").replace("'", "''") + "'"


def version_dir(bronze_root: Path, entry: ManifestEntry) -> Path:
    return (
        Path(bronze_root)
        / entry.dataset
        / f"period={entry.period}"
        / f"version={entry.sha256[:12]}"
    )


def _relative(path: Path) -> str:
    """Store paths relative to the repository when possible, with forward slashes."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def load_file(
    con: duckdb.DuckDBPyConnection,
    entry: ManifestEntry,
    source: Path,
    bronze_root: Path,
    run_id: str,
    now: datetime | None = None,
    force: bool = False,
) -> BronzeResult:
    """Copy one registered file into Bronze and mark it LOADED in the manifest."""
    source = Path(source)
    if manifest.sha256_of(source) != entry.sha256:
        raise BronzeError(
            f"{source.name} changed since it was registered; register it again before loading"
        )

    target_dir = version_dir(bronze_root, entry)
    target = target_dir / PART_FILE
    if target.exists() and not force:
        rows = con.execute(f"SELECT count(*) FROM read_parquet({_sql_path(target)})").fetchone()[0]
        return BronzeResult(target, rows, "skipped")

    columns = [
        name
        for name, *_ in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet({_sql_path(source)})"
        ).fetchall()
    ]
    clash = sorted({c.lower() for c in columns} & set(LINEAGE_COLUMNS))
    if clash:
        raise BronzeError(f"{source.name} already has lineage column(s): {', '.join(clash)}")

    tmp_dir = target_dir.with_name(target_dir.name + ".tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    tmp = tmp_dir / PART_FILE
    loaded_at = now or utc_now()
    try:
        con.execute(
            f"""
            COPY (
                SELECT * EXCLUDE (file_row_number),
                       file_row_number + 1           AS bronze_row_id,
                       ?::VARCHAR                    AS source_file,
                       ?::TIMESTAMP                  AS ingestion_timestamp,
                       ?::VARCHAR                    AS pipeline_run_id
                FROM read_parquet({_sql_path(source)}, file_row_number = true)
                ORDER BY file_row_number
            ) TO {_sql_path(tmp)} (FORMAT parquet, COMPRESSION zstd)
            """,
            [source.name, loaded_at, run_id],
        )
        rows = con.execute(f"SELECT count(*) FROM read_parquet({_sql_path(tmp)})").fetchone()[0]
        if entry.row_count is not None and rows != entry.row_count:
            raise BronzeError(
                f"{source.name}: wrote {rows} rows but the source has {entry.row_count}"
            )
        if target_dir.exists():
            shutil.rmtree(target_dir)
        os.replace(tmp_dir, target_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    manifest.set_status(con, entry, "LOADED", bronze_path=_relative(target), run_id=run_id)
    return BronzeResult(target, rows, "loaded")


def current_files(con: duckdb.DuckDBPyConnection, dataset: Dataset | str) -> dict[str, Path]:
    """The Bronze file to use per period: the most recently registered loaded version."""
    name = dataset.name if isinstance(dataset, Dataset) else dataset
    rows = con.execute(
        """
        SELECT period, bronze_path FROM file_manifest
        WHERE dataset = ? AND status = 'LOADED' AND bronze_path IS NOT NULL
        QUALIFY row_number() OVER (PARTITION BY period ORDER BY registered_at DESC) = 1
        ORDER BY period
        """,
        [name],
    ).fetchall()
    return {
        period: (REPO_ROOT / path if not Path(path).is_absolute() else Path(path))
        for period, path in rows
    }
