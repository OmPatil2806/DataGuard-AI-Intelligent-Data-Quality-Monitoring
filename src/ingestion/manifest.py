"""Register incoming source files in the file_manifest control table.

Each distinct file version (dataset, period, SHA-256 of the content) is registered once.
Registering the same file again returns the existing entry; a corrected re-export of
the same month has a different checksum and becomes a new version.

    python src/ingestion/manifest.py                     # register every configured month
    python src/ingestion/manifest.py --months 2023-01
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow.parquet as pq

if __package__ in (None, ""):
    # Allow running this file directly: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import db  # noqa: E402
from common.config import ConfigError, Dataset, load_datasets_config  # noqa: E402
from common.runs import utc_now  # noqa: E402
from data_download.download_data import parse_months  # noqa: E402
from profiling.taxi_profile import period_of  # noqa: E402

log = logging.getLogger("manifest")

HASH_CHUNK = 8 * 1024 * 1024


@dataclass(frozen=True)
class ManifestEntry:
    dataset: str
    period: str
    file_name: str
    sha256: str
    size_bytes: int
    row_count: int | None
    expected_at: datetime | None
    delivered_at: datetime
    registered_at: datetime
    status: str
    bronze_path: str | None = None
    loaded_run_id: str | None = None
    source_url: str | None = None
    simulated_delivered_at: datetime | None = None


@dataclass(frozen=True)
class Registration:
    entry: ManifestEntry
    is_new: bool  # False: this exact file version was already registered
    other_versions: tuple[ManifestEntry, ...]  # earlier versions of the same period


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def parquet_row_count(path: Path) -> int:
    """Row count from the Parquet footer, without reading the data."""
    return pq.ParquetFile(path).metadata.num_rows


def expected_at(period: str, lag_days: int | None) -> datetime | None:
    """When a monthly file should arrive: end of its month plus the publication lag."""
    if lag_days is None:
        return None
    month_end = pd.Period(period, freq="M").end_time.normalize() + timedelta(days=1)
    return (month_end + timedelta(days=lag_days)).to_pydatetime()


def delivered_at(path: Path) -> datetime:
    """When the file arrived in incoming/: its modification time, in UTC."""
    return datetime.fromtimestamp(Path(path).stat().st_mtime, UTC).replace(tzinfo=None)


def _rows_to_entries(cursor: duckdb.DuckDBPyConnection) -> list[ManifestEntry]:
    names = [d[0] for d in cursor.description]
    fields = set(ManifestEntry.__dataclass_fields__)
    return [
        ManifestEntry(**{k: v for k, v in zip(names, row, strict=True) if k in fields})
        for row in cursor.fetchall()
    ]


def get_entries(
    con: duckdb.DuckDBPyConnection, dataset: str, period: str | None = None
) -> list[ManifestEntry]:
    query = "SELECT * FROM file_manifest WHERE dataset = ?"
    params: list[Any] = [dataset]
    if period:
        query += " AND period = ?"
        params.append(period)
    return _rows_to_entries(con.execute(query + " ORDER BY period, registered_at", params))


def register_file(
    con: duckdb.DuckDBPyConnection,
    dataset: Dataset,
    path: Path,
    period: str | None = None,
    now: datetime | None = None,
) -> Registration:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"file not found: {path}")
    period = period or period_of(path)
    if not period:
        raise ConfigError(f"cannot tell the period of '{path.name}'; pass period='YYYY-MM'")

    checksum = sha256_of(path)
    versions = get_entries(con, dataset.name, period)
    for existing in versions:
        if existing.sha256 == checksum:
            others = tuple(v for v in versions if v.sha256 != checksum)
            return Registration(existing, is_new=False, other_versions=others)

    row_count = parquet_row_count(path) if dataset.format == "parquet" else None
    source_url = dataset.url_for(period) if dataset.url_template else None
    con.execute(
        """
        INSERT INTO file_manifest (dataset, period, file_name, sha256, size_bytes, source_url,
                                   row_count, expected_at, delivered_at, registered_at, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'REGISTERED')
        """,
        [
            dataset.name,
            period,
            path.name,
            checksum,
            path.stat().st_size,
            source_url,
            row_count,
            expected_at(period, dataset.expected_lag_days),
            delivered_at(path),
            now or utc_now(),
        ],
    )
    entry = next(e for e in get_entries(con, dataset.name, period) if e.sha256 == checksum)
    return Registration(entry, is_new=True, other_versions=tuple(versions))


def set_status(
    con: duckdb.DuckDBPyConnection,
    entry: ManifestEntry,
    status: str,
    bronze_path: str | None = None,
    run_id: str | None = None,
) -> None:
    if status not in db.FILE_STATUSES:
        raise ValueError(f"status must be one of {', '.join(db.FILE_STATUSES)}, got '{status}'")
    con.execute(
        """
        UPDATE file_manifest
        SET status = ?, bronze_path = coalesce(?, bronze_path),
            loaded_run_id = coalesce(?, loaded_run_id)
        WHERE dataset = ? AND period = ? AND sha256 = ?
        """,
        [status, bronze_path, run_id, entry.dataset, entry.period, entry.sha256],
    )


def set_simulated_delivery(
    con: duckdb.DuckDBPyConnection, entry: ManifestEntry, when: datetime | None
) -> None:
    """Set (or with None clear) the delivery time the freshness check uses in replay mode."""
    con.execute(
        "UPDATE file_manifest SET simulated_delivered_at = ? "
        "WHERE dataset = ? AND period = ? AND sha256 = ?",
        [when, entry.dataset, entry.period, entry.sha256],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Register incoming files in file_manifest.")
    parser.add_argument("--dataset", default="yellow_taxi", help="dataset name in datasets.yml")
    parser.add_argument("--months", help="e.g. 2023-01, 2023-01:2023-03 or 2023-01,2019-01")
    parser.add_argument("--source", type=Path, help="input folder (default: paths.incoming)")
    parser.add_argument("--database", type=Path, help="DuckDB file (default: paths.duckdb)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load_datasets_config()
        dataset = config.dataset(args.dataset)
        months = parse_months(args.months, dataset.periods)
    except ConfigError as exc:
        log.error("%s", exc)
        return 2
    source_dir = args.source or config.paths["incoming"]

    missing = 0
    with db.connect(args.database) as con:
        db.init_schema(con)
        for month in months:
            path = source_dir / dataset.file_for(month)
            if not path.is_file():
                log.warning("missing    %s (run download_data.py first)", path.name)
                missing += 1
                continue
            result = register_file(con, dataset, path, period=month)
            e = result.entry
            if result.is_new:
                note = (
                    f", new version ({len(result.other_versions)} earlier)"
                    if result.other_versions
                    else ""
                )
                log.info(
                    "registered %s: %s rows, sha256 %s...%s",
                    e.file_name,
                    e.row_count,
                    e.sha256[:12],
                    note,
                )
            else:
                log.info("known      %s: already registered (%s)", e.file_name, e.status)
        table = con.execute(
            "SELECT period, file_name, row_count, round(size_bytes / 1048576, 1) AS mb, "
            "left(sha256, 12) AS sha256, expected_at, status FROM file_manifest "
            "WHERE dataset = ? ORDER BY period, registered_at",
            [dataset.name],
        ).df()
    if not table.empty:
        print(table.to_string(index=False))
    return 1 if missing and missing == len(months) else 0


if __name__ == "__main__":
    sys.exit(main())
