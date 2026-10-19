"""Verify that Bronze matches the source files recorded in file_manifest.

    python src/ingestion/verify_bronze.py

For every loaded file version it checks, without changing anything:

    rows        Bronze row count equals the source row count in the manifest
    row_ids     bronze_row_id runs from 1 to the row count with no gaps or repeats
    lineage     source_file, ingestion_timestamp and pipeline_run_id are filled and match
                the manifest
    schema      the source columns (name and type) are present, unchanged, plus the
                lineage columns, when the source file is still in data/incoming/
    checksum    the source file in data/incoming/ still has the registered SHA-256

Exit code 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

if __package__ in (None, ""):
    # Allow running this file directly: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import db  # noqa: E402
from common.config import REPO_ROOT, load_datasets_config  # noqa: E402
from ingestion import manifest  # noqa: E402
from ingestion.bronze_loader import LINEAGE_COLUMNS, read_bronze_sql  # noqa: E402
from ingestion.manifest import ManifestEntry  # noqa: E402

CHECKS = ("rows", "row_ids", "lineage", "schema", "checksum")


@dataclass(frozen=True)
class CheckResult:
    period: str
    version: str
    check: str
    passed: bool | None  # None: not checked (for example the source file is gone)
    detail: str


def _bronze_path(entry: ManifestEntry) -> Path:
    path = Path(entry.bronze_path)
    return path if path.is_absolute() else REPO_ROOT / path


def _schema(con: duckdb.DuckDBPyConnection, path: Path) -> list[tuple[str, str]]:
    rows = con.execute(f"DESCRIBE SELECT * FROM {read_bronze_sql(path)}").fetchall()
    return [(name, dtype) for name, dtype, *_ in rows]


def verify_entry(
    con: duckdb.DuckDBPyConnection, entry: ManifestEntry, incoming_dir: Path
) -> list[CheckResult]:
    version = entry.sha256[:12]

    def result(check: str, passed: bool | None, detail: str) -> CheckResult:
        return CheckResult(entry.period, version, check, passed, detail)

    bronze = _bronze_path(entry)
    if not bronze.is_file():
        return [result("rows", False, f"Bronze file missing: {entry.bronze_path}")]

    rows, min_id, max_id, distinct_ids, bad_lineage, run_ids = con.execute(
        f"""
        SELECT count(*), min(bronze_row_id), max(bronze_row_id), count(DISTINCT bronze_row_id),
               count_if(source_file IS DISTINCT FROM ? OR ingestion_timestamp IS NULL
                        OR pipeline_run_id IS NULL),
               list(DISTINCT pipeline_run_id)
        FROM {read_bronze_sql(bronze)}
        """,
        [entry.file_name],
    ).fetchone()

    results = [
        result(
            "rows",
            rows == entry.row_count,
            f"{rows:,} in Bronze, {entry.row_count:,} in source"
            if entry.row_count is not None
            else f"{rows:,} in Bronze, source count unknown",
        ),
        result(
            "row_ids",
            rows == 0 or (min_id == 1 and max_id == rows and distinct_ids == rows),
            f"{min_id}..{max_id}, {distinct_ids:,} distinct",
        ),
    ]
    lineage_ok = bad_lineage == 0 and run_ids == [entry.loaded_run_id]
    results.append(
        result(
            "lineage",
            lineage_ok,
            f"run {entry.loaded_run_id}"
            if lineage_ok
            else f"{bad_lineage:,} rows with wrong or empty lineage; runs {run_ids}",
        )
    )

    source = incoming_dir / entry.file_name
    if not source.is_file():
        results.append(result("schema", None, "source not in incoming/, not compared"))
        results.append(result("checksum", None, "source not in incoming/, not compared"))
        return results

    source_schema = _schema(con, source)
    bronze_schema = _schema(con, bronze)
    expected = [c for c, _ in source_schema] + list(LINEAGE_COLUMNS)
    schema_ok = [c for c, _ in bronze_schema] == expected and bronze_schema[
        : len(source_schema)
    ] == source_schema
    results.append(
        result(
            "schema",
            schema_ok,
            f"{len(source_schema)} source columns + {len(LINEAGE_COLUMNS)} lineage"
            if schema_ok
            else "columns or types differ from the source",
        )
    )
    checksum = manifest.sha256_of(source)
    if checksum == entry.sha256:
        results.append(result("checksum", True, "source unchanged since registration"))
    else:
        # A newer download of the same month: it is a different version, not an error in Bronze.
        results.append(result("checksum", None, "incoming/ now holds a different version"))
    return results


def verify(con: duckdb.DuckDBPyConnection, dataset: str, incoming_dir: Path) -> list[CheckResult]:
    loaded = [e for e in manifest.get_entries(con, dataset) if e.status == "LOADED"]
    return [r for entry in loaded for r in verify_entry(con, entry, incoming_dir)]


def report(results: list[CheckResult]) -> pd.DataFrame:
    """One row per file version, one column per check: PASS, FAIL or '-' (not checked)."""
    if not results:
        return pd.DataFrame()
    frame = pd.DataFrame([r.__dict__ for r in results])
    frame["outcome"] = frame["passed"].map({True: "PASS", False: "FAIL"}).fillna("-")
    table = frame.pivot(index=["period", "version"], columns="check", values="outcome")
    return (
        table.reindex(columns=[c for c in CHECKS if c in table.columns]).fillna("-").reset_index()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify Bronze against the file manifest.")
    parser.add_argument("--dataset", default="yellow_taxi", help="dataset name in datasets.yml")
    parser.add_argument("--source", type=Path, help="input folder (default: paths.incoming)")
    parser.add_argument("--database", type=Path, help="DuckDB file (default: paths.duckdb)")
    args = parser.parse_args(argv)

    config = load_datasets_config()
    dataset = config.dataset(args.dataset)
    with db.connect(args.database, read_only=True) as con:
        results = verify(con, dataset.name, args.source or config.paths["incoming"])

    if not results:
        print(f"nothing to verify: no loaded files for '{dataset.name}' (run ingest_data.py)")
        return 1
    print(report(results).to_string(index=False))
    failures = [r for r in results if r.passed is False]
    for r in failures:
        print(f"FAIL {r.period} {r.version} {r.check}: {r.detail}")
    versions = len({(r.period, r.version) for r in results})
    print(f"\n{versions} file version(s) checked, {len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
