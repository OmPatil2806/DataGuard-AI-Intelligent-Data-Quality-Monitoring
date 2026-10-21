"""Lineage: where every month and every row came from, and whether the row counts add up.

    python src/gold/lineage.py --period 2023-01 --row 42     # trace one row of a month

gold_lineage (built by monitoring_tables.py) has one row per dataset and month of the current
Bronze version: source URL -> file and SHA-256 -> Bronze file -> Silver and Quarantine files ->
the runs that wrote and checked them. The row counts are read from the files themselves and
compared with the manifest and silver_log:

    ok             manifest rows = Bronze file rows = Silver + Quarantine file rows, as logged
    not_processed  in Bronze, but no Silver yet
    missing_file   a file the control tables point to is not on disk
    mismatch       a count differs (detail says which)

A row is identified by its month and bronze_row_id (its 1-based position in the source file);
trace() follows it from the source file to Silver or Quarantine.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

if __package__ in (None, ""):
    # Allow running this file directly: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import db  # noqa: E402
from common.config import REPO_ROOT  # noqa: E402
from ingestion.bronze_loader import read_bronze_sql  # noqa: E402

LINEAGE_COLUMNS = {
    "dataset": "string",
    "period": "string",
    "source_url": "string",
    "file_name": "string",
    "sha256": "string",
    "size_bytes": "Int64",
    "versions_registered": "Int64",  # file versions of the month in the manifest
    "bronze_path": "string",
    "silver_path": "string",
    "quarantine_path": "string",
    "manifest_rows": "Int64",
    "bronze_rows": "Int64",  # counted in the files
    "silver_rows": "Int64",
    "quarantine_rows": "Int64",
    "logged_silver_rows": "Int64",  # as silver_log recorded them
    "logged_quarantine_rows": "Int64",
    "status": "string",
    "detail": "string",
    "loaded_run_id": "string",
    "silver_run_id": "string",
    "rule_run_id": "string",
    "check_run_id": "string",
    "scored": "boolean",
}
# Run ids differ between rebuilds, and where the files are on disk is not content (the
# SHA-256 is): both are left out of the Gold checksum.
VOLATILE = (
    "bronze_path",
    "silver_path",
    "quarantine_path",
    "loaded_run_id",
    "silver_run_id",
    "rule_run_id",
    "check_run_id",
)


def _absolute(path: str | None) -> Path | None:
    if path is None:
        return None
    return Path(path) if Path(path).is_absolute() else REPO_ROOT / path


def _count(con: duckdb.DuckDBPyConnection, path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    return con.execute(f"SELECT count(*) FROM {read_bronze_sql(path)}").fetchone()[0]


def lineage_frame(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    months = con.execute(
        """
        WITH silver AS (
            SELECT dataset, period, sha256, silver_path, quarantine_path
            FROM silver_log
            QUALIFY row_number() OVER (
                PARTITION BY dataset, period, sha256 ORDER BY logged_at DESC, run_id DESC) = 1
        ),
        processed AS (
            SELECT dataset, period, sha256, run_id, silver_rows, quarantine_rows
            FROM silver_log WHERE status = 'processed'
            QUALIFY row_number() OVER (
                PARTITION BY dataset, period, sha256 ORDER BY logged_at DESC, run_id DESC) = 1
        ),
        rules AS (
            SELECT dataset, period, sha256, max(run_id) AS run_id
            FROM rule_run_summary GROUP BY ALL
        ),
        versions AS (
            SELECT dataset, period, count(*) AS n FROM file_manifest GROUP BY ALL
        )
        SELECT m.dataset, m.period, m.source_url, m.file_name, m.sha256, m.size_bytes,
               v.n, f.bronze_path, s.silver_path, s.quarantine_path, m.bronze_rows,
               p.silver_rows, p.quarantine_rows, m.loaded_run_id, p.run_id, r.run_id,
               h.check_run_id, h.score IS NOT NULL
        FROM gold_months m
        JOIN file_manifest f USING (dataset, period, sha256)
        JOIN versions v USING (dataset, period)
        LEFT JOIN silver s USING (dataset, period, sha256)
        LEFT JOIN processed p USING (dataset, period, sha256)
        LEFT JOIN rules r USING (dataset, period, sha256)
        LEFT JOIN gold_dataset_health h USING (dataset, period, sha256)
        ORDER BY m.dataset, m.period
        """
    ).fetchall()

    records = []
    for (
        dataset, period, url, name, sha, size, versions, bronze, silver, quarantine,
        manifest_rows, logged_silver, logged_quarantine, loaded_run, silver_run, rule_run,
        check_run, scored,
    ) in months:  # fmt: skip
        paths = {"Bronze": _absolute(bronze)}
        if silver is not None:
            paths |= {"Silver": _absolute(silver), "Quarantine": _absolute(quarantine)}
        counts = {kind: _count(con, path) for kind, path in paths.items()}
        missing = [kind for kind, path in paths.items() if not path.exists()]
        problems = []
        if missing:
            status = "missing_file"
            problems = [f"{kind} file not found: {paths[kind]}" for kind in missing]
        else:
            if counts["Bronze"] != manifest_rows:
                problems.append(
                    f"Bronze file has {counts['Bronze']} rows, manifest {manifest_rows}"
                )
            if silver is not None:
                split = counts["Silver"] + counts["Quarantine"]
                if split != counts["Bronze"]:
                    problems.append(
                        f"Silver + Quarantine = {split} rows, Bronze {counts['Bronze']}"
                    )
                if (counts["Silver"], counts["Quarantine"]) != (logged_silver, logged_quarantine):
                    problems.append(
                        f"files have {counts['Silver']} + {counts['Quarantine']} rows, "
                        f"silver_log {logged_silver} + {logged_quarantine}"
                    )
            status = "mismatch" if problems else ("ok" if silver is not None else "not_processed")
        records.append(
            {
                "dataset": dataset,
                "period": period,
                "source_url": url,
                "file_name": name,
                "sha256": sha,
                "size_bytes": size,
                "versions_registered": versions,
                "bronze_path": bronze,
                "silver_path": silver,
                "quarantine_path": quarantine,
                "manifest_rows": manifest_rows,
                "bronze_rows": counts["Bronze"],
                "silver_rows": counts.get("Silver"),
                "quarantine_rows": counts.get("Quarantine"),
                "logged_silver_rows": logged_silver,
                "logged_quarantine_rows": logged_quarantine,
                "status": status,
                "detail": "; ".join(problems) or None,
                "loaded_run_id": loaded_run,
                "silver_run_id": silver_run,
                "rule_run_id": rule_run,
                "check_run_id": check_run,
                "scored": scored,
            }
        )
    frame = pd.DataFrame(records, columns=list(LINEAGE_COLUMNS))
    return frame.astype(LINEAGE_COLUMNS)


def lineage_sql(con: duckdb.DuckDBPyConnection) -> str:
    con.register("gold_lineage_frame", lineage_frame(con))
    return "SELECT * FROM gold_lineage_frame"


class TraceError(LookupError):
    """Raised when a row cannot be traced."""


def trace(
    con: duckdb.DuckDBPyConnection, period: str, row_id: int, dataset: str = "yellow_taxi"
) -> dict[str, Any]:
    """Follow one row of the current version of a month from its source file to Silver or
    Quarantine. Reads the control tables and the files, not Gold, so it is never stale."""
    file = con.execute(
        """
        SELECT source_url, file_name, sha256, bronze_path, loaded_run_id FROM file_manifest
        WHERE dataset = ? AND period = ? AND status = 'LOADED' AND bronze_path IS NOT NULL
        ORDER BY registered_at DESC LIMIT 1
        """,
        [dataset, period],
    ).fetchone()
    if file is None:
        raise TraceError(f"{dataset} {period} is not in Bronze")
    url, name, sha, bronze, loaded_run = file
    result: dict[str, Any] = {
        "dataset": dataset,
        "period": period,
        "bronze_row_id": row_id,
        "source_url": url,
        "source_file": name,
        "sha256": sha,
        "bronze_path": bronze,
        "loaded_run_id": loaded_run,
    }
    bronze_file = _absolute(bronze)
    if not bronze_file.exists():
        raise TraceError(f"Bronze file not found: {bronze_file}")
    row = con.execute(
        f"SELECT * FROM {read_bronze_sql(bronze_file)} WHERE bronze_row_id = ?", [row_id]
    ).df()
    if row.empty:
        raise TraceError(f"{period} has no row {row_id} (bronze_row_id is 1-based)")
    result["bronze_row"] = row.iloc[0].to_dict()

    logged = con.execute(
        """
        SELECT run_id, silver_path, quarantine_path FROM silver_log
        WHERE dataset = ? AND period = ? AND sha256 = ?
        ORDER BY logged_at DESC, run_id DESC LIMIT 1
        """,
        [dataset, period, sha],
    ).fetchone()
    result["layer"] = None
    if logged:
        result["silver_run_id"] = logged[0]
        for layer, path in (("Silver", logged[1]), ("Quarantine", logged[2])):
            found = con.execute(
                f"SELECT failed_rules, reasons FROM {read_bronze_sql(_absolute(path))} "
                "WHERE bronze_row_id = ?",
                [row_id],
            ).fetchone()
            if found:
                result |= {
                    "layer": layer,
                    "layer_path": path,
                    "failed_rules": list(found[0]),
                    "reasons": list(found[1]),
                }
                break
    checks = con.execute(
        "SELECT max(run_id) FROM dataset_check_results WHERE dataset = ? AND period = ? "
        "AND sha256 = ?",
        [dataset, period, sha],
    ).fetchone()[0]
    score = con.execute(
        "SELECT score FROM quality_scores WHERE dataset = ? AND period = ? AND sha256 = ?",
        [dataset, period, sha],
    ).fetchone()
    result["check_run_id"] = checks
    result["month_score"] = score[0] if score else None
    return result


def format_trace(t: dict[str, Any]) -> str:
    lines = [
        f"row {t['bronze_row_id']} of {t['dataset']} {t['period']}",
        f"  source     {t['source_url'] or '-'}",
        f"  file       {t['source_file']} (sha256 {t['sha256'][:12]})",
        f"  Bronze     {t['bronze_path']} (written by {t['loaded_run_id']})",
    ]
    if t["layer"] is None:
        lines.append("  Silver     not processed yet (run process_silver.py)")
    else:
        lines.append(f"  {t['layer']:<10} {t['layer_path']} (written by {t['silver_run_id']})")
        if t["reasons"]:
            lines += [f"    - {r}" for r in t["reasons"]]
        else:
            lines.append("    passed every rule")
    if t["check_run_id"]:
        lines.append(f"  checks     {t['check_run_id']}")
    if t["month_score"] is not None:
        lines.append(f"  score      {t['month_score']:.1f} for the month")
    row = t["bronze_row"]
    values = {k: v for k, v in row.items() if k not in ("bronze_row_id", "source_file")}
    lines.append("  values     " + ", ".join(f"{k}={v}" for k, v in values.items()))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Trace one row from its source to Silver.")
    parser.add_argument("--period", required=True, help="month, YYYY-MM")
    parser.add_argument("--row", type=int, required=True, help="bronze_row_id (1-based)")
    parser.add_argument("--dataset", default="yellow_taxi", help="dataset name")
    parser.add_argument("--database", type=Path, help="DuckDB file (default: paths.duckdb)")
    args = parser.parse_args(argv)
    try:
        with db.connect(args.database, read_only=True) as con:
            print(format_trace(trace(con, args.period, args.row, args.dataset)))
    except (TraceError, FileNotFoundError) as exc:
        print(f"error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
