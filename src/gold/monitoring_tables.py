"""Gold monitoring tables: small summary tables in the DuckDB file, read by the dashboard.

    python src/gold/monitoring_tables.py           # rebuild every Gold table in one tracked run
    python src/gold/monitoring_tables.py --list    # the Gold tables and what they hold

Gold is always rebuilt in full from the control tables (file_manifest, silver_log, checks,
incidents, scores), never updated piece by piece, so it cannot drift away from them. All
tables of a build are replaced in one transaction: a reader sees either the previous build or
the new one, and a failed build leaves the previous one untouched.

Every build is recorded in gold_build_log with the row count and a checksum per table. The
checksum leaves out the table's volatile columns (run ids, timestamps), so building twice from
the same data gives the same checksums.

Exit codes: 0 success, 1 build failed.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

if __package__ in (None, ""):
    # Allow running this file directly: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import db  # noqa: E402
from common.runs import track_run, utc_now  # noqa: E402

RUN_DATASET = "gold"  # pipeline_runs.dataset of a Gold build (Gold covers every dataset)
NAME = re.compile(r"^gold_[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class GoldTable:
    name: str
    description: str
    sql: str  # SELECT over the control tables
    order_by: str  # stored order, so the table reads the same after every build
    volatile: tuple[str, ...] = ()  # columns left out of the checksum

    def __post_init__(self) -> None:
        if not NAME.match(self.name) or self.name == "gold_build_log":
            raise ValueError(f"invalid Gold table name '{self.name}'")


@dataclass(frozen=True)
class BuiltTable:
    name: str
    rows: int
    checksum: str


# The spine of Gold: one row per dataset and month, for the current Bronze version.
MONTHS = GoldTable(
    name="gold_months",
    description="One row per dataset and month: the current Bronze file version",
    sql="""
        SELECT dataset, period, sha256, file_name, source_url, row_count AS bronze_rows,
               size_bytes, loaded_run_id, registered_at
        FROM file_manifest
        WHERE status = 'LOADED' AND bronze_path IS NOT NULL
        QUALIFY row_number() OVER (PARTITION BY dataset, period ORDER BY registered_at DESC) = 1
    """,
    order_by="dataset, period",
    volatile=("loaded_run_id", "registered_at"),
)

TABLES: list[GoldTable] = [MONTHS]


def checksum(con: duckdb.DuckDBPyConnection, table: str, volatile: tuple[str, ...] = ()) -> str:
    """MD5 over the table's rows as text, sorted, without the volatile columns."""
    columns = f"* EXCLUDE ({', '.join(volatile)})" if volatile else "*"
    return con.execute(
        "SELECT md5(coalesce(string_agg(CAST(t AS VARCHAR), chr(10) "
        f"ORDER BY CAST(t AS VARCHAR)), '')) FROM (SELECT {columns} FROM {table}) t"
    ).fetchone()[0]


def build(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    tables: list[GoldTable] | None = None,
    now: datetime | None = None,
) -> list[BuiltTable]:
    """Replace every Gold table and record the build, all in one transaction."""
    tables = TABLES if tables is None else tables
    names = [t.name for t in tables]
    if len(set(names)) != len(names):
        raise ValueError("Gold table names must be unique")
    now = now or utc_now()
    built = []
    con.execute("BEGIN TRANSACTION")
    try:
        for table in tables:
            con.execute(
                f"CREATE OR REPLACE TABLE {table.name} AS "
                f"SELECT * FROM ({table.sql}) ORDER BY {table.order_by}"
            )
            rows = con.execute(f"SELECT count(*) FROM {table.name}").fetchone()[0]
            built.append(BuiltTable(table.name, rows, checksum(con, table.name, table.volatile)))
        con.executemany(
            "INSERT INTO gold_build_log VALUES (?, ?, ?, ?, ?)",
            [[run_id, b.name, b.rows, b.checksum, now] for b in built],
        )
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise
    return built


def rebuild(
    con: duckdb.DuckDBPyConnection, tables: list[GoldTable] | None = None
) -> tuple[str, list[BuiltTable]]:
    """Build Gold in a tracked run (pipeline_runs, dataset 'gold')."""
    with track_run(con, RUN_DATASET, command="monitoring_tables") as run:
        built = build(con, run.run_id, tables)
    return run.run_id, built


def latest_build(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        "SELECT * FROM gold_build_log WHERE run_id = (SELECT max(run_id) FROM gold_build_log) "
        "ORDER BY table_name"
    ).df()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild the Gold monitoring tables.")
    parser.add_argument("--database", type=Path, help="DuckDB file (default: paths.duckdb)")
    parser.add_argument("--list", action="store_true", help="list the Gold tables and exit")
    args = parser.parse_args(argv)
    if args.list:
        for table in TABLES:
            print(f"{table.name:<24} {table.description}")
        return 0
    with db.connect(args.database) as con:
        db.init_schema(con)
        try:
            run_id, built = rebuild(con)
        except duckdb.Error as exc:
            print(f"Gold build failed, previous Gold kept: {exc}")
            return 1
    print(f"{run_id}: {len(built)} Gold table(s) built")
    for b in built:
        print(f"  {b.name:<24} {b.rows:>8} row(s)  checksum {b.checksum[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
