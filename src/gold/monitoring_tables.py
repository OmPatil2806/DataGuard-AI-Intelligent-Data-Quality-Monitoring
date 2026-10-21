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
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

if __package__ in (None, ""):
    # Allow running this file directly: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import db  # noqa: E402
from common.config import SEVERITIES, load_severity_weights  # noqa: E402
from common.runs import track_run, utc_now  # noqa: E402

RUN_DATASET = "gold"  # pipeline_runs.dataset of a Gold build (Gold covers every dataset)
NAME = re.compile(r"^gold_[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class GoldTable:
    name: str
    description: str
    sql: str | Callable[[], str]  # SELECT over the control tables (a function: built when run)
    order_by: str  # stored order, so the table reads the same after every build
    volatile: tuple[str, ...] = ()  # columns left out of the checksum

    def __post_init__(self) -> None:
        if not NAME.match(self.name) or self.name == "gold_build_log":
            raise ValueError(f"invalid Gold table name '{self.name}'")

    def query(self) -> str:
        return self.sql() if callable(self.sql) else self.sql


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


def _rank(column: str) -> str:
    """critical 4 ... low 1, else 0 (the order of SEVERITIES)."""
    cases = " ".join(f"WHEN '{s}' THEN {len(SEVERITIES) - i}" for i, s in enumerate(SEVERITIES))
    return f"CASE {column} {cases} ELSE 0 END"


def dataset_health_sql(health: dict | None = None) -> str:
    """One row per month of gold_months: rows, score, findings and a health label."""
    health = health or load_severity_weights().get("health") or {}
    warning_below = float(health.get("warning_below", 95))
    critical_below = float(health.get("critical_below", 80))
    worst = "greatest(coalesce(c.worst_check, 0), coalesce(f.worst_incident, 0))"
    names = " ".join(f"WHEN {len(SEVERITIES) - i} THEN '{s}'" for i, s in enumerate(SEVERITIES))
    return f"""
        WITH silver AS (
            SELECT dataset, period, sha256, run_id, silver_rows, quarantine_rows, warned_rows
            FROM silver_log WHERE status = 'processed'
            QUALIFY row_number() OVER (
                PARTITION BY dataset, period, sha256 ORDER BY logged_at DESC, run_id DESC) = 1
        ),
        check_run AS (
            SELECT dataset, period, sha256, max(run_id) AS run_id
            FROM dataset_check_results GROUP BY ALL
        ),
        checks AS (
            SELECT r.dataset, r.period, r.sha256, r.run_id,
                   count_if(r.status <> 'skipped') AS checks_run,
                   count_if(r.status = 'fail') AS checks_failed,
                   {", ".join(f"count_if(r.severity = '{s}') AS failed_{s}" for s in SEVERITIES)},
                   max({_rank("r.severity")}) AS worst_check
            FROM dataset_check_results r JOIN check_run USING (dataset, period, sha256, run_id)
            GROUP BY ALL
        ),
        found AS (
            SELECT i.dataset, e.period,
                   count(DISTINCT e.incident_id) AS incidents_found,
                   count_if(e.event = 'opened') AS incidents_opened,
                   count(DISTINCT e.incident_id) FILTER (WHERE i.status <> 'RESOLVED')
                       AS incidents_active,
                   max({_rank("e.severity")}) AS worst_incident
            FROM incident_events e JOIN incidents i USING (incident_id)
            WHERE e.event IN ('opened', 'seen_again')
            GROUP BY ALL
        )
        SELECT m.dataset, m.period, m.sha256, m.bronze_rows,
               s.silver_rows, s.quarantine_rows, s.warned_rows,
               round(100.0 * s.quarantine_rows / nullif(m.bronze_rows, 0), 4) AS quarantine_pct,
               q.score,
               coalesce(c.checks_run, 0)::BIGINT AS checks_run,
               coalesce(c.checks_failed, 0)::BIGINT AS checks_failed,
               {", ".join(f"coalesce(c.failed_{s}, 0)::BIGINT AS failed_{s}" for s in SEVERITIES)},
               coalesce(f.incidents_found, 0)::BIGINT AS incidents_found,
               coalesce(f.incidents_opened, 0)::BIGINT AS incidents_opened,
               coalesce(f.incidents_active, 0)::BIGINT AS incidents_active,
               CASE {worst} {names} END AS worst_severity,
               CASE
                   WHEN q.score IS NULL THEN 'not_scored'
                   WHEN q.score < {critical_below} OR {worst} = {len(SEVERITIES)} THEN 'critical'
                   WHEN q.score < {warning_below} OR {worst} >= {len(SEVERITIES) - 2}
                       THEN 'warning'
                   ELSE 'healthy'
               END AS health,
               s.run_id AS silver_run_id, c.run_id AS check_run_id
        FROM gold_months m
        LEFT JOIN silver s USING (dataset, period, sha256)
        LEFT JOIN quality_scores q USING (dataset, period, sha256)
        LEFT JOIN checks c USING (dataset, period, sha256)
        LEFT JOIN found f USING (dataset, period)
    """


DATASET_HEALTH = GoldTable(
    name="gold_dataset_health",
    description="One row per dataset and month: rows, score, findings and a health label",
    sql=dataset_health_sql,
    order_by="dataset, period",
    volatile=("silver_run_id", "check_run_id"),
)

TABLES: list[GoldTable] = [MONTHS, DATASET_HEALTH]  # in build order: later tables may read earlier


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
                f"SELECT * FROM ({table.query()}) ORDER BY {table.order_by}"
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
