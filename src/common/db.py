"""DuckDB connection and control tables (data/dataguard.duckdb).

Control tables record what the pipeline did, so every Bronze, Silver and Quarantine
row can be traced back to a run and a source file.

    python src/common/db.py          # create the database and tables, list them and recent runs
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

if __package__ in (None, ""):
    # Allow running this file directly: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import load_datasets_config  # noqa: E402

SCHEMA_VERSION = 10

RUN_STATUSES = ("RUNNING", "SUCCESS", "WARNING", "FAILED")
FILE_STATUSES = ("REGISTERED", "LOADED", "SKIPPED", "FAILED")
SEVERITY_LEVELS = ("critical", "high", "medium", "low")
INCIDENT_SOURCES = ("row_rule", "dataset_check")
INCIDENT_STATUSES = ("OPEN", "INVESTIGATING", "RESOLVED")
INCIDENT_EVENTS = (
    "opened",  # a new incident
    "seen_again",  # the same problem failed again (de-duplicated)
    "clean",  # a run in which the problem did not occur
    "investigating",  # someone is working on it
    "reopened",  # back to OPEN from INVESTIGATING
    "resolved",  # closed after enough clean runs, or by hand
    "note",  # free text
)


def _in(values: tuple[str, ...]) -> str:
    return "(" + ", ".join(f"'{v}'" for v in values) + ")"


SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version     INTEGER PRIMARY KEY,
        applied_at  TIMESTAMP NOT NULL DEFAULT current_timestamp
    )
    """,
    # One row per pipeline run.
    f"""
    CREATE TABLE IF NOT EXISTS pipeline_runs (
        run_id         VARCHAR PRIMARY KEY,        -- run_0001, run_0002, ...
        dataset        VARCHAR NOT NULL,
        command        VARCHAR,                    -- how the run was started
        status         VARCHAR NOT NULL CHECK (status IN {_in(RUN_STATUSES)}),
        started_at     TIMESTAMP NOT NULL,
        finished_at    TIMESTAMP,
        files_total    INTEGER NOT NULL DEFAULT 0,
        files_loaded   INTEGER NOT NULL DEFAULT 0,
        files_skipped  INTEGER NOT NULL DEFAULT 0,
        rows_loaded    BIGINT NOT NULL DEFAULT 0,
        error_message  VARCHAR,
        CHECK (finished_at IS NULL OR finished_at >= started_at)
    )
    """,
    # One row per distinct source file version (a corrected re-export gets a new row).
    f"""
    CREATE TABLE IF NOT EXISTS file_manifest (
        dataset        VARCHAR NOT NULL,
        period         VARCHAR NOT NULL,           -- YYYY-MM
        file_name      VARCHAR NOT NULL,
        sha256         VARCHAR NOT NULL,
        size_bytes     BIGINT NOT NULL,
        source_url     VARCHAR,
        row_count      BIGINT,
        expected_at    TIMESTAMP,                  -- when the file should arrive (freshness)
        delivered_at   TIMESTAMP,                  -- when it arrived
        registered_at  TIMESTAMP NOT NULL,
        status         VARCHAR NOT NULL CHECK (status IN {_in(FILE_STATUSES)}),
        bronze_path    VARCHAR,
        loaded_run_id  VARCHAR,
        PRIMARY KEY (dataset, period, sha256)
    )
    """,
    # Schema version 2: row rule results (Phase 2).
    # One row per rule per checked file version and run.
    """
    CREATE TABLE IF NOT EXISTS rule_run_log (
        run_id         VARCHAR NOT NULL,
        dataset        VARCHAR NOT NULL,
        period         VARCHAR NOT NULL,
        sha256         VARCHAR NOT NULL,           -- file version that was checked
        rule_id        VARCHAR NOT NULL,
        category       VARCHAR NOT NULL,
        action         VARCHAR NOT NULL CHECK (action IN ('quarantine', 'warn')),
        severity       VARCHAR NOT NULL,
        rows_checked   BIGINT NOT NULL,
        rows_failed    BIGINT NOT NULL,
        PRIMARY KEY (run_id, dataset, period, rule_id),
        CHECK (rows_failed BETWEEN 0 AND rows_checked)
    )
    """,
    # One row per checked file version and run: how many rows each action affects.
    """
    CREATE TABLE IF NOT EXISTS rule_run_summary (
        run_id            VARCHAR NOT NULL,
        dataset           VARCHAR NOT NULL,
        period            VARCHAR NOT NULL,
        sha256            VARCHAR NOT NULL,
        rows_checked      BIGINT NOT NULL,
        rows_quarantined  BIGINT NOT NULL,
        rows_warned       BIGINT NOT NULL,
        standardization   VARCHAR,                 -- renames, casts, lossy values
        PRIMARY KEY (run_id, dataset, period),
        CHECK (rows_quarantined + rows_warned <= rows_checked)
    )
    """,
    # Schema version 3: dataset-level metrics per file version (Phase 3).
    # metric: rows | rows_per_day | raw_schema | null_rate | histogram | mean | median | p99
    # column_name and bin are '' when they do not apply; text_value holds raw_schema as JSON.
    """
    CREATE TABLE IF NOT EXISTS dataset_metrics (
        dataset       VARCHAR NOT NULL,
        period        VARCHAR NOT NULL,
        sha256        VARCHAR NOT NULL,
        metric        VARCHAR NOT NULL,
        column_name   VARCHAR NOT NULL DEFAULT '',
        bin           VARCHAR NOT NULL DEFAULT '',
        value         DOUBLE,
        text_value    VARCHAR,
        computed_at   TIMESTAMP NOT NULL,
        PRIMARY KEY (dataset, period, sha256, metric, column_name, bin)
    )
    """,
    # Schema version 4: the baseline each dataset check compared a month with (Phase 3).
    """
    CREATE TABLE IF NOT EXISTS baselines (
        run_id          VARCHAR NOT NULL,
        dataset         VARCHAR NOT NULL,
        period          VARCHAR NOT NULL,          -- month that was checked
        metric          VARCHAR NOT NULL,          -- rows_per_day | null_rate | distribution
        column_name     VARCHAR NOT NULL DEFAULT '',
        bin             VARCHAR NOT NULL DEFAULT '',
        value           DOUBLE,
        source_periods  VARCHAR NOT NULL,          -- earlier months it was built from
        trusted         BOOLEAN NOT NULL,          -- false: cold start, fixed limits apply
        computed_at     TIMESTAMP NOT NULL,
        PRIMARY KEY (run_id, dataset, period, metric, column_name, bin)
    )
    """,
    # Schema version 5: a delivery time set on purpose (freshness replay and fault injection).
    "ALTER TABLE file_manifest ADD COLUMN IF NOT EXISTS simulated_delivered_at TIMESTAMP",
    # Schema version 6: results of the dataset-level checks (Phase 3), one row per month,
    # check and column (column_name is '' for checks on the whole month).
    """
    CREATE TABLE IF NOT EXISTS dataset_check_results (
        run_id       VARCHAR NOT NULL,
        dataset      VARCHAR NOT NULL,
        period       VARCHAR NOT NULL,
        sha256       VARCHAR NOT NULL,
        "check"      VARCHAR NOT NULL,      -- schema, volume, freshness, null_rate, drift
        column_name  VARCHAR NOT NULL DEFAULT '',
        status       VARCHAR NOT NULL CHECK (status IN ('pass', 'fail', 'skipped')),
        severity     VARCHAR CHECK (severity IN ('critical', 'high', 'medium', 'low')),
        observed     VARCHAR,
        expected     VARCHAR,
        detail       VARCHAR,
        baseline     VARCHAR,
        PRIMARY KEY (run_id, dataset, period, "check", column_name),
        CHECK ((status = 'fail') = (severity IS NOT NULL))
    )
    """,
    # Schema version 7: Silver and Quarantine files written per run (Phase 4).
    """
    CREATE TABLE IF NOT EXISTS silver_log (
        run_id           VARCHAR NOT NULL,
        dataset          VARCHAR NOT NULL,
        period           VARCHAR NOT NULL,
        sha256           VARCHAR NOT NULL,
        status           VARCHAR NOT NULL CHECK (status IN ('processed', 'skipped')),
        rows             BIGINT NOT NULL,
        silver_rows      BIGINT NOT NULL,
        quarantine_rows  BIGINT NOT NULL,
        warned_rows      BIGINT NOT NULL,
        silver_path      VARCHAR NOT NULL,
        quarantine_path  VARCHAR NOT NULL,
        logged_at        TIMESTAMP NOT NULL,
        PRIMARY KEY (run_id, dataset, period),
        CHECK (silver_rows + quarantine_rows = rows),
        CHECK (warned_rows <= silver_rows)
    )
    """,
    # Schema version 8: incidents (Phase 4). One row per incident; at most one incident per
    # (dataset, source, rule_id, column_name) is not RESOLVED (enforced in incident_manager).
    f"""
    CREATE TABLE IF NOT EXISTS incidents (
        incident_id         VARCHAR PRIMARY KEY,     -- INC-0001, INC-0002, ...
        dataset             VARCHAR NOT NULL,
        source              VARCHAR NOT NULL CHECK (source IN {_in(INCIDENT_SOURCES)}),
        rule_id             VARCHAR NOT NULL,        -- row rule id or dataset check name
        column_name         VARCHAR NOT NULL DEFAULT '',
        title               VARCHAR NOT NULL,
        severity            VARCHAR NOT NULL CHECK (severity IN {_in(SEVERITY_LEVELS)}),
        status              VARCHAR NOT NULL CHECK (status IN {_in(INCIDENT_STATUSES)}),
        first_seen_period   VARCHAR NOT NULL,
        last_seen_period    VARCHAR NOT NULL,
        first_seen_run      VARCHAR NOT NULL,
        last_seen_run       VARCHAR NOT NULL,
        occurrences         INTEGER NOT NULL DEFAULT 1 CHECK (occurrences >= 1),
        clean_streak        INTEGER NOT NULL DEFAULT 0 CHECK (clean_streak >= 0),
        latest_observed     VARCHAR,
        latest_expected     VARCHAR,
        latest_detail       VARCHAR,
        recurrence_of       VARCHAR,                 -- earlier incident of the same problem
        opened_at           TIMESTAMP NOT NULL,
        updated_at          TIMESTAMP NOT NULL,
        resolved_at         TIMESTAMP,
        resolved_period     VARCHAR,
        CHECK ((status = 'RESOLVED') = (resolved_at IS NOT NULL))
    )
    """,
    # Every change to an incident, in order: the incident's history.
    f"""
    CREATE TABLE IF NOT EXISTS incident_events (
        incident_id   VARCHAR NOT NULL,
        seq           INTEGER NOT NULL,              -- 1, 2, 3, ... per incident
        event         VARCHAR NOT NULL CHECK (event IN {_in(INCIDENT_EVENTS)}),
        period        VARCHAR,
        run_id        VARCHAR,
        severity      VARCHAR,
        status_after  VARCHAR NOT NULL,
        detail        VARCHAR,
        created_at    TIMESTAMP NOT NULL,
        PRIMARY KEY (incident_id, seq)
    )
    """,
    # Schema version 9: quality score per month and file version (Phase 4).
    """
    CREATE TABLE IF NOT EXISTS quality_scores (
        dataset      VARCHAR NOT NULL,
        period       VARCHAR NOT NULL,
        sha256       VARCHAR NOT NULL,
        score        DOUBLE NOT NULL CHECK (score BETWEEN 0 AND 100),
        rows         BIGINT NOT NULL,
        check_run    VARCHAR,                      -- dataset check run used, if any
        computed_at  TIMESTAMP NOT NULL,
        PRIMARY KEY (dataset, period, sha256)
    )
    """,
    # The parts the score is made of, one row per category.
    """
    CREATE TABLE IF NOT EXISTS quality_score_parts (
        dataset      VARCHAR NOT NULL,
        period       VARCHAR NOT NULL,
        sha256       VARCHAR NOT NULL,
        category     VARCHAR NOT NULL,
        weight       DOUBLE NOT NULL,
        pass_rate    DOUBLE NOT NULL CHECK (pass_rate BETWEEN 0 AND 1),
        source       VARCHAR NOT NULL,             -- what the pass rate was computed from
        PRIMARY KEY (dataset, period, sha256, category)
    )
    """,
    # Schema version 10: one row per Gold table per build (Phase 5). The checksum covers the
    # table's content without run ids and timestamps, so two builds of the same data match.
    """
    CREATE TABLE IF NOT EXISTS gold_build_log (
        run_id      VARCHAR NOT NULL,
        table_name  VARCHAR NOT NULL,
        rows        BIGINT NOT NULL,
        checksum    VARCHAR NOT NULL,
        built_at    TIMESTAMP NOT NULL,
        PRIMARY KEY (run_id, table_name)
    )
    """,
]


def database_path() -> Path:
    return load_datasets_config().paths["duckdb"]


def connect(path: Path | str | None = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open the DataGuard database (default: paths.duckdb in configs/datasets.yml).

    Use read_only=True for readers such as the dashboard: DuckDB allows one writing
    process at a time, and a read-only connection never blocks the pipeline's writes.
    """
    path = Path(path) if path else database_path()
    if read_only:
        if not path.exists():
            raise FileNotFoundError(f"database not found: {path} (run the pipeline first)")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only)


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Create the control tables if they do not exist. Safe to run any number of times."""
    for statement in SCHEMA:
        con.execute(statement)
    for version in range(1, SCHEMA_VERSION + 1):
        con.execute(
            "INSERT INTO schema_version (version) SELECT ? WHERE NOT EXISTS "
            "(SELECT 1 FROM schema_version WHERE version = ?)",
            [version, version],
        )


def list_tables(con: duckdb.DuckDBPyConnection) -> list[str]:
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' ORDER BY table_name"
    ).fetchall()
    return [name for (name,) in rows]


def main() -> int:
    path = database_path()
    with connect(path) as con:
        init_schema(con)
        print(f"database: {path}")
        for table in list_tables(con):
            count = con.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
            print(f"  {table:<16} {count} row(s)")
        runs = con.execute(
            "SELECT run_id, dataset, status, started_at, rows_loaded, error_message "
            "FROM pipeline_runs ORDER BY started_at DESC, run_id DESC LIMIT 5"
        ).df()
        if not runs.empty:
            print("\nlatest runs (UTC):")
            print(runs.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
