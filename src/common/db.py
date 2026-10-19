"""DuckDB connection and control tables (data/dataguard.duckdb).

Control tables record what the pipeline did, so every Bronze, Silver and Quarantine
row can be traced back to a run and a source file.

    python src/common/db.py          # create the database and tables, then list them
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

if __package__ in (None, ""):
    # Allow running this file directly: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import load_datasets_config  # noqa: E402

SCHEMA_VERSION = 1

RUN_STATUSES = ("RUNNING", "SUCCESS", "WARNING", "FAILED")
FILE_STATUSES = ("REGISTERED", "LOADED", "SKIPPED", "FAILED")


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
    con.execute(
        "INSERT INTO schema_version (version) SELECT ? WHERE NOT EXISTS "
        "(SELECT 1 FROM schema_version WHERE version = ?)",
        [SCHEMA_VERSION, SCHEMA_VERSION],
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
