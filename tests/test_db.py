from datetime import datetime

import duckdb
import pytest

from common import db

NOW = datetime(2023, 3, 1, 6, 0, 0)


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "nested" / "dataguard.duckdb"


@pytest.fixture
def con(db_path):
    with db.connect(db_path) as connection:
        db.init_schema(connection)
        yield connection


def columns(con, table):
    return [r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()]


def insert_run(con, run_id="run_0001", status="RUNNING", started=NOW, finished=None):
    con.execute(
        "INSERT INTO pipeline_runs (run_id, dataset, status, started_at, finished_at) "
        "VALUES (?, 'yellow_taxi', ?, ?, ?)",
        [run_id, status, started, finished],
    )


def insert_file(con, sha="abc", status="REGISTERED", period="2023-01"):
    con.execute(
        "INSERT INTO file_manifest (dataset, period, file_name, sha256, size_bytes, "
        "registered_at, status) VALUES ('yellow_taxi', ?, 'f.parquet', ?, 10, ?, ?)",
        [period, sha, NOW, status],
    )


def test_connect_creates_parent_folder(db_path):
    with db.connect(db_path):
        pass
    assert db_path.exists()


def test_init_creates_control_tables(con):
    assert db.list_tables(con) == ["file_manifest", "pipeline_runs", "schema_version"]
    assert "rows_loaded" in columns(con, "pipeline_runs")
    assert {"sha256", "expected_at", "delivered_at", "loaded_run_id"} <= set(
        columns(con, "file_manifest")
    )


def test_init_is_idempotent_and_keeps_data(con):
    insert_run(con)
    db.init_schema(con)
    db.init_schema(con)
    assert con.execute("SELECT count(*) FROM pipeline_runs").fetchone()[0] == 1
    assert con.execute("SELECT version FROM schema_version").fetchall() == [(db.SCHEMA_VERSION,)]


def test_run_defaults(con):
    insert_run(con)
    row = con.execute(
        "SELECT files_total, files_loaded, files_skipped, rows_loaded FROM pipeline_runs"
    ).fetchone()
    assert row == (0, 0, 0, 0)


def test_run_status_must_be_known(con):
    with pytest.raises(duckdb.ConstraintException):
        insert_run(con, status="DONE")


def test_run_cannot_finish_before_it_starts(con):
    with pytest.raises(duckdb.ConstraintException):
        insert_run(con, finished=datetime(2023, 2, 1))


def test_run_id_is_unique(con):
    insert_run(con)
    with pytest.raises(duckdb.ConstraintException):
        insert_run(con)


def test_file_status_must_be_known(con):
    with pytest.raises(duckdb.ConstraintException):
        insert_file(con, status="DONE")


def test_same_file_version_registered_once(con):
    insert_file(con, sha="abc")
    with pytest.raises(duckdb.ConstraintException):
        insert_file(con, sha="abc")
    insert_file(con, sha="def")  # a corrected re-export of the same month is a new version
    insert_file(con, sha="abc", period="2023-02")
    assert con.execute("SELECT count(*) FROM file_manifest").fetchone()[0] == 3


def test_read_only_connection_reads_but_cannot_write(con, db_path):
    insert_run(con)
    con.close()
    with db.connect(db_path, read_only=True) as reader:
        assert reader.execute("SELECT count(*) FROM pipeline_runs").fetchone()[0] == 1
        with pytest.raises(duckdb.Error):
            insert_run(reader, run_id="run_0002")


def test_read_only_needs_existing_database(tmp_path):
    with pytest.raises(FileNotFoundError, match="run the pipeline first"):
        db.connect(tmp_path / "missing.duckdb", read_only=True)
