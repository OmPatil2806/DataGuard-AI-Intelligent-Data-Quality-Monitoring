from datetime import datetime, timedelta

import pytest

from common import db
from common.runs import (
    INTERRUPTED,
    MAX_ERROR_LENGTH,
    RunError,
    add_counts,
    finish_run,
    get_run,
    next_run_id,
    start_run,
    track_run,
)

T0 = datetime(2023, 3, 1, 6, 0, 0)


@pytest.fixture
def con(tmp_path):
    with db.connect(tmp_path / "dataguard.duckdb") as connection:
        db.init_schema(connection)
        yield connection


def ticking_clock(start=T0, step=timedelta(seconds=30)):
    times = iter(start + i * step for i in range(1000))
    return lambda: next(times)


def test_run_ids_count_up(con):
    assert next_run_id(con) == "run_0001"
    assert start_run(con, "yellow_taxi", now=T0) == "run_0001"
    finish_run(con, "run_0001", "SUCCESS", now=T0)
    assert start_run(con, "yellow_taxi", now=T0) == "run_0002"


def test_run_ids_go_past_9999(con):
    con.execute(
        "INSERT INTO pipeline_runs (run_id, dataset, status, started_at) "
        "VALUES ('run_9999', 'yellow_taxi', 'SUCCESS', ?)",
        [T0],
    )
    assert next_run_id(con) == "run_10000"


def test_start_records_running_run(con):
    run_id = start_run(con, "yellow_taxi", command="ingest --months 2023-01", now=T0)
    run = get_run(con, run_id)
    assert run["status"] == "RUNNING"
    assert run["command"] == "ingest --months 2023-01"
    assert run["started_at"] == T0
    assert run["finished_at"] is None


def test_counts_accumulate(con):
    run_id = start_run(con, "yellow_taxi", now=T0)
    add_counts(con, run_id, files_total=2)
    add_counts(con, run_id, files_loaded=1, rows_loaded=100)
    add_counts(con, run_id, files_loaded=1, rows_loaded=50)
    run = get_run(con, run_id)
    assert (run["files_total"], run["files_loaded"], run["rows_loaded"]) == (2, 2, 150)


def test_counts_reject_unknown_and_negative(con):
    run_id = start_run(con, "yellow_taxi", now=T0)
    with pytest.raises(RunError, match="unknown counter"):
        add_counts(con, run_id, rows=1)
    with pytest.raises(RunError, match="only increase"):
        add_counts(con, run_id, rows_loaded=-1)


def test_finish_rules(con):
    run_id = start_run(con, "yellow_taxi", now=T0)
    with pytest.raises(RunError, match="status must be one of"):
        finish_run(con, run_id, "RUNNING")
    with pytest.raises(RunError, match="needs an error message"):
        finish_run(con, run_id, "FAILED")
    finish_run(con, run_id, "SUCCESS", now=T0 + timedelta(minutes=2))
    with pytest.raises(RunError, match="already finished"):
        finish_run(con, run_id, "SUCCESS")
    with pytest.raises(RunError, match="already finished"):
        add_counts(con, run_id, files_loaded=1)
    with pytest.raises(RunError, match="unknown run"):
        get_run(con, "run_0099")


def test_long_errors_are_truncated(con):
    run_id = start_run(con, "yellow_taxi", now=T0)
    finish_run(con, run_id, "FAILED", error="x" * 5000, now=T0)
    error = get_run(con, run_id)["error_message"]
    assert len(error) == MAX_ERROR_LENGTH
    assert error.endswith("...")


def test_track_run_success(con):
    with track_run(con, "yellow_taxi", command="test", clock=ticking_clock()) as run:
        run.add(files_total=1, files_loaded=1, rows_loaded=3_066_766)
    stored = get_run(con, run.run_id)
    assert stored["status"] == "SUCCESS"
    assert stored["rows_loaded"] == 3_066_766
    assert stored["finished_at"] - stored["started_at"] == timedelta(seconds=30)
    assert stored["error_message"] is None


def test_track_run_warning(con):
    with track_run(con, "yellow_taxi", clock=ticking_clock()) as run:
        run.warn("2023-01 arrived late")
        run.warn("2023-02 is small")
    stored = get_run(con, run.run_id)
    assert stored["status"] == "WARNING"
    assert stored["error_message"] == "2023-01 arrived late; 2023-02 is small"


def test_track_run_failure_is_recorded_and_reraised(con):
    with pytest.raises(FileNotFoundError):
        with track_run(con, "yellow_taxi", clock=ticking_clock()) as run:
            run.add(files_loaded=1)
            raise FileNotFoundError("yellow_tripdata_2023-13.parquet")
    stored = get_run(con, run.run_id)
    assert stored["status"] == "FAILED"
    assert stored["error_message"] == "FileNotFoundError: yellow_tripdata_2023-13.parquet"
    assert stored["files_loaded"] == 1  # work done before the failure stays visible


def test_track_run_records_keyboard_interrupt(con):
    with pytest.raises(KeyboardInterrupt):
        with track_run(con, "yellow_taxi", clock=ticking_clock()) as run:
            raise KeyboardInterrupt
    assert get_run(con, run.run_id)["error_message"] == "KeyboardInterrupt"


def test_interrupted_run_is_failed_by_the_next_start(con):
    killed = start_run(con, "yellow_taxi", now=T0)  # process killed: never finished
    other = start_run(con, "ved_signals", now=T0)
    new = start_run(con, "yellow_taxi", now=T0 + timedelta(hours=1))
    stale = get_run(con, killed)
    assert stale["status"] == "FAILED"
    assert stale["error_message"] == INTERRUPTED
    assert stale["finished_at"] == T0 + timedelta(hours=1)
    assert get_run(con, other)["status"] == "RUNNING"  # other datasets are left alone
    assert get_run(con, new)["status"] == "RUNNING"
