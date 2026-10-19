import shutil
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from common import db
from common.runs import get_run
from ingestion import ingest_data, manifest

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "yellow_tripdata_fixture.parquet"


@pytest.fixture
def dirs(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    shutil.copy(FIXTURE, incoming / "yellow_tripdata_2023-01.parquet")
    # February in the 2023-02 schema: Airport_fee renamed, 150 rows.
    duckdb.sql(
        f"COPY (SELECT * EXCLUDE (airport_fee), airport_fee AS Airport_fee "
        f"FROM read_parquet('{FIXTURE}') LIMIT 150) "
        f"TO '{incoming / 'yellow_tripdata_2023-02.parquet'}' (FORMAT parquet)"
    )
    return {
        "incoming": incoming,
        "bronze": tmp_path / "bronze",
        "database": tmp_path / "dataguard.duckdb",
    }


def run_main(dirs, *extra):
    argv = ["--source", str(dirs["incoming"]), "--bronze", str(dirs["bronze"])]
    argv += ["--database", str(dirs["database"]), *extra]
    return ingest_data.main(argv)


def query(dirs, sql):
    with db.connect(dirs["database"], read_only=True) as con:
        return con.execute(sql).fetchall()


def runs(dirs):
    return query(
        dirs,
        "SELECT run_id, status, files_total, files_loaded, files_skipped, rows_loaded "
        "FROM pipeline_runs ORDER BY run_id",
    )


def bronze_rows(dirs):
    pattern = (dirs["bronze"] / "yellow_taxi" / "**" / "*.parquet").as_posix()
    return duckdb.sql(
        f"SELECT count(*) FROM read_parquet('{pattern}', union_by_name = true)"
    ).fetchone()[0]


def test_first_run_loads_every_month(dirs):
    assert run_main(dirs, "--months", "2023-01:2023-02") == 0
    assert runs(dirs) == [("run_0001", "SUCCESS", 2, 2, 0, 350)]
    assert bronze_rows(dirs) == 350
    statuses = query(
        dirs, "SELECT period, status, loaded_run_id FROM file_manifest ORDER BY period"
    )
    assert statuses == [("2023-01", "LOADED", "run_0001"), ("2023-02", "LOADED", "run_0001")]


def test_second_run_changes_nothing(dirs):
    run_main(dirs, "--months", "2023-01:2023-02")
    assert run_main(dirs, "--months", "2023-01:2023-02") == 0
    assert runs(dirs)[1] == ("run_0002", "SUCCESS", 2, 0, 2, 0)
    assert bronze_rows(dirs) == 350  # no duplicates in Bronze
    assert query(dirs, "SELECT count(*) FROM file_manifest") == [(2,)]


def test_force_rewrites_the_same_version(dirs):
    run_main(dirs, "--months", "2023-01")
    assert run_main(dirs, "--months", "2023-01", "--force") == 0
    assert runs(dirs)[1] == ("run_0002", "SUCCESS", 1, 1, 0, 200)
    assert bronze_rows(dirs) == 200  # rewritten in place, not added
    assert query(dirs, "SELECT loaded_run_id FROM file_manifest") == [("run_0002",)]


def test_corrected_file_is_loaded_as_new_version(dirs):
    run_main(dirs, "--months", "2023-01")
    path = dirs["incoming"] / "yellow_tripdata_2023-01.parquet"
    trips = pd.read_parquet(path)
    trips.loc[0, "fare_amount"] += 1.0
    trips.to_parquet(path, index=False)
    assert run_main(dirs, "--months", "2023-01") == 0
    assert runs(dirs)[1][3] == 1  # loaded again
    assert query(dirs, "SELECT count(*) FROM file_manifest WHERE status = 'LOADED'") == [(2,)]
    assert bronze_rows(dirs) == 400  # both versions kept in Bronze


def test_missing_month_gives_warning(dirs):
    assert run_main(dirs, "--months", "2023-01:2023-03") == 0
    run_id, status, total, loaded, skipped, rows = runs(dirs)[0]
    assert (status, total, loaded, skipped, rows) == ("WARNING", 3, 2, 1, 350)
    assert "yellow_tripdata_2023-03.parquet missing" in get_run_error(dirs, run_id)


def test_only_missing_months_exit_with_error(dirs):
    assert run_main(dirs, "--months", "2023-05") == 1
    assert runs(dirs)[0][1] == "WARNING"


def test_bad_file_fails_the_run_but_others_load(dirs):
    (dirs["incoming"] / "yellow_tripdata_2023-02.parquet").write_bytes(b"PAR1 broken PAR1")
    assert run_main(dirs, "--months", "2023-01:2023-02") == 1
    run_id, status, total, loaded, skipped, rows = runs(dirs)[0]
    assert (status, loaded, rows) == ("FAILED", 1, 200)
    assert "yellow_tripdata_2023-02.parquet" in get_run_error(dirs, run_id)
    assert query(dirs, "SELECT period, status FROM file_manifest") == [("2023-01", "LOADED")]


def test_bronze_deleted_by_hand_is_reloaded(dirs):
    run_main(dirs, "--months", "2023-01")
    shutil.rmtree(dirs["bronze"])
    assert run_main(dirs, "--months", "2023-01") == 0
    assert runs(dirs)[1][3] == 1
    assert bronze_rows(dirs) == 200


def test_invalid_arguments(dirs):
    assert run_main(dirs, "--months", "2023-13") == 2
    assert run_main(dirs, "--dataset", "green_taxi") == 2


def test_command_is_recorded(dirs):
    run_main(dirs, "--months", "2023-01")
    command = query(dirs, "SELECT command FROM pipeline_runs")[0][0]
    assert command.startswith("ingest_data.py ") and "--months 2023-01" in command


def get_run_error(dirs, run_id):
    with db.connect(dirs["database"], read_only=True) as con:
        return get_run(con, run_id)["error_message"]


def test_manifest_entries_are_shared_with_register_command(dirs):
    # Files registered earlier by manifest.py are reused, not registered twice.
    manifest.main(
        [
            "--months",
            "2023-01",
            "--source",
            str(dirs["incoming"]),
            "--database",
            str(dirs["database"]),
        ]
    )
    run_main(dirs, "--months", "2023-01")
    assert query(dirs, "SELECT count(*), max(status) FROM file_manifest") == [(1, "LOADED")]
