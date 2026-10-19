import shutil
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from common import db
from ingestion import ingest_data, verify_bronze

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "yellow_tripdata_fixture.parquet"


@pytest.fixture
def setup(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    shutil.copy(FIXTURE, incoming / "yellow_tripdata_2023-01.parquet")
    duckdb.sql(
        f"COPY (SELECT * EXCLUDE (airport_fee), airport_fee AS Airport_fee "
        f"FROM read_parquet('{FIXTURE}') LIMIT 150) "
        f"TO '{incoming / 'yellow_tripdata_2023-02.parquet'}' (FORMAT parquet)"
    )
    paths = {"incoming": incoming, "bronze": tmp_path / "bronze", "db": tmp_path / "d.duckdb"}
    argv = ["--months", "2023-01:2023-02", "--source", str(incoming)]
    argv += ["--bronze", str(paths["bronze"]), "--database", str(paths["db"])]
    assert ingest_data.main(argv) == 0
    return paths


def verify(paths):
    with db.connect(paths["db"], read_only=True) as con:
        return verify_bronze.verify(con, "yellow_taxi", paths["incoming"])


def outcomes(results):
    return {(r.period, r.check): r.passed for r in results}


def bronze_file(paths, period):
    return next((paths["bronze"] / "yellow_taxi" / f"period={period}").rglob("*.parquet"))


def test_clean_bronze_passes_every_check(setup):
    results = verify(setup)
    assert len(results) == 10  # 2 files x 5 checks
    assert all(r.passed for r in results)
    table = verify_bronze.report(results)
    assert table["period"].tolist() == ["2023-01", "2023-02"]
    assert set(table[list(verify_bronze.CHECKS)].to_numpy().ravel()) == {"PASS"}


def test_main_exit_codes(setup, capsys):
    argv = ["--source", str(setup["incoming"]), "--database", str(setup["db"])]
    assert verify_bronze.main(argv) == 0
    assert "2 file version(s) checked, 0 failure(s)" in capsys.readouterr().out


def test_lost_rows_are_detected(setup):
    path = bronze_file(setup, "2023-01")
    duckdb.sql(
        f"COPY (SELECT * FROM read_parquet('{path}') WHERE bronze_row_id <= 190) "
        f"TO '{path}.new' (FORMAT parquet)"
    )
    Path(f"{path}.new").replace(path)
    checks = outcomes(verify(setup))
    assert checks[("2023-01", "rows")] is False
    assert checks[("2023-02", "rows")] is True


def test_duplicated_rows_are_detected(setup):
    path = bronze_file(setup, "2023-02")
    duckdb.sql(
        f"COPY (SELECT * FROM read_parquet('{path}') UNION ALL "
        f"SELECT * FROM read_parquet('{path}') WHERE bronze_row_id = 1) "
        f"TO '{path}.new' (FORMAT parquet)"
    )
    Path(f"{path}.new").replace(path)
    checks = outcomes(verify(setup))
    assert checks[("2023-02", "rows")] is False
    assert checks[("2023-02", "row_ids")] is False


def test_changed_type_is_detected(setup):
    path = bronze_file(setup, "2023-01")
    duckdb.sql(
        f"COPY (SELECT * REPLACE (CAST(passenger_count AS BIGINT) AS passenger_count) "
        f"FROM read_parquet('{path}')) TO '{path}.new' (FORMAT parquet)"
    )
    Path(f"{path}.new").replace(path)
    assert outcomes(verify(setup))[("2023-01", "schema")] is False


def test_wrong_lineage_is_detected(setup):
    path = bronze_file(setup, "2023-01")
    duckdb.sql(
        f"COPY (SELECT * REPLACE ('run_9999' AS pipeline_run_id) FROM read_parquet('{path}')) "
        f"TO '{path}.new' (FORMAT parquet)"
    )
    Path(f"{path}.new").replace(path)
    assert outcomes(verify(setup))[("2023-01", "lineage")] is False


def test_missing_bronze_file_fails(setup, capsys):
    bronze_file(setup, "2023-01").unlink()
    results = verify(setup)
    assert outcomes(results)[("2023-01", "rows")] is False
    argv = ["--source", str(setup["incoming"]), "--database", str(setup["db"])]
    assert verify_bronze.main(argv) == 1
    assert "Bronze file missing" in capsys.readouterr().out


def test_source_gone_or_replaced_is_not_a_failure(setup):
    (setup["incoming"] / "yellow_tripdata_2023-01.parquet").unlink()
    path = setup["incoming"] / "yellow_tripdata_2023-02.parquet"
    trips = pd.read_parquet(path)
    trips.loc[0, "fare_amount"] += 1.0
    trips.to_parquet(path, index=False)  # a newer download, not yet ingested
    checks = outcomes(verify(setup))
    assert checks[("2023-01", "schema")] is None
    assert checks[("2023-01", "checksum")] is None
    assert checks[("2023-02", "checksum")] is None
    assert all(passed is not False for passed in checks.values())


def test_nothing_loaded(tmp_path, capsys):
    database = tmp_path / "empty.duckdb"
    with db.connect(database) as con:
        db.init_schema(con)
    assert verify_bronze.main(["--database", str(database)]) == 1
    assert "nothing to verify" in capsys.readouterr().out
