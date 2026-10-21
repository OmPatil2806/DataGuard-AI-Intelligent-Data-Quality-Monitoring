import shutil
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from common import db
from common.runs import get_run
from gold import monitoring_tables as gold
from gold.monitoring_tables import GoldTable
from ingestion import ingest_data

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "yellow_tripdata_fixture.parquet"


@pytest.fixture
def dirs(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    shutil.copy(FIXTURE, incoming / "yellow_tripdata_2023-01.parquet")
    shutil.copy(FIXTURE, incoming / "yellow_tripdata_2023-02.parquet")
    return {"incoming": incoming, "bronze": tmp_path / "bronze", "db": tmp_path / "d.duckdb"}


def ingest(dirs, months="2023-01:2023-02"):
    argv = ["--months", months, "--source", str(dirs["incoming"]), "--bronze", str(dirs["bronze"])]
    assert ingest_data.main([*argv, "--database", str(dirs["db"])]) == 0


@pytest.fixture
def con(dirs):
    ingest(dirs)
    with db.connect(dirs["db"]) as connection:
        db.init_schema(connection)
        yield connection


def test_months_table_holds_the_current_version(dirs, con):
    gold.rebuild(con)
    months = con.execute("SELECT period, bronze_rows FROM gold_months ORDER BY period").fetchall()
    assert months == [("2023-01", 200), ("2023-02", 200)]

    path = dirs["incoming"] / "yellow_tripdata_2023-01.parquet"  # a corrected re-export
    trips = pd.read_parquet(path)
    trips.loc[0, "fare_amount"] += 1.0
    trips.to_parquet(path, index=False)
    con.close()
    ingest(dirs, "2023-01")
    with db.connect(dirs["db"]) as con:
        gold.rebuild(con)
        sha = con.execute("SELECT sha256 FROM gold_months WHERE period = '2023-01'").fetchone()[0]
        newest = con.execute(
            "SELECT sha256 FROM file_manifest WHERE period = '2023-01' "
            "ORDER BY registered_at DESC LIMIT 1"
        ).fetchone()[0]
        assert sha == newest
        assert con.execute("SELECT count(*) FROM gold_months").fetchone()[0] == 2


def test_build_is_logged_in_a_tracked_run(con):
    run_id, built = gold.rebuild(con)
    assert get_run(con, run_id)["status"] == "SUCCESS"
    assert get_run(con, run_id)["dataset"] == "gold"
    log = gold.latest_build(con)
    assert log["run_id"].unique().tolist() == [run_id]
    assert log["table_name"].tolist() == sorted(t.name for t in gold.TABLES)
    assert {b.name: b.rows for b in built} == dict(zip(log["table_name"], log["rows"], strict=True))


def test_rebuild_gives_identical_checksums(con):
    first_run, first = gold.rebuild(con)
    second_run, second = gold.rebuild(con)
    assert first_run != second_run
    assert first == second
    # the volatile columns (run id, registered_at) are left out; changing them changes nothing
    con.execute("UPDATE gold_months SET loaded_run_id = 'run_9999', registered_at = now()")
    assert gold.checksum(con, "gold_months", gold.MONTHS.volatile) == first[0].checksum


def test_checksum_ignores_row_order_but_not_content():
    con = duckdb.connect()
    con.execute("CREATE TABLE a AS SELECT * FROM (VALUES (1, 'x'), (2, 'y')) t(n, s)")
    con.execute("CREATE TABLE b AS SELECT * FROM (VALUES (2, 'y'), (1, 'x')) t(n, s)")
    con.execute("CREATE TABLE c AS SELECT * FROM (VALUES (1, 'x'), (2, 'z')) t(n, s)")
    con.execute("CREATE TABLE e AS SELECT * FROM a WHERE false")
    assert gold.checksum(con, "a") == gold.checksum(con, "b")
    assert gold.checksum(con, "a") != gold.checksum(con, "c")
    assert gold.checksum(con, "a", ("s",)) == gold.checksum(con, "c", ("s",))
    assert gold.checksum(con, "e")  # an empty table still has a checksum


def test_failed_build_keeps_the_previous_gold(con):
    run_id, before = gold.rebuild(con)
    broken = GoldTable("gold_broken", "fails", "SELECT * FROM no_such_table", "1")
    with pytest.raises(duckdb.Error):
        gold.rebuild(con, [gold.MONTHS, broken])
    failed = con.execute("SELECT max(run_id) FROM pipeline_runs").fetchone()[0]
    assert get_run(con, failed)["status"] == "FAILED"
    assert "gold_broken" not in db.list_tables(con)
    assert (
        con.execute("SELECT count(*) FROM gold_build_log WHERE run_id = ?", [failed]).fetchone()[0]
        == 0
    )
    assert gold.latest_build(con)["run_id"].unique().tolist() == [run_id]
    assert gold.checksum(con, "gold_months", gold.MONTHS.volatile) == before[0].checksum


@pytest.mark.parametrize("name", ["months", "gold_Months", "gold_build_log", "gold_x; DROP"])
def test_invalid_table_names(name):
    with pytest.raises(ValueError, match="invalid Gold table name"):
        GoldTable(name, "", "SELECT 1", "1")


def test_duplicate_table_names(con):
    with pytest.raises(ValueError, match="unique"):
        gold.build(con, "run_x", [gold.MONTHS, gold.MONTHS])


def test_main(dirs, capsys):
    ingest(dirs)
    assert gold.main(["--database", str(dirs["db"])]) == 0
    out = capsys.readouterr().out
    assert (
        f"{len(gold.TABLES)} Gold table(s) built" in out
        and "gold_months" in out
        and "2 row(s)" in out
    )
    assert gold.main(["--list"]) == 0
    assert "gold_months" in capsys.readouterr().out


def test_main_on_an_empty_database(tmp_path, capsys):
    assert gold.main(["--database", str(tmp_path / "e.duckdb")]) == 0
    assert "0 row(s)" in capsys.readouterr().out
