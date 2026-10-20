import json
import shutil
from pathlib import Path

import duckdb
import pytest

from common import db
from ingestion import ingest_data
from quality import run_rules
from quality.rules import load_rules

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ISSUES = json.loads((FIXTURES / "yellow_tripdata_fixture_issues.json").read_text())
# Rows the real rules quarantine / only warn in the fixture (see tests/test_taxi_rules.py).
QUARANTINED = {
    r
    for k in (
        "negative_fare",
        "dropoff_before_pickup",
        "pickup_outside_month",
        "unknown_location_id",
        "duplicate_row",
    )
    for r in ISSUES[k]
}
WARNED = {
    r
    for k in ("zero_passengers", "null_group", "zero_distance", "unknown_zone_264_265")
    for r in ISSUES[k]
} - QUARANTINED


@pytest.fixture
def paths(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    shutil.copy(
        FIXTURES / "yellow_tripdata_fixture.parquet", incoming / "yellow_tripdata_2023-01.parquet"
    )
    shutil.copy(FIXTURES / "taxi_zone_lookup_fixture.csv", incoming / "taxi_zone_lookup.csv")
    # February in the 2023-02 schema, so standardization has work to do.
    duckdb.sql(
        f"COPY (SELECT * EXCLUDE (airport_fee), airport_fee AS Airport_fee "
        f"FROM read_parquet('{incoming / 'yellow_tripdata_2023-01.parquet'}')) "
        f"TO '{incoming / 'yellow_tripdata_2023-02.parquet'}' (FORMAT parquet)"
    )
    p = {"incoming": incoming, "bronze": tmp_path / "bronze", "db": tmp_path / "d.duckdb"}
    argv = ["--months", "2023-01:2023-02", "--source", str(incoming)]
    argv += ["--bronze", str(p["bronze"]), "--database", str(p["db"])]
    assert ingest_data.main(argv) == 0
    return p


def run(paths, *extra):
    argv = ["--lookups", str(paths["incoming"]), "--database", str(paths["db"]), *extra]
    return run_rules.main(argv)


def query(paths, sql):
    with db.connect(paths["db"], read_only=True) as con:
        return con.execute(sql).fetchall()


def test_checks_every_month_in_bronze(paths):
    assert run(paths) == 0
    runs = query(
        paths,
        "SELECT run_id, status, files_total, files_loaded, rows_loaded, command "
        "FROM pipeline_runs ORDER BY run_id",
    )
    assert runs[-1][:5] == ("run_0002", "SUCCESS", 2, 2, 400)  # run_0001 was the ingest
    assert runs[-1][5].startswith("run_rules.py")


def test_summary_matches_the_answer_key(paths):
    run(paths)
    rows = query(
        paths,
        "SELECT period, rows_checked, rows_quarantined, rows_warned "
        "FROM rule_run_summary ORDER BY period",
    )
    assert rows[0] == ("2023-01", 200, len(QUARANTINED), len(WARNED))


def test_file_under_the_wrong_month_is_quarantined(paths):
    """The '2023-02' file holds January trips: every pickup is outside its month."""
    run(paths)
    assert query(
        paths, "SELECT rows_quarantined FROM rule_run_summary WHERE period = '2023-02'"
    ) == [(200,)]
    failed = dict(
        query(paths, "SELECT rule_id, rows_failed FROM rule_run_log WHERE period = '2023-02'")
    )
    assert failed["pickup_in_file_month"] == 200


def test_rule_log_has_every_enabled_rule(paths):
    run(paths)
    rules = load_rules("yellow_taxi")
    logged = query(paths, "SELECT rule_id, rows_failed FROM rule_run_log WHERE period = '2023-01'")
    assert sorted(r for r, _ in logged) == sorted(r.id for r in rules)
    failed = dict(logged)
    assert failed["fare_not_negative"] == len(ISSUES["negative_fare"])
    assert failed["trip_unique"] == len(ISSUES["duplicate_row"])
    assert failed["distance_max_100_miles"] == 0
    assert "total_matches_parts" not in failed  # pending O1, disabled


def test_standardization_is_recorded(paths):
    run(paths)
    notes = dict(query(paths, "SELECT period, standardization FROM rule_run_summary"))
    assert "renamed Airport_fee->airport_fee" in notes["2023-02"]
    assert "renamed" not in notes["2023-01"]


def test_versions_are_recorded(paths):
    run(paths)
    shas = query(
        paths,
        "SELECT DISTINCT s.period, s.sha256 = m.sha256 FROM rule_run_summary s "
        "JOIN file_manifest m USING (dataset, period)",
    )
    assert sorted(shas) == [("2023-01", True), ("2023-02", True)]


def test_each_run_adds_its_own_results(paths):
    run(paths)
    run(paths, "--months", "2023-02")
    assert query(
        paths, "SELECT run_id, count(*) FROM rule_run_summary GROUP BY run_id ORDER BY run_id"
    ) == [("run_0002", 2), ("run_0003", 1)]


def test_month_missing_from_bronze_gives_warning(paths):
    assert run(paths, "--months", "2023-01:2023-03") == 0
    status, error = query(
        paths, "SELECT status, error_message FROM pipeline_runs ORDER BY run_id DESC LIMIT 1"
    )[0]
    assert status == "WARNING" and "2023-03 not in Bronze" in error


def test_only_missing_months_exit_with_error(paths):
    assert run(paths, "--months", "2023-05") == 1


def test_lossy_cast_gives_warning(paths):
    # A Bronze month where passenger_count holds 1.5: the cast to INTEGER loses it.
    feb = next((paths["bronze"] / "yellow_taxi" / "period=2023-02").rglob("*.parquet"))
    duckdb.sql(
        f"COPY (SELECT * REPLACE (CASE WHEN bronze_row_id = 1 THEN 1.5 ELSE passenger_count END "
        f"AS passenger_count) FROM read_parquet('{feb}', hive_partitioning = false)) "
        f"TO '{feb}.new' (FORMAT parquet)"
    )
    Path(f"{feb}.new").replace(feb)
    assert run(paths) == 0
    status, error = query(
        paths, "SELECT status, error_message FROM pipeline_runs ORDER BY run_id DESC LIMIT 1"
    )[0]
    assert status == "WARNING" and "LOSSY passenger_count: 1" in error


def test_unreadable_month_fails_run_but_others_are_checked(paths):
    feb = next((paths["bronze"] / "yellow_taxi" / "period=2023-02").rglob("*.parquet"))
    feb.write_bytes(b"not parquet")
    assert run(paths) == 1
    assert query(paths, "SELECT period FROM rule_run_summary") == [("2023-01",)]
    status, error = query(
        paths, "SELECT status, error_message FROM pipeline_runs ORDER BY run_id DESC LIMIT 1"
    )[0]
    assert status == "FAILED" and "2023-02" in error


def test_missing_lookup_and_bad_arguments(paths, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run_rules.main(["--lookups", str(empty), "--database", str(paths["db"])]) == 2
    assert run(paths, "--months", "2023-13") == 2
    assert run(paths, "--dataset", "green_taxi") == 2


def test_rule_table(paths):
    run(paths)
    with db.connect(paths["db"], read_only=True) as con:
        table = run_rules.rule_table(con, "run_0002")
    top = table.iloc[0]
    assert top["rows_failed"] == max(table["rows_failed"])
    assert set(table["action"]) == {"quarantine", "warn"}
