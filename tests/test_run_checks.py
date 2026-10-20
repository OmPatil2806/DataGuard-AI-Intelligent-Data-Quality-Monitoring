"""run_checks.py end to end on a Bronze with planted month-level problems."""

from dataclasses import replace
from datetime import datetime
from pathlib import Path

import duckdb
import pytest

from common import db
from common.config import load_datasets_config
from ingestion import ingest_data, manifest
from quality import baselines, run_checks

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "yellow_tripdata_fixture.parquet"
FEB_SCHEMA = (
    "SELECT * EXCLUDE (airport_fee, VendorID), airport_fee AS Airport_fee, "
    "CAST(VendorID AS INTEGER) AS VendorID FROM '{src}'"
)
MONTHS = {
    "2023-01": "SELECT * FROM '{src}'",
    "2023-02": FEB_SCHEMA,  # schema change
    "2023-03": FEB_SCHEMA,
    "2023-04": FEB_SCHEMA + " LIMIT 100",  # volume drop (-48%)
    "2023-05": (  # NULL spike + miles -> km
        "SELECT * REPLACE (CASE WHEN fare_amount > 25 THEN NULL ELSE fare_amount END "
        "AS fare_amount, trip_distance * 1.609 AS trip_distance) FROM (" + FEB_SCHEMA + ")"
    ),
}


@pytest.fixture(scope="module")
def taxi():
    return load_datasets_config().dataset("yellow_taxi")


@pytest.fixture(scope="module")
def limits(taxi):
    # The fixture has a few hundred rows a month, far below the real cold-start limits.
    return replace(baselines.settings_for(taxi.name), rows_per_day_min=1, rows_per_day_max=100)


@pytest.fixture(scope="module")
def paths(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("checks")
    incoming = tmp / "incoming"
    incoming.mkdir()
    for period, query in MONTHS.items():
        target = incoming / f"yellow_tripdata_{period}.parquet"
        duckdb.sql(f"COPY ({query.format(src=FIXTURE)}) TO '{target}' (FORMAT parquet)")
    database = tmp / "d.duckdb"
    argv = [
        "--months",
        "2023-01:2023-05",
        "--source",
        str(incoming),
        "--bronze",
        str(tmp / "bronze"),
        "--database",
        str(database),
    ]
    assert ingest_data.main(argv) == 0
    return {"incoming": incoming, "db": database, "tmp": tmp}


@pytest.fixture(scope="module")
def checked(paths, taxi, limits):
    with db.connect(paths["db"]) as con:
        run_id = run_checks.run_checks(con, taxi, limits=limits, command="test")
        results = con.execute(
            'SELECT period, "check", column_name, status, severity, detail '
            "FROM dataset_check_results WHERE run_id = ?",
            [run_id],
        ).fetchall()
        run = con.execute(
            "SELECT status, files_loaded FROM pipeline_runs WHERE run_id = ?", [run_id]
        ).fetchone()
        stored_baselines = con.execute(
            "SELECT DISTINCT period, trusted FROM baselines WHERE run_id = ? ORDER BY period",
            [run_id],
        ).fetchall()
        table = run_checks.overview(con, run_id)
        found = run_checks.findings(con, run_id)
    return {
        "run_id": run_id,
        "results": results,
        "run": run,
        "baselines": stored_baselines,
        "table": table,
        "findings": found,
    }


def result(checked, period, check, column=""):
    return next(r for r in checked["results"] if r[:3] == (period, check, column))


def test_run_succeeds_even_with_findings(checked):
    assert checked["run"] == ("SUCCESS", 5)


def test_every_month_gets_every_check(checked, taxi):
    per_month = {}
    for period, check, *_ in checked["results"]:
        per_month.setdefault(period, set()).add(check)
    assert set(per_month) == set(MONTHS)
    assert all(
        c == {"schema", "volume", "freshness", "null_rate", "drift"} for c in per_month.values()
    )
    nulls = [r for r in checked["results"] if r[0] == "2023-01" and r[1] == "null_rate"]
    assert len(nulls) == len(taxi.columns)


def test_schema_change_in_february(checked):
    period, check, column, status, severity, detail = result(checked, "2023-02", "schema")
    assert (status, severity) == ("fail", "high")
    assert (
        "renamed airport_fee -> Airport_fee" in detail
        and "VendorID type BIGINT -> INTEGER" in detail
    )
    assert result(checked, "2023-03", "schema")[3] == "pass"


def test_volume_drop_in_april(checked):
    *_, status, severity, detail = result(checked, "2023-04", "volume")
    assert (status, severity) == ("fail", "medium")
    assert detail.startswith("volume drop: -4")


def test_null_spike_and_unit_change_in_may(checked):
    *_, status, severity, detail = result(checked, "2023-05", "null_rate", "fare_amount")
    assert (status, severity) == ("fail", "high")
    assert "normally complete column" in detail
    *_, status, severity, _ = result(checked, "2023-05", "drift", "trip_distance")
    assert (status, severity) == ("fail", "high")


def test_cold_start_months_skip_drift(checked):
    for period in ("2023-01", "2023-02", "2023-03"):
        assert result(checked, period, "drift", "trip_distance")[3] == "skipped"
    assert checked["baselines"] == [
        ("2023-01", False),
        ("2023-02", False),
        ("2023-03", False),
        ("2023-04", True),
        ("2023-05", True),
    ]


def test_clean_months_have_no_findings(checked):
    failed = {(r[0], r[1]) for r in checked["results"] if r[3] == "fail"}
    assert failed == {
        ("2023-02", "schema"),
        ("2023-04", "volume"),
        ("2023-05", "null_rate"),
        ("2023-05", "drift"),
    }


def test_overview_and_findings(checked):
    table = checked["table"].set_index("period")
    assert list(table.columns) == list(run_checks.CHECK_ORDER)
    assert table.loc["2023-02", "schema"] == "FAIL high"
    assert table.loc["2023-01", "drift"] == "skipped"
    assert table.loc["2023-01", "volume"] == "pass"
    found = checked["findings"]
    assert found.iloc[0]["severity"] == "high" and found.iloc[-1]["severity"] == "medium"
    # Emptying every fare above $25 (68.5% of the fixture) also shifts the fare distribution,
    # so May has drift findings on both trip_distance and fare_amount.
    assert sorted(zip(found["period"], found["check"], found["column_name"], strict=True)) == [
        ("2023-02", "schema", ""),
        ("2023-04", "volume", ""),
        ("2023-05", "drift", "fare_amount"),
        ("2023-05", "drift", "trip_distance"),
        ("2023-05", "null_rate", "fare_amount"),
    ]


def test_late_file_is_found(paths, taxi, limits):
    with db.connect(paths["db"]) as con:
        entry = next(e for e in manifest.get_entries(con, taxi.name, "2023-03"))
        manifest.set_simulated_delivery(con, entry, datetime(2023, 6, 20))  # expected 2023-05-30
        run_id = run_checks.run_checks(con, taxi, months=["2023-03"], limits=limits)
        row = con.execute(
            "SELECT status, severity FROM dataset_check_results "
            "WHERE run_id = ? AND \"check\" = 'freshness'",
            [run_id],
        ).fetchone()
        manifest.set_simulated_delivery(con, entry, None)
    assert row == ("fail", "medium")


def test_missing_month_gives_warning(paths, taxi, limits):
    with db.connect(paths["db"]) as con:
        run_id = run_checks.run_checks(con, taxi, months=["2023-05", "2023-09"], limits=limits)
        status, error = con.execute(
            "SELECT status, error_message FROM pipeline_runs WHERE run_id = ?", [run_id]
        ).fetchone()
    assert status == "WARNING" and "2023-09 not in Bronze" in error


def test_main_with_repository_config(paths, capsys):
    # With the real cold-start limits the tiny fixture months are below 50,000 rows/day.
    assert run_checks.main(["--database", str(paths["db"]), "--months", "2023-01"]) == 0
    out = capsys.readouterr().out
    assert "below the cold-start limits" in out
    assert run_checks.main(["--database", str(paths["db"]), "--months", "2023-13"]) == 2


def test_main_without_bronze(tmp_path):
    assert run_checks.main(["--database", str(tmp_path / "empty.duckdb")]) == 1
