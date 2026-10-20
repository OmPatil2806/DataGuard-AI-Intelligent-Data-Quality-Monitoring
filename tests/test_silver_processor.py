import json
import shutil
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from common import db
from common.config import load_datasets_config
from ingestion import ingest_data
from ingestion.bronze_loader import LINEAGE_COLUMNS, current_versions
from quality import row_rules
from quality.row_rules import RuleError
from quality.rules import load_rules
from quality.standardize import standardize_file
from silver.silver_processor import SilverError, process_version

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TRIPS = FIXTURES / "yellow_tripdata_fixture.parquet"
ISSUES = json.loads((FIXTURES / "yellow_tripdata_fixture_issues.json").read_text())
# Rows the taxi rules quarantine / only warn (see tests/test_taxi_rules.py), 0-based.
QUARANTINED = sorted(
    {
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
)
WARNED = sorted(
    {
        r
        for k in ("zero_passengers", "null_group", "zero_distance", "unknown_zone_264_265")
        for r in ISSUES[k]
    }
    - set(QUARANTINED)
)


@pytest.fixture
def taxi():
    return load_datasets_config().dataset("yellow_taxi")


@pytest.fixture
def setup(tmp_path, taxi):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    shutil.copy(TRIPS, incoming / "yellow_tripdata_2023-01.parquet")
    shutil.copy(FIXTURES / "taxi_zone_lookup_fixture.csv", incoming / "taxi_zone_lookup.csv")
    duckdb.sql(
        f"COPY (SELECT * EXCLUDE (airport_fee), airport_fee AS Airport_fee FROM '{TRIPS}' "
        "WHERE tpep_pickup_datetime >= '2023-01-01') "
        f"TO '{incoming / 'yellow_tripdata_2023-02.parquet'}' (FORMAT parquet)"
    )
    database = tmp_path / "d.duckdb"
    argv = [
        "--months",
        "2023-01:2023-02",
        "--source",
        str(incoming),
        "--bronze",
        str(tmp_path / "bronze"),
        "--database",
        str(database),
    ]
    assert ingest_data.main(argv) == 0
    con = db.connect(database)
    row_rules.load_lookups(con, taxi, incoming)
    yield {"con": con, "tmp": tmp_path, "versions": current_versions(con, taxi)}
    con.close()


def process(setup, taxi, period="2023-01", rules=None, force=False, run_id="run_0009"):
    path, sha = setup["versions"][period]
    return process_version(
        setup["con"],
        taxi,
        load_rules(taxi) if rules is None else rules,
        period,
        path,
        sha,
        setup["tmp"] / "silver",
        setup["tmp"] / "quarantine",
        run_id,
        force=force,
    )


def read(path):
    return pd.read_parquet(path)


def test_counts_match_the_answer_key(setup, taxi):
    r = process(setup, taxi)
    assert r.status == "processed"
    assert (r.rows, r.quarantine_rows, r.silver_rows) == (
        200,
        len(QUARANTINED),
        200 - len(QUARANTINED),
    )
    assert r.warned_rows == len(WARNED)


def test_every_bronze_row_lands_exactly_once(setup, taxi):
    r = process(setup, taxi)
    ids = sorted([*read(r.silver_path)["bronze_row_id"], *read(r.quarantine_path)["bronze_row_id"]])
    assert ids == list(range(1, 201))


def test_quarantine_holds_exactly_the_failing_rows(setup, taxi):
    r = process(setup, taxi)
    q = read(r.quarantine_path)
    assert sorted(q["bronze_row_id"] - 1) == QUARANTINED
    assert all(len(rules) > 0 for rules in q["quarantine_rules"])
    assert all(
        set(qr) <= set(fr) for qr, fr in zip(q["quarantine_rules"], q["failed_rules"], strict=True)
    )
    assert all(len(r) == len(f) for r, f in zip(q["reasons"], q["failed_rules"], strict=True))
    dup = q[q["bronze_row_id"] - 1 == ISSUES["duplicate_row"][0]]
    assert list(dup["quarantine_rules"].iloc[0]) == ["trip_unique"]


def test_silver_keeps_warned_rows_with_their_reasons(setup, taxi):
    s = read(process(setup, taxi).silver_path)
    warned = s[s["failed_rules"].apply(len) > 0]
    assert sorted(warned["bronze_row_id"] - 1) == WARNED
    quarantine_ids = {r.id for r in load_rules(taxi) if r.action == "quarantine"}
    assert not any(set(f) & quarantine_ids for f in s["failed_rules"])
    row = warned[warned["bronze_row_id"] - 1 == ISSUES["zero_passengers"][0]].iloc[0]
    assert list(row["failed_rules"]) == ["passengers_not_zero"]
    assert list(row["reasons"]) == ["passenger_count is 0"]


def test_columns(setup, taxi):
    r = process(setup, taxi)
    standard = [c for c, _ in taxi.columns]
    lineage = list(LINEAGE_COLUMNS)
    assert list(read(r.silver_path).columns) == [
        *standard,
        *lineage,
        "failed_rules",
        "reasons",
        "silver_run_id",
    ]
    assert list(read(r.quarantine_path).columns) == [
        *standard,
        *lineage,
        "failed_rules",
        "reasons",
        "quarantine_rules",
        "silver_run_id",
    ]
    assert set(read(r.silver_path)["silver_run_id"]) == {"run_0009"}


def test_values_equal_the_standardized_rows(setup, taxi):
    r = process(setup, taxi)
    path, _ = setup["versions"]["2023-01"]
    sql, _ = standardize_file(setup["con"], path, taxi)
    expected = setup["con"].execute(f"SELECT * FROM ({sql}) ORDER BY bronze_row_id").df()
    got = pd.concat([read(r.silver_path), read(r.quarantine_path)]).sort_values("bronze_row_id")
    got = got[expected.columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(got, expected, check_dtype=False)


def test_february_schema_is_standardized(setup, taxi):
    r = process(setup, taxi, period="2023-02")
    s = read(r.silver_path)
    assert "airport_fee" in s.columns and "Airport_fee" not in s.columns
    assert r.rows == 198  # the two December pickups were left out of this file


def test_second_run_skips_and_force_rewrites(setup, taxi):
    first = process(setup, taxi)
    again = process(setup, taxi, run_id="run_0010")
    assert again.status == "skipped"
    assert (again.silver_rows, again.quarantine_rows, again.warned_rows) == (
        first.silver_rows,
        first.quarantine_rows,
        first.warned_rows,
    )
    assert set(read(again.silver_path)["silver_run_id"]) == {"run_0009"}
    forced = process(setup, taxi, run_id="run_0011", force=True)
    assert forced.status == "processed"
    assert set(read(forced.silver_path)["silver_run_id"]) == {"run_0011"}


def test_layout_mirrors_bronze(setup, taxi):
    r = process(setup, taxi)
    _, sha = setup["versions"]["2023-01"]
    expected = Path("yellow_taxi") / "period=2023-01" / f"version={sha[:12]}" / "part-0.parquet"
    assert r.silver_path == setup["tmp"] / "silver" / expected
    assert r.quarantine_path == setup["tmp"] / "quarantine" / expected


def test_no_rules_means_everything_is_silver(setup, taxi):
    r = process(setup, taxi, rules=[])
    assert (r.silver_rows, r.quarantine_rows, r.warned_rows) == (200, 0, 0)


def test_failure_leaves_nothing_behind(setup, taxi, monkeypatch):
    def broken(*args, **kwargs):
        raise RuleError("rule could not be compiled")

    monkeypatch.setattr(row_rules, "evaluate_sql", broken)
    with pytest.raises(RuleError):
        process(setup, taxi)
    assert not (setup["tmp"] / "silver").exists()
    assert not (setup["tmp"] / "quarantine").exists()


def test_count_mismatch_is_refused(setup, taxi, monkeypatch):
    from silver import silver_processor

    real = silver_processor._count
    monkeypatch.setattr(
        silver_processor, "_count", lambda con, path, where="TRUE": real(con, path, where) + 1
    )
    with pytest.raises(SilverError, match="Bronze rows"):
        process(setup, taxi)
    period_dir = setup["tmp"] / "silver" / "yellow_taxi" / "period=2023-01"
    assert not period_dir.exists() or list(period_dir.iterdir()) == []
