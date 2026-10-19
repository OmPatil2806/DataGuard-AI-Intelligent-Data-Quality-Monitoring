"""The real taxi rules (configs/quality_rules.yml) against the fixture's answer key."""

import json
from dataclasses import replace
from pathlib import Path

import duckdb
import pytest

from common.config import load_datasets_config
from quality import row_rules
from quality.rules import load_rules
from quality.standardize import standardize

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TRIPS = FIXTURES / "yellow_tripdata_fixture.parquet"
ISSUES = json.loads((FIXTURES / "yellow_tripdata_fixture_issues.json").read_text())

# Which rules must catch each planted problem.
EXPECTED = {
    "passengers_not_zero": ISSUES["zero_passengers"],
    "trip_metadata_missing": ISSUES["null_group"],
    "fare_not_negative": ISSUES["negative_fare"],
    "total_not_negative": ISSUES["negative_fare"],
    "distance_not_zero": ISSUES["zero_distance"],
    "dropoff_not_before_pickup": ISSUES["dropoff_before_pickup"],
    "pickup_in_file_month": ISSUES["pickup_outside_month"],
    "pickup_zone_known": ISSUES["unknown_location_id"],
    "dropoff_zone_not_unknown": ISSUES["unknown_zone_264_265"],
    "trip_unique": ISSUES["duplicate_row"],
}
QUIET = [  # rules with no planted problem: must flag nothing
    "trip_times_present",
    "fare_not_zero",
    "distance_not_negative",
    "distance_max_100_miles",
    "passengers_max_6",
    "trip_has_duration",
    "trip_under_24_hours",
    "dropoff_zone_known",
    "pickup_zone_not_unknown",
]
PENDING = ["total_matches_parts"]  # open item O1


@pytest.fixture(scope="module")
def taxi():
    return load_datasets_config().dataset("yellow_taxi")


@pytest.fixture(scope="module")
def rules(taxi):
    return load_rules(taxi)


@pytest.fixture(scope="module")
def con(taxi, tmp_path_factory):
    folder = tmp_path_factory.mktemp("lookups")
    (folder / "taxi_zone_lookup.csv").write_bytes(
        (FIXTURES / "taxi_zone_lookup_fixture.csv").read_bytes()
    )
    with duckdb.connect() as connection:
        row_rules.load_lookups(connection, taxi, folder)
        yield connection


@pytest.fixture(scope="module")
def source(con, taxi):
    bronze_like = (
        f"(SELECT * EXCLUDE (file_row_number), file_row_number + 1 AS bronze_row_id "
        f"FROM read_parquet('{TRIPS}', file_row_number = true))"
    )
    sql, _ = standardize(con, bronze_like, taxi)
    return f"({sql})"


def flagged_rows(con, rules, source):
    sql = row_rules.evaluate_sql(con, rules, source, period="2023-01")
    rows = con.execute(
        f"SELECT bronze_row_id - 1, failed_rules, quarantine FROM ({sql})"
    ).fetchall()
    by_rule: dict[str, list[int]] = {}
    for row, failed, _ in rows:
        for rule_id in failed:
            by_rule.setdefault(rule_id, []).append(row)
    return {k: sorted(v) for k, v in by_rule.items()}, rows


def test_rule_set_is_complete(rules):
    assert sorted(r.id for r in rules) == sorted([*EXPECTED, *QUIET])
    everything = load_rules("yellow_taxi", include_disabled=True)
    assert [r.id for r in everything if not r.enabled] == PENDING


def test_every_rule_flags_exactly_its_planted_rows(con, rules, source):
    flagged, _ = flagged_rows(con, rules, source)
    for rule_id, rows in EXPECTED.items():
        assert flagged.get(rule_id, []) == rows, rule_id
    for rule_id in QUIET:
        assert rule_id not in flagged, rule_id


def test_quarantine_and_warn_split(con, rules, source):
    _, rows = flagged_rows(con, rules, source)
    quarantined = sorted(r for r, _, q in rows if q)
    warned = sorted(r for r, failed, q in rows if failed and not q)
    expected_quarantine = sorted(
        {
            r
            for rule_id in (
                "fare_not_negative",
                "dropoff_not_before_pickup",
                "pickup_in_file_month",
                "pickup_zone_known",
                "trip_unique",
            )
            for r in EXPECTED[rule_id]
        }
    )
    assert quarantined == expected_quarantine
    expected_warn = sorted(
        {
            r
            for rule_id in (
                "passengers_not_zero",
                "trip_metadata_missing",
                "distance_not_zero",
                "dropoff_zone_not_unknown",
            )
            for r in EXPECTED[rule_id]
        }
    )
    assert warned == expected_warn


def test_pending_total_rule_works_when_enabled(con, source):
    pending = [
        replace(r, enabled=True)
        for r in load_rules("yellow_taxi", include_disabled=True)
        if r.id in PENDING
    ]
    flagged, _ = flagged_rows(con, pending, source)
    assert flagged == {"total_matches_parts": ISSUES["total_mismatch"]}


def test_reasons_carry_values(con, rules, source):
    sql = row_rules.evaluate_sql(con, rules, source, period="2023-01")
    reasons = con.execute(
        f"SELECT reasons FROM ({sql}) WHERE list_contains(failed_rules, 'pickup_zone_known')"
    ).fetchall()
    assert all("PULocationID 999 not in zone lookup" in list(r[0]) for r in reasons)


def test_counts_match(con, rules, source):
    counts = {c.rule_id: c.failed for c in row_rules.count_failures(con, rules, source, "2023-01")}
    assert counts == {r.id: len(EXPECTED.get(r.id, [])) for r in rules}
