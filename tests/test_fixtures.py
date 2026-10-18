"""The committed fixtures contain exactly the documented problems, and nothing else."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
import make_fixtures  # noqa: E402

AMOUNT_PARTS = make_fixtures.AMOUNT_PARTS


def flagged(mask: pd.Series) -> list[int]:
    return sorted(mask[mask].index.tolist())


def detectors(trips: pd.DataFrame, zones: pd.DataFrame) -> dict[str, pd.Series]:
    known = set(zones["LocationID"])
    unknown_zone = {264, 265}
    parts = trips[AMOUNT_PARTS].fillna(0).sum(axis=1)
    pickup = trips["tpep_pickup_datetime"]
    return {
        "zero_passengers": trips["passenger_count"] == 0,
        "null_group": trips["passenger_count"].isna(),
        "negative_fare": trips["fare_amount"] < 0,
        "zero_distance": trips["trip_distance"] == 0,
        "dropoff_before_pickup": trips["tpep_dropoff_datetime"] < pickup,
        "pickup_outside_month": (pickup < "2023-01-01") | (pickup >= "2023-02-01"),
        "unknown_location_id": ~trips["PULocationID"].isin(known)
        | ~trips["DOLocationID"].isin(known),
        "unknown_zone_264_265": trips["PULocationID"].isin(unknown_zone)
        | trips["DOLocationID"].isin(unknown_zone),
        "total_mismatch": (trips["total_amount"] - parts).abs() > 0.01,
        "duplicate_row": trips.duplicated(keep="first"),
    }


def test_shape_and_schema(fixture_trips):
    assert fixture_trips.shape == (200, 19)
    assert list(fixture_trips.columns) == make_fixtures.COLUMNS
    assert str(fixture_trips["tpep_pickup_datetime"].dtype) == "datetime64[us]"
    assert fixture_trips["VendorID"].dtype == "int64"
    assert fixture_trips["passenger_count"].dtype == "float64"


def test_manifest_lists_every_issue(fixture_issues):
    assert set(fixture_issues) == set(make_fixtures.ISSUES)
    all_rows = [r for rows in fixture_issues.values() for r in rows]
    assert len(all_rows) == len(set(all_rows)) == sum(make_fixtures.ISSUES.values())


@pytest.mark.parametrize("issue", sorted(make_fixtures.ISSUES))
def test_issue_rows_match_manifest(issue, fixture_trips, fixture_zones, fixture_issues):
    assert flagged(detectors(fixture_trips, fixture_zones)[issue]) == fixture_issues[issue]


def test_null_group_blanks_related_columns(fixture_trips, fixture_issues):
    rows = fixture_trips.loc[fixture_issues["null_group"]]
    for col in ["RatecodeID", "store_and_fwd_flag", "congestion_surcharge", "airport_fee"]:
        assert rows[col].isna().all()


def test_fixture_file_matches_generator(fixture_trips, fixture_issues, fixture_zones):
    trips, issues, zones = make_fixtures.build()
    pd.testing.assert_frame_equal(trips, fixture_trips)
    assert issues == fixture_issues
    pd.testing.assert_frame_equal(zones, fixture_zones)
