import shutil
from pathlib import Path

import duckdb
import pytest

from profiling import taxi_profile as tp

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def files(tmp_path):
    """The fixture under a real file name, plus a copy with a changed schema."""
    jan = tmp_path / "yellow_tripdata_2023-01.parquet"
    shutil.copy(FIXTURES / "yellow_tripdata_fixture.parquet", jan)
    feb = tmp_path / "yellow_tripdata_2023-02.parquet"
    duckdb.sql(
        f"COPY (SELECT * EXCLUDE (airport_fee, store_and_fwd_flag), airport_fee AS Airport_fee, "
        f"CAST(VendorID AS INTEGER) AS VendorID_int FROM read_parquet('{jan}')) "
        f"TO '{feb}' (FORMAT parquet)"
    )
    return [jan, feb]


@pytest.fixture
def con():
    with tp.connect() as connection:
        yield connection


def test_period_of():
    assert tp.period_of(Path("yellow_tripdata_2019-01.parquet")) == "2019-01"
    assert tp.period_of(Path("yellow_tripdata_fixture.parquet")) is None


def test_rows_per_file(con, files):
    df = tp.rows_per_file(con, files)
    assert df["rows"].tolist() == [200, 200]
    assert df["days"].tolist() == [31, 28]
    assert df["rows_per_day"].tolist() == [6, 7]


def test_schema_differences(con, files):
    diff = tp.schema_differences(tp.schema_by_file(con, files)).set_index("column")
    assert sorted(diff.index) == [
        "Airport_fee",
        "VendorID_int",
        "airport_fee",
        "store_and_fwd_flag",
    ]
    assert diff.loc["store_and_fwd_flag", "yellow_tripdata_2023-02.parquet"] == "missing"


def test_null_rates(con, files):
    rates = tp.null_rates(con, files[:1]).set_index("column")
    assert rates.loc["passenger_count", files[0].name] == 2.0  # 4 of 200 rows
    assert rates.loc["fare_amount", files[0].name] == 0.0


def test_null_group(con, files):
    group = tp.null_group(con, files).set_index("file")
    jan = group.loc[files[0].name]
    assert (jan["passenger_count_null"], jan["whole_group_null"]) == (4, 4)
    # The second file has no store_and_fwd_flag; the group check still runs.
    assert group.loc[files[1].name, "whole_group_null"] == 4


def test_issue_counts_match_fixture_manifest(con, files, fixture_issues):
    issues = tp.issue_counts(con, files[:1], FIXTURES / "taxi_zone_lookup_fixture.csv")
    counts = dict(zip(issues["check"], issues["rows"], strict=True))
    for name in [
        "zero_passengers",
        "negative_fare",
        "zero_distance",
        "dropoff_before_pickup",
        "pickup_outside_month",
        "unknown_location_id",
        "unknown_zone_264_265",
        "total_mismatch",
    ]:
        assert counts[name] == len(fixture_issues[name]), name
    for name in ["many_passengers", "long_distance", "zero_duration", "long_duration"]:
        assert counts[name] == 0, name
    assert counts["negative_total"] == len(fixture_issues["negative_fare"])


def test_issue_counts_handle_renamed_amount_column(con, files):
    issues = tp.issue_counts(con, files, FIXTURES / "taxi_zone_lookup_fixture.csv")
    feb = issues[issues["file"] == files[1].name].set_index("check")
    assert feb.loc["total_mismatch", "rows"] == 2  # Airport_fee is still summed


def test_duplicate_counts(con, files):
    dups = tp.duplicate_counts(con, files[:1]).iloc[0]
    assert (dups["exact_duplicates"], dups["key_duplicates"]) == (2, 2)


def test_distribution(con, files):
    dist = tp.distribution(con, files[:1]).set_index("column")
    assert dist.loc["trip_distance", "p01"] <= dist.loc["trip_distance", "p50"]
    assert dist.loc["fare_amount", "p01"] < 0  # the negative fares


def test_summary(con, files):
    issues = tp.issue_counts(con, files, FIXTURES / "taxi_zone_lookup_fixture.csv")
    summary = tp.summary(issues).set_index("check")
    assert summary.loc["zero_passengers", "total_rows"] == 6
    assert summary.loc["zero_passengers", "files_affected"] == 2
    assert summary.loc["many_passengers", "files_affected"] == 0
