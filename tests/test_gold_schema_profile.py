"""gold_schema_history and gold_column_profile."""

import json

import duckdb
import pandas as pd
import pytest
from pipeline_fixture import run_pipeline

from common import db
from common.config import load_datasets_config
from gold import monitoring_tables as gold
from gold.monitoring_tables import schema_history_frame

STANDARD = list(load_datasets_config().dataset("yellow_taxi").columns)


@pytest.fixture(scope="module")
def database(tmp_path_factory):
    database = run_pipeline(tmp_path_factory.mktemp("gold_schema"))
    with db.connect(database) as con:
        gold.rebuild(con)
    return database


@pytest.fixture(scope="module")
def history(database):
    with db.connect(database, read_only=True) as con:
        return con.execute("SELECT * FROM gold_schema_history").df()


@pytest.fixture(scope="module")
def profile(database):
    with db.connect(database, read_only=True) as con:
        return con.execute("SELECT * FROM gold_column_profile").df()


# --- schema history on the pipeline ----------------------------------------------------------


def test_every_column_every_month(history):
    assert history.groupby("period").size().to_dict() == {
        p: len(STANDARD) for p in ["2023-01", "2023-02", "2023-03", "2023-04", "2023-05"]
    }
    assert not history.duplicated(["dataset", "period", "column_key"]).any()
    assert history["in_standard"].all()


def test_february_changes_match_the_schema_check(history, database):
    changed = history[history["change"] != "unchanged"]
    assert changed[["period", "column_name", "change", "severity"]].values.tolist() == [
        ["2023-02", "Airport_fee", "renamed", "high"],
        ["2023-02", "VendorID", "type_changed", "medium"],
    ]
    vendor = changed[changed["column_key"] == "vendorid"].iloc[0]
    assert (vendor.previous_dtype, vendor["dtype"], vendor.previous_period) == (
        "BIGINT",
        "INTEGER",
        "2023-01",
    )
    with db.connect(database, read_only=True) as con:
        detail = con.execute(
            "SELECT detail FROM dataset_check_results "
            "WHERE \"check\" = 'schema' AND status = 'fail'"
        ).fetchone()[0]
    for text in changed["detail"]:
        assert text in detail  # the same wording as the check


def test_first_month_has_no_previous(history):
    jan = history[history["period"] == "2023-01"]
    assert jan["previous_period"].isna().all() and (jan["change"] == "unchanged").all()
    assert jan.sort_values("position")["column_name"].tolist()[:2] == [
        "VendorID",
        "tpep_pickup_datetime",
    ]


# --- schema history on made-up schemas -------------------------------------------------------


def frame_for(schemas: dict[str, list[tuple[str, str]]]):
    con = duckdb.connect()
    db.init_schema(con)
    con.execute("CREATE TABLE gold_months (dataset VARCHAR, period VARCHAR, sha256 VARCHAR)")
    for period, schema in schemas.items():
        con.execute("INSERT INTO gold_months VALUES ('yellow_taxi', ?, ?)", [period, period])
        con.execute(
            "INSERT INTO dataset_metrics (dataset, period, sha256, metric, text_value, "
            "computed_at) VALUES ('yellow_taxi', ?, ?, 'raw_schema', ?, now())",
            [period, period, json.dumps(schema)],
        )
    return schema_history_frame(con).set_index(["period", "column_key"])


def test_missing_extra_and_removed_columns():
    jan = [*STANDARD, ("extra_col", "VARCHAR")]
    feb = [c for c in STANDARD if c[0] != "fare_amount"]  # fare missing, extra column removed
    frame = frame_for({"2023-01": jan, "2023-02": feb})

    extra = frame.loc[("2023-01", "extra_col")]
    assert (extra.change, extra.severity, bool(extra.in_standard)) == ("unexpected", "low", False)
    assert extra.position == len(STANDARD) + 1

    fare = frame.loc[("2023-02", "fare_amount")]
    assert (fare.change, fare.severity) == ("missing", "critical")
    assert pd.isna(fare["dtype"]) and pd.isna(fare.position) and fare.column_name == "fare_amount"
    assert fare.previous_dtype == "DOUBLE"

    removed = frame.loc[("2023-02", "extra_col")]
    assert (removed.change, removed.severity) == ("removed", "medium")
    assert removed.column_name == "extra_col" and removed.previous_dtype == "VARCHAR"
    assert len(frame.loc["2023-02"]) == len(STANDARD) + 1  # 18 delivered, fare, extra_col


def test_rename_and_type_change_of_one_column_are_joined():
    feb = [("VENDORID", "BIGINT") if c[0] == "VendorID" else c for c in STANDARD]
    frame = frame_for({"2023-01": STANDARD, "2023-02": feb})
    vendor = frame.loc[("2023-02", "vendorid")]
    assert vendor.change == "renamed, type_changed" and vendor.severity == "high"
    assert vendor.detail == "renamed VendorID -> VENDORID; VENDORID type INTEGER -> BIGINT"


def test_previous_month_follows_the_baseline_gap():
    frame = frame_for({"2019-01": STANDARD, "2023-01": STANDARD})
    assert frame.loc["2023-01"]["previous_period"].isna().all()  # more than 12 months apart


def test_empty_database_gives_an_empty_typed_frame():
    frame = frame_for({})
    assert frame.empty and list(frame.index.names) == ["period", "column_key"]


# --- column profile --------------------------------------------------------------------------


def test_profile_has_every_column_every_month(profile):
    assert len(profile) == 5 * len(STANDARD)
    assert not profile.duplicated(["dataset", "period", "column_name"]).any()
    binned = profile.dropna(subset=["mean"])["column_name"].unique().tolist()
    assert sorted(binned) == ["fare_amount", "total_amount", "trip_distance"]
    assert profile[~profile["column_name"].isin(binned)]["histogram"].isna().all()


def test_profile_values_match_the_metrics(profile, database):
    fare = profile[profile["column_name"] == "fare_amount"].set_index("period")
    with db.connect(database, read_only=True) as con:
        rows = con.execute(
            "SELECT m.period, d.metric, d.value FROM gold_months m JOIN dataset_metrics d "
            "USING (dataset, period, sha256) WHERE d.column_name = 'fare_amount' "
            "AND d.metric IN ('mean', 'median', 'p99', 'null_rate')"
        ).fetchall()
        counts = dict(
            con.execute(
                "SELECT m.period, sum(d.value) FROM gold_months m JOIN dataset_metrics d "
                "USING (dataset, period, sha256) WHERE d.column_name = 'fare_amount' "
                "AND d.metric = 'histogram' GROUP BY 1"
            ).fetchall()
        )
    for period, metric, value in rows:
        assert fare.loc[period, metric] == pytest.approx(value)
    for period, total in counts.items():
        assert sum(fare.loc[period, "histogram"]) == total


def test_profile_flags_the_may_problems(profile):
    may = profile[profile["period"] == "2023-05"].set_index("column_name")
    assert may.loc["fare_amount", "null_rate"] == pytest.approx(0.685)
    assert may.loc["fare_amount", ["null_rate_status", "null_rate_severity"]].tolist() == [
        "fail",
        "high",
    ]
    assert may.loc["fare_amount", "psi"] == pytest.approx(5.93)
    assert may.loc["trip_distance", "drift_severity"] == "high"
    assert may.loc["total_amount", "drift_status"] == "pass"
    jan = profile[profile["period"] == "2023-01"]
    assert jan["psi"].isna().all()  # drift is skipped during the cold start


def test_rebuild_checksums_are_stable(database):
    with db.connect(database) as con:
        _, first = gold.rebuild(con)
        _, second = gold.rebuild(con)
    assert first == second
    names = [b.name for b in first]
    assert "gold_schema_history" in names and "gold_column_profile" in names
