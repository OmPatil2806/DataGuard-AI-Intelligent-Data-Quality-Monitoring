"""concentration.py (Phase 8.2): where and when a problem sits, and "unknown" when it is spread."""

import duckdb
import numpy as np
import pandas as pd
import pytest

from common.config import ConfigError
from investigation import concentration as conc

SETTINGS = conc.load_settings("yellow_taxi")
ROWS = 3000


def month(seed=0, rows=ROWS):
    """A synthetic month: vendor 2 has 75% of the trips, 20 zones, every day and hour."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2023-06-01", "2023-06-30", freq="D")
    pickup = days[rng.integers(0, len(days), rows)] + pd.to_timedelta(
        rng.integers(0, 24 * 60, rows), unit="min"
    )
    return pd.DataFrame(
        {
            "bronze_row_id": np.arange(1, rows + 1),
            "VendorID": rng.choice([1, 2], rows, p=[0.25, 0.75]),
            "pickup_borough": rng.choice(
                ["Manhattan", "Queens", "Brooklyn"], rows, p=[0.8, 0.1, 0.1]
            ),
            "pickup_zone": [f"zone {z}" for z in rng.integers(1, 21, rows)],
            "dropoff_zone": [f"zone {z}" for z in rng.integers(1, 21, rows)],
            "payment_type": rng.choice([1, 2, 3, 4], rows, p=[0.7, 0.25, 0.03, 0.02]),
            "RatecodeID": rng.choice([1, 2, 5], rows, p=[0.9, 0.07, 0.03]),
            "tpep_pickup_datetime": pickup,
            "pickup_date": pickup.date,
            "pickup_hour": pickup.hour,
            "broken": False,
        }
    )


def analyse(frame, settings=SETTINGS):
    con = duckdb.connect()
    con.register("trips", frame)
    return conc.segments(con, "trips", "broken", settings)


def test_one_vendor_only(seed=1):
    frame = month(seed)
    vendor1 = frame.index[frame["VendorID"] == 1]
    frame.loc[vendor1[: len(vendor1) // 2], "broken"] = True
    found = analyse(frame)
    top = found["findings"][0]
    assert (top["dimension"], top["kind"], top["value"]) == ("vendor", "where", "1")
    assert top["share"] == 1.0 and top["ratio"] is None and top["rate_outside"] == 0
    assert "and nowhere else" in conc.verdict(found)


def test_large_segment_is_found_by_ratio_not_lift():
    frame = month(2)
    vendor2 = frame.index[frame["VendorID"] == 2]
    frame.loc[vendor2[::10], "broken"] = True  # 10% of vendor 2, 75% of all rows
    top = analyse(frame)["findings"][0]
    assert (top["dimension"], top["value"], top["share"]) == ("vendor", "2", 1.0)


def test_one_zone_and_one_day():
    frame = month(3)
    frame.loc[frame["pickup_zone"] == "zone 7", "broken"] = True
    found = analyse(frame)
    assert found["findings"][0]["dimension"] == "pickup_zone"
    assert found["findings"][0]["value"] == "zone 7"
    assert conc.verdict(found).startswith("where: pickup_zone = zone 7 holds 100%")
    frame["broken"] = frame["pickup_date"] == pd.Timestamp("2023-06-15").date()
    top = analyse(frame)["findings"][0]
    assert (top["dimension"], top["kind"], top["value"]) == ("pickup_date", "when", "2023-06-15")


@pytest.mark.parametrize("share", [0.01, 0.05, 0.2, 0.5])
def test_random_problems_get_no_where(share):
    for seed in range(10):
        frame = month(100 + seed)
        frame["broken"] = np.random.default_rng(seed).random(len(frame)) < share
        found = analyse(frame)
        assert found["analysed"] and found["findings"] == [], (share, seed, found["findings"])
        assert conc.verdict(found).startswith("unknown")


def test_a_mixed_problem_reports_only_the_concentrated_part():
    frame = month(4)
    rng = np.random.default_rng(4)
    frame["broken"] = rng.random(len(frame)) < 0.01  # background noise
    frame.loc[frame["payment_type"] == 3, "broken"] = True  # 3% of rows, all broken
    found = analyse(frame)
    assert [f["value"] for f in found["findings"] if f["dimension"] == "payment_type"] == ["3"]
    assert all(f["share"] >= 0.5 for f in found["findings"])


def test_small_segments_and_trivial_months():
    frame = month(5)
    frame.loc[frame.index[:5], "pickup_zone"] = "tiny"
    frame.loc[frame.index[:5], "broken"] = True  # 5 rows: below min_rows
    few = analyse(frame)
    assert all(f["value"] != "tiny" for f in few["findings"])
    assert few["few_affected"] and "only 5 affected row(s), too few" in conc.verdict(few)
    frame["broken"] = False
    assert analyse(frame)["reason"] == "no affected rows"
    frame["broken"] = True
    found = analyse(frame)
    assert found["reason"] == "every row is affected"
    assert conc.verdict(found) == "not analysed: every row is affected"


def test_low_days():
    frame = month(6, rows=6000)
    con = duckdb.connect()
    con.register("all_rows", frame)
    assert conc.low_days(con, "all_rows", "2023-06", SETTINGS, "tpep_pickup_datetime")["days"] == []
    missing = frame[~frame["pickup_date"].astype(str).isin(["2023-06-10", "2023-06-11"])]
    con.register("missing", missing)
    days = conc.low_days(con, "missing", "2023-06", SETTINGS, "tpep_pickup_datetime")
    assert [d["day"] for d in days["days"]] == ["2023-06-10", "2023-06-11"]
    assert days["days"][0]["rows"] == 0 and days["median_rows_per_day"] >= 150
    thin = frame.sample(frac=0.5, random_state=1)  # half the rows, evenly
    con.register("thin", thin)
    assert conc.low_days(con, "thin", "2023-06", SETTINGS, "tpep_pickup_datetime")["days"] == []
    con.register("few", frame.head(300))
    assert not conc.low_days(con, "few", "2023-06", SETTINGS, "tpep_pickup_datetime")["analysed"]


def test_z_statistic():
    assert conc.z_statistic(10, 10, 0, 90) > 4
    assert conc.z_statistic(5, 50, 45, 450) == pytest.approx(0)
    assert conc.z_statistic(0, 10, 0, 10) is None and conc.z_statistic(1, 0, 1, 10) is None


def test_settings(tmp_path):
    assert {d.name for d in SETTINGS.dimensions} >= {"vendor", "pickup_zone", "pickup_hour"}
    assert {d.kind for d in SETTINGS.dimensions} == {"where", "when"}
    assert conc.load_settings("ved_signals").min_z == SETTINGS.min_z  # thresholds are shared
    assert conc.load_settings("not_configured").dimensions == ()  # no where/when
    assert conc.sample_columns("yellow_taxi")[0] == "tpep_pickup_datetime"
    assert conc.sample_columns("not_configured") is None
    bad = tmp_path / "investigation.yml"
    bad.write_text(
        "concentration: {}\ndatasets:\n  d:\n    dimensions:\n      v: {sql: V, kind: why}\n"
    )
    with pytest.raises(ConfigError, match="kind"):
        conc.load_settings("d", bad)
    bad.write_text("concentration:\n  min_z: -1\n")
    with pytest.raises(ConfigError, match="min_z"):
        conc.load_settings("d", bad)
    bad.write_text("concentration: {}\ndatasets:\n  d:\n    sample_columns: a\n")
    with pytest.raises(ConfigError, match="sample_columns"):
        conc.sample_columns("d", bad)
    bad.write_text("other: 1\n")
    with pytest.raises(ConfigError, match="mapping"):
        conc.load_settings("d", bad)
