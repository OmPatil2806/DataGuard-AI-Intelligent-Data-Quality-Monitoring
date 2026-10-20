import random
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from common import db
from common.config import ConfigError, load_quality_rules
from quality import baselines
from quality.baselines import BaselineSettings, build
from quality.metrics import MonthMetrics

SETTINGS = BaselineSettings(min_runs=3, window=6, max_gap_months=12)
REAL_PERIODS = ["2019-01"] + [f"2023-{m:02d}" for m in range(1, 13)]


def month(period, rows_per_day=100.0, null=0.02, hist=(10, 30, 60), schema=None):
    return MonthMetrics(
        period=period,
        sha256=period.replace("-", "") + "0" * 58,
        rows=int(rows_per_day * 30),
        rows_per_day=rows_per_day,
        raw_schema=schema or [("VendorID", "INTEGER"), ("airport_fee", "DOUBLE")],
        null_rates={"passenger_count": null, "fare_amount": 0.0},
        histograms={"trip_distance": list(hist)},
    )


@pytest.fixture
def real_like():
    return {p: month(p, rows_per_day=100 + i) for i, p in enumerate(REAL_PERIODS)}


def test_2019_never_feeds_2023(real_like):
    assert build(real_like, "2019-01", SETTINGS).source_periods == []
    assert build(real_like, "2023-01", SETTINGS).source_periods == []  # 48 months back


@pytest.mark.parametrize(
    ("period", "sources", "trusted"),
    [
        ("2023-02", ["2023-01"], False),
        ("2023-03", ["2023-01", "2023-02"], False),
        ("2023-04", ["2023-01", "2023-02", "2023-03"], True),
        ("2023-10", ["2023-04", "2023-05", "2023-06", "2023-07", "2023-08", "2023-09"], True),
    ],
)
def test_window_and_cold_start(real_like, period, sources, trusted):
    b = build(real_like, period, SETTINGS)
    assert (b.source_periods, b.trusted) == (sources, trusted)


def test_cold_start_still_knows_the_previous_schema(real_like):
    b = build(real_like, "2023-02", SETTINGS)
    assert not b.trusted
    assert b.previous_period == "2023-01"
    assert b.previous_schema == real_like["2023-01"].raw_schema
    assert b.rows_per_day is None and b.null_rates == {} and b.distributions == {}
    assert b.note == "cold start: 1 earlier month(s) available"


def test_baseline_values_are_medians_and_mean_shares():
    m = {
        "2023-01": month("2023-01", 90, null=0.01, hist=(0, 50, 50)),
        "2023-02": month("2023-02", 120, null=0.05, hist=(25, 25, 50)),
        "2023-03": month("2023-03", 100, null=0.02, hist=(10, 40, 50)),
    }
    b = build(m, "2023-04", SETTINGS)
    assert b.trusted
    assert b.rows_per_day == 100  # median of 90, 120, 100
    assert b.null_rates == {"fare_amount": 0.0, "passenger_count": 0.02}
    assert b.distributions["trip_distance"] == pytest.approx([35 / 300, 115 / 300, 0.5])
    assert b.note == "baseline from 3 month(s): 2023-01, 2023-02, 2023-03"


def test_one_extreme_month_does_not_move_the_median():
    m = {p: month(p, 100) for p in ["2023-01", "2023-02", "2023-03", "2023-04"]}
    m["2023-03"] = month("2023-03", 10_000)
    assert build(m, "2023-05", SETTINGS).rows_per_day == 100


def test_gaps_and_future_months_are_ignored():
    m = {p: month(p) for p in ["2021-12", "2022-06", "2022-07", "2022-08", "2023-02", "2023-09"]}
    b = build(m, "2023-01", SETTINGS)
    assert b.source_periods == ["2022-06", "2022-07", "2022-08"]  # not 2021-12, not later


def test_load_order_does_not_matter(real_like):
    shuffled = list(real_like.items())
    random.Random(7).shuffle(shuffled)
    assert build(dict(shuffled), "2023-08", SETTINGS) == build(real_like, "2023-08", SETTINGS)


def test_changed_bins_are_not_compared():
    m = {
        "2023-01": month("2023-01", hist=(1, 2, 3)),
        "2023-02": month("2023-02", hist=(1, 2, 3, 4)),
        "2023-03": month("2023-03", hist=(1, 2, 3)),
    }
    assert "trip_distance" not in build(m, "2023-04", SETTINGS).distributions


def test_settings_from_the_repository_config():
    s = baselines.settings_for("yellow_taxi")
    assert (s.min_runs, s.window, s.max_gap_months) == (3, 6, 12)
    assert (s.rows_per_day_min, s.rows_per_day_max, s.null_rate_max) == (50000, 400000, 0.10)
    assert baselines.settings_for("ved_signals").rows_per_day_min is None


def write_config(tmp_path, change):
    data = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "quality_rules.yml").read_text()
    )
    change(data)
    path = tmp_path / "quality_rules.yml"
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d["defaults"]["baseline"].update(window=2), "window must be >= min_runs"),
        (lambda d: d["defaults"]["baseline"].update(max_gap_months=0), "max_gap_months must be"),
        (
            lambda d: d["cold_start"]["yellow_taxi"].update(rows_per_day={"min": 5, "max": 1}),
            "needs 0 <= min < max",
        ),
        (lambda d: d["cold_start"]["yellow_taxi"].update(null_rate_max=2), "between 0 and 1"),
    ],
)
def test_config_is_validated(tmp_path, change, message):
    with pytest.raises(ConfigError, match=message):
        load_quality_rules(write_config(tmp_path, change))


def test_store_records_what_was_used(tmp_path, real_like):
    b = build(real_like, "2023-05", SETTINGS)
    with db.connect(tmp_path / "d.duckdb") as con:
        db.init_schema(con)
        baselines.store(con, "run_0007", "yellow_taxi", b, now=datetime(2024, 1, 1))
        baselines.store(con, "run_0007", "yellow_taxi", b)  # same run again: replaced
        rows = con.execute(
            "SELECT metric, column_name, bin, value, source_periods, trusted FROM baselines "
            "ORDER BY metric, column_name, bin"
        ).fetchall()
    sources = "2023-01,2023-02,2023-03,2023-04"  # all four earlier months fit the window of 6
    assert ("rows_per_day", "", "", b.rows_per_day, sources, True) in rows
    assert sum(r[0] == "distribution" for r in rows) == 3
    assert sum(r[0] == "null_rate" for r in rows) == 2


def test_store_cold_start(tmp_path, real_like):
    with db.connect(tmp_path / "d.duckdb") as con:
        db.init_schema(con)
        baselines.store(con, "run_0001", "yellow_taxi", build(real_like, "2023-01", SETTINGS))
        assert con.execute("SELECT value, source_periods, trusted FROM baselines").fetchall() == [
            (None, "", False)
        ]


def test_overview(real_like):
    table = baselines.overview(real_like, SETTINGS).set_index("period")
    assert table.loc["2023-04", "trusted"]
    assert not table.loc["2023-03", "trusted"]
    assert table.loc["2023-01", "earlier_months"] == "-"
    assert table.loc["2023-02", "previous_schema_from"] == "2023-01"
