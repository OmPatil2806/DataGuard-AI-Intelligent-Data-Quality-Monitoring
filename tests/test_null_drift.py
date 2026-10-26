from pathlib import Path

import numpy as np
import pytest
import yaml

from common.config import ConfigError, load_quality_rules
from quality import baselines
from quality.baselines import BaselineSettings
from quality.dataset_checks import (
    CheckSettings,
    bin_labels,
    check_drift,
    check_null_rates,
    check_settings,
    psi,
)
from quality.metrics import MonthMetrics, drift_bins

SETTINGS = check_settings()
LIMITS = baselines.settings_for("yellow_taxi")
EDGES = drift_bins("yellow_taxi")["trip_distance"]

# Share of rows with the trip metadata missing, measured in profiling (data_profile.md, section 3).
REAL_NULLS = {
    "2023-01": 0.0234,
    "2023-02": 0.0264,
    "2023-03": 0.0257,
    "2023-04": 0.0276,
    "2023-05": 0.0290,
    "2023-06": 0.0302,
    "2023-07": 0.0293,
    "2023-08": 0.0311,
    "2023-09": 0.0493,
    "2023-10": 0.0440,
    "2023-11": 0.0397,
    "2023-12": 0.0533,
}


def month(period, nulls=None, distances=None):
    m = MonthMetrics(period, period * 4, 1000, 33.0, [])
    m.null_rates = nulls or {"passenger_count": 0.03, "fare_amount": 0.0}
    if distances is not None:
        idx = np.searchsorted(EDGES, distances, side="right")
        m.histograms = {"trip_distance": np.bincount(idx, minlength=len(EDGES) + 1).tolist()}
        m.stats = {"trip_distance": {"median": float(np.median(distances))}}
    return m


def run_nulls(months, period, limits=LIMITS):
    b = baselines.build(months, period, BaselineSettings(**{**LIMITS.__dict__}))
    return {r.column: r for r in check_null_rates(months[period], b, SETTINGS, limits)}


def test_settings_from_the_repository_config():
    assert (SETTINGS.null_max_increase, SETTINGS.null_max_increase_complete) == (0.02, 0.001)
    assert (SETTINGS.psi_medium, SETTINGS.psi_high) == (0.10, 0.20)


@pytest.mark.parametrize("period", [p for p in REAL_NULLS if p >= "2023-04"])
def test_real_2023_null_rates(period):
    months = {p: month(p, {"passenger_count": v}) for p, v in REAL_NULLS.items()}
    result = run_nulls(months, period)["passenger_count"]
    if period == "2023-09":  # +2.01 points over the median of Mar-Aug: a real, small change
        assert (result.status, result.severity) == ("fail", "low")
        assert result.detail.startswith("+2.01 points")
    else:
        assert result.status == "pass", (period, result.detail)


@pytest.mark.parametrize(
    ("rate", "status", "severity"),
    [
        (0.04, "pass", None),
        (0.065, "fail", "low"),
        (0.10, "fail", "medium"),
        (0.25, "fail", "high"),
    ],
)
def test_null_increase_severity(rate, status, severity):
    months = {p: month(p) for p in ["2023-01", "2023-02", "2023-03"]}
    months["2023-04"] = month("2023-04", {"passenger_count": rate, "fare_amount": 0.0})
    result = run_nulls(months, "2023-04")["passenger_count"]
    assert (result.status, result.severity) == (status, severity)


def test_one_percent_spike_in_a_complete_column_is_caught():
    months = {p: month(p) for p in ["2023-01", "2023-02", "2023-03"]}
    months["2023-04"] = month("2023-04", {"passenger_count": 0.03, "fare_amount": 0.01})
    result = run_nulls(months, "2023-04")["fare_amount"]
    assert (result.status, result.severity) == ("fail", "high")  # 1 point = 10x the 0.1 limit
    assert "normally complete column" in result.detail


def test_fewer_nulls_pass():
    months = {p: month(p, {"passenger_count": 0.05}) for p in ["2023-01", "2023-02", "2023-03"]}
    months["2023-04"] = month("2023-04", {"passenger_count": 0.0})
    assert run_nulls(months, "2023-04")["passenger_count"].status == "pass"


def test_cold_start_uses_the_fixed_limit():
    months = {"2023-01": month("2023-01", {"passenger_count": 0.12, "fare_amount": 0.0})}
    results = run_nulls(months, "2023-01")
    assert (results["passenger_count"].status, results["passenger_count"].severity) == (
        "fail",
        "medium",
    )
    assert results["fare_amount"].status == "pass"
    assert (
        run_nulls(months, "2023-01", limits=BaselineSettings())["fare_amount"].status == "skipped"
    )


def test_psi_basics():
    assert psi([0.5, 0.5], [0.5, 0.5]) == 0
    assert psi([0.5, 0.5], [0.4, 0.6]) == pytest.approx(
        0.1 * np.log(1.5) + 0.1 * np.log(0.6 / 0.4) * 0 + 0.0405, abs=0.05
    )
    assert psi([1.0, 0.0], [0.0, 1.0]) > 1  # empty bins stay finite
    with pytest.raises(ValueError, match="same bins"):
        psi([1.0], [0.5, 0.5])


def test_bin_labels():
    assert bin_labels([0, 1, 5]) == ["< 0", "0 to 1", "1 to 5", ">= 5"]


@pytest.fixture
def distance_months():
    """Three baseline months and candidates drawn from a taxi-like distance distribution."""
    rng = np.random.default_rng(1)

    def draw(n=50_000):
        return rng.lognormal(mean=np.log(1.8), sigma=0.9, size=n)  # median about 1.8 miles

    months = {p: month(p, distances=draw()) for p in ["2023-01", "2023-02", "2023-03"]}
    return months, draw


def drift(months, period):
    b = baselines.build(months, period, BaselineSettings())
    return check_drift(months[period], b, SETTINGS, {"trip_distance": EDGES})[0]


def test_a_normal_month_does_not_drift(distance_months):
    months, draw = distance_months
    months["2023-04"] = month("2023-04", distances=draw())
    result = drift(months, "2023-04")
    assert result.status == "pass"
    assert float(result.observed.split()[1]) < 0.01


def test_miles_to_kilometres_is_high_drift(distance_months):
    months, draw = distance_months
    months["2023-04"] = month("2023-04", distances=draw() * 1.609)
    result = drift(months, "2023-04")
    assert (result.status, result.severity) == ("fail", "high")
    assert "median 2.89" in result.detail
    assert float(result.observed.split()[1]) > 0.25


@pytest.mark.parametrize(
    ("factor", "status", "severity"),
    # Measured with these bins: PSI 0.023 at x1.15, 0.136 at x1.4, 0.2 at x1.5, 0.28 at x1.61.
    [(1.15, "pass", None), (1.4, "fail", "medium"), (1.609, "fail", "high")],
)
def test_how_big_a_shift_must_be(distance_months, factor, status, severity):
    months, draw = distance_months
    months["2023-04"] = month("2023-04", distances=draw() * factor)
    result = drift(months, "2023-04")
    assert (result.status, result.severity) == (status, severity), result.observed


def test_drift_is_skipped_during_a_cold_start(distance_months):
    months, _ = distance_months
    result = drift(months, "2023-02")
    assert result.status == "skipped"


@pytest.mark.parametrize("value", [0, 1, -0.1, "a"])
def test_null_config_is_validated(tmp_path, value):
    data = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "quality_rules.yml").read_text()
    )
    data["defaults"]["null_rate"]["max_increase"] = value
    path = tmp_path / "quality_rules.yml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="between 0 and 1"):
        load_quality_rules(path)


def test_check_settings_type():
    assert isinstance(SETTINGS, CheckSettings)
