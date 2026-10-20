from datetime import datetime, timedelta
from pathlib import Path

import pytest
import yaml

from common.config import ConfigError, load_quality_rules
from ingestion.manifest import ManifestEntry
from quality import baselines
from quality.baselines import BaselineSettings
from quality.dataset_checks import CheckSettings, check_freshness, check_settings, check_volume
from quality.metrics import MonthMetrics

LIMITS = baselines.settings_for("yellow_taxi")
SETTINGS = CheckSettings(volume_tolerance=0.30, freshness_mode="replay", grace_days=3)

# Rows per day measured in profiling (docs/data_profile.md, section 1).
REAL = {
    "2019-01": 248278,
    "2023-01": 98928,
    "2023-02": 104070,
    "2023-03": 109799,
    "2023-04": 109608,
    "2023-05": 113344,
    "2023-06": 110241,
    "2023-07": 93778,
    "2023-08": 91104,
    "2023-09": 94891,
    "2023-10": 113622,
    "2023-11": 111324,
    "2023-12": 108922,
}


def month(period, rows_per_day):
    return MonthMetrics(period, period * 4, int(rows_per_day * 30), float(rows_per_day), [])


def volume(months, period, limits=LIMITS):
    data = {p: month(p, v) for p, v in months.items()}
    return check_volume(data[period], baselines.build(data, period, limits), SETTINGS, limits)


def test_check_settings_from_the_repository_config():
    assert check_settings() == CheckSettings(0.30, "replay", 3)


@pytest.mark.parametrize("period", [p for p in REAL if p.startswith("2023")])
def test_every_real_2023_month_passes(period):
    assert volume(REAL, period).status == "pass"


def test_october_jump_is_within_tolerance():
    # Against the median of the six months before (Apr-Sep, 102,250/day). The +21% in the
    # profiling doc compared October with the three summer months only.
    result = volume(REAL, "2023-10")
    assert result.detail.startswith("+11.1% against the median of 2023-04")


def test_2019_is_above_the_cold_start_limits():
    result = volume(REAL, "2019-01")
    assert (result.status, result.severity) == ("fail", "medium")
    assert result.detail == "above the cold-start limits"
    assert "50,000 to 200,000" in result.expected


@pytest.mark.parametrize(
    ("change", "status", "severity", "word"),
    [
        (-0.25, "pass", None, ""),
        (-0.40, "fail", "medium", "drop"),
        (-0.70, "fail", "high", "drop"),
        (0.45, "fail", "medium", "rise"),
    ],
)
def test_deviation_from_the_baseline(change, status, severity, word):
    months = {p: 100_000 for p in ["2023-01", "2023-02", "2023-03"]}
    months["2023-04"] = 100_000 * (1 + change)
    result = volume(months, "2023-04")
    assert (result.status, result.severity) == (status, severity)
    assert word in result.detail
    assert result.expected == "100,000 rows/day ± 30%"


def test_cold_start_without_limits_is_skipped():
    result = volume({"2023-01": 100_000}, "2023-01", limits=BaselineSettings())
    assert result.status == "skipped"


def test_cold_start_below_limits():
    result = volume({"2023-01": 20_000}, "2023-01")
    assert (result.status, result.detail) == ("fail", "below the cold-start limits")


def entry(expected=datetime(2023, 4, 2), delivered=datetime(2026, 10, 18, 9), simulated=None):
    return ManifestEntry(
        dataset="yellow_taxi",
        period="2023-01",
        file_name="f.parquet",
        sha256="a" * 64,
        size_bytes=1,
        row_count=1,
        expected_at=expected,
        delivered_at=delivered,
        registered_at=delivered,
        status="LOADED",
        simulated_delivered_at=simulated,
    )


def test_replay_counts_historical_files_as_on_time():
    result = check_freshness(entry(), SETTINGS)
    assert (result.status, result.detail) == ("pass", "on time")
    assert "(replay mode)" in result.observed


@pytest.mark.parametrize(
    ("late_days", "status", "severity"),
    [
        (-5, "pass", None),
        (2, "pass", None),
        (5, "fail", "low"),
        (20, "fail", "medium"),
        (60, "fail", "high"),
    ],
)
def test_simulated_late_delivery(late_days, status, severity):
    expected = datetime(2023, 4, 2)
    result = check_freshness(entry(simulated=expected + timedelta(days=late_days)), SETTINGS)
    assert (result.status, result.severity) == (status, severity)
    assert "(simulated delivery)" in result.observed


def test_live_mode_uses_the_real_arrival_time():
    live = CheckSettings(freshness_mode="live", grace_days=3)
    result = check_freshness(entry(), live)
    assert (result.status, result.severity) == ("fail", "high")  # downloaded years later
    on_time = check_freshness(entry(delivered=datetime(2023, 4, 1)), live)
    assert on_time.status == "pass"


def test_no_expected_time_is_skipped():
    assert check_freshness(entry(expected=None), SETTINGS).status == "skipped"


@pytest.mark.parametrize(
    ("freshness", "message"),
    [({"mode": "realtime"}, "must be 'replay' or 'live'"), ({"grace_days": -1}, "0 or more")],
)
def test_freshness_config_is_validated(tmp_path, freshness, message):
    data = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "quality_rules.yml").read_text()
    )
    data["defaults"]["freshness"] = freshness
    path = tmp_path / "quality_rules.yml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match=message):
        load_quality_rules(path)
