from pathlib import Path

import pytest
import yaml

from common import db
from common.config import ConfigError, load_quality_rules
from incidents import incident_manager as im
from incidents import lifecycle
from incidents.severity_engine import RowRuleSettings, row_rule_settings, rule_findings, rule_rates

SETTINGS = RowRuleSettings(
    relative_increase=1.0,
    min_increase=0.001,
    cold_start_max_rate={"quarantine": 0.05, "warn": 0.10},
)
# Monthly share of negative fares, roughly as profiled (data_profile.md): steady, then a jump.
NEGATIVE_FARES = {
    "2023-01": 0.0100,
    "2023-02": 0.0105,
    "2023-03": 0.0110,
    "2023-04": 0.0108,
    "2023-05": 0.0112,
    "2023-06": 0.0118,
    "2023-07": 0.0126,
    "2023-08": 0.0300,
    "2023-09": 0.0121,
    "2023-10": 0.0119,
}


@pytest.fixture
def con(tmp_path):
    with db.connect(tmp_path / "d.duckdb") as connection:
        db.init_schema(connection)
        yield connection


def log(con, run_id, period, rule_id, rate, action="quarantine", severity="high", rows=100_000):
    con.execute(
        "INSERT INTO rule_run_log VALUES (?, 'yellow_taxi', ?, 'sha', ?, 'validity', ?, ?, ?, ?)",
        [run_id, period, rule_id, action, severity, rows, round(rate * rows)],
    )


def fill(con, run_id="run_0002", rates=NEGATIVE_FARES, **kwargs):
    for period, rate in rates.items():
        log(con, run_id, period, "fare_not_negative", rate, **kwargs)


def findings(con, run_id="run_0002"):
    return {
        p: [f.rule_id for f in fs]
        for p, (fs, _) in rule_findings(con, run_id, "yellow_taxi", SETTINGS).items()
    }


def test_settings_from_the_repository_config():
    s = row_rule_settings()
    assert (s.relative_increase, s.min_increase) == (1.0, 0.001)
    assert s.cold_start_limit("quarantine") == 0.05 and s.cold_start_limit("warn") == 0.10


def test_steady_rates_raise_nothing_and_a_jump_does(con):
    fill(con)
    result = findings(con)
    assert [p for p, ids in result.items() if ids] == ["2023-08"]


def test_the_finding_explains_itself(con):
    fill(con)
    [f] = rule_findings(con, "run_0002", "yellow_taxi", SETTINGS)["2023-08"][0]
    assert (f.source, f.rule_id, f.severity, f.column) == (
        "row_rule",
        "fare_not_negative",
        "high",
        "",
    )
    assert f.title == "Rule failure rate up: fare_not_negative"
    assert f.observed == "3.00%"
    # Baseline: median of Feb-Jul = 1.11%; doubled = 2.22%, and 3.00% is above it.
    assert f.expected == "at most 2.22% (baseline 1.11%)"
    assert "points against the median of 2023-02, 2023-03" in f.detail


def test_every_rule_counts_as_checked(con):
    fill(con)
    log(con, "run_0002", "2023-08", "distance_not_zero", 0.03, action="warn", severity="low")
    _, checked = rule_findings(con, "run_0002", "yellow_taxi", SETTINGS)["2023-08"]
    assert checked == {
        ("yellow_taxi", "row_rule", "fare_not_negative", ""),
        ("yellow_taxi", "row_rule", "distance_not_zero", ""),
    }


def test_cold_start_uses_the_fixed_limit_per_action(con):
    log(con, "run_0002", "2023-01", "fare_not_negative", 0.06)  # quarantine limit 5%
    log(con, "run_0002", "2023-01", "distance_not_zero", 0.06, action="warn", severity="low")
    assert findings(con) == {"2023-01": ["fare_not_negative"]}


def test_small_increases_near_zero_do_not_flicker(con):
    rates = {f"2023-0{m}": 0.0001 for m in range(1, 5)}
    rates["2023-05"] = 0.0009  # 9x the baseline, but only +0.08 points
    fill(con, rates=rates)
    assert findings(con)["2023-05"] == []
    fill(con, run_id="run_0003", rates={"2023-05": 0.0030})  # +0.29 points: a real jump
    assert findings(con, "run_0003")["2023-05"] == ["fare_not_negative"]


def test_the_latest_run_per_month_is_used(con):
    fill(con)
    log(con, "run_0005", "2023-08", "fare_not_negative", 0.0120)  # month re-processed, normal now
    assert rule_rates(con, "yellow_taxi")["2023-08"]["fare_not_negative"].run_id == "run_0005"
    assert findings(con, "run_0005") == {"2023-08": []}


def test_rule_incident_lifecycle(con):
    fill(con)
    r = lifecycle.apply_rule_run(con, "run_0002", "yellow_taxi", resolve_after=2)
    [incident] = im.list_incidents(con)
    assert r.opened == [incident.incident_id] and r.resolved == [incident.incident_id]
    assert (incident.source, incident.first_seen_period, incident.resolved_period) == (
        "row_rule",
        "2023-08",
        "2023-10",
    )  # clean in September and October


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"relative_increase": -1}, "must be a number >= 0"),
        ({"cold_start_max_rate": {"drop": 0.1}}, "unknown action 'drop'"),
        ({"cold_start_max_rate": {"warn": 0}}, "between 0 and 1"),
    ],
)
def test_config_is_validated(tmp_path, change, message):
    data = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "quality_rules.yml").read_text()
    )
    data["defaults"]["incidents"]["row_rules"].update(change)
    path = tmp_path / "quality_rules.yml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match=message):
        load_quality_rules(path)
