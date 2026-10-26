"""notifier.py (Phase 10.1): alerts for new critical and high incidents, once each.

The planted pipeline's incidents: INC-0003, INC-0004 and INC-0005 are high and open (May's NULL
fares and drift); INC-0002 and INC-0006 are high but resolved; INC-0001 is medium.
"""

import json

import duckdb
import pytest
from pipeline_fixture import run_pipeline

from alerts import notifier as nt
from common.config import ConfigError
from gold import rebuild
from investigation import investigate

OPEN_HIGH = ["INC-0003", "INC-0004", "INC-0005"]


@pytest.fixture
def env(tmp_path_factory, tmp_path):
    root = tmp_path_factory.mktemp("alerts")
    database = run_pipeline(root)
    assert investigate.main(["--database", str(database), "--no-reports"]) == 0
    return {"db": str(database), "outbox": tmp_path / "outbox"}


def run(env, *args):
    return nt.main(["--database", env["db"], "--outbox", str(env["outbox"]), *args])


def log(env):
    with duckdb.connect(env["db"], read_only=True) as con:
        return con.execute(
            "SELECT incident_id, severity, channel, status, error, message FROM alert_log "
            "ORDER BY incident_id, channel"
        ).fetchall()


def outbox(env):
    path = env["outbox"] / nt.OUTBOX
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_open_critical_and_high_incidents_are_alerted_once(env, capsys):
    assert run(env) == 0
    lines = outbox(env)
    assert [line["incident_id"] for line in lines] == OPEN_HIGH  # not resolved, not medium
    first = lines[0]
    assert first["severity"] == "high" and first["period"] == "2023-05"
    assert "[HIGH] INC-0003 NULL rate increase: fare_amount" in first["text"]
    assert "most likely" in first["text"] and "dashboard: http://localhost:8501" in first["text"]
    assert "suggested action(s) waiting for approval" in first["text"]
    assert [(r[0], r[2], r[3]) for r in log(env)] == [(i, "outbox", "sent") for i in OPEN_HIGH]
    assert run(env) == 0  # again: nothing new
    assert len(outbox(env)) == 3 and "0 alert(s) sent" in capsys.readouterr().out


def test_a_failed_write_is_retried(env, capsys):
    env["outbox"].write_text("not a folder")  # the outbox folder cannot be made
    assert run(env) == 1
    assert {(r[3], r[4].split(":")[0]) for r in log(env)} == {("failed", "FileExistsError")}
    assert "3 failed" in capsys.readouterr().out
    env["outbox"].unlink()
    assert run(env) == 0
    assert [line["incident_id"] for line in outbox(env)] == OPEN_HIGH
    assert {r[3] for r in log(env)} == {"sent"}


def test_escalation_alerts_again(env):
    assert run(env) == 0
    with duckdb.connect(env["db"]) as con:
        con.execute("UPDATE incidents SET severity = 'critical' WHERE incident_id = 'INC-0004'")
        con.execute("UPDATE incidents SET status = 'OPEN' WHERE incident_id = 'INC-0001'")
    assert run(env) == 0
    (line,) = outbox(env)[3:]
    assert (line["incident_id"], line["severity"], line["escalated"]) == (
        "INC-0004",
        "critical",
        True,
    )
    assert "escalated to critical" in line["text"]  # medium INC-0001 still never alerts


def test_silent_records_without_sending(env, capsys):
    assert run(env, "--silent") == 0
    assert outbox(env) == []
    assert {r[3] for r in log(env)} == {"skipped"} and len(log(env)) == 3
    assert run(env) == 0 and outbox(env) == []  # already counted as alerted
    assert "recorded as alerted, nothing sent" in capsys.readouterr().out


def test_dataset_filter_and_settings(env, tmp_path):
    assert run(env, "--dataset", "ved_signals") == 0 and outbox(env) == []
    assert nt.load_settings().severities == ("critical", "high")
    bad = tmp_path / "alerts.yml"
    bad.write_text("severities: [urgent]\nchannels: [outbox]\n")
    with pytest.raises(ConfigError, match="severities"):
        nt.load_settings(bad)
    bad.write_text("severities: [high]\nchannels: [pager]\n")
    with pytest.raises(ConfigError, match="channels"):
        nt.load_settings(bad)


def test_rebuild_stage_and_experiments_without_alerts():
    names = [s[0] for s in rebuild.stages(rebuild.parse_args([]))]
    assert names[names.index("investigate") + 1] == "alert"
    assert "alert" not in [s[0] for s in rebuild.stages(rebuild.parse_args(["--no-alerts"]))]
