"""investigate.py (Phase 8.4): every incident investigated, actions kept, reports, Gold."""

import duckdb
import pytest
from pipeline_fixture import run_pipeline

from gold import monitoring_tables as gold
from incidents.incident_manager import IncidentError
from investigation import investigate as iv

IDS = [f"INC-000{i}" for i in range(1, 7)]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("investigate")
    database = run_pipeline(root)
    reports = root / "reports"
    with duckdb.connect(str(database)) as con:
        run_id, results = iv.investigate_all(con, "yellow_taxi", reports=reports)
    return {"db": database, "reports": reports, "run_id": run_id, "results": results,
            "folder": reports / "investigations" / "yellow_taxi"}  # fmt: skip


def query(env, sql, params=()):
    with duckdb.connect(str(env["db"])) as con:
        return con.execute(sql, list(params)).fetchall()


def test_every_incident_is_investigated(env):
    rows = query(env, "SELECT incident_id, run_id, explained, top_cause, confidence, period "
                      "FROM investigations ORDER BY 1")  # fmt: skip
    assert [r[0] for r in rows] == IDS
    assert {r[1] for r in rows} == {env["run_id"]} and all(r[2] for r in rows)
    assert dict((r[0], r[3]) for r in rows)["INC-0004"] == "unit_change"
    assert query(env, "SELECT count(*) FROM investigation_evidence")[0][0] == 6
    status = query(env, "SELECT status, command FROM pipeline_runs WHERE run_id = ?",
                   [env["run_id"]])  # fmt: skip
    assert status == [("SUCCESS", "investigate")]


def test_actions_are_pending(env):
    actions = query(env, "SELECT incident_id, action_id, status FROM investigation_actions")
    assert actions and {a[2] for a in actions} == {"pending_approval"}
    suggested = sum(len(e["actions"]) for e in env["results"])
    assert len(actions) == suggested


def test_reports(env):
    files = sorted(p.name for p in env["folder"].glob("*.md"))
    assert files == [f"{i}.md" for i in IDS]
    text = (env["folder"] / "INC-0004.md").read_text(encoding="utf-8")
    for section in ("## Likely cause", "## Where and when", "## Suggested actions",
                    "## Since when", "## Impact", "## Metrics", "## File versions",
                    "## Related incidents", "## What is tested"):  # fmt: skip
        assert section in text, section
    assert "kilometres" in text and "pending_approval" in text
    nulls = (env["folder"] / "INC-0003.md").read_text(encoding="utf-8")
    assert "## Sample rows" in nulls and "unknown: the affected rows are spread" in nulls
    rule = (env["folder"] / "INC-0006.md").read_text(encoding="utf-8")
    assert "Rule `pickup_in_file_month`" in rule
    paths = query(env, "SELECT report_path FROM investigations WHERE incident_id = 'INC-0004'")
    assert paths[0][0].endswith("investigations/yellow_taxi/INC-0004.md")


def test_again_gives_the_same_and_keeps_decisions(env):
    before = (env["folder"] / "INC-0004.md").read_text(encoding="utf-8")
    md5 = query(env, "SELECT evidence_md5 FROM investigations WHERE incident_id = 'INC-0004'")
    with duckdb.connect(str(env["db"])) as con:
        iv.decide(con, "INC-0004", "contact_provider", "approved", "om")
        run_id, _ = iv.investigate_all(con, "yellow_taxi", reports=env["reports"])
    assert run_id != env["run_id"]
    assert (
        query(env, "SELECT evidence_md5 FROM investigations WHERE incident_id = 'INC-0004'") == md5
    )
    decided = query(
        env,
        "SELECT status, decided_by FROM investigation_actions "
        "WHERE incident_id = 'INC-0004' AND action_id = 'contact_provider'",
    )
    assert decided == [("approved", "om")]
    after = (env["folder"] / "INC-0004.md").read_text(encoding="utf-8")
    assert after == before.replace(
        "| contact_provider | contact_provider | pending_approval |",
        "| contact_provider | contact_provider | approved |",
    )


def test_actions_no_longer_suggested(env):
    with duckdb.connect(str(env["db"])) as con:
        iv.store_actions(con, "INC-0002", [])  # nothing suggested any more
        assert iv.stored_actions(con, "INC-0002") == []  # pending ones are dropped
        iv.store_actions(con, "INC-0004", [])
        kept = iv.stored_actions(con, "INC-0004")  # a decided one stays as a record
        assert [(a["action_id"], a["status"]) for a in kept] == [("contact_provider", "approved")]
        with pytest.raises(ValueError, match="decision"):
            iv.decide(con, "INC-0004", "contact_provider", "pending_approval", "om")
        with pytest.raises(IncidentError, match="no suggested action"):
            iv.decide(con, "INC-0004", "launch", "approved", "om")
        iv.investigate_all(con, "yellow_taxi", ["INC-0002", "INC-0004"], env["reports"])
        assert len(iv.stored_actions(con, "INC-0002")) == 2


def test_gold_investigations(env):
    with duckdb.connect(str(env["db"])) as con:
        gold.rebuild(con)
        rows = con.execute(
            "SELECT incident_id, top_cause, actions, actions_pending, actions_approved "
            "FROM gold_investigations ORDER BY 1"
        ).fetchall()
        first = con.execute(
            "SELECT checksum FROM gold_build_log WHERE table_name = 'gold_investigations' "
            "ORDER BY run_id DESC LIMIT 1"
        ).fetchone()[0]
        iv.investigate_all(con, "yellow_taxi", reports=env["reports"])
        gold.rebuild(con)
        second = con.execute(
            "SELECT checksum FROM gold_build_log WHERE table_name = 'gold_investigations' "
            "ORDER BY run_id DESC LIMIT 1"
        ).fetchone()[0]
    assert [r[0] for r in rows] == IDS
    approved = {r[0]: r[4] for r in rows}
    assert approved["INC-0004"] == 1 and sum(approved.values()) == 1
    assert all(r[2] == r[3] + r[4] for r in rows)
    assert first == second  # run ids and report paths are volatile


def test_main(env, capsys, tmp_path):
    db = ["--database", str(env["db"])]
    assert iv.main([*db, "--incident", "INC-0003", "--reports", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "1 incident(s) investigated, 0 with cause unknown" in out
    assert (tmp_path / "investigations" / "yellow_taxi" / "INC-0003.md").exists()
    assert iv.main([*db, "--no-reports"]) == 0
    assert "reports in" not in capsys.readouterr().out
    assert iv.main([*db, "--incident", "INC-9999"]) == 2
    assert iv.main([*db, "--decide", "INC-0003", "contact_provider", "done"]) == 2
    assert "--by" in capsys.readouterr().out
    assert iv.main([*db, "--decide", "INC-0003", "contact_provider", "done", "--by", "om"]) == 0
    assert "done by om" in capsys.readouterr().out
    assert iv.main([*db, "--decide", "INC-0003", "x", "done", "--by", "om"]) == 2


def test_reports_dir_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(iv.REPORTS_DIR_ENV, str(tmp_path))
    assert iv.reports_dir() == tmp_path
    monkeypatch.delenv(iv.REPORTS_DIR_ENV)
    assert iv.reports_dir().parts[-2:] == ("data", "reports")
