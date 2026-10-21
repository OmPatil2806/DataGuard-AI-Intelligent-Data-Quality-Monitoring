"""Phase 5 'done when': Gold rebuilds from Bronze in one step with identical results."""

import shutil

import pytest
from pipeline_fixture import run_pipeline

from common import db
from gold import rebuild
from ingestion import ingest_data


def checksums(database):
    with db.connect(database, read_only=True) as con:
        return dict(
            con.execute(
                "SELECT table_name, checksum FROM gold_build_log "
                "WHERE run_id = (SELECT max(run_id) FROM gold_build_log)"
            ).fetchall()
        )


@pytest.fixture(scope="module")
def stage_by_stage(tmp_path_factory):
    """Every command run by hand, then Gold."""
    database = run_pipeline(tmp_path_factory.mktemp("by_hand"))
    from gold import monitoring_tables

    assert monitoring_tables.main(["--database", str(database)]) == 0
    return database


def one_step(root, incoming):
    """Bronze from the same files in a new folder, then everything else in one command."""
    database = root / "dataguard.duckdb"
    argv = ["--months", "2023-01:2023-05", "--source", str(incoming)]
    assert (
        ingest_data.main([*argv, "--bronze", str(root / "bronze"), "--database", str(database)])
        == 0
    )
    args = ["--lookups", str(incoming), "--silver", str(root / "silver")]
    args += ["--quarantine", str(root / "quarantine"), "--database", str(database)]
    return database, args


@pytest.fixture(scope="module")
def rebuilt(stage_by_stage, tmp_path_factory):
    root = tmp_path_factory.mktemp("one_step")
    incoming = root / "incoming"
    shutil.copytree(stage_by_stage.parent / "incoming", incoming)
    database, args = one_step(root, incoming)
    assert rebuild.main(args) == 0
    return database, args


def test_one_step_gives_the_same_gold_as_stage_by_stage(stage_by_stage, rebuilt):
    by_hand, one = checksums(stage_by_stage), checksums(rebuilt[0])
    assert len(by_hand) == 8
    assert by_hand == one  # other folders, other timestamps: identical content


def test_rebuilding_again_changes_nothing(rebuilt):
    database, args = rebuilt
    before = checksums(database)
    with db.connect(database, read_only=True) as con:
        incidents = con.execute("SELECT count(*) FROM incidents").fetchone()[0]
        builds = con.execute("SELECT count(DISTINCT run_id) FROM gold_build_log").fetchone()[0]
    assert rebuild.main(args) == 0
    assert checksums(database) == before
    with db.connect(database, read_only=True) as con:
        assert con.execute("SELECT count(*) FROM incidents").fetchone()[0] == incidents
        assert (
            con.execute("SELECT count(DISTINCT run_id) FROM gold_build_log").fetchone()[0]
            == builds + 1
        )


def test_force_rewrites_silver_with_the_same_result(rebuilt):
    database, args = rebuilt
    before = checksums(database)
    assert rebuild.main([*args, "--force"]) == 0
    assert checksums(database) == before


def test_stages_run_in_order(monkeypatch, tmp_path, capsys):
    calls = []
    for module, name in [
        (rebuild.process_silver, "silver"),
        (rebuild.run_checks, "checks"),
        (rebuild.run_rules, "rules"),
        (rebuild.incidents_cli, "incidents"),
        (rebuild.score, "score"),
        (rebuild.monitoring_tables, "gold"),
    ]:
        monkeypatch.setattr(module, "main", lambda argv, name=name: calls.append((name, argv)) or 0)
    assert (
        rebuild.main(["--database", str(tmp_path / "d.duckdb"), "--lookups", "L", "--force"]) == 0
    )
    assert [c[0] for c in calls] == ["silver", "checks", "rules", "incidents", "score", "gold"]
    argv = dict(calls)
    assert "--force" in argv["silver"] and "--force" not in argv["checks"]
    assert argv["rules"][argv["rules"].index("--lookups") + 1] == "L"
    assert argv["incidents"][-1] == "update"
    assert argv["gold"] == ["--database", str(tmp_path / "d.duckdb")]
    assert "--months" not in argv["silver"]
    calls.clear()
    assert rebuild.main(["--months", "2023-06"]) == 0
    argv = dict(calls)
    for stage in ("silver", "checks", "rules"):
        assert argv[stage][argv[stage].index("--months") + 1] == "2023-06", stage
    for stage in ("incidents", "score", "gold"):
        assert "--months" not in argv[stage], stage
    out = capsys.readouterr().out
    assert "rebuild summary:" in out and out.count(" ok") >= 6


def test_failed_stage_continues_but_config_error_stops(monkeypatch, capsys):
    calls = []

    def stage(name, code):
        return lambda argv: calls.append(name) or code

    monkeypatch.setattr(rebuild.process_silver, "main", stage("silver", 1))
    monkeypatch.setattr(rebuild.run_checks, "main", stage("checks", 0))
    monkeypatch.setattr(rebuild.run_rules, "main", stage("rules", 2))
    monkeypatch.setattr(rebuild.incidents_cli, "main", stage("incidents", 0))
    monkeypatch.setattr(rebuild.score, "main", stage("score", 0))
    monkeypatch.setattr(rebuild.monitoring_tables, "main", stage("gold", 0))
    assert rebuild.main([]) == 2
    assert calls == ["silver", "checks", "rules"]
    out = capsys.readouterr().out
    assert "rules stopped the rebuild" in out and "FAILED" in out and "STOPPED" in out
