"""The experiment runner (Phase 6.3) on clean test months: history once, faults on June."""

import json

import duckdb
import pytest
from evaluation_fixture import make_config, make_sources

from evaluation import fault_injector as fi
from evaluation import run_experiments as rx

SOME = [
    "control",
    "missing_column",
    "null_spike-20pct",
    "duplicate_rows-5pct",
    "volume_drop-50pct",
    "stale_file-20d",
]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("experiments")
    source = make_sources(root / "source")
    config = make_config(root, root / "workspace")
    lines = []
    records = rx.run_all(config, source, SOME, out=lines.append)
    return {"config": config, "source": source, "records": records, "lines": lines, "root": root}


def con_for(env, exp_id):
    path = fi.experiment_dir(env["config"], exp_id) / rx.DATABASE
    return duckdb.connect(str(path), read_only=True)


def june_findings(env, exp_id):
    with con_for(env, exp_id) as con:
        return con.execute(
            """
            SELECT "check", column_name, severity FROM dataset_check_results
            WHERE period = '2023-06' AND status = 'fail' AND run_id = (
                SELECT max(run_id) FROM dataset_check_results WHERE period = '2023-06')
            ORDER BY 1, 2
            """
        ).fetchall()


def june_incidents(env, exp_id):
    with con_for(env, exp_id) as con:
        return con.execute(
            "SELECT rule_id, column_name, severity FROM incidents "
            "WHERE first_seen_period = '2023-06' ORDER BY 1, 2"
        ).fetchall()


def test_every_experiment_ran(env):
    plan = [exp_id for exp_id, _, _ in env["config"].experiments() if exp_id in SOME]
    assert [r.experiment_id for r in env["records"]] == plan  # in the order of faults.yml
    for record in env["records"]:
        assert record.ran and record.error is None, record
        assert list(record.stages) == [
            "ingest", "silver", "checks", "rules", "ml", "incidents", "score",
            "investigate", "gold",
        ]  # fmt: skip
        assert set(record.stages.values()) == {0}
    assert env["lines"][0].startswith("history 2023-01 to 2023-05: built")
    assert len(env["lines"]) == 1 + len(SOME)


def test_experiment_folders(env):
    config = env["config"]
    for exp_id in SOME:
        folder = fi.experiment_dir(config, exp_id)
        manifest = fi.Manifest.read(folder / "incoming" / fi.MANIFEST)
        assert manifest.experiment_id == exp_id
        assert rx.ExperimentRecord.read(folder) == next(
            r for r in env["records"] if r.experiment_id == exp_id
        )
        assert (folder / "log.txt").stat().st_size > 0
        with con_for(env, exp_id) as con:
            sha = con.execute("SELECT sha256 FROM gold_months WHERE period = '2023-06'").fetchone()[
                0
            ]
        assert sha == manifest.output_sha256  # the broken file is the month in the pipeline


def test_history_is_shared_and_untouched(env):
    history = fi.history_dir(env["config"])
    with duckdb.connect(str(history / rx.DATABASE), read_only=True) as con:
        periods = [
            p for (p,) in con.execute("SELECT period FROM gold_months ORDER BY 1").fetchall()
        ]
    assert periods == ["2023-01", "2023-02", "2023-03", "2023-04", "2023-05"]
    for exp_id in SOME:
        with con_for(env, exp_id) as con:
            months = con.execute("SELECT count(*) FROM gold_months").fetchone()[0]
            history_runs = con.execute(
                "SELECT count(*) FROM pipeline_runs WHERE run_id <= 'run_0007'"
            ).fetchone()[0]
        assert months == 6 and history_runs == 7  # the same history in every experiment


def test_control_is_clean(env):
    assert june_findings(env, "control") == []
    assert june_incidents(env, "control") == []
    with con_for(env, "control") as con:
        assert (
            con.execute(
                "SELECT health FROM gold_dataset_health WHERE period = '2023-06'"
            ).fetchone()[0]
            == "healthy"
        )


def test_faults_reach_the_pipeline(env):
    assert ("schema", "", "critical") in june_findings(env, "missing_column")
    assert june_findings(env, "null_spike-20pct") == [("null_rate", "fare_amount", "high")]
    assert june_findings(env, "volume_drop-50pct") == [("volume", "", "medium")]
    assert june_findings(env, "stale_file-20d") == [("freshness", "", "medium")]
    assert june_incidents(env, "duplicate_rows-5pct") == [("trip_unique", "", "medium")]


def test_stale_file_delivery_delay(env):
    record = next(r for r in env["records"] if r.experiment_id == "stale_file-20d")
    assert record.delivery_delay_days == 20
    with con_for(env, "stale_file-20d") as con:
        days = con.execute(
            "SELECT date_diff('day', expected_at, simulated_delivered_at) FROM file_manifest "
            "WHERE period = '2023-06'"
        ).fetchone()[0]
    assert days == 20


def test_history_is_reused_until_something_changes(env, tmp_path):
    config, source = env["config"], env["source"]
    lines = []
    rx.run_all(config, source, ["control"], out=lines.append)
    assert lines[0].startswith("history 2023-01 to 2023-05: reused")
    rx.run_all(config, source, ["control"], fresh=True, out=lines.append)
    assert lines[2].startswith("history 2023-01 to 2023-05: built")
    changed = make_config(tmp_path, config.workspace, seed=7)  # another sample
    rx.run_all(changed, source, ["control"], out=lines.append)
    assert lines[4].startswith("history 2023-01 to 2023-05: built")
    marker = json.loads((fi.history_dir(config) / "history.json").read_text())
    assert marker["fingerprint"] == rx.fingerprint(changed, rx.source_files(changed, source))
    rx.run_all(config, source, ["control"], out=lines.append)  # back to the module's history
    assert lines[6].startswith("history 2023-01 to 2023-05: built")


def test_a_broken_setup_is_recorded_not_raised(env, monkeypatch):
    def broken(frame, strength, params, rng):
        raise ValueError("cannot break this")

    monkeypatch.setitem(fi.FAULTS, "out_of_range", broken)
    (record,) = rx.run_all(env["config"], env["source"], ["out_of_range-5pct"], out=lambda _: None)
    assert not record.ran and "cannot break this" in record.error
    assert rx.ExperimentRecord.read(fi.experiment_dir(env["config"], "out_of_range-5pct")) == record
    assert rx.summary([record])["errors"] == {"out_of_range-5pct": record.error}


def test_input_errors(env, tmp_path):
    with pytest.raises(rx.ExperimentError, match="unknown experiment"):
        rx.run_all(env["config"], env["source"], ["speeding-5pct"])
    with pytest.raises(rx.ExperimentError, match="missing source file"):
        rx.run_all(env["config"], tmp_path)


def test_main(env, tmp_path, capsys):
    config_file = env["root"] / "faults.yml"
    argv = ["--config", str(config_file), "--source", str(env["source"])]
    assert rx.main([*argv, "--only", "control"]) == 0
    out = capsys.readouterr().out
    assert "1 of 1 experiment(s) ran" in out and "control" in out
    assert rx.main([*argv[:2], "--source", str(tmp_path), "--only", "control"]) == 2
    assert "missing source file" in capsys.readouterr().out
