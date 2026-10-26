"""Fault experiments on VED (Phase 9.6): configs/faults_ved.yml on the VED fixture."""

import shutil

import pytest
import yaml
from ved_fixture import make_unpacked

from common.config import REPO_ROOT, load_datasets_config
from evaluation import evaluate as ev
from evaluation import explanation_accuracy as xa
from evaluation import fault_injector as fi
from evaluation import run_experiments as rx
from ingestion import prepare_source

RUN = ["control", "missing_column", "duplicate_rows-20pct", "orphan_location-20pct"]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("ved_experiments")
    incoming = root / "incoming"
    make_unpacked(incoming / "ved_signals")
    dataset = load_datasets_config().dataset("ved_signals")
    prepare_source.prepare(dataset, incoming, ["2017-11", "2017-12"])
    shutil.rmtree(incoming / "ved_signals" / "staged")  # month files and lookups remain
    for week in (incoming / "ved_signals").glob("VED_*_week.csv"):
        week.unlink()
    data = yaml.safe_load((REPO_ROOT / "configs" / "faults_ved.yml").read_text())
    data |= {"target_month": "2017-12", "history": {"start": "2017-11", "end": "2017-11"},
             "sample_fraction": 1.0, "workspace": str(root / "workspace")}  # fmt: skip
    path = root / "faults_ved.yml"
    path.write_text(yaml.safe_dump(data))
    config = fi.load_faults_config(path)
    records = rx.run_all(config, incoming, RUN, out=lambda _: None)
    return {"config": config, "records": {r.experiment_id: r for r in records}}


def test_the_faults_config_matches_ved():
    config = fi.load_faults_config(REPO_ROOT / "configs" / "faults_ved.yml")
    assert config.dataset == "ved_signals" and config.target_month == "2018-02"
    assert len(config.experiments()) == 25 and all(f.causes for f in config.faults.values())
    vendor = config.faults["null_spike_vendor"]
    assert vendor.where == {"dimension": "vehicle", "value": "560"}
    assert vendor.expect == (fi.Expect("rule", "battery_signals_present_ev"),)


def test_experiments_run_without_the_model(env):
    for exp_id, record in env["records"].items():
        assert record.ran and record.error is None, exp_id
    history = rx.ExperimentRecord.read(fi.history_dir(env["config"]))
    assert not any(stage.endswith("_after_training") for stage in history.stages)  # no ML


def test_faults_are_caught_and_explained(env):
    detected = ev.evaluate(env["config"]).set_index("experiment_id")
    assert set(detected["outcome"]) == {"caught"}
    assert ev.control_findings(env["config"]) == [] or all(
        "volume" in f for f in ev.control_findings(env["config"])
    )  # the fixture's few rows a day fail the volume limits in every month alike
    explained = xa.evaluate_explanations(env["config"]).set_index("experiment_id")
    assert explained["top_cause"].to_dict() == {
        "missing_column": "column_missing",
        "duplicate_rows-20pct": "duplicate_delivery",
        "orphan_location-20pct": "unknown_location",
    }
    assert set(explained["outcome"]) == {"top1"}
