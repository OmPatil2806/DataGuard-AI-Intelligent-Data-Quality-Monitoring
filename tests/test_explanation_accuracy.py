"""explanation_accuracy.py (Phase 8.5): the investigation against the injected cause."""

import duckdb
import pytest
from evaluation_fixture import make_config, make_sources

from evaluation import explanation_accuracy as xa
from evaluation import run_experiments as rx

RUN = [
    "control",
    "missing_column",
    "null_spike-20pct",
    "null_spike_vendor-1pct",
    "null_spike_vendor-20pct",
    "volume_drop-20pct",
    "volume_drop-50pct",
]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("explanations")
    config = make_config(root, root / "workspace")
    rx.run_all(config, make_sources(root / "source"), RUN, out=lambda _: None)
    frame = xa.evaluate_explanations(config).set_index("experiment_id")
    return {"config": config, "frame": frame, "root": root}


def test_one_row_per_experiment_that_ran(env):
    assert sorted(env["frame"].index) == sorted(e for e in RUN if e != "control")
    assert set(env["frame"]["outcome"]) <= set(xa.OUTCOMES)


def test_right_causes(env):
    f = env["frame"]
    assert f.loc["missing_column", "outcome"] == "top1"
    assert f.loc["missing_column", "top_cause"] == "column_missing"
    assert f.loc["volume_drop-50pct", "top_cause"] == "rows_missing_evenly"
    assert f.loc["volume_drop-20pct", "outcome"] == "not_detected"  # within the volume limit
    assert pd_isna(f.loc["volume_drop-20pct", "incident_id"])


def test_where_and_when(env):
    f = env["frame"]
    spread = f.loc["null_spike-20pct"]
    assert (spread.outcome, spread.where_outcome) == ("top1", "ok")  # no where is claimed
    vendor = f.loc["null_spike_vendor-20pct"]
    assert (vendor.top_cause, vendor.where_claimed, vendor.where_outcome) == (
        "segment_sends_nulls", "vendor = 2", "found",
    )  # fmt: skip
    assert "vendor = 2" in vendor.headline
    one_row = f.loc["null_spike_vendor-1pct"]  # one broken row cannot show where it sits
    assert (one_row.where_claimed, one_row.where_outcome) == (None, "missed")
    assert pd_isna(f.loc["missing_column", "where_outcome"])  # month level: not judged


def test_accuracy_and_store(env):
    a = xa.accuracy(env["frame"].reset_index())
    assert a["detected"] == 5 and a["top1"] >= 4 and a["false_where"] == 0
    assert a["top1_accuracy"] == round(a["top1"] / 5, 4)
    csv, database = xa.store(env["config"], env["frame"].reset_index())
    assert csv.is_file()
    with duckdb.connect(str(database), read_only=True) as con:
        assert con.execute("SELECT count(*) FROM explanation_results").fetchone()[0] == 6


def test_main(env, capsys, tmp_path):
    assert xa.main(["--config", str(env["root"] / "faults.yml")]) == 0
    out = capsys.readouterr().out
    assert "5 of 6 experiment(s) detected: top-1" in out and "0 false where" in out
    make_config(tmp_path, tmp_path / "empty")
    assert xa.main(["--config", str(tmp_path / "faults.yml")]) == 2


def pd_isna(value):
    import pandas as pd

    return pd.isna(value)


# --- the judging rules, on made-up explanations ------------------------------------------------


def explanation(*leads, explained=True, where=None):
    facts = {"concentrated": where is not None}
    if where:
        facts |= {"where_dimension": where[0], "where_value": where[1]}
    return {"explained": explained, "facts": facts, "headline": "h",
            "causes": [{"cause_id": c, "confidence": "medium"} for c in leads]}  # fmt: skip


def test_judge_cause():
    right = ("unit_change",)
    assert xa.judge_cause(None, right) == "not_detected"
    assert xa.judge_cause(explanation("unit_change", "real_change"), right) == "top1"
    assert xa.judge_cause(explanation("scale_error", "unit_change"), right) == "top3"
    assert xa.judge_cause(explanation("a", "b", "c", "unit_change"), right) == "wrong"
    assert xa.judge_cause(explanation("scale_error", explained=False), right) == "unknown"
    assert xa.judge_cause(explanation("unit_change", explained=False), right) == "top3"
    assert xa.judge_cause(explanation(explained=False), right) == "unknown"


def test_judge_where():
    spread, vendor = "none", {"dimension": "vendor", "value": "2"}
    assert xa.judge_where(explanation("x"), spread) == "ok"
    assert xa.judge_where(explanation("x", where=("pickup_zone", "z")), spread) == "false_where"
    assert xa.judge_where(explanation("x", where=("vendor", "2")), vendor) == "found"
    assert xa.judge_where(explanation("x", where=("vendor", "1")), vendor) == "missed"
    assert xa.judge_where(explanation("x"), vendor) == "missed"
    assert xa.judge_where(explanation("x"), None) is None
    assert xa.judge_where(None, spread) is None
