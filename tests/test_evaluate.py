"""evaluate.py (Phase 6.4): outcomes against the ground truth and the control."""

import duckdb
import pandas as pd
import pytest
from evaluation_fixture import make_config, make_sources

from evaluation import evaluate as ev
from evaluation import fault_injector as fi
from evaluation import run_experiments as rx
from evaluation.evaluate import MonthView

RUN = [
    "control",
    "missing_column",
    "duplicate_rows-5pct",
    "out_of_range-1pct",
    "out_of_range-20pct",
    "unit_change-1pct",
    "unit_change-100pct",
    "volume_drop-20pct",
    "volume_drop-50pct",
    "stale_file-1d",
    "stale_file-20d",
    "plausible_but_wrong-20pct",
]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("evaluate")
    config = make_config(root, root / "workspace")
    rx.run_all(config, make_sources(root / "source"), RUN, out=lambda _: None)
    frame = ev.evaluate(config)
    return {"config": config, "frame": frame.set_index("experiment_id"), "root": root}


def row(env, exp_id):
    return env["frame"].loc[exp_id]


def test_one_row_per_experiment_that_ran(env):
    assert sorted(env["frame"].index) == sorted(e for e in RUN if e != "control")
    assert set(env["frame"]["outcome"]) <= set(ev.OUTCOMES)


def test_rule_fault_rows(env):
    big = row(env, "out_of_range-20pct")
    assert big.detector == "rule fare_not_negative" and big.outcome == "caught"
    assert big.recall == 1.0 and big.precision == 1.0 and big.extra_flags == 0
    assert big.caught_rows == big.broken_rows > 0 and big.incident_severity == "high"
    small = row(env, "out_of_range-1pct")  # rows quarantined, the rate stays within its normal
    assert small.recall == 1.0 and not small.incident and small.outcome == "flagged"
    assert row(env, "duplicate_rows-5pct").outcome == "caught"


def test_check_faults(env):
    schema = row(env, "missing_column")
    assert (schema.outcome, schema.check_severity, schema.incident_severity) == (
        "caught", "critical", "critical",
    )  # fmt: skip
    assert pd.isna(schema.broken_rows) and "null_rate passenger_count" in schema.side_findings
    assert row(env, "unit_change-100pct").outcome == "caught"
    assert row(env, "volume_drop-50pct").check_severity == "medium"
    assert row(env, "stale_file-20d").outcome == "caught"


def test_misses_keep_what_the_check_saw(env):
    for exp_id in ("unit_change-1pct", "volume_drop-20pct", "stale_file-1d"):
        miss = row(env, exp_id)
        assert miss.outcome == "missed" and not miss.incident and not miss.check_fired, exp_id
        assert miss.check_observed and miss.check_expected, exp_id
    assert row(env, "unit_change-1pct").check_observed.startswith("PSI 0.0")
    assert "± 30%" in row(env, "volume_drop-20pct").check_expected
    assert "grace" in row(env, "stale_file-1d").check_expected


def test_score_drop_against_the_control(env):
    frame = env["frame"]
    assert frame["control_score"].nunique() == 1
    assert (frame["score_drop"] == (frame["control_score"] - frame["score"]).round(4)).all()
    assert row(env, "missing_column").score_drop > 10  # critical schema finding
    assert row(env, "missing_column").health == "critical"


def test_control_raises_nothing(env):
    assert ev.control_findings(env["config"]) == []  # no ML incident either
    rate, flagged = ev.control_ml(env["config"])
    assert 0 < rate <= 0.02 and flagged >= 1  # the model's calibrated ~1% of normal flags


def test_ml_catches_what_no_rule_sees(env):
    row = env["frame"].loc["plausible_but_wrong-20pct"]
    assert row.detector == "ml anomaly"
    assert (row.rules_outcome, row.ml_outcome, row.combined_outcome) == (
        "missed", "caught", "caught",
    )  # fmt: skip
    assert row.outcome == "caught" and row.incident and row.caught_rows >= 1
    table = ev.comparison(env["frame"].reset_index()).set_index("experiment_id")
    assert table.loc["plausible_but_wrong-20pct", "ml_adds"]
    rule_faults = ["duplicate_rows-5pct", "out_of_range-20pct", "missing_column"]
    assert not table.loc[rule_faults, "ml_adds"].any()  # the rules already catch these
    assert (table["ml_outcome"].notna()).all()  # every experiment was scored by the model


def test_results_are_stored(env):
    csv, database = ev.store(env["config"], env["frame"].reset_index())
    assert len(pd.read_csv(csv)) == len(env["frame"])
    with duckdb.connect(str(database), read_only=True) as con:
        stored = con.execute(
            "SELECT count(*), count_if(outcome = 'caught') FROM evaluation_results"
        ).fetchone()
    assert stored == (len(env["frame"]), int((env["frame"]["outcome"] == "caught").sum()))


def test_main(env, capsys):
    assert ev.main(["--config", str(env["root"] / "faults.yml")]) == 0
    out = capsys.readouterr().out
    assert "rules only:" in out and "rules + ML:" in out and "of the control month" in out
    assert "detector(s)" in out and "control month raises: nothing" in out
    assert "evaluation_results" in out


def test_main_without_control(tmp_path, capsys):
    config = make_config(tmp_path, tmp_path / "empty")
    assert ev.main(["--config", str(tmp_path / "faults.yml")]) == 2
    assert "no control experiment" in capsys.readouterr().out
    assert config.workspace == tmp_path / "empty"


# --- the rules of attribution, on made-up views ------------------------------------------------


def view(checks=None, incidents=None, rule_rows=None, score=90.0):
    return MonthView(checks or {}, incidents or {}, rule_rows or {}, score, "warning")


def manifest(expect, affected=(), month_level=False):
    return fi.Manifest(
        "f-5pct", "f", "validity", "share", 0.05, "", "yellow_taxi", "2023-06", 1.0, 42,
        "s", "x", "o", "y", 10, 10, 10, month_level, list(affected), [], None, expect, "",
    )  # fmt: skip


RECORD = rx.ExperimentRecord("f-5pct", "f", 0.05, stages={"ingest": 0, "silver": 0})


def test_a_check_failing_in_the_control_is_not_credited():
    failing = {("volume", ""): ("fail", "medium", "1", "2")}
    (r,) = ev.evaluate_experiment(
        manifest([{"kind": "check", "id": "volume"}], month_level=True),
        RECORD,
        view(checks=failing, incidents={("dataset_check", "volume", ""): "medium"}),
        view(checks=failing, incidents={("dataset_check", "volume", ""): "medium"}, score=95),
    )
    assert (r.outcome, r.check_fired, r.incident) == ("missed", False, False)
    assert r.score_drop == 5


def test_rule_rows_and_extra_flags():
    (r,) = ev.evaluate_experiment(
        manifest([{"kind": "rule", "id": "fare_not_negative"}], affected=[1, 2, 3, 4]),
        RECORD,
        view(rule_rows={"fare_not_negative": {1, 2, 3, 9, 10}}),
        view(rule_rows={"fare_not_negative": {10}}),
    )
    # 3 of 4 broken rows caught; row 10 the control flags too; row 9 is an extra flag
    assert (r.caught_rows, r.extra_flags, r.recall, r.precision) == (3, 1, 0.75, 0.75)
    assert r.outcome == "flagged"


def test_side_findings_and_errors():
    (r,) = ev.evaluate_experiment(
        manifest([{"kind": "check", "id": "schema"}], month_level=True),
        RECORD,
        view(
            checks={
                ("schema", ""): ("fail", "critical", "a", "b"),
                ("drift", "fare_amount"): ("fail", "high", "PSI 1", "PSI < 0.1"),
            },
            incidents={("row_rule", "trip_unique", ""): "medium"},
        ),  # fmt: skip
        view(),
    )
    assert r.outcome == "caught" and r.check_severity == "critical"
    assert r.side_findings == "drift fare_amount (high); incident trip_unique"
    failed = rx.ExperimentRecord("f-5pct", "f", 0.05, error="ValueError: no")
    (e,) = ev.evaluate_experiment(
        manifest([{"kind": "check", "id": "schema"}]), failed, None, view()
    )
    assert (e.outcome, e.error) == ("error", "ValueError: no")


def test_compare_counts_only_new_flags():
    # rows 1-4 broken; the rule already fails row 2 in the control, the model row 3
    m = manifest([{"kind": "ml", "id": "anomaly"}], affected=[1, 2, 3, 4])
    control = view(rule_rows={"fare_not_negative": {2}})
    control = MonthView(**{**control.__dict__, "ml_rows": frozenset({3}), "ml_rate": 0.01})
    seen = view(rule_rows={"fare_not_negative": {2}})
    seen = MonthView(**{**seen.__dict__, "ml_rows": frozenset({3, 4, 9}), "ml_rate": 0.03})
    result = ev.compare(m, seen, control)
    assert result["rules_outcome"] == "missed"  # row 2 failed the rule before the fault
    assert (result["ml_caught_rows"], result["ml_recall"]) == (1, 0.25)  # only row 4 is new
    assert result["ml_extra_flags"] == 1  # row 9, beyond the control's one flag
    assert (result["ml_outcome"], result["combined_outcome"]) == ("flagged", "flagged")
    alert = MonthView(
        **{**seen.__dict__, "incidents": {("dataset_check", "ml_anomaly", ""): "medium"}}
    )
    assert ev.compare(m, alert, control)["ml_outcome"] == "caught"
    without_model = view()
    assert ev.compare(m, without_model, view())["ml_outcome"] is None
