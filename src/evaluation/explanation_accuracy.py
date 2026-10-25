"""Explanation accuracy: does the investigation name the cause that was injected? (Phase 8)

    python src/evaluation/explanation_accuracy.py       # after run_experiments.py

Every fault in configs/faults.yml says which playbook causes count as the right explanation
(`explain.causes`) and where the problem sits (`explain.where`). The rebuild of each experiment
has already investigated its incidents (the investigate stage); this reads, per experiment, the
investigation of the incident the fault's detector raised, if it is new against the control:

    top1           the best cause is a right one
    top3           a right cause is among the three best
    wrong          a cause was given, none of the three is right
    unknown        the explainer said "cause unknown" and no right cause is among its leads
    not_detected   no new incident of the expected detector: nothing to explain

and the place:

    ok             where = none and no where/when was claimed
    false_where    where = none, but a concentration was claimed (a guess the data does not hold)
    found          the expected segment is the strongest concentration
    missed         the expected segment was not named

Accuracy is counted over the detected experiments only: a fault no detector caught is the
detection evaluation's miss (evaluate.py), not the explainer's.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import pandas as pd

if __package__ in (None, ""):
    # Allow running this file directly: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import ConfigError  # noqa: E402
from evaluation import fault_injector as fi  # noqa: E402
from evaluation.evaluate import RESULTS_DB  # noqa: E402
from evaluation.run_experiments import DATABASE, ExperimentRecord  # noqa: E402

OUTCOMES = ("top1", "top3", "wrong", "unknown", "not_detected")
WHERE_OUTCOMES = ("ok", "false_where", "found", "missed")
CSV = "explanation_results.csv"


@dataclass
class ExplanationResult:
    experiment_id: str
    fault_id: str
    strength: float | None
    incident_id: str | None
    detector: str | None
    expected_causes: str
    top_cause: str | None
    confidence: str | None
    leads: str  # the three best causes, best first
    outcome: str
    expected_where: str | None
    where_claimed: str | None
    where_outcome: str | None
    headline: str | None


def _key(expect: fi.Expect) -> tuple[str, str, str]:
    if expect.kind == "rule":
        return ("row_rule", expect.id, "")
    if expect.kind == "ml":
        return ("dataset_check", "ml_anomaly", "")
    return ("dataset_check", expect.id, expect.column)


def _incident_keys(database: Path, dataset: str, period: str) -> set[tuple[str, str, str]]:
    with duckdb.connect(str(database), read_only=True) as con:
        return set(
            con.execute(
                "SELECT source, rule_id, column_name FROM incidents WHERE dataset = ? "
                "AND last_seen_period = ?",
                [dataset, period],
            ).fetchall()
        )


def _investigations(database: Path, dataset: str, period: str) -> dict[tuple, dict]:
    """Per incident key of the month: the incident id and its stored explanation."""
    with duckdb.connect(str(database), read_only=True) as con:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
        if "investigations" not in tables:
            return {}
        rows = con.execute(
            """
            SELECT i.source, i.rule_id, i.column_name, i.incident_id, v.explanation
            FROM incidents i JOIN investigations v USING (incident_id)
            WHERE i.dataset = ? AND i.last_seen_period = ? AND v.period = ?
            """,
            [dataset, period, period],
        ).fetchall()
    return {(s, r, c): {"incident_id": i, **json.loads(e)} for s, r, c, i, e in rows}


def judge_cause(explanation: dict | None, causes: tuple[str, ...]) -> str:
    if explanation is None:
        return "not_detected"
    leads = [c["cause_id"] for c in explanation["causes"]]
    if leads and explanation["explained"] and leads[0] in causes:
        return "top1"
    if any(c in causes for c in leads[:3]):
        return "top3"
    return "wrong" if explanation["explained"] else "unknown"


def claimed_where(explanation: dict) -> str | None:
    facts = explanation.get("facts") or {}
    if not facts.get("concentrated"):
        return None
    return f"{facts.get('where_dimension')} = {facts.get('where_value')}"


def judge_where(explanation: dict | None, expected: str | dict | None) -> str | None:
    if explanation is None or expected is None:
        return None
    claimed = claimed_where(explanation)
    if expected == "none":
        return "ok" if claimed is None else "false_where"
    return "found" if claimed == f"{expected['dimension']} = {expected['value']}" else "missed"


def evaluate_explanations(config: fi.FaultsConfig) -> pd.DataFrame:
    """One row per experiment that ran (the control aside)."""
    experiments = config.workspace / "experiments"
    control = experiments / fi.CONTROL / DATABASE
    if not control.is_file():
        raise FileNotFoundError(f"no control experiment in {experiments} (run run_experiments.py)")
    in_control = _incident_keys(control, config.dataset, config.target_month)
    results = []
    for exp_id, fault_id, strength in config.experiments():
        folder = experiments / exp_id
        if exp_id == fi.CONTROL or not (folder / DATABASE).is_file():
            continue
        if not ExperimentRecord.read(folder).ran:
            continue
        spec = config.faults[fault_id]
        found = _investigations(folder / DATABASE, config.dataset, config.target_month)
        explanation, detector, incident = None, None, None
        for expect in spec.expect:
            key = _key(expect)
            if key in found and key not in in_control:
                explanation, incident = found[key], found[key]["incident_id"]
                detector = " ".join(k for k in key[1:] if k)
                break
        leads = [c["cause_id"] for c in explanation["causes"]] if explanation else []
        expected_where = spec.where
        results.append(
            ExplanationResult(
                exp_id,
                fault_id,
                strength,
                incident,
                detector,
                ", ".join(spec.causes),
                leads[0] if leads else None,
                explanation["causes"][0]["confidence"] if leads else None,
                ", ".join(leads[:3]),
                judge_cause(explanation, spec.causes),
                None
                if expected_where is None
                else expected_where
                if isinstance(expected_where, str)
                else f"{expected_where['dimension']} = {expected_where['value']}",
                claimed_where(explanation) if explanation else None,
                judge_where(explanation, expected_where),
                explanation["headline"] if explanation else None,
            )
        )
    return pd.DataFrame(
        [asdict(r) for r in results], columns=list(ExplanationResult.__dataclass_fields__)
    )


def accuracy(frame: pd.DataFrame) -> dict[str, object]:
    """Top-1 and top-3 accuracy over the detected experiments, and the where/when results."""
    detected = frame[frame["outcome"] != "not_detected"]
    n = len(detected)
    top1 = int((detected["outcome"] == "top1").sum())
    top3 = top1 + int((detected["outcome"] == "top3").sum())
    where = frame["where_outcome"].value_counts().to_dict()
    return {
        "experiments": len(frame),
        "detected": n,
        "top1": top1,
        "top3": top3,
        "top1_accuracy": round(top1 / n, 4) if n else None,
        "top3_accuracy": round(top3 / n, 4) if n else None,
        "wrong": int((detected["outcome"] == "wrong").sum()),
        "unknown": int((detected["outcome"] == "unknown").sum()),
        "false_where": int(where.get("false_where", 0)),
        "where_found": int(where.get("found", 0)),
        "where_missed": int(where.get("missed", 0)),
        "where_ok": int(where.get("ok", 0)),
    }


def store(config: fi.FaultsConfig, frame: pd.DataFrame) -> tuple[Path, Path]:
    csv = config.workspace / CSV
    frame.to_csv(csv, index=False)
    database = config.workspace / RESULTS_DB
    with duckdb.connect(str(database)) as con:
        con.register("results", frame)
        con.execute("CREATE OR REPLACE TABLE explanation_results AS SELECT * FROM results")
    return csv, database


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Judge the explanations of the experiments.")
    parser.add_argument("--config", type=Path, help="faults config (default: configs/faults.yml)")
    args = parser.parse_args(argv)
    try:
        config = fi.load_faults_config(args.config)
        frame = evaluate_explanations(config)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"error: {exc}")
        return 2
    csv, database = store(config, frame)
    columns = ["experiment_id", "detector", "top_cause", "confidence", "outcome",
               "where_claimed", "where_outcome"]  # fmt: skip
    with pd.option_context("display.width", 220, "display.max_colwidth", 40):
        print(frame[columns].fillna("").to_string(index=False))
    a = accuracy(frame)
    if a["detected"]:
        print(
            f"\n{a['detected']} of {a['experiments']} experiment(s) detected: top-1 "
            f"{a['top1']} ({a['top1_accuracy']:.0%}), top-3 {a['top3']} "
            f"({a['top3_accuracy']:.0%}), {a['wrong']} wrong, {a['unknown']} unknown"
        )
    else:
        print(f"\nnone of {a['experiments']} experiment(s) detected: nothing to explain")
    print(
        f"where/when: {a['where_ok']} spread faults left unexplained as they should, "
        f"{a['false_where']} false where, {a['where_found']} found, {a['where_missed']} missed"
    )
    print(f"-> {csv}\n-> {database} (explanation_results)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
