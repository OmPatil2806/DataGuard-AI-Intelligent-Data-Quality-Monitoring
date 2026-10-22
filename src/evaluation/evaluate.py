"""Score the detection experiments against their ground truth and the control run.

    python src/evaluation/evaluate.py            # after run_experiments.py

For every experiment and every detector its manifest expects, on the target month only:

    rows        rule detectors: broken rows the rule flagged (caught_rows, recall) and rows it
                flagged beyond the broken rows and what the control already flags (extra_flags,
                precision)
    check       check detectors: did the expected check fail, with what severity; observed and
                expected values are kept either way, so a miss shows how far below the limit
                the fault stayed
    incident    was an incident opened or updated for the detector this month
    score       quality score of the month, and the drop against the control

Findings the control raises by itself are never credited: a check failing or an incident
seen in the control too does not count. Outcome per detector:

    caught    an incident was raised: the fault would reach someone
    flagged   rows were flagged (quarantined or warned) but no incident was raised: the bad rows
              are kept out of Silver, yet nobody is alerted
    missed    nothing
    error     the experiment could not run

Results: evaluation_results in <workspace>/evaluation.duckdb and <workspace>/evaluation_results.csv.
"""

from __future__ import annotations

import argparse
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
from evaluation.run_experiments import DATABASE, ExperimentRecord  # noqa: E402
from ingestion.bronze_loader import read_bronze_sql  # noqa: E402
from silver.process_silver import current_silver  # noqa: E402

OUTCOMES = ("caught", "flagged", "missed", "error")
CSV = "evaluation_results.csv"
RESULTS_DB = "evaluation.duckdb"


@dataclass(frozen=True)
class MonthView:
    """What the pipeline found in the target month of one experiment."""

    checks: dict[tuple[str, str], tuple[str, str | None, str | None, str | None]]
    incidents: dict[tuple[str, str, str], str]  # (source, rule_id, column) -> severity
    rule_rows: dict[str, set[int]]  # rule id -> bronze_row_ids that failed it
    score: float | None
    health: str | None
    ml_rows: frozenset[int] = frozenset()  # bronze_row_ids the anomaly model flagged
    ml_rate: float | None = None  # share of Silver rows flagged; None without a model

    def failed_checks(self) -> set[tuple[str, str]]:
        return {key for key, (status, *_rest) in self.checks.items() if status == "fail"}


def month_view(database: Path, dataset: str, period: str) -> MonthView:
    with duckdb.connect(str(database), read_only=True) as con:
        version = con.execute(
            "SELECT sha256 FROM gold_months WHERE dataset = ? AND period = ?", [dataset, period]
        ).fetchone()
        sha = version[0] if version else None
        checks = {
            (check, column): (status, severity, observed, expected)
            for check, column, status, severity, observed, expected in con.execute(
                """
                SELECT "check", column_name, status, severity, observed, expected
                FROM dataset_check_results
                WHERE dataset = ? AND period = ? AND sha256 = ? AND run_id = (
                    SELECT max(run_id) FROM dataset_check_results
                    WHERE dataset = ? AND period = ? AND sha256 = ?)
                """,
                [dataset, period, sha] * 2,
            ).fetchall()
        }
        incidents = {}
        for source, rule_id, column, severity in con.execute(
            """
            SELECT i.source, i.rule_id, i.column_name, e.severity
            FROM incident_events e JOIN incidents i USING (incident_id)
            WHERE i.dataset = ? AND e.period = ? AND e.event IN ('opened', 'seen_again')
            """,
            [dataset, period],
        ).fetchall():
            incidents[(source, rule_id, column)] = severity
        rule_rows: dict[str, set[int]] = {}
        files = current_silver(con, dataset).get(period)
        for path in files or ():
            if not path.exists():
                continue
            for row_id, rules in con.execute(
                f"SELECT bronze_row_id, failed_rules FROM {read_bronze_sql(path)} "
                "WHERE len(failed_rules) > 0"
            ).fetchall():
                for rule in rules:
                    rule_rows.setdefault(rule, set()).add(int(row_id))
        health = con.execute(
            "SELECT score, health FROM gold_dataset_health WHERE dataset = ? AND period = ?",
            [dataset, period],
        ).fetchone()
        scored = con.execute(
            "SELECT model_id, flag_rate FROM ml_month_summary WHERE dataset = ? AND period = ? "
            "AND sha256 = ? ORDER BY scored_at DESC LIMIT 1",
            [dataset, period, sha],
        ).fetchone()
        ml_rows: frozenset[int] = frozenset()
        if scored:
            ml_rows = frozenset(
                int(r)
                for (r,) in con.execute(
                    "SELECT bronze_row_id FROM ml_anomalies WHERE dataset = ? AND period = ? "
                    "AND sha256 = ? AND model_id = ?",
                    [dataset, period, sha, scored[0]],
                ).fetchall()
            )
    return MonthView(
        checks,
        incidents,
        rule_rows,
        *(health or (None, None)),
        ml_rows=ml_rows,
        ml_rate=float(scored[1]) if scored else None,
    )


@dataclass
class Result:
    experiment_id: str
    fault_id: str
    category: str
    strength: float | None
    detector: str
    outcome: str
    broken_rows: int | None = None
    removed_rows: int | None = None
    caught_rows: int | None = None
    extra_flags: int | None = None
    recall: float | None = None
    precision: float | None = None
    check_fired: bool | None = None
    check_severity: str | None = None
    check_observed: str | None = None
    check_expected: str | None = None
    incident: bool = False
    incident_severity: str | None = None
    score: float | None = None
    control_score: float | None = None
    score_drop: float | None = None
    health: str | None = None
    side_findings: str | None = None
    stage_failures: str | None = None
    error: str | None = None
    # rules only against rules + ML (Phase 7), per experiment
    ml_caught_rows: int | None = None
    ml_recall: float | None = None
    ml_extra_flags: int | None = None
    ml_rate: float | None = None
    rules_outcome: str | None = None
    ml_outcome: str | None = None
    combined_outcome: str | None = None


def _label(expect: dict[str, str]) -> str:
    column = f" ({expect['column']})" if expect.get("column") else ""
    return f"{expect['kind']} {expect['id']}{column}"


ML_INCIDENT = ("dataset_check", "ml_anomaly", "")
RANK = {"caught": 3, "flagged": 2, "missed": 1, "error": 0}


def _incident_key(expect: dict[str, str]) -> tuple[str, str, str]:
    if expect["kind"] == "rule":
        return ("row_rule", expect["id"], "")
    if expect["kind"] == "ml":
        return ML_INCIDENT
    return ("dataset_check", expect["id"], expect.get("column", ""))


def _rows(view: MonthView, expect: dict[str, str]) -> set[int]:
    if expect["kind"] == "ml":
        return set(view.ml_rows)
    return view.rule_rows.get(expect["id"], set())


def compare(manifest: fi.Manifest, view: MonthView, control: MonthView) -> dict[str, object]:
    """Rules only against rules + ML for one experiment.

    rules: caught when any check or row-rule incident is new against the control, flagged when
    a rule flagged a broken row that the control row does not already fail. ML: caught when an
    ml_anomaly incident is new, flagged when the model flagged a broken row that it does not
    already flag in the control. Combined: the better of the two."""
    broken = set(manifest.affected_rows)
    new_incidents = set(view.incidents) - set(control.incidents)
    rule_alert = bool(view.failed_checks() - control.failed_checks()) or any(
        key != ML_INCIDENT for key in new_incidents
    )
    newly_flagged = {
        row
        for rule, rows in view.rule_rows.items()
        for row in (rows & broken) - control.rule_rows.get(rule, set())
    }
    rules = "caught" if rule_alert else ("flagged" if newly_flagged else "missed")
    ml_caught = len((broken & view.ml_rows) - control.ml_rows)
    if ML_INCIDENT in new_incidents:
        ml = "caught"
    else:
        ml = "flagged" if ml_caught else "missed"
    return {
        "ml_caught_rows": ml_caught if broken else None,
        "ml_recall": round(ml_caught / len(broken), 4) if broken else None,
        "ml_extra_flags": max(0, len(view.ml_rows) - ml_caught - len(control.ml_rows))
        if view.ml_rate is not None
        else None,
        "ml_rate": view.ml_rate,
        "rules_outcome": rules,
        "ml_outcome": ml if view.ml_rate is not None else None,
        "combined_outcome": max(rules, ml, key=RANK.get) if view.ml_rate is not None else rules,
    }


def evaluate_experiment(
    manifest: fi.Manifest, record: ExperimentRecord, view: MonthView | None, control: MonthView
) -> list[Result]:
    base = {
        "experiment_id": manifest.experiment_id,
        "fault_id": manifest.fault_id,
        "category": manifest.category,
        "strength": manifest.strength,
        "broken_rows": None if manifest.month_level else len(manifest.affected_rows),
        "removed_rows": len(manifest.removed_rows) or None,
        "control_score": control.score,
        "stage_failures": ", ".join(s for s, c in record.stages.items() if c != 0) or None,
    }
    if view is None or not record.ran:
        failed = {"rules_outcome": "error", "ml_outcome": "error", "combined_outcome": "error"}
        return [
            Result(**base, **failed, detector=_label(e), outcome="error", error=record.error)
            for e in manifest.expect
        ]

    expected_checks = {
        (e["id"], e.get("column", "")) for e in manifest.expect if e["kind"] == "check"
    }
    expected_incidents = {_incident_key(e) for e in manifest.expect}
    new_checks = view.failed_checks() - control.failed_checks() - expected_checks
    new_incidents = set(view.incidents) - set(control.incidents) - expected_incidents
    side = sorted(
        {f"{c} {col}".strip() + f" ({view.checks[(c, col)][1]})" for c, col in new_checks}
        | {
            f"incident {r} {col}".strip()
            for s, r, col in new_incidents
            if s == "row_rule" or r == "ml_anomaly"
        }
    )
    common = (
        base
        | {
            "score": view.score,
            "score_drop": None
            if view.score is None or control.score is None
            else round(control.score - view.score, 4),
            "health": view.health,
            "side_findings": "; ".join(side) or None,
        }
        | compare(manifest, view, control)
    )

    results = []
    for expect in manifest.expect:
        result = Result(**common, detector=_label(expect), outcome="missed")
        key = _incident_key(expect)
        if key in view.incidents and key not in control.incidents:
            result.incident, result.incident_severity = True, view.incidents[key]
        if expect["kind"] in ("rule", "ml"):
            flagged = _rows(view, expect)
            broken = set(manifest.affected_rows)
            already_flagged = _rows(control, expect)
            if expect["kind"] == "ml":  # a row the model flags anyway is not a detection
                caught = len((flagged & broken) - already_flagged)
            else:  # a fault only breaks rows that do not fail its rule already
                caught = len(flagged & broken)
            already = len(already_flagged)
            result.caught_rows = caught
            result.extra_flags = max(0, len(flagged) - caught - already)
            if broken:
                result.recall = round(caught / len(broken), 4)
            if caught + result.extra_flags:
                result.precision = round(caught / (caught + result.extra_flags), 4)
            rows_flagged = caught > 0
        else:
            check = view.checks.get((expect["id"], expect.get("column", "")))
            if check:
                status, severity, observed, wanted = check
                result.check_observed, result.check_expected = observed, wanted
                in_control = (expect["id"], expect.get("column", "")) in control.failed_checks()
                result.check_fired = status == "fail" and not in_control
                result.check_severity = severity if result.check_fired else None
            rows_flagged = False
        if result.incident or result.check_fired:
            result.outcome = "caught"
        elif rows_flagged:
            result.outcome = "flagged"
        results.append(result)
    return results


def evaluate(config: fi.FaultsConfig) -> pd.DataFrame:
    """Every experiment folder under the workspace, against the control."""
    experiments = config.workspace / "experiments"
    control_dir = experiments / fi.CONTROL
    if not (control_dir / DATABASE).is_file():
        raise FileNotFoundError(f"no control experiment in {experiments} (run run_experiments.py)")
    control = month_view(control_dir / DATABASE, config.dataset, config.target_month)

    results: list[Result] = []
    for exp_id, _fault, _strength in config.experiments():
        folder = experiments / exp_id
        if exp_id == fi.CONTROL or not (folder / "incoming" / fi.MANIFEST).is_file():
            continue
        manifest = fi.Manifest.read(folder / "incoming" / fi.MANIFEST)
        record = ExperimentRecord.read(folder)
        database = folder / DATABASE
        view = (
            month_view(database, config.dataset, config.target_month)
            if record.ran and database.is_file()
            else None
        )
        results += evaluate_experiment(manifest, record, view, control)
    frame = pd.DataFrame([asdict(r) for r in results], columns=list(Result.__dataclass_fields__))
    return frame


def control_findings(config: fi.FaultsConfig) -> list[str]:
    """What the unbroken target month raises by itself (false alarms of the setup)."""
    database = config.workspace / "experiments" / fi.CONTROL / DATABASE
    view = month_view(database, config.dataset, config.target_month)
    return sorted(
        [f"check {c} {col}".strip() for c, col in view.failed_checks()]
        + [f"incident {r} {col}".strip() for _s, r, col in view.incidents]
    )


def control_ml(config: fi.FaultsConfig) -> tuple[float | None, int]:
    """The anomaly model on the unbroken target month: flag rate and rows flagged."""
    database = config.workspace / "experiments" / fi.CONTROL / DATABASE
    view = month_view(database, config.dataset, config.target_month)
    return view.ml_rate, len(view.ml_rows)


def comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per experiment: rules only against rules + ML."""
    columns = ["experiment_id", "fault_id", "strength", "rules_outcome", "ml_outcome",
               "combined_outcome", "ml_recall", "ml_extra_flags"]  # fmt: skip
    table = frame.drop_duplicates("experiment_id")[columns].reset_index(drop=True)
    table["ml_adds"] = [
        RANK.get(c, 0) > RANK.get(r, 0)
        for r, c in zip(table["rules_outcome"], table["combined_outcome"], strict=True)
    ]
    return table


def store(config: fi.FaultsConfig, frame: pd.DataFrame) -> tuple[Path, Path]:
    csv = config.workspace / CSV
    frame.to_csv(csv, index=False)
    database = config.workspace / RESULTS_DB
    with duckdb.connect(str(database)) as con:
        con.register("results", frame)
        con.execute("CREATE OR REPLACE TABLE evaluation_results AS SELECT * FROM results")
    return csv, database


def overview(frame: pd.DataFrame) -> pd.DataFrame:
    view = frame[["experiment_id", "detector", "outcome", "recall", "check_severity",
                  "incident_severity", "score_drop", "check_observed"]].copy()  # fmt: skip
    view["recall"] = view["recall"].map(lambda v: "" if pd.isna(v) else f"{v:.0%}")
    view["score_drop"] = view["score_drop"].map(lambda v: "" if pd.isna(v) else f"{v:.1f}")
    return view.fillna("")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the fault injection experiments.")
    parser.add_argument("--config", type=Path, help="faults config (default: configs/faults.yml)")
    args = parser.parse_args(argv)
    try:
        config = fi.load_faults_config(args.config)
        frame = evaluate(config)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"error: {exc}")
        return 2
    csv, database = store(config, frame)
    with pd.option_context("display.width", 220, "display.max_colwidth", 50):
        print(overview(frame).to_string(index=False))
    counts = frame["outcome"].value_counts().reindex(OUTCOMES, fill_value=0)
    print("\n" + ", ".join(f"{n} {o}" for o, n in counts.items()) + f" of {len(frame)} detector(s)")
    alarms = control_findings(config)
    print("control month raises: " + (", ".join(alarms) if alarms else "nothing"))
    rate, flagged = control_ml(config)
    if rate is not None:
        table = comparison(frame)
        print(
            f"rules only: {(table['rules_outcome'] == 'caught').sum()} caught; rules + ML: "
            f"{(table['combined_outcome'] == 'caught').sum()} caught; ML adds detection in "
            f"{int(table['ml_adds'].sum())} experiment(s); the model flags {rate:.2%} "
            f"({flagged} rows) of the control month"
        )
    print(f"-> {csv}\n-> {database} (evaluation_results)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
