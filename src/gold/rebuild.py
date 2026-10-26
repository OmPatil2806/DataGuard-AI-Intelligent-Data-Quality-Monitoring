"""Rebuild everything after Bronze in one step: Silver to incidents, score, investigations, Gold.

    python src/gold/rebuild.py                  # every month in Bronze
    python src/gold/rebuild.py --force          # also rewrite Silver versions already written
    python src/gold/rebuild.py --months 2023-06 # Silver, checks and rules for these months only

Each stage is the same command a user would run by hand, in this order:

    1 silver     src/silver/process_silver.py        Silver and Quarantine (skips versions done)
    2 checks     src/quality/run_checks.py           dataset-level checks, one tracked run
    3 rules      src/quality/run_rules.py            row rule results, one tracked run
    4 ml         src/ml/score.py --if-trained        anomaly flags (skipped until a model is
                                                     trained with src/ml/train.py, and for a
                                                     dataset other than configs/ml.yml's)
    5 incidents  src/incidents/incidents_cli.py update
    6 score      src/quality/score.py
    7 investigate src/investigation/investigate.py   evidence, explanation, actions, report
                                                     for every incident
    8 alert      src/alerts/notifier.py              new critical/high incidents to the alert
                                                     outbox (--no-alerts to skip)
    9 gold       src/gold/monitoring_tables.py       every Gold table, one transaction

--months limits the stages that work month by month (silver, checks, rules, ml); the checks still
compare with every earlier month in the database, and the incidents, score and Gold stages
always cover every month. Every stage is safe to repeat, so running the rebuild again on the
same Bronze gives the same Gold: the same checksums in gold_build_log (run ids and timestamps
aside).

A stage that exits with 2 (invalid config or arguments) stops the rebuild. A stage that exits
with 1 (for example one month failed) is reported and the later stages still run, on what is
there. Exit code: the highest stage exit code.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    # Allow running this file directly: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alerts import notifier  # noqa: E402
from gold import monitoring_tables  # noqa: E402
from incidents import incidents_cli  # noqa: E402
from investigation import investigate  # noqa: E402
from ml import score as ml_score  # noqa: E402
from quality import run_checks, run_rules, score  # noqa: E402
from silver import process_silver  # noqa: E402


@dataclass(frozen=True)
class StageResult:
    name: str
    exit_code: int
    seconds: float


def stages(args: argparse.Namespace) -> list[tuple[str, Callable[[list[str]], int], list[str]]]:
    database = ["--database", str(args.database)] if args.database else []
    dataset = ["--dataset", args.dataset]
    lookups = ["--lookups", str(args.lookups)] if args.lookups else []
    silver = ["--silver", str(args.silver)] if args.silver else []
    quarantine = ["--quarantine", str(args.quarantine)] if args.quarantine else []
    force = ["--force"] if args.force else []
    months = ["--months", args.months] if args.months else []
    reports = ["--reports", str(args.reports)] if args.reports else []
    return [
        (
            "silver",
            process_silver.main,
            [*dataset, *months, *lookups, *silver, *quarantine, *force, *database],
        ),
        ("checks", run_checks.main, [*dataset, *months, *database]),
        ("rules", run_rules.main, [*dataset, *months, *lookups, *database]),
        ("ml", ml_score.main, [*dataset, *months, *database, "--if-trained"]),
        ("incidents", incidents_cli.main, [*database, *dataset, "update"]),
        ("score", score.main, [*dataset, *database]),
        ("investigate", investigate.main, [*dataset, *reports, *database]),
        *([] if args.no_alerts else [("alert", notifier.main, [*dataset, *database])]),
        ("gold", monitoring_tables.main, database),
    ]


def run(
    args: argparse.Namespace, clock: Callable[[], float] = time.perf_counter
) -> list[StageResult]:
    results = []
    for name, main, argv in stages(args):
        print(f"\n=== {name} ===", flush=True)
        start = clock()
        code = main(argv)
        results.append(StageResult(name, code, clock() - start))
        if code == 2:
            print(f"{name} stopped the rebuild (exit 2: fix the config or arguments)")
            break
    return results


def summary(results: list[StageResult]) -> str:
    status = {0: "ok", 1: "FAILED (later stages ran on what is there)", 2: "STOPPED"}
    lines = ["", "rebuild summary:"]
    lines += [
        f"  {r.name:<11} {r.seconds:7.1f}s  {status.get(r.exit_code, r.exit_code)}" for r in results
    ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild Silver, checks, incidents, score, investigations and Gold."
    )
    parser.add_argument("--dataset", default="yellow_taxi", help="dataset name in datasets.yml")
    parser.add_argument("--lookups", type=Path, help="folder with lookup files (default: incoming)")
    parser.add_argument("--silver", type=Path, help="Silver folder (default: paths.silver)")
    parser.add_argument(
        "--quarantine", type=Path, help="Quarantine folder (default: paths.quarantine)"
    )
    parser.add_argument(
        "--force", action="store_true", help="rewrite Silver versions already written"
    )
    parser.add_argument(
        "--months", help="e.g. 2023-06 or 2023-01:2023-03: silver, checks and rules only"
    )
    parser.add_argument(
        "--reports", type=Path, help="investigation reports folder (default: paths.reports)"
    )
    parser.add_argument(
        "--no-alerts", action="store_true", help="skip the alert stage (e.g. for experiments)"
    )
    parser.add_argument("--database", type=Path, help="DuckDB file (default: paths.duckdb)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    results = run(parse_args(argv))
    print(summary(results))
    return max(r.exit_code for r in results)


if __name__ == "__main__":
    sys.exit(main())
