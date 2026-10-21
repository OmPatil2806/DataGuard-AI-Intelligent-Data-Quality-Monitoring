"""Rebuild everything after Bronze in one step: Silver, checks, rules, incidents, score, Gold.

    python src/gold/rebuild.py                  # every month in Bronze
    python src/gold/rebuild.py --force          # also rewrite Silver versions already written

Each stage is the same command a user would run by hand, in this order:

    1 silver     src/silver/process_silver.py        Silver and Quarantine (skips versions done)
    2 checks     src/quality/run_checks.py           dataset-level checks, one tracked run
    3 rules      src/quality/run_rules.py            row rule results, one tracked run
    4 incidents  src/incidents/incidents_cli.py update
    5 score      src/quality/score.py
    6 gold       src/gold/monitoring_tables.py       every Gold table, one transaction

Every stage is safe to repeat, so running the rebuild again on the same Bronze gives the same
Gold: the same checksums in gold_build_log (run ids and timestamps aside).

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

from gold import monitoring_tables  # noqa: E402
from incidents import incidents_cli  # noqa: E402
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
    return [
        (
            "silver",
            process_silver.main,
            [*dataset, *lookups, *silver, *quarantine, *force, *database],
        ),
        ("checks", run_checks.main, [*dataset, *database]),
        ("rules", run_rules.main, [*dataset, *lookups, *database]),
        ("incidents", incidents_cli.main, [*database, *dataset, "update"]),
        ("score", score.main, [*dataset, *database]),
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
        f"  {r.name:<10} {r.seconds:7.1f}s  {status.get(r.exit_code, r.exit_code)}" for r in results
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild Silver, checks, incidents, score and Gold."
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
    parser.add_argument("--database", type=Path, help="DuckDB file (default: paths.duckdb)")
    args = parser.parse_args(argv)
    results = run(args)
    print(summary(results))
    return max(r.exit_code for r in results)


if __name__ == "__main__":
    sys.exit(main())
