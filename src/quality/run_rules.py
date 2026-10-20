"""Run the row rules over Bronze and record the results.

    python src/quality/run_rules.py                     # every month in Bronze
    python src/quality/run_rules.py --months 2023-01

For each month the current Bronze version is standardized, checked against the standard
schema (Pandera) and evaluated with the enabled rules of configs/quality_rules.yml. Results
go to rule_run_log (failures per rule) and rule_run_summary (rows quarantined, warned and
clean), all within one tracked pipeline run. Nothing is written to Silver yet (Phase 4).

Run status: SUCCESS; WARNING when a month is missing from Bronze or a cast loses values;
FAILED when a month cannot be checked (the other months still are).
Exit codes: 0 success or warning, 1 failure, 2 invalid arguments, config or rules.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import duckdb

if __package__ in (None, ""):
    # Allow running this file directly: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import db  # noqa: E402
from common.config import ConfigError, Dataset, load_datasets_config  # noqa: E402
from common.runs import Run, get_run, track_run  # noqa: E402
from data_download.download_data import parse_months  # noqa: E402
from ingestion.bronze_loader import current_versions  # noqa: E402
from quality import row_rules  # noqa: E402
from quality.rules import Rule, load_rules  # noqa: E402
from quality.standardize import standardize_file  # noqa: E402

log = logging.getLogger("run_rules")


class RulesRunError(RuntimeError):
    """Raised at the end of a run in which at least one month could not be checked."""


def check_month(
    con: duckdb.DuckDBPyConnection,
    dataset: Dataset,
    rules: list[Rule],
    period: str,
    path: Path,
    sha256: str,
    run: Run,
) -> row_rules.Summary:
    sql, report = standardize_file(con, path, dataset)
    if report.lossy:
        run.warn(f"{period}: {report.summary()}")
    row_rules.check_schema(con, sql, dataset)
    summary = row_rules.summarize(con, rules, f"({sql})", period=period)

    by_id = {r.id: r for r in rules}
    con.executemany(
        "INSERT INTO rule_run_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            [
                run.run_id,
                dataset.name,
                period,
                sha256,
                c.rule_id,
                by_id[c.rule_id].category,
                c.action,
                by_id[c.rule_id].severity,
                c.rows,
                c.failed,
            ]
            for c in summary.counts
        ],
    )
    con.execute(
        "INSERT INTO rule_run_summary VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            run.run_id,
            dataset.name,
            period,
            sha256,
            summary.rows,
            summary.quarantined,
            summary.warned,
            report.summary(),
        ],
    )
    run.add(files_loaded=1, rows_loaded=summary.rows)
    return summary


def run_rules(
    con: duckdb.DuckDBPyConnection,
    dataset: Dataset,
    rules: list[Rule],
    months: list[str] | None,
    command: str | None = None,
) -> str:
    """Check the given months (default: every month in Bronze) in one tracked run."""
    versions = current_versions(con, dataset)
    wanted = months or list(versions)
    run_id = None
    try:
        with track_run(con, dataset.name, command=command) as run:
            run_id = run.run_id
            run.add(files_total=len(wanted))
            failed = []
            for period in wanted:
                if period not in versions:
                    run.add(files_skipped=1)
                    run.warn(f"{period} not in Bronze")
                    log.warning("missing    %s (run ingest_data.py first)", period)
                    continue
                path, sha = versions[period]
                try:
                    s = check_month(con, dataset, rules, period, path, sha, run)
                except Exception as exc:  # one bad month must not stop the others
                    failed.append(period)
                    log.error("failed     %s: %s: %s", period, type(exc).__name__, exc)
                    continue
                log.info(
                    "checked    %s: %s rows, %s quarantined (%.2f%%), %s warned (%.2f%%)",
                    period,
                    f"{s.rows:,}",
                    f"{s.quarantined:,}",
                    100 * s.quarantined / s.rows if s.rows else 0,
                    f"{s.warned:,}",
                    100 * s.warned / s.rows if s.rows else 0,
                )
            if failed:
                raise RulesRunError(
                    f"{len(failed)} month(s) could not be checked: {', '.join(failed)}"
                )
    except RulesRunError:
        pass  # recorded as FAILED by track_run
    return run_id


def rule_table(con: duckdb.DuckDBPyConnection, run_id: str):
    """Failures per rule over all months of a run, most frequent first."""
    return con.execute(
        """
        SELECT rule_id, action, sum(rows_failed) AS rows_failed,
               round(100.0 * sum(rows_failed) / sum(rows_checked), 4) AS pct,
               round(100.0 * max(rows_failed / rows_checked), 4) AS worst_month_pct
        FROM rule_run_log WHERE run_id = ?
        GROUP BY rule_id, action ORDER BY rows_failed DESC, rule_id
        """,
        [run_id],
    ).df()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the row rules over Bronze.")
    parser.add_argument("--dataset", default="yellow_taxi", help="dataset name in datasets.yml")
    parser.add_argument("--months", help="e.g. 2023-01, 2023-01:2023-03 (default: all in Bronze)")
    parser.add_argument("--lookups", type=Path, help="folder with lookup files (default: incoming)")
    parser.add_argument("--database", type=Path, help="DuckDB file (default: paths.duckdb)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load_datasets_config()
        dataset = config.dataset(args.dataset)
        months = parse_months(args.months, ()) if args.months else None
        rules = load_rules(dataset)
    except ConfigError as exc:
        log.error("%s", exc)
        return 2

    command = "run_rules.py " + " ".join(argv if argv is not None else sys.argv[1:])
    with db.connect(args.database) as con:
        db.init_schema(con)
        try:
            row_rules.load_lookups(con, dataset, args.lookups or config.paths["incoming"])
        except FileNotFoundError as exc:
            log.error("%s", exc)
            return 2
        log.info("%d enabled rule(s) for '%s'", len(rules), dataset.name)
        run_id = run_rules(con, dataset, rules, months, command=command.strip())
        run = get_run(con, run_id)
        table = rule_table(con, run_id)

    if not table.empty:
        print(table.to_string(index=False))
    log.info(
        "%s %s: %d month(s) checked, %s rows",
        run_id,
        run["status"],
        run["files_loaded"],
        f"{run['rows_loaded']:,}",
    )
    if run["error_message"]:
        log.log(
            logging.ERROR if run["status"] == "FAILED" else logging.WARNING,
            "%s",
            run["error_message"],
        )
    if run["status"] == "FAILED" or run["files_loaded"] == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
