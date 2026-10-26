"""Run the dataset-level checks on every month in Bronze and record the results.

    python src/quality/run_checks.py                    # every month in Bronze
    python src/quality/run_checks.py --months 2023-09

For each month: metrics are collected (computed once per file version), the baseline is
built from the earlier months, and five checks run: schema, volume, freshness, NULL rate
and drift. Results go to dataset_check_results and the baselines used to baselines, all
within one tracked pipeline run.

The run itself is SUCCESS when every month was checked, even if checks failed: failed
checks are findings about the data (Phase 4 turns them into incidents), not pipeline
errors. A month missing from Bronze makes the run WARNING; a month that cannot be checked
makes it FAILED while the other months are still checked.
Exit codes: 0 success or warning, 1 failure, 2 invalid arguments or config.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import duckdb
import pandas as pd

if __package__ in (None, ""):
    # Allow running this file directly: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import db  # noqa: E402
from common.config import ConfigError, Dataset, load_datasets_config  # noqa: E402
from common.runs import get_run, track_run  # noqa: E402
from data_download.download_data import parse_months  # noqa: E402
from ingestion.manifest import get_entries  # noqa: E402
from quality import baselines, dataset_checks, metrics  # noqa: E402
from quality.baselines import BaselineSettings  # noqa: E402
from quality.dataset_checks import CheckResult, CheckSettings  # noqa: E402

log = logging.getLogger("run_checks")
CHECK_ORDER = ("schema", "volume", "freshness", "null_rate", "drift")


class ChecksRunError(RuntimeError):
    """Raised at the end of a run in which at least one month could not be checked."""


def check_month(
    m: metrics.MonthMetrics,
    all_metrics: dict[str, metrics.MonthMetrics],
    entry,
    dataset: Dataset,
    settings: CheckSettings,
    limits: BaselineSettings,
    bins: dict[str, list[float]],
) -> tuple[baselines.Baseline, list[CheckResult]]:
    baseline = baselines.build(all_metrics, m.period, limits)
    results = [
        dataset_checks.check_schema(m, baseline, dataset),
        dataset_checks.check_volume(m, baseline, settings, limits),
        dataset_checks.check_freshness(entry, settings),
        *dataset_checks.check_null_rates(m, baseline, settings, limits),
        *dataset_checks.check_drift(m, baseline, settings, bins),
    ]
    return baseline, results


def store_results(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    dataset: str,
    sha256: str,
    results: list[CheckResult],
) -> None:
    con.executemany(
        "INSERT INTO dataset_check_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            [
                run_id,
                dataset,
                r.period,
                sha256,
                r.check,
                r.column,
                r.status,
                r.severity,
                r.observed,
                r.expected,
                r.detail,
                r.baseline,
            ]
            for r in results
        ],
    )


def run_checks(
    con: duckdb.DuckDBPyConnection,
    dataset: Dataset,
    months: list[str] | None = None,
    settings: CheckSettings | None = None,
    limits: BaselineSettings | None = None,
    bins: dict[str, list[float]] | None = None,
    command: str | None = None,
) -> str:
    """Check the given months (default: every month in Bronze) in one tracked run."""
    settings = settings or dataset_checks.check_settings()
    limits = limits or baselines.settings_for(dataset.name)
    bins = metrics.drift_bins(dataset.name) if bins is None else bins
    run_id = None
    try:
        with track_run(con, dataset.name, command=command) as run:
            run_id = run.run_id
            # Baselines need every month, so metrics are collected for all of Bronze.
            all_metrics = metrics.collect(con, dataset, bins=bins)
            entries = {
                (e.period, e.sha256): e
                for e in get_entries(con, dataset.name)
                if e.status == "LOADED"
            }
            wanted = months or sorted(all_metrics)
            run.add(files_total=len(wanted))
            failed = []
            for period in wanted:
                if period not in all_metrics:
                    run.add(files_skipped=1)
                    run.warn(f"{period} not in Bronze")
                    log.warning("missing    %s (run ingest_data.py first)", period)
                    continue
                m = all_metrics[period]
                try:
                    baseline, results = check_month(
                        m,
                        all_metrics,
                        entries[(period, m.sha256)],
                        dataset,
                        settings,
                        limits,
                        bins,
                    )
                    baselines.store(con, run_id, dataset.name, baseline)
                    store_results(con, run_id, dataset.name, m.sha256, results)
                except Exception as exc:  # one bad month must not stop the others
                    failed.append(period)
                    log.error("failed     %s: %s: %s", period, type(exc).__name__, exc)
                    continue
                run.add(files_loaded=1, rows_loaded=m.rows)
                findings = [r for r in results if r.status == "fail"]
                if findings:
                    worst = dataset_checks.worst([r.severity for r in findings])
                    log.info("checked    %s: %d finding(s), worst %s", period, len(findings), worst)
                else:
                    log.info("checked    %s: no findings", period)
            if failed:
                raise ChecksRunError(
                    f"{len(failed)} month(s) could not be checked: {', '.join(failed)}"
                )
    except ChecksRunError:
        pass  # recorded as FAILED by track_run
    return run_id


def _cell(statuses: pd.DataFrame) -> str:
    failed = statuses[statuses["status"] == "fail"]
    if not failed.empty:
        return "FAIL " + dataset_checks.worst(failed["severity"].tolist())
    if (statuses["status"] == "pass").any():
        return "pass"
    return "skipped"


def overview(con: duckdb.DuckDBPyConnection, run_id: str) -> pd.DataFrame:
    """One line per month, one column per check: pass, skipped or FAIL <worst severity>."""
    results = con.execute(
        'SELECT period, "check", status, severity FROM dataset_check_results WHERE run_id = ?',
        [run_id],
    ).df()
    if results.empty:
        return results
    table = (
        results.groupby(["period", "check"])
        .apply(_cell, include_groups=False)
        .unstack("check")
        .reindex(columns=list(CHECK_ORDER))
        .fillna("-")
    )
    return table.reset_index()


def findings(con: duckdb.DuckDBPyConnection, run_id: str) -> pd.DataFrame:
    """Every failed check of a run, most severe first."""
    return con.execute(
        """
        SELECT period, "check", column_name, severity, observed, expected, detail
        FROM dataset_check_results WHERE run_id = ? AND status = 'fail'
        ORDER BY CASE severity WHEN 'critical' THEN 0 WHEN 'high' THEN 1
                               WHEN 'medium' THEN 2 ELSE 3 END, period, "check", column_name
        """,
        [run_id],
    ).df()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the dataset-level checks on Bronze.")
    parser.add_argument("--dataset", default="yellow_taxi", help="dataset name in datasets.yml")
    parser.add_argument("--months", help="e.g. 2023-09 or 2023-01:2023-03 (default: all in Bronze)")
    parser.add_argument("--database", type=Path, help="DuckDB file (default: paths.duckdb)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        dataset = load_datasets_config().dataset(args.dataset)
        months = parse_months(args.months, ()) if args.months else None
        settings = dataset_checks.check_settings()
        limits = baselines.settings_for(dataset.name)
    except ConfigError as exc:
        log.error("%s", exc)
        return 2

    command = "run_checks.py " + " ".join(argv if argv is not None else sys.argv[1:])
    with db.connect(args.database) as con:
        db.init_schema(con)
        run_id = run_checks(con, dataset, months, settings, limits, command=command.strip())
        run = get_run(con, run_id)
        table, found = overview(con, run_id), findings(con, run_id)

    with pd.option_context("display.max_colwidth", 90, "display.width", 250):
        if not table.empty:
            print(table.to_string(index=False))
        if not found.empty:
            print(f"\n{len(found)} finding(s):")
            print(found.to_string(index=False))
    log.info("%s %s: %d month(s) checked", run_id, run["status"], run["files_loaded"])
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
