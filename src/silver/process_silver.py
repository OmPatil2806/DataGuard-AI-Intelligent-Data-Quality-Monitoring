"""Write Silver and Quarantine for every month in Bronze, in one tracked run.

    python src/silver/process_silver.py                  # every month in Bronze
    python src/silver/process_silver.py --months 2023-01
    python src/silver/process_silver.py --force          # rewrite versions already processed

For each month the current Bronze version is split by silver_processor (standardize,
schema check, rules, enrichment, verified write). Every month is recorded in silver_log.
Versions already written are skipped unless --force is given.

Run status: SUCCESS; WARNING when a month is missing from Bronze or a cast loses values;
FAILED when a month cannot be processed (the other months still are).
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
from common.config import REPO_ROOT, ConfigError, Dataset, load_datasets_config  # noqa: E402
from common.runs import get_run, track_run, utc_now  # noqa: E402
from data_download.download_data import parse_months  # noqa: E402
from ingestion.bronze_loader import current_versions  # noqa: E402
from quality import row_rules  # noqa: E402
from quality.rules import Rule, load_rules  # noqa: E402
from quality.standardize import standardize_file  # noqa: E402
from silver.silver_processor import SilverResult, process_version  # noqa: E402

log = logging.getLogger("process_silver")


class SilverRunError(RuntimeError):
    """Raised at the end of a run in which at least one month could not be processed."""


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def log_result(
    con: duckdb.DuckDBPyConnection, run_id: str, dataset: str, sha256: str, r: SilverResult
) -> None:
    con.execute(
        "INSERT INTO silver_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            run_id,
            dataset,
            r.period,
            sha256,
            r.status,
            r.rows,
            r.silver_rows,
            r.quarantine_rows,
            r.warned_rows,
            _relative(r.silver_path),
            _relative(r.quarantine_path),
            utc_now(),
        ],
    )


def current_silver(
    con: duckdb.DuckDBPyConnection, dataset: Dataset | str
) -> dict[str, tuple[Path, Path]]:
    """Per period: the Silver and Quarantine files of the current Bronze version."""
    name = dataset.name if isinstance(dataset, Dataset) else dataset
    versions = current_versions(con, name)
    rows = con.execute(
        """
        SELECT period, sha256, silver_path, quarantine_path FROM silver_log WHERE dataset = ?
        QUALIFY row_number() OVER (PARTITION BY period, sha256 ORDER BY logged_at DESC) = 1
        """,
        [name],
    ).fetchall()

    def absolute(p: str) -> Path:
        return Path(p) if Path(p).is_absolute() else REPO_ROOT / p

    return {
        period: (absolute(s), absolute(q))
        for period, sha, s, q in sorted(rows)
        if period in versions and versions[period][1] == sha
    }


def process(
    con: duckdb.DuckDBPyConnection,
    dataset: Dataset,
    rules: list[Rule],
    months: list[str] | None,
    silver_root: Path,
    quarantine_root: Path,
    force: bool = False,
    command: str | None = None,
) -> str:
    """Process the given months (default: every month in Bronze) in one tracked run."""
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
                    _, report = standardize_file(con, path, dataset)
                    if report.lossy:
                        run.warn(f"{period}: {report.summary()}")
                    result = process_version(
                        con,
                        dataset,
                        rules,
                        period,
                        path,
                        sha,
                        silver_root,
                        quarantine_root,
                        run_id,
                        force=force,
                    )
                    log_result(con, run_id, dataset.name, sha, result)
                except Exception as exc:  # one bad month must not stop the others
                    failed.append(period)
                    log.error("failed     %s: %s: %s", period, type(exc).__name__, exc)
                    continue
                if result.status == "skipped":
                    run.add(files_skipped=1)
                else:
                    run.add(files_loaded=1, rows_loaded=result.rows)
                log.info(
                    "%-10s %s: %s rows -> %s Silver (%s warned), %s Quarantine (%.2f%%)",
                    result.status,
                    period,
                    f"{result.rows:,}",
                    f"{result.silver_rows:,}",
                    f"{result.warned_rows:,}",
                    f"{result.quarantine_rows:,}",
                    100 * result.quarantine_rows / result.rows if result.rows else 0,
                )
            if failed:
                raise SilverRunError(
                    f"{len(failed)} month(s) could not be processed: {', '.join(failed)}"
                )
    except SilverRunError:
        pass  # recorded as FAILED by track_run
    return run_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write Silver and Quarantine from Bronze.")
    parser.add_argument("--dataset", default="yellow_taxi", help="dataset name in datasets.yml")
    parser.add_argument("--months", help="e.g. 2023-01 or 2023-01:2023-03 (default: all in Bronze)")
    parser.add_argument("--force", action="store_true", help="rewrite versions already processed")
    parser.add_argument("--lookups", type=Path, help="folder with lookup files (default: incoming)")
    parser.add_argument("--silver", type=Path, help="Silver folder (default: paths.silver)")
    parser.add_argument(
        "--quarantine", type=Path, help="Quarantine folder (default: paths.quarantine)"
    )
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

    command = "process_silver.py " + " ".join(argv if argv is not None else sys.argv[1:])
    with db.connect(args.database) as con:
        db.init_schema(con)
        try:
            row_rules.load_lookups(con, dataset, args.lookups or config.paths["incoming"])
        except FileNotFoundError as exc:
            log.error("%s", exc)
            return 2
        run_id = process(
            con,
            dataset,
            rules,
            months,
            args.silver or config.paths["silver"],
            args.quarantine or config.paths["quarantine"],
            force=args.force,
            command=command.strip(),
        )
        run = get_run(con, run_id)
        totals = con.execute(
            "SELECT count(*), coalesce(sum(silver_rows), 0), coalesce(sum(quarantine_rows), 0) "
            "FROM silver_log WHERE run_id = ?",
            [run_id],
        ).fetchone()

    log.info(
        "%s %s: %d month(s) written, %d skipped; %s Silver and %s Quarantine rows",
        run_id,
        run["status"],
        run["files_loaded"],
        run["files_skipped"],
        f"{totals[1]:,}",
        f"{totals[2]:,}",
    )
    if run["error_message"]:
        log.log(
            logging.ERROR if run["status"] == "FAILED" else logging.WARNING,
            "%s",
            run["error_message"],
        )
    if run["status"] == "FAILED" or totals[0] == 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
