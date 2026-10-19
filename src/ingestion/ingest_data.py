"""Ingest source files into Bronze: register, load and record the run.

    python src/ingestion/ingest_data.py                        # every configured month
    python src/ingestion/ingest_data.py --months 2023-01
    python src/ingestion/ingest_data.py --months 2023-01 --force

For each month the file in data/incoming/ is registered in file_manifest, written to
Bronze unless that exact version is already there, and counted in one pipeline run.
A missing file makes the run WARNING; a file that cannot be loaded makes it FAILED,
while the other files are still loaded.

Exit codes: 0 success or warning, 1 failure, 2 invalid arguments or config.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb

if __package__ in (None, ""):
    # Allow running this file directly: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import db  # noqa: E402
from common.config import ConfigError, Dataset, load_datasets_config  # noqa: E402
from common.runs import Run, get_run, track_run  # noqa: E402
from data_download.download_data import parse_months  # noqa: E402
from ingestion import bronze_loader, manifest  # noqa: E402

log = logging.getLogger("ingest_data")


class IngestError(RuntimeError):
    """Raised at the end of a run in which at least one file could not be loaded."""


@dataclass(frozen=True)
class FileOutcome:
    period: str
    file_name: str
    status: str  # loaded, skipped, missing or failed
    rows: int = 0
    detail: str = ""


def ingest_month(
    con: duckdb.DuckDBPyConnection,
    dataset: Dataset,
    period: str,
    source_dir: Path,
    bronze_root: Path,
    run: Run,
    force: bool = False,
) -> FileOutcome:
    path = source_dir / dataset.file_for(period)
    if not path.is_file():
        run.add(files_skipped=1)
        run.warn(f"{path.name} missing")
        return FileOutcome(period, path.name, "missing", detail="run download_data.py first")

    entry = None
    try:
        registration = manifest.register_file(con, dataset, path, period=period)
        entry = registration.entry
        target = bronze_loader.version_dir(bronze_root, entry) / bronze_loader.PART_FILE
        if entry.status == "LOADED" and target.exists() and not force:
            run.add(files_skipped=1)
            return FileOutcome(
                period, path.name, "skipped", entry.row_count or 0, "already in Bronze"
            )

        result = bronze_loader.load_file(con, entry, path, bronze_root, run.run_id, force=force)
    except Exception as exc:  # one bad file must not stop the others
        if entry is not None:
            manifest.set_status(con, entry, "FAILED", run_id=run.run_id)
        return FileOutcome(period, path.name, "failed", detail=f"{type(exc).__name__}: {exc}")

    if result.status == "skipped":
        run.add(files_skipped=1)
        return FileOutcome(period, path.name, "skipped", result.rows, "already in Bronze")
    run.add(files_loaded=1, rows_loaded=result.rows)
    note = "new version" if registration.is_new and registration.other_versions else ""
    return FileOutcome(period, path.name, "loaded", result.rows, note)


def ingest(
    con: duckdb.DuckDBPyConnection,
    dataset: Dataset,
    months: list[str],
    source_dir: Path,
    bronze_root: Path,
    command: str | None = None,
    force: bool = False,
) -> tuple[str, list[FileOutcome]]:
    """Ingest the given months in one tracked run; returns the run ID and per-file outcomes."""
    outcomes: list[FileOutcome] = []
    run_id = None
    try:
        with track_run(con, dataset.name, command=command) as run:
            run_id = run.run_id
            run.add(files_total=len(months))
            for period in months:
                outcome = ingest_month(con, dataset, period, source_dir, bronze_root, run, force)
                outcomes.append(outcome)
                _log_outcome(outcome)
            failed = [o for o in outcomes if o.status == "failed"]
            if failed:
                names = ", ".join(o.file_name for o in failed)
                raise IngestError(f"{len(failed)} file(s) could not be loaded: {names}")
    except IngestError:
        pass  # already recorded as FAILED by track_run; the caller reads the run status
    return run_id, outcomes


def _log_outcome(o: FileOutcome) -> None:
    detail = f" ({o.detail})" if o.detail else ""
    if o.status == "failed":
        log.error("failed     %s%s", o.file_name, detail)
    elif o.status == "missing":
        log.warning("missing    %s%s", o.file_name, detail)
    else:
        log.info("%-10s %s: %s rows%s", o.status, o.file_name, f"{o.rows:,}", detail)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest source files into Bronze.")
    parser.add_argument("--dataset", default="yellow_taxi", help="dataset name in datasets.yml")
    parser.add_argument("--months", help="e.g. 2023-01, 2023-01:2023-03 or 2023-01,2019-01")
    parser.add_argument("--force", action="store_true", help="rewrite versions already in Bronze")
    parser.add_argument("--source", type=Path, help="input folder (default: paths.incoming)")
    parser.add_argument("--bronze", type=Path, help="Bronze folder (default: paths.bronze)")
    parser.add_argument("--database", type=Path, help="DuckDB file (default: paths.duckdb)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load_datasets_config()
        dataset = config.dataset(args.dataset)
        months = parse_months(args.months, dataset.periods)
    except ConfigError as exc:
        log.error("%s", exc)
        return 2

    command = "ingest_data.py " + " ".join(argv if argv is not None else sys.argv[1:])
    with db.connect(args.database) as con:
        db.init_schema(con)
        run_id, outcomes = ingest(
            con,
            dataset,
            months,
            args.source or config.paths["incoming"],
            args.bronze or config.paths["bronze"],
            command=command.strip(),
            force=args.force,
        )
        run = get_run(con, run_id)

    log.info(
        "%s %s: %d loaded, %d skipped, %d failed, %s rows written",
        run_id,
        run["status"],
        run["files_loaded"],
        run["files_skipped"] - sum(o.status == "missing" for o in outcomes),
        sum(o.status == "failed" for o in outcomes),
        f"{run['rows_loaded']:,}",
    )
    if run["error_message"]:
        log.log(
            logging.ERROR if run["status"] == "FAILED" else logging.WARNING,
            "%s",
            run["error_message"],
        )
    if run["status"] == "FAILED":
        return 1
    if outcomes and all(o.status == "missing" for o in outcomes):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
