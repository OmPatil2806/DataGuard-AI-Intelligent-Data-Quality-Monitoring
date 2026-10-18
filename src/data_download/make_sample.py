"""Write a reproducible fraction of each downloaded file to data/sample/ for fast development.

Rows are chosen by hashing their position in the file with a seed, so the same
fraction and seed always select the same rows, independent of thread count.

Examples:
    python src/data_download/make_sample.py
    python src/data_download/make_sample.py --fraction 0.05 --months 2023-01:2023-03
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import duckdb

if __package__ in (None, ""):
    # Allow running this file directly: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import ConfigError, load_datasets_config  # noqa: E402
from data_download.download_data import is_valid_file, parse_months  # noqa: E402

log = logging.getLogger("make_sample")

DEFAULT_FRACTION = 0.10
DEFAULT_SEED = 42
HASH_BUCKETS = 1_000_000


@dataclass(frozen=True)
class SampleResult:
    source: Path
    dest: Path
    status: str  # "created", "skipped" or "missing"
    source_rows: int = 0
    sample_rows: int = 0


def _sql_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def sample_parquet(
    source: Path,
    dest: Path,
    fraction: float = DEFAULT_FRACTION,
    seed: int = DEFAULT_SEED,
    force: bool = False,
) -> SampleResult:
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if not is_valid_file(source, "parquet"):
        return SampleResult(source, dest, "missing")
    if dest.exists() and not force:
        return SampleResult(source, dest, "skipped")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    threshold = round(fraction * HASH_BUCKETS)
    with duckdb.connect() as con:
        source_rows = con.execute(
            f"SELECT count(*) FROM read_parquet('{_sql_path(source)}')"
        ).fetchone()[0]
        con.execute(
            f"""
            COPY (
                SELECT * EXCLUDE (file_row_number)
                FROM read_parquet('{_sql_path(source)}', file_row_number = true)
                WHERE hash(file_row_number, {int(seed)}) % {HASH_BUCKETS} < {threshold}
                ORDER BY file_row_number
            ) TO '{_sql_path(tmp)}' (FORMAT parquet)
            """
        )
        sample_rows = con.execute(
            f"SELECT count(*) FROM read_parquet('{_sql_path(tmp)}')"
        ).fetchone()[0]
    tmp.replace(dest)
    return SampleResult(source, dest, "created", source_rows, sample_rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create reproducible samples in data/sample/.")
    parser.add_argument("--dataset", default="yellow_taxi", help="dataset name in datasets.yml")
    parser.add_argument("--months", help="e.g. 2023-01, 2023-01:2023-03 or 2023-01,2019-01")
    parser.add_argument("--fraction", type=float, default=DEFAULT_FRACTION, help="default 0.10")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="default 42")
    parser.add_argument("--config", type=Path, help="path to datasets.yml")
    parser.add_argument("--source", type=Path, help="input folder (default: paths.incoming)")
    parser.add_argument("--dest", type=Path, help="output folder (default: paths.sample)")
    parser.add_argument("--force", action="store_true", help="recreate existing samples")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if not 0 < args.fraction <= 1:
            raise ConfigError(f"--fraction must be in (0, 1], got {args.fraction}")
        config = load_datasets_config(args.config)
        dataset = config.dataset(args.dataset)
        months = parse_months(args.months, dataset.periods)
        source_dir = args.source or config.paths["incoming"]
        dest_dir = args.dest or config.paths["sample"]
    except (ConfigError, KeyError) as exc:
        log.error("%s", exc)
        return 2

    log.info(
        "sampling %.0f%% (seed %d): %s -> %s", args.fraction * 100, args.seed, source_dir, dest_dir
    )
    results = []
    for month in months:
        name = dataset.file_for(month)
        result = sample_parquet(
            source_dir / name, dest_dir / name, args.fraction, args.seed, force=args.force
        )
        results.append(result)
        if result.status == "missing":
            log.warning("missing    %s (run download_data.py first)", name)
        elif result.status == "skipped":
            log.info("skipped    %s (already sampled; use --force to recreate)", name)
        else:
            log.info("created    %s: %d of %d rows", name, result.sample_rows, result.source_rows)

    # Lookups are small: copy them so data/sample/ is self-contained.
    for lookup in dataset.lookups:
        src = source_dir / lookup.file
        if src.is_file():
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest_dir / lookup.file)
        else:
            log.warning("missing    %s (run download_data.py first)", lookup.file)

    created = sum(r.status == "created" for r in results)
    skipped = sum(r.status == "skipped" for r in results)
    missing = sum(r.status == "missing" for r in results)
    log.info("done: %d created, %d skipped, %d missing", created, skipped, missing)
    return 1 if missing and not (created or skipped) else 0


if __name__ == "__main__":
    sys.exit(main())
