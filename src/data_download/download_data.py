"""Download source files listed in configs/datasets.yml into data/incoming/.

Examples:
    python src/data_download/download_data.py
    python src/data_download/download_data.py --months 2023-01
    python src/data_download/download_data.py --months 2023-01:2023-03,2019-01
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import requests

if __package__ in (None, ""):
    # Allow running this file directly: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import (  # noqa: E402
    PERIOD_PATTERN,
    ConfigError,
    Dataset,
    expand_periods,
    load_datasets_config,
)

log = logging.getLogger("download_data")

CHUNK_SIZE = 1024 * 1024
TIMEOUT_SECONDS = 60
RETRIES = 3
BACKOFF_SECONDS = 2
PARQUET_MAGIC = b"PAR1"


@dataclass(frozen=True)
class Target:
    url: str
    dest: Path
    kind: str  # "parquet" or "csv"


@dataclass(frozen=True)
class Result:
    target: Target
    status: str  # "downloaded", "skipped" or "failed"
    size: int = 0
    error: str = ""


def parse_months(spec: str | None, configured: tuple[str, ...]) -> list[str]:
    """Turn '2023-01', '2023-01:2023-03' or a comma-separated mix into months."""
    if not spec:
        return list(configured)
    months: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if ":" in part:
            start, end = (p.strip() for p in part.split(":", 1))
            months.extend(expand_periods(start, end))
        elif PERIOD_PATTERN.match(part):
            months.append(part)
        else:
            raise ConfigError(f"invalid month '{part}', expected YYYY-MM or YYYY-MM:YYYY-MM")
    return sorted(set(months))


def build_targets(
    dataset: Dataset, months: list[str], dest_dir: Path, lookups: bool = True
) -> list[Target]:
    targets = [
        Target(dataset.url_for(m), dest_dir / dataset.file_for(m), dataset.format) for m in months
    ]
    if lookups:
        targets += [
            Target(lookup.url, dest_dir / lookup.file, Path(lookup.file).suffix.lstrip("."))
            for lookup in dataset.lookups
        ]
    return targets


def is_valid_file(path: Path, kind: str) -> bool:
    """Cheap integrity check: Parquet magic bytes, or a CSV that is not an HTML page."""
    if not path.is_file():
        return False
    size = path.stat().st_size
    if kind == "parquet":
        if size < 2 * len(PARQUET_MAGIC):
            return False
        with path.open("rb") as f:
            head = f.read(len(PARQUET_MAGIC))
            f.seek(-len(PARQUET_MAGIC), os.SEEK_END)
            tail = f.read(len(PARQUET_MAGIC))
        return head == PARQUET_MAGIC and tail == PARQUET_MAGIC
    if kind == "csv":
        with path.open("rb") as f:
            first_line = f.readline().strip()
        return b"," in first_line and not first_line.startswith(b"<")
    raise ValueError(f"unsupported file kind '{kind}'")


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        status = exc.response.status_code
        return status >= 500 or status == 429
    return isinstance(exc, requests.RequestException)


def download(
    target: Target,
    session: requests.Session,
    force: bool = False,
    retries: int = RETRIES,
    sleep: Callable[[float], None] = time.sleep,
) -> Result:
    """Download one file to a .part file, validate it, then move it into place."""
    if not force and is_valid_file(target.dest, target.kind):
        return Result(target, "skipped", target.dest.stat().st_size)

    target.dest.parent.mkdir(parents=True, exist_ok=True)
    part = target.dest.with_name(target.dest.name + ".part")
    error = ""
    for attempt in range(1, retries + 1):
        try:
            with session.get(target.url, stream=True, timeout=TIMEOUT_SECONDS) as response:
                response.raise_for_status()
                with part.open("wb") as f:
                    for chunk in response.iter_content(CHUNK_SIZE):
                        f.write(chunk)
            if not is_valid_file(part, target.kind):
                part.unlink(missing_ok=True)
                return Result(target, "failed", error=f"downloaded file is not valid {target.kind}")
            os.replace(part, target.dest)
            return Result(target, "downloaded", target.dest.stat().st_size)
        except requests.RequestException as exc:
            part.unlink(missing_ok=True)
            error = str(exc)
            if not _is_retryable(exc) or attempt == retries:
                break
            wait = BACKOFF_SECONDS * 2 ** (attempt - 1)
            log.warning(
                "attempt %d failed for %s (%s); retrying in %ss", attempt, target.url, exc, wait
            )
            sleep(wait)
    return Result(target, "failed", error=error)


def _format_size(size: int) -> str:
    return f"{size / 1024 / 1024:.1f} MB" if size >= 1024 * 1024 else f"{size / 1024:.1f} KB"


def main(
    argv: list[str] | None = None,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    parser = argparse.ArgumentParser(description="Download source data into data/incoming/.")
    parser.add_argument("--dataset", default="yellow_taxi", help="dataset name in datasets.yml")
    parser.add_argument("--months", help="e.g. 2023-01, 2023-01:2023-03 or 2023-01,2019-01")
    parser.add_argument("--config", type=Path, help="path to datasets.yml")
    parser.add_argument("--dest", type=Path, help="target folder (default: paths.incoming)")
    parser.add_argument("--force", action="store_true", help="download even if the file exists")
    parser.add_argument("--no-lookups", action="store_true", help="skip lookup files")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load_datasets_config(args.config)
        dataset = config.dataset(args.dataset)
        months = parse_months(args.months, dataset.periods)
        dest_dir = args.dest or config.paths["incoming"]
        targets = build_targets(dataset, months, dest_dir, lookups=not args.no_lookups)
    except ConfigError as exc:
        log.error("%s", exc)
        return 2

    session = session or requests.Session()
    log.info("%d file(s) for '%s' -> %s", len(targets), dataset.name, dest_dir)
    results = []
    for target in targets:
        result = download(target, session, force=args.force, sleep=sleep)
        results.append(result)
        name = target.dest.name
        if result.status == "failed":
            log.error("failed     %s: %s", name, result.error)
        else:
            log.info("%-10s %s (%s)", result.status, name, _format_size(result.size))

    counts = {s: sum(r.status == s for r in results) for s in ("downloaded", "skipped", "failed")}
    log.info(
        "done: %d downloaded, %d skipped, %d failed",
        counts["downloaded"],
        counts["skipped"],
        counts["failed"],
    )
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
