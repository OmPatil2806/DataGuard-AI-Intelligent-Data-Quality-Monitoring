"""Prepare a source that is not one file per period: one Parquet file per period (Phase 9.2).

    python src/ingestion/prepare_source.py --dataset ved_signals
    python src/ingestion/prepare_source.py --dataset ved_signals --months 2018-01

For a dataset with a `source` block in configs/datasets.yml (`kind: csv_files`), the files
matching `source.files` in <incoming>/<dataset>/ are read and their rows regrouped into
periods by `source.period_sql`, a SQL expression over the source row. Each period becomes one
file in <incoming>/, named by `file_template`, which ingest_data.py then registers and loads
like any other source file (it calls this first for such datasets).

    1 stage     every CSV is read with pyarrow, `source.null_values` read as NULL; its row
                count must equal its line count (a CSV reader that loses rows silently, as
                DuckDB does on the VED files, stops the run); written to
                <incoming>/<dataset>/staged/<file>.parquet with origin_file, origin_line
                (1-based line after the header) and the period
    2 regroup   per period: the staged rows of that period, ordered by origin_file and
                origin_line, written to <incoming>/<file_template> without the period column

Source columns keep their names and values; origin_file and origin_line trace every row back
to its source line. The output is deterministic: the same CSVs give the same files and the
same SHA-256, so ingest_data.py registers no new version. Rows of periods not configured
(`periods`) are counted and left out.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

if __package__ in (None, ""):
    # Allow running this file directly: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import ConfigError, Dataset, load_datasets_config  # noqa: E402

log = logging.getLogger("prepare_source")
ORIGIN_COLUMNS = ("origin_file", "origin_line")
PERIOD = "__period"
STAGED = "staged"


class PrepareError(RuntimeError):
    """A source file cannot be read completely."""


@dataclass
class Prepared:
    staged: list[str] = field(default_factory=list)  # files staged now (others were current)
    files: dict[str, int] = field(default_factory=dict)  # period -> rows written
    unchanged: list[str] = field(default_factory=list)  # periods whose file was up to date
    rows_outside: dict[str, int] = field(default_factory=dict)  # unconfigured period -> rows
    empty: list[str] = field(default_factory=list)  # wanted periods without a row: no file


def _sql_path(path: Path) -> str:
    return "'" + path.as_posix().replace("'", "''") + "'"


def data_lines(path: Path) -> int:
    """Lines after the header, counted in the file (a last line without newline counts)."""
    lines, last = 0, b"\n"
    with path.open("rb") as f:
        f.readline()
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            lines += chunk.count(b"\n")
            last = chunk[-1:]
    return lines + (last != b"\n")


def stage(dataset: Dataset, path: Path, staged_dir: Path) -> Path:
    """One source CSV as staged Parquet, with its origin and period, once per change."""
    target = staged_dir / path.with_suffix(".parquet").name
    if target.is_file() and target.stat().st_mtime >= path.stat().st_mtime:
        return target
    nulls = ["", *dataset.source.null_values]
    table = pacsv.read_csv(path, convert_options=pacsv.ConvertOptions(null_values=nulls))
    expected = data_lines(path)
    if table.num_rows != expected:
        raise PrepareError(f"{path.name}: read {table.num_rows:,} rows, the file has {expected:,}")
    clash = sorted({c.lower() for c in table.column_names} & {*ORIGIN_COLUMNS, PERIOD})
    if clash:
        raise PrepareError(f"{path.name} already has column(s): {', '.join(clash)}")
    table = table.append_column("origin_file", pa.array([path.name] * table.num_rows))
    table = table.append_column("origin_line", pa.array(range(1, table.num_rows + 1), pa.int64()))
    con = duckdb.connect()
    try:
        con.register("rows", table)
        result = con.execute(
            f"SELECT *, CAST(({dataset.source.period_sql}) AS VARCHAR) AS {PERIOD} FROM rows"
        ).arrow()
        # a table, or a stream of batches in newer DuckDB versions
        with_period = result.read_all() if hasattr(result, "read_all") else result
    finally:
        con.close()
    staged_dir.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".parquet.part")
    pq.write_table(with_period, tmp, compression="zstd")
    os.replace(tmp, target)
    return target


def prepare(
    dataset: Dataset,
    incoming: Path,
    periods: list[str] | None = None,
    force: bool = False,
) -> Prepared:
    """Stage the dataset's source files and write one file per period into incoming."""
    if dataset.source is None:
        raise ConfigError(f"dataset '{dataset.name}' has no source block in datasets.yml")
    folder = incoming / dataset.name
    sources = sorted(folder.glob(dataset.source.files))
    if not sources:
        raise ConfigError(
            f"no files matching {dataset.source.files} in {folder} "
            "(python src/data_download/fetch_archives.py)"
        )
    staged_dir = folder / STAGED
    if force:
        shutil.rmtree(staged_dir, ignore_errors=True)
    result = Prepared()
    staged = []
    for path in sources:
        before = (staged_dir / path.with_suffix(".parquet").name).exists()
        staged.append(stage(dataset, path, staged_dir))
        if force or not before:
            result.staged.append(path.name)
    wanted = list(periods if periods is not None else dataset.periods)
    files = ", ".join(_sql_path(p) for p in staged)
    con = duckdb.connect()
    try:
        con.execute(f"CREATE VIEW staged AS SELECT * FROM read_parquet([{files}])")
        counts = dict(con.execute(f"SELECT {PERIOD}, count(*) FROM staged GROUP BY 1").fetchall())
        configured = set(dataset.periods) | set(wanted)
        result.rows_outside = {
            p: n for p, n in sorted(counts.items(), key=str) if p not in configured
        }
        newest = max(p.stat().st_mtime for p in staged)
        for period in wanted:
            target = incoming / dataset.file_for(period)
            if not counts.get(period):  # no file: ingest_data.py reports the month missing
                result.empty.append(period)
                continue
            if target.is_file() and target.stat().st_mtime >= newest and not force:
                result.unchanged.append(period)
                continue
            tmp = target.with_name(target.name + ".part")
            con.execute(
                f"COPY (SELECT * EXCLUDE ({PERIOD}) FROM staged WHERE {PERIOD} = ? "
                f"ORDER BY origin_file, origin_line) TO {_sql_path(tmp)} "
                "(FORMAT parquet, COMPRESSION zstd)",
                [period],
            )
            os.replace(tmp, target)
            result.files[period] = counts.get(period, 0)
    finally:
        con.close()
    return result


def main(argv: list[str] | None = None) -> int:
    from data_download.download_data import parse_months

    parser = argparse.ArgumentParser(description="Write one source file per period.")
    parser.add_argument("--dataset", required=True, help="dataset name in datasets.yml")
    parser.add_argument("--months", help="e.g. 2018-01 or 2017-11:2018-03 (default: periods)")
    parser.add_argument("--source", type=Path, help="incoming folder (default: paths.incoming)")
    parser.add_argument("--force", action="store_true", help="stage and write everything again")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        config = load_datasets_config()
        dataset = config.dataset(args.dataset)
        periods = parse_months(args.months, dataset.periods)
        result = prepare(dataset, args.source or config.paths["incoming"], periods, args.force)
    except (ConfigError, PrepareError) as exc:
        log.error("%s", exc)
        return 2 if isinstance(exc, ConfigError) else 1
    for period, rows in result.files.items():
        log.info("written    %s: %s rows", dataset.file_for(period), f"{rows:,}")
    for period in result.unchanged:
        log.info("unchanged  %s", dataset.file_for(period))
    for period in result.empty:
        log.warning("no rows for %s: no file written", period)
    for period, rows in result.rows_outside.items():
        log.warning("outside the configured periods: %s, %s rows left out", period, f"{rows:,}")
    log.info("%d source file(s) staged, %d period file(s) written, %d unchanged",
             len(result.staged), len(result.files), len(result.unchanged))  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
