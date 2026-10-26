"""Standardize a raw file to the dataset's standard schema (configs/datasets.yml: columns).

Source columns are matched to standard columns by name, ignoring case (or by the column's
`source` names, e.g. "Vehicle Speed[km/h]" -> speed_kmh), then renamed and cast. Computed
columns (`sql`) are added from the standardized columns, so rules can use them. Values are
never altered beyond the type change, and every change is reported:

    renamed      Airport_fee -> airport_fee
    computed     a column computed by its sql (e.g. a row time from a day number)
    cast         passenger_count DOUBLE -> INTEGER
    added        a standard column the source does not have (filled with NULL)
    dropped      a source column that is not in the standard schema
    lossy        values that would not survive a cast (e.g. 1.5 -> INTEGER); they become NULL

Bronze lineage columns (bronze_row_id, source_file, ingestion_timestamp, pipeline_run_id, and
origin_file and origin_line of a prepared source) pass through unchanged after them.

    python src/quality/standardize.py             # report for every month in Bronze
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

if __package__ in (None, ""):
    # Allow running this file directly: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import db  # noqa: E402
from common.config import ConfigError, Dataset, load_datasets_config  # noqa: E402
from ingestion.bronze_loader import PASS_THROUGH, current_files, read_bronze_sql  # noqa: E402


class StandardizeError(ValueError):
    """Raised when a source cannot be mapped to the standard schema."""


@dataclass
class Report:
    renamed: list[tuple[str, str]] = field(default_factory=list)
    cast: list[tuple[str, str, str]] = field(default_factory=list)  # column, from, to
    added: list[str] = field(default_factory=list)
    computed: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    lossy: dict[str, int] = field(default_factory=dict)  # column -> values lost in the cast

    @property
    def changed(self) -> bool:
        return bool(
            self.renamed or self.cast or self.added or self.computed or self.dropped or self.lossy
        )

    def summary(self) -> str:
        parts = []
        if self.renamed:
            parts.append("renamed " + ", ".join(f"{a}->{b}" for a, b in self.renamed))
        if self.cast:
            parts.append("cast " + ", ".join(f"{c} {a}->{b}" for c, a, b in self.cast))
        if self.added:
            parts.append("added " + ", ".join(self.added))
        if self.computed:
            parts.append("computed " + ", ".join(self.computed))
        if self.dropped:
            parts.append("dropped " + ", ".join(self.dropped))
        if self.lossy:
            parts.append("LOSSY " + ", ".join(f"{c}: {n:,}" for c, n in self.lossy.items()))
        return "; ".join(parts) or "already standard"


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def plan(source: dict[str, str], dataset: Dataset) -> tuple[list[str], Report]:
    """Build the SELECT list that standardizes a source with the given columns and types.

    Without computed columns this selects from the source directly; with them, the list is
    meant for select_sql(), which computes them from the standardized columns."""
    if not dataset.columns:
        raise StandardizeError(f"dataset '{dataset.name}' has no standard schema (columns:)")

    by_lower: dict[str, str] = {}
    for name in source:
        if name.lower() in by_lower:
            raise StandardizeError(
                f"source has both '{by_lower[name.lower()]}' and '{name}'; names must differ "
                "by more than case"
            )
        by_lower[name.lower()] = name

    report = Report()
    select: list[str] = []
    used: set[str] = set()
    computed = dict(dataset.computed)
    for target, target_type in dataset.columns:
        if target in computed:
            report.computed.append(target)
            continue
        found = next(
            (by_lower[n.lower()] for n in dataset.source_names(target) if n.lower() in by_lower),
            None,
        )
        if found is None:
            report.added.append(target)
            select.append(f"CAST(NULL AS {target_type}) AS {_quote(target)}")
            continue
        used.add(found)
        if found != target:
            report.renamed.append((found, target))
        if source[found] != target_type:
            report.cast.append((target, source[found], target_type))
            select.append(f"TRY_CAST({_quote(found)} AS {target_type}) AS {_quote(target)}")
        else:
            select.append(f"{_quote(found)} AS {_quote(target)}")

    lineage = [c for c in PASS_THROUGH if c in source]
    select += [_quote(c) for c in lineage]
    report.dropped = [c for c in source if c not in used and c not in lineage]
    return select, report


def select_sql(select: list[str], source_sql: str, dataset: Dataset, source: list[str]) -> str:
    """The standardized SELECT; computed columns are added from the standardized columns, and
    every column comes in the order of the standard schema, then the lineage."""
    inner = f"SELECT {', '.join(select)} FROM {source_sql}"
    if not dataset.computed:
        return inner
    computed = dict(dataset.computed)
    outer = [
        f"CAST(({computed[name]}) AS {dtype}) AS {_quote(name)}"
        if name in computed
        else _quote(name)
        for name, dtype in dataset.columns
    ]
    outer += [_quote(c) for c in PASS_THROUGH if c in source]
    return f"SELECT {', '.join(outer)} FROM ({inner}) AS standardized"


def _lossy_counts(
    con: duckdb.DuckDBPyConnection, source_sql: str, source: dict[str, str], report: Report
) -> dict[str, int]:
    """Values that are not NULL before a cast but would not round-trip through it."""
    checks = {}
    lookup = {name.lower(): name for name in source}
    for found, target in report.renamed:
        lookup.setdefault(target.lower(), found)  # a renamed column is cast by its source name
    for column, _, target_type in report.cast:
        original = _quote(lookup[column.lower()])
        converted = f"TRY_CAST({original} AS {target_type})"
        checks[column] = (
            f"count_if({original} IS NOT NULL AND ({converted} IS NULL "
            f"OR CAST({converted} AS {source[lookup[column.lower()]]}) <> {original}))"
        )
    if not checks:
        return {}
    values = con.execute(f"SELECT {', '.join(checks.values())} FROM {source_sql}").fetchone()
    return {c: n for c, n in zip(checks, values, strict=True) if n}


def standardize(
    con: duckdb.DuckDBPyConnection, source_sql: str, dataset: Dataset
) -> tuple[str, Report]:
    """Return SQL selecting the standardized rows of `source_sql` (a FROM-able expression)."""
    described = con.execute(f"DESCRIBE SELECT * FROM {source_sql}").fetchall()
    source = {name: dtype for name, dtype, *_ in described}
    select, report = plan(source, dataset)
    report.lossy = _lossy_counts(con, source_sql, source, report)
    return select_sql(select, source_sql, dataset, list(source)), report


def standardize_file(
    con: duckdb.DuckDBPyConnection, path: Path, dataset: Dataset
) -> tuple[str, Report]:
    return standardize(con, read_bronze_sql(path), dataset)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show how each Bronze month is standardized.")
    parser.add_argument("--dataset", default="yellow_taxi", help="dataset name in datasets.yml")
    parser.add_argument("--database", type=Path, help="DuckDB file (default: paths.duckdb)")
    args = parser.parse_args(argv)

    try:
        dataset = load_datasets_config().dataset(args.dataset)
    except ConfigError as exc:
        print(exc)
        return 2
    with db.connect(args.database, read_only=True) as con:
        files = current_files(con, dataset)
        if not files:
            print(f"no Bronze files for '{dataset.name}' (run ingest_data.py)")
            return 1
        records, lossy = [], False
        for period, path in files.items():
            _, report = standardize_file(con, path, dataset)
            lossy = lossy or bool(report.lossy)
            records.append({"period": period, "changes": report.summary()})
    with pd.option_context("display.max_colwidth", None):
        print(pd.DataFrame(records).to_string(index=False))
    return 1 if lossy else 0


if __name__ == "__main__":
    sys.exit(main())
