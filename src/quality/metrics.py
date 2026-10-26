"""Dataset-level metrics per month and file version (input for the Phase 3 checks).

For each current Bronze version the following is computed once and stored in
dataset_metrics:

    rows, rows_per_day       volume (rows per day evens out short and long months)
    raw_schema               columns and types exactly as delivered (JSON), before standardization
    null_rate                share of NULLs per standard column, 0..1
    histogram                row counts per fixed bin (configs/quality_rules.yml drift.bins);
                             bin 0 is below the first edge, bin n at or above the last edge
    mean, median, p99        per binned column, for reports and the investigation evidence

    python src/quality/metrics.py            # collect for every month in Bronze and show them
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

if __package__ in (None, ""):
    # Allow running this file directly: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import db  # noqa: E402
from common.config import ConfigError, Dataset, load_datasets_config, load_quality_rules  # noqa: E402
from common.runs import utc_now  # noqa: E402
from ingestion.bronze_loader import PASS_THROUGH, current_versions, read_bronze_sql  # noqa: E402
from quality.standardize import standardize_file  # noqa: E402


@dataclass
class MonthMetrics:
    period: str
    sha256: str
    rows: int
    rows_per_day: float
    raw_schema: list[tuple[str, str]]
    null_rates: dict[str, float] = field(default_factory=dict)
    histograms: dict[str, list[int]] = field(default_factory=dict)
    stats: dict[str, dict[str, float | None]] = field(default_factory=dict)  # mean/median/p99


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def days_in_month(period: str) -> int:
    return pd.Period(period, freq="M").days_in_month


def drift_bins(path: Path | None = None) -> dict[str, list[float]]:
    return load_quality_rules(path)["defaults"]["drift"].get("bins") or {}


def bin_sql(column: str, edges: list[float]) -> str:
    """SQL giving the bin index: 0 below edges[0], i for edges[i-1] <= x < edges[i], n above."""
    cases = " ".join(f"WHEN {_q(column)} < {e!r} THEN {i}" for i, e in enumerate(edges))
    return f"CASE {cases} ELSE {len(edges)} END"


def compute(
    con: duckdb.DuckDBPyConnection,
    bronze_path: Path,
    dataset: Dataset,
    period: str,
    sha256: str,
    bins: dict[str, list[float]],
) -> MonthMetrics:
    raw = con.execute(f"DESCRIBE SELECT * FROM {read_bronze_sql(bronze_path)}").fetchall()
    raw_schema = [(n, t) for n, t, *_ in raw if n not in PASS_THROUGH]
    sql, _ = standardize_file(con, bronze_path, dataset)
    source = f"({sql})"

    columns = [c for c, _ in dataset.columns]
    unknown = sorted(set(bins) - set(columns))
    if unknown:
        raise ConfigError(f"drift.bins: unknown column(s) {', '.join(unknown)}")

    nulls = ", ".join(f"count_if({_q(c)} IS NULL)" for c in columns)
    values = con.execute(f"SELECT count(*), {nulls} FROM {source}").fetchone()
    rows = values[0]
    null_rates = {c: (n / rows if rows else 0.0) for c, n in zip(columns, values[1:], strict=True)}

    histograms, stats = {}, {}
    for column, edges in bins.items():
        counts = dict(
            con.execute(
                f"SELECT {bin_sql(column, edges)} AS b, count(*) FROM {source} "
                f"WHERE {_q(column)} IS NOT NULL GROUP BY b"
            ).fetchall()
        )
        histograms[column] = [counts.get(i, 0) for i in range(len(edges) + 1)]
        mean, median, p99 = con.execute(
            f"SELECT avg({_q(column)}), quantile_cont({_q(column)}, 0.5), "
            f"quantile_cont({_q(column)}, 0.99) FROM {source}"
        ).fetchone()
        stats[column] = {"mean": mean, "median": median, "p99": p99}

    return MonthMetrics(
        period=period,
        sha256=sha256,
        rows=rows,
        rows_per_day=rows / days_in_month(period),
        raw_schema=raw_schema,
        null_rates=null_rates,
        histograms=histograms,
        stats=stats,
    )


def store(
    con: duckdb.DuckDBPyConnection,
    dataset: Dataset,
    m: MonthMetrics,
    now: datetime | None = None,
) -> None:
    """Replace the stored metrics of this file version with `m`."""
    now = now or utc_now()
    key = [dataset.name, m.period, m.sha256]
    con.execute("DELETE FROM dataset_metrics WHERE dataset = ? AND period = ? AND sha256 = ?", key)
    records = [
        [*key, "rows", "", "", float(m.rows), None, now],
        [*key, "rows_per_day", "", "", m.rows_per_day, None, now],
        [*key, "raw_schema", "", "", None, json.dumps(m.raw_schema), now],
    ]
    records += [[*key, "null_rate", c, "", v, None, now] for c, v in m.null_rates.items()]
    for column, counts in m.histograms.items():
        records += [
            [*key, "histogram", column, str(i), float(n), None, now] for i, n in enumerate(counts)
        ]
    for column, values in m.stats.items():
        records += [[*key, name, column, "", v, None, now] for name, v in values.items()]
    con.executemany("INSERT INTO dataset_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", records)


def has_metrics(con: duckdb.DuckDBPyConnection, dataset: str, period: str, sha256: str) -> bool:
    return (
        con.execute(
            "SELECT count(*) FROM dataset_metrics WHERE dataset = ? AND period = ? AND sha256 = ?",
            [dataset, period, sha256],
        ).fetchone()[0]
        > 0
    )


def load(con: duckdb.DuckDBPyConnection, dataset: str, period: str, sha256: str) -> MonthMetrics:
    rows = con.execute(
        "SELECT metric, column_name, bin, value, text_value FROM dataset_metrics "
        "WHERE dataset = ? AND period = ? AND sha256 = ?",
        [dataset, period, sha256],
    ).fetchall()
    if not rows:
        raise KeyError(f"no metrics for {dataset} {period} {sha256[:12]}")
    m = MonthMetrics(period, sha256, 0, 0.0, [])
    histograms: dict[str, dict[int, int]] = {}
    for metric, column, bin_, value, text in rows:
        if metric == "rows":
            m.rows = int(value)
        elif metric == "rows_per_day":
            m.rows_per_day = value
        elif metric == "raw_schema":
            m.raw_schema = [tuple(x) for x in json.loads(text)]
        elif metric == "null_rate":
            m.null_rates[column] = value
        elif metric == "histogram":
            histograms.setdefault(column, {})[int(bin_)] = int(value)
        else:
            m.stats.setdefault(column, {})[metric] = value
    m.histograms = {c: [b[i] for i in range(len(b))] for c, b in histograms.items()}
    return m


def collect(
    con: duckdb.DuckDBPyConnection,
    dataset: Dataset,
    months: list[str] | None = None,
    force: bool = False,
    bins: dict[str, list[float]] | None = None,
) -> dict[str, MonthMetrics]:
    """Metrics for the current Bronze version of each month, computed only when missing."""
    bins = drift_bins() if bins is None else bins
    versions = current_versions(con, dataset)
    result = {}
    for period in months or list(versions):
        if period not in versions:
            continue
        path, sha = versions[period]
        if force or not has_metrics(con, dataset.name, period, sha):
            store(con, dataset, compute(con, path, dataset, period, sha, bins))
        result[period] = load(con, dataset.name, period, sha)
    return result


def overview(metrics: dict[str, MonthMetrics]) -> pd.DataFrame:
    records = []
    for period, m in metrics.items():
        worst = max(m.null_rates.items(), key=lambda kv: kv[1]) if m.null_rates else ("", 0.0)
        distance = m.stats.get("trip_distance", {})
        records.append(
            {
                "period": period,
                "version": m.sha256[:12],
                "rows": m.rows,
                "rows_per_day": round(m.rows_per_day),
                "raw_columns": len(m.raw_schema),
                "most_nulls": f"{worst[0]} {100 * worst[1]:.2f}%",
                "distance_median": distance.get("median"),
            }
        )
    return pd.DataFrame(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect dataset-level metrics from Bronze.")
    parser.add_argument("--dataset", default="yellow_taxi", help="dataset name in datasets.yml")
    parser.add_argument("--force", action="store_true", help="recompute stored metrics")
    parser.add_argument("--database", type=Path, help="DuckDB file (default: paths.duckdb)")
    args = parser.parse_args(argv)

    try:
        dataset = load_datasets_config().dataset(args.dataset)
    except ConfigError as exc:
        print(exc)
        return 2
    with db.connect(args.database) as con:
        db.init_schema(con)
        metrics = collect(con, dataset, force=args.force)
    if not metrics:
        print(f"no Bronze files for '{dataset.name}' (run ingest_data.py)")
        return 1
    print(overview(metrics).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
