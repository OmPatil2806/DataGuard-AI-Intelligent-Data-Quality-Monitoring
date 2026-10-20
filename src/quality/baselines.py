"""Baselines: what a month is compared with in the dataset-level checks.

The baseline of a month is built from the months *before it in time* (by period, not by
load order, because historical files are loaded all at once):

    - only earlier months at most `max_gap_months` back (so 2019-01 never shapes 2023)
    - the most recent `window` of those
    - trusted once there are at least `min_runs`; until then it is a cold start and the
      checks use the fixed limits in configs/quality_rules.yml (cold_start)

Baseline values:
    rows_per_day      median of the earlier months
    null_rates        median per column
    distributions     mean share of rows per bin, per binned column
    previous_schema   raw schema of the latest earlier month (for the schema check), even
                      during a cold start
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from common.config import load_quality_rules
from common.runs import utc_now
from quality.metrics import MonthMetrics


@dataclass(frozen=True)
class BaselineSettings:
    min_runs: int = 3
    window: int = 6
    max_gap_months: int = 12
    rows_per_day_min: float | None = None
    rows_per_day_max: float | None = None
    null_rate_max: float | None = None


@dataclass
class Baseline:
    period: str
    source_periods: list[str]
    trusted: bool
    rows_per_day: float | None = None
    null_rates: dict[str, float] = field(default_factory=dict)
    distributions: dict[str, list[float]] = field(default_factory=dict)
    previous_period: str | None = None
    previous_schema: list[tuple[str, str]] | None = None

    @property
    def note(self) -> str:
        if self.trusted:
            months = ", ".join(self.source_periods)
            return f"baseline from {len(self.source_periods)} month(s): {months}"
        return f"cold start: {len(self.source_periods)} earlier month(s) available"


def settings_for(dataset: str, path: Path | None = None) -> BaselineSettings:
    config = load_quality_rules(path)
    baseline = config["defaults"]["baseline"]
    limits = (config.get("cold_start") or {}).get(dataset) or {}
    rows = limits.get("rows_per_day") or {}
    return BaselineSettings(
        min_runs=baseline["min_runs"],
        window=baseline.get("window", 6),
        max_gap_months=baseline.get("max_gap_months", 12),
        rows_per_day_min=rows.get("min"),
        rows_per_day_max=rows.get("max"),
        null_rate_max=limits.get("null_rate_max"),
    )


def month_number(period: str) -> int:
    year, month = map(int, period.split("-"))
    return year * 12 + month - 1


def earlier_months(periods: list[str], period: str, settings: BaselineSettings) -> list[str]:
    """The earlier months a baseline for `period` is built from, oldest first."""
    target = month_number(period)
    candidates = sorted(
        p for p in periods if 0 < target - month_number(p) <= settings.max_gap_months
    )
    return candidates[-settings.window :]


def shares(counts: list[int]) -> list[float]:
    total = sum(counts)
    return [c / total for c in counts] if total else [0.0] * len(counts)


def build(metrics: dict[str, MonthMetrics], period: str, settings: BaselineSettings) -> Baseline:
    sources = earlier_months(list(metrics), period, settings)
    earlier = [metrics[p] for p in sources]
    baseline = Baseline(period, sources, trusted=len(sources) >= settings.min_runs)
    if earlier:
        latest = earlier[-1]
        baseline.previous_period = latest.period
        baseline.previous_schema = list(latest.raw_schema)
    if not baseline.trusted:
        return baseline

    baseline.rows_per_day = statistics.median(m.rows_per_day for m in earlier)
    columns = set.intersection(*(set(m.null_rates) for m in earlier))
    baseline.null_rates = {
        c: statistics.median(m.null_rates[c] for m in earlier) for c in sorted(columns)
    }
    binned = set.intersection(*(set(m.histograms) for m in earlier))
    for column in sorted(binned):
        lengths = {len(m.histograms[column]) for m in earlier}
        if len(lengths) != 1:  # bin edges changed between months: not comparable
            continue
        per_month = [shares(m.histograms[column]) for m in earlier]
        baseline.distributions[column] = [statistics.fmean(b) for b in zip(*per_month, strict=True)]
    return baseline


def store(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    dataset: str,
    baseline: Baseline,
    now: datetime | None = None,
) -> None:
    """Record the baseline a run used for a month (for audit and the evidence pack)."""
    now = now or utc_now()
    key = [run_id, dataset, baseline.period]
    sources = ",".join(baseline.source_periods)
    con.execute("DELETE FROM baselines WHERE run_id = ? AND dataset = ? AND period = ?", key)
    records = [
        [*key, "rows_per_day", "", "", baseline.rows_per_day, sources, baseline.trusted, now]
    ]
    records += [
        [*key, "null_rate", c, "", v, sources, baseline.trusted, now]
        for c, v in baseline.null_rates.items()
    ]
    for column, values in baseline.distributions.items():
        records += [
            [*key, "distribution", column, str(i), v, sources, baseline.trusted, now]
            for i, v in enumerate(values)
        ]
    con.executemany("INSERT INTO baselines VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", records)


def overview(metrics: dict[str, MonthMetrics], settings: BaselineSettings) -> pd.DataFrame:
    """For every month: which earlier months its baseline uses and whether it is trusted."""
    records = []
    for period in sorted(metrics):
        b = build(metrics, period, settings)
        records.append(
            {
                "period": period,
                "trusted": b.trusted,
                "earlier_months": ", ".join(b.source_periods) or "-",
                "baseline_rows_per_day": round(b.rows_per_day) if b.rows_per_day else None,
                "previous_schema_from": b.previous_period or "-",
            }
        )
    return pd.DataFrame(records)
