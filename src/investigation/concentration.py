"""Where and when: is an incident's problem concentrated in one part of the month? (Phase 8)

The affected rows of a row-level incident (evidence_pack.month_rows) are compared with all rows
the rule or check looked at, segment by segment, along the dimensions of
configs/investigation.yml (vendor, zones, payment type, rate code, date, weekday, hour). A
segment is a concentration only when all of these hold:

    rows    >= min_rows       enough rows in the segment to judge
    share   >= min_share      it holds most of the affected rows
    ratio   >= min_ratio      its affected rate is several times the rate outside it
    z       >= min_z          and that difference is far beyond chance (two-proportion z test)

A problem spread evenly over the month has no such segment: the answer is "unknown", never a
guess. For volume incidents, the days of the month with far fewer rows than the month's median
day show whether rows are missing on some days or thinly everywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from common.config import ConfigError, config_dir, load_yaml

KINDS = ("where", "when")


@dataclass(frozen=True)
class Dimension:
    name: str
    sql: str
    kind: str


@dataclass(frozen=True)
class ConcentrationSettings:
    dimensions: tuple[Dimension, ...]
    min_rows: int = 20
    min_share: float = 0.5
    min_ratio: float = 3.0
    min_z: float = 4.0
    max_findings: int = 3
    low_day_share: float = 0.5
    min_rows_per_day: int = 50


THRESHOLDS = (
    "min_rows", "min_share", "min_ratio", "min_z", "max_findings", "low_day_share",
    "min_rows_per_day",
)  # fmt: skip


def _config(path: Path | None) -> tuple[Path, dict[str, Any]]:
    path = Path(path) if path else config_dir() / "investigation.yml"
    data = load_yaml(path)
    if not isinstance(data.get("concentration"), dict):
        raise ConfigError(f"{path}: 'concentration' must be a mapping")
    datasets = data.get("datasets") or {}
    if not isinstance(datasets, dict):
        raise ConfigError(f"{path}: 'datasets' must map dataset names to settings")
    return path, data


def load_settings(dataset: str | None = None, path: Path | None = None) -> ConcentrationSettings:
    """The thresholds, and the where/when dimensions of a dataset (none when not configured)."""
    path, data = _config(path)
    raw = data["concentration"]
    dims = ((data.get("datasets") or {}).get(dataset) or {}).get("dimensions") or {}
    if not isinstance(dims, dict):
        raise ConfigError(f"{path}: datasets.{dataset}.dimensions must be a mapping")
    dimensions = []
    for name, spec in dims.items():
        if not isinstance(spec, dict) or not spec.get("sql"):
            raise ConfigError(f"{path}: dimension '{name}' needs sql")
        kind = spec.get("kind", "where")
        if kind not in KINDS:
            raise ConfigError(f"{path}: dimension '{name}': kind must be one of {KINDS}")
        dimensions.append(Dimension(name, spec["sql"], kind))
    numbers = {k: raw[k] for k in THRESHOLDS if k in raw}
    for key, value in numbers.items():
        if not isinstance(value, int | float) or isinstance(value, bool) or value < 0:
            raise ConfigError(f"{path}: concentration.{key} must be a number >= 0")
    return ConcentrationSettings(tuple(dimensions), **numbers)


def sample_columns(dataset: str, path: Path | None = None) -> list[str] | None:
    """The columns shown for a dataset's sample rows, or None when not configured."""
    path, data = _config(path)
    columns = ((data.get("datasets") or {}).get(dataset) or {}).get("sample_columns")
    if columns is not None and not (
        isinstance(columns, list) and all(isinstance(c, str) for c in columns)
    ):
        raise ConfigError(f"{path}: datasets.{dataset}.sample_columns must be a list of columns")
    return columns


def z_statistic(k_in: int, n_in: int, k_out: int, n_out: int) -> float | None:
    """Two-proportion z statistic of the rate inside against the rate outside (pooled)."""
    if n_in == 0 or n_out == 0:
        return None
    pooled = (k_in + k_out) / (n_in + n_out)
    if pooled in (0.0, 1.0):
        return None
    se = math.sqrt(pooled * (1 - pooled) * (1 / n_in + 1 / n_out))
    return (k_in / n_in - k_out / n_out) / se


def segments(
    con: duckdb.DuckDBPyConnection, source: str, where: str, settings: ConcentrationSettings
) -> dict[str, Any]:
    """The concentrations of the affected rows (where = SQL condition) in the rows of source."""
    total, affected = con.execute(
        f"SELECT count(*), count_if(coalesce({where}, false)) FROM {source}"
    ).fetchone()
    result: dict[str, Any] = {"rows": total, "affected_rows": affected, "findings": []}
    if affected == 0 or affected == total:
        result["analysed"] = False
        result["reason"] = "no affected rows" if affected == 0 else "every row is affected"
        return result
    result["analysed"] = True
    result["overall_rate"] = round(affected / total, 6)
    # fewer affected rows than a segment needs cannot show a concentration beyond chance
    result["few_affected"] = affected < settings.min_rows
    found = []
    for dim in settings.dimensions:
        rows = con.execute(
            f"SELECT CAST(({dim.sql}) AS VARCHAR) AS v, count(*), "
            f"count_if(coalesce({where}, false)) FROM {source} GROUP BY 1"
        ).fetchall()
        for value, n, k in rows:
            if value is None or n < settings.min_rows or k == 0:
                continue
            k_out, n_out = affected - k, total - n
            z = z_statistic(k, n, k_out, n_out)
            rate, outside = k / n, (k_out / n_out if n_out else 0.0)
            share = k / affected
            ratio = rate / outside if outside > 0 else None  # None: only in this segment
            if (
                z is not None
                and z >= settings.min_z
                and share >= settings.min_share
                and (ratio is None or ratio >= settings.min_ratio)
            ):
                found.append(
                    {
                        "dimension": dim.name,
                        "kind": dim.kind,
                        "value": value,
                        "rows": n,
                        "affected_rows": k,
                        "rate": round(rate, 6),
                        "rate_outside": round(outside, 6),
                        "ratio": None if ratio is None else round(ratio, 2),
                        "share": round(share, 6),
                        "z": round(z, 2),
                    }
                )
    found.sort(key=lambda f: (-f["z"], f["dimension"], f["value"]))
    result["findings"] = found[: settings.max_findings]
    return result


def low_days(
    con: duckdb.DuckDBPyConnection,
    source: str,
    period: str,
    settings: ConcentrationSettings,
    timestamp: str,
) -> dict[str, Any]:
    """Days of the month with far fewer rows than the month's median day (volume incidents);
    a row's day comes from `timestamp`, the dataset's timestamp column."""
    first = f"{period}-01"
    days = con.execute(
        f"""
        WITH days AS (
            SELECT CAST(d AS DATE) AS day FROM generate_series(DATE '{first}',
                DATE '{first}' + INTERVAL 1 MONTH - INTERVAL 1 DAY, INTERVAL 1 DAY) t(d)
        ), counts AS (
            SELECT CAST("{timestamp}" AS DATE) AS day, count(*) AS n FROM {source} GROUP BY 1
        )
        SELECT day, coalesce(n, 0) FROM days LEFT JOIN counts USING (day) ORDER BY day
        """
    ).fetchall()
    counts = [n for _, n in days]
    median = sorted(counts)[len(counts) // 2] if counts else 0
    result: dict[str, Any] = {"median_rows_per_day": median, "days": []}
    if median < settings.min_rows_per_day:
        result["analysed"] = False
        result["reason"] = f"median day has {median} rows, fewer than {settings.min_rows_per_day}"
        return result
    result["analysed"] = True
    limit = settings.low_day_share * median
    result["days"] = [
        {"day": day.isoformat(), "rows": n, "share_of_median": round(n / median, 4)}
        for day, n in days
        if n < limit
    ]
    return result


def verdict(where: dict[str, Any]) -> str:
    """One line on the strongest concentration, or why there is none."""
    if not where.get("analysed"):
        return f"not analysed: {where.get('reason', 'no affected rows')}"
    if not where["findings"] and where.get("few_affected"):
        return (
            f"unknown: only {where['affected_rows']} affected row(s), too few to tell whether "
            "they gather in one part of the data"
        )
    if not where["findings"]:
        return (
            "unknown: the affected rows are spread over the month, no segment holds most of "
            "them at a clearly higher rate"
        )
    f = where["findings"][0]
    outside = (
        "and nowhere else"
        if f["ratio"] is None
        else f"against {100 * f['rate_outside']:.1f}% elsewhere"
    )
    return (
        f"{f['kind']}: {f['dimension']} = {f['value']} holds {100 * f['share']:.0f}% of the "
        f"affected rows; {100 * f['rate']:.1f}% of its rows are affected {outside}"
    )
