"""Load and validate the YAML files in configs/."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

PERIOD_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
SEVERITIES = ("critical", "high", "medium", "low")
CHECK_CATEGORIES = (
    "schema",
    "completeness",
    "uniqueness",
    "validity",
    "pattern",
    "referential",
    "consistency",
    "freshness",
    "volume",
    "drift",
    "ml_anomaly",
)


class ConfigError(ValueError):
    """Raised when a config file is missing, unreadable or invalid."""


@dataclass(frozen=True)
class Lookup:
    name: str
    url: str
    file: str


@dataclass(frozen=True)
class Dataset:
    name: str
    description: str
    enabled: bool
    format: str
    frequency: str
    url_template: str | None = None
    file_template: str | None = None
    timestamp_column: str | None = None
    expected_lag_days: int | None = None
    columns: tuple[tuple[str, str], ...] = ()  # standard schema: (name, DuckDB type)
    periods: tuple[str, ...] = ()
    lookups: tuple[Lookup, ...] = ()

    def url_for(self, period: str) -> str:
        if not self.url_template:
            raise ConfigError(f"dataset '{self.name}' has no url_template")
        return self.url_template.format(period=period)

    def file_for(self, period: str) -> str:
        if not self.file_template:
            raise ConfigError(f"dataset '{self.name}' has no file_template")
        return self.file_template.format(period=period)


@dataclass(frozen=True)
class DatasetsConfig:
    paths: dict[str, Path]
    datasets: dict[str, Dataset] = field(default_factory=dict)

    def dataset(self, name: str) -> Dataset:
        try:
            return self.datasets[name]
        except KeyError:
            known = ", ".join(sorted(self.datasets))
            raise ConfigError(f"unknown dataset '{name}' (known: {known})") from None


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML file that must contain a mapping at the top level."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if not isinstance(mapping, dict) or key not in mapping:
        raise ConfigError(f"missing required key '{key}' in {where}")
    return mapping[key]


def expand_periods(start: str, end: str) -> list[str]:
    """Return every month from start to end inclusive, as 'YYYY-MM' strings."""
    for value in (start, end):
        if not PERIOD_PATTERN.match(str(value)):
            raise ConfigError(f"invalid period '{value}', expected YYYY-MM")
    year, month = map(int, start.split("-"))
    end_year, end_month = map(int, end.split("-"))
    if (year, month) > (end_year, end_month):
        raise ConfigError(f"period start {start} is after end {end}")

    periods = []
    while (year, month) <= (end_year, end_month):
        periods.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return periods


def _parse_periods(raw: dict[str, Any] | None, where: str) -> tuple[str, ...]:
    if not raw:
        return ()
    periods = expand_periods(str(_require(raw, "start", where)), str(_require(raw, "end", where)))
    for extra in raw.get("extra") or []:
        if not PERIOD_PATTERN.match(str(extra)):
            raise ConfigError(f"invalid period '{extra}' in {where}.extra, expected YYYY-MM")
        if extra not in periods:
            periods.append(str(extra))
    return tuple(sorted(periods))


def _parse_lag(value: Any, where: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{where}.expected_lag_days must be a whole number of days >= 0")
    return value


COLUMN_TYPES = ("BOOLEAN", "INTEGER", "BIGINT", "DOUBLE", "VARCHAR", "DATE", "TIMESTAMP")


def _parse_columns(raw: Any, where: str) -> tuple[tuple[str, str], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, dict) or not raw:
        raise ConfigError(f"{where} must map column names to types")
    columns = []
    seen: set[str] = set()
    for name, dtype in raw.items():
        dtype = str(dtype).upper()
        if dtype not in COLUMN_TYPES:
            raise ConfigError(
                f"{where}.{name}: unknown type '{dtype}' (use {', '.join(COLUMN_TYPES)})"
            )
        if name.lower() in seen:
            raise ConfigError(f"{where}: column '{name}' is listed twice (names ignore case)")
        seen.add(name.lower())
        columns.append((str(name), dtype))
    return tuple(columns)


def _parse_dataset(name: str, raw: dict[str, Any]) -> Dataset:
    where = f"datasets.{name}"
    lookups = tuple(
        Lookup(
            name=lookup_name,
            url=_require(lookup, "url", f"{where}.lookups.{lookup_name}"),
            file=_require(lookup, "file", f"{where}.lookups.{lookup_name}"),
        )
        for lookup_name, lookup in (raw.get("lookups") or {}).items()
    )
    dataset = Dataset(
        name=name,
        description=_require(raw, "description", where),
        enabled=bool(_require(raw, "enabled", where)),
        format=_require(raw, "format", where),
        frequency=_require(raw, "frequency", where),
        url_template=raw.get("url_template"),
        file_template=raw.get("file_template"),
        timestamp_column=raw.get("timestamp_column"),
        expected_lag_days=_parse_lag(raw.get("expected_lag_days"), where),
        periods=_parse_periods(raw.get("periods"), f"{where}.periods"),
        lookups=lookups,
        columns=_parse_columns(raw.get("columns"), f"{where}.columns"),
    )
    if dataset.enabled:
        for key in ("url_template", "file_template", "timestamp_column"):
            if not getattr(dataset, key):
                raise ConfigError(f"missing required key '{key}' in {where} (dataset is enabled)")
        if not dataset.periods:
            raise ConfigError(f"missing required key 'periods' in {where} (dataset is enabled)")
    return dataset


def load_datasets_config(path: Path | None = None) -> DatasetsConfig:
    path = Path(path) if path else CONFIG_DIR / "datasets.yml"
    data = load_yaml(path)

    raw_paths = _require(data, "paths", path.name)
    for key in ("incoming", "bronze", "silver", "quarantine", "duckdb"):
        _require(raw_paths, key, f"{path.name}: paths")
    paths = {key: REPO_ROOT / value for key, value in raw_paths.items()}

    raw_datasets = _require(data, "datasets", path.name)
    if not isinstance(raw_datasets, dict) or not raw_datasets:
        raise ConfigError(f"{path.name}: 'datasets' must list at least one dataset")
    datasets = {name: _parse_dataset(name, raw or {}) for name, raw in raw_datasets.items()}
    return DatasetsConfig(paths=paths, datasets=datasets)


def load_quality_rules(path: Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else CONFIG_DIR / "quality_rules.yml"
    data = load_yaml(path)
    defaults = _require(data, "defaults", path.name)
    for section in ("baseline", "volume", "drift", "incidents"):
        _require(defaults, section, f"{path.name}: defaults")
    drift = defaults["drift"]
    if not 0 < drift["psi_medium"] < drift["psi_high"]:
        raise ConfigError(f"{path.name}: drift thresholds need 0 < psi_medium < psi_high")
    for column, edges in (drift.get("bins") or {}).items():
        where = f"{path.name}: defaults.drift.bins.{column}"
        if not isinstance(edges, list) or len(edges) < 2:
            raise ConfigError(f"{where} needs a list of at least two bin edges")
        if any(not isinstance(e, int | float) or isinstance(e, bool) for e in edges):
            raise ConfigError(f"{where} must contain only numbers")
        if any(b <= a for a, b in zip(edges, edges[1:], strict=False)):
            raise ConfigError(f"{where} must be strictly increasing")
    freshness = defaults.get("freshness") or {}
    if freshness.get("mode", "replay") not in ("replay", "live"):
        raise ConfigError(f"{path.name}: defaults.freshness.mode must be 'replay' or 'live'")
    grace = freshness.get("grace_days", 0)
    if isinstance(grace, bool) or not isinstance(grace, int | float) or grace < 0:
        raise ConfigError(f"{path.name}: defaults.freshness.grace_days must be 0 or more")
    null_rate = defaults.get("null_rate") or {}
    for key in ("max_increase", "complete_below", "max_increase_complete"):
        value = null_rate.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int | float) or not 0 < value < 1
        ):
            raise ConfigError(f"{path.name}: defaults.null_rate.{key} must be between 0 and 1")
    baseline = defaults["baseline"]
    for key, minimum in (("min_runs", 1), ("window", 1), ("max_gap_months", 1)):
        value = baseline.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < minimum
        ):
            raise ConfigError(
                f"{path.name}: defaults.baseline.{key} must be a whole number >= {minimum}"
            )
    if baseline.get("window", baseline["min_runs"]) < baseline["min_runs"]:
        raise ConfigError(f"{path.name}: defaults.baseline.window must be >= min_runs")
    for dataset, limits in (data.get("cold_start") or {}).items():
        where = f"{path.name}: cold_start.{dataset}"
        rows = (limits or {}).get("rows_per_day")
        if rows is not None and not (
            isinstance(rows, dict) and 0 <= rows.get("min", 0) < rows.get("max", float("inf"))
        ):
            raise ConfigError(f"{where}.rows_per_day needs 0 <= min < max")
        null_max = (limits or {}).get("null_rate_max")
        if null_max is not None and not (isinstance(null_max, int | float) and 0 <= null_max <= 1):
            raise ConfigError(f"{where}.null_rate_max must be between 0 and 1")
    _require(data, "rules", path.name)
    return data


def load_severity_weights(path: Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else CONFIG_DIR / "severity_weights.yml"
    data = load_yaml(path)

    levels = _require(data, "severity_levels", path.name)
    missing = [s for s in SEVERITIES if s not in levels]
    if missing:
        raise ConfigError(f"{path.name}: severity_levels missing {', '.join(missing)}")

    weights = _require(data, "category_weights", path.name)
    unknown = sorted(set(weights) - set(CHECK_CATEGORIES))
    if unknown:
        raise ConfigError(f"{path.name}: unknown check categories {', '.join(unknown)}")
    for category, weight in weights.items():
        if not isinstance(weight, int | float) or weight <= 0:
            raise ConfigError(f"{path.name}: weight for '{category}' must be a positive number")
    return data
