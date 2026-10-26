"""Load and validate the YAML files in configs/."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
CONFIG_DIR_ENV = "DATAGUARD_CONFIG_DIR"  # another folder of config files (Phase 9.4: tests)


def config_dir() -> Path:
    """The folder the config files are read from: configs/, or $DATAGUARD_CONFIG_DIR."""
    override = os.environ.get(CONFIG_DIR_ENV)
    return Path(override) if override else CONFIG_DIR


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
    url: str | None  # where download_data.py fetches it (None: it comes another way)
    file: str  # the first (often only) file
    # Phase 9.4: a lookup made of several files with the same columns (stacked by name), and
    # columns renamed per file first, where the files name the same column differently
    files: tuple[str, ...] = ()
    rename: tuple[tuple[str, str], ...] = ()

    @property
    def paths(self) -> tuple[str, ...]:
        return self.files or (self.file,)


ARCHIVE_KINDS = ("7z", "xlsx", "csv")
SOURCE_KINDS = ("csv_files",)


@dataclass(frozen=True)
class Source:
    """Source files that are not one file per period (Phase 9.2): prepare_source.py reads them
    and writes one Parquet file per period (file_template) into the incoming folder."""

    kind: str  # csv_files: CSV files in <incoming>/<dataset>/, grouped into periods by row
    files: str  # glob of the files
    period_sql: str  # SQL over a source row giving its period, YYYY-MM
    null_values: tuple[str, ...] = ()  # text read as NULL (besides an empty field)
    url: str | None = None  # where the source is published (lineage)


@dataclass(frozen=True)
class Archive:
    """A source file that is not one file per period (Phase 9): a 7z archive of member files,
    or a spreadsheet converted to CSV. Fetched by src/data_download/fetch_archives.py."""

    name: str
    url: str
    file: str
    kind: str  # 7z | xlsx | csv
    members: str | None = None  # glob of the 7z members to unpack (default: all)


@dataclass(frozen=True)
class SilverJoin:
    lookup: str
    key: str  # column in the lookup
    column: str  # standard column matched against the key
    columns: tuple[tuple[str, str], ...]  # (lookup column, Silver column name)


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
    # Phase 9.3: source names of columns whose source name differs, and computed columns
    column_sources: tuple[tuple[str, tuple[str, ...]], ...] = ()
    computed: tuple[tuple[str, str], ...] = ()  # (column, SQL over the other standard columns)
    silver_derived: tuple[tuple[str, str], ...] = ()  # (column, SQL expression)
    silver_joins: tuple[SilverJoin, ...] = ()
    periods: tuple[str, ...] = ()
    lookups: tuple[Lookup, ...] = ()
    archives: tuple[Archive, ...] = ()
    source: Source | None = None

    def source_names(self, column: str) -> tuple[str, ...]:
        """The names a standard column may have in a source file, in order of preference."""
        return dict(self.column_sources).get(column, (column,))

    @property
    def computed_names(self) -> set[str]:
        return {name for name, _ in self.computed}

    def raw_columns(self) -> list[tuple[str, tuple[str, ...]]]:
        """(standard column, accepted source names) for every column read from the source."""
        return [
            (name, self.source_names(name))
            for name, _ in self.columns
            if name not in self.computed_names
        ]

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


def _parse_columns(
    raw: Any, where: str
) -> tuple[
    tuple[tuple[str, str], ...],
    tuple[tuple[str, tuple[str, ...]], ...],
    tuple[tuple[str, str], ...],
]:
    """The standard schema: (name, type) per column, the source names of renamed columns, and
    the SQL of computed columns.

        VendorID: INTEGER                                    source column of the same name
        speed_kmh: {type: DOUBLE, source: "Vehicle Speed[km/h]"}         or a list of names
        row_time: {type: TIMESTAMP, sql: "..."}              computed from the other columns
    """
    if raw is None:
        return (), (), ()
    if not isinstance(raw, dict) or not raw:
        raise ConfigError(f"{where} must map column names to types")
    columns, sources, computed = [], [], []
    seen: set[str] = set()
    for name, spec in raw.items():
        at = f"{where}.{name}"
        if isinstance(spec, dict):
            unknown = set(spec) - {"type", "source", "sql"}
            if unknown:
                raise ConfigError(f"{at}: unknown key(s) {', '.join(sorted(unknown))}")
            if "source" in spec and "sql" in spec:
                raise ConfigError(f"{at}: a column has a source or a sql, not both")
            dtype = str(_require(spec, "type", at)).upper()
            if "source" in spec:
                names = spec["source"] if isinstance(spec["source"], list) else [spec["source"]]
                if not names or not all(isinstance(n, str) and n for n in names):
                    raise ConfigError(f"{at}.source must be a column name or a list of them")
                sources.append((str(name), tuple(names)))
            if "sql" in spec:
                if not isinstance(spec["sql"], str) or not spec["sql"].strip():
                    raise ConfigError(f"{at}.sql must be an SQL expression")
                computed.append((str(name), spec["sql"]))
        else:
            dtype = str(spec).upper()
        if dtype not in COLUMN_TYPES:
            raise ConfigError(f"{at}: unknown type '{dtype}' (use {', '.join(COLUMN_TYPES)})")
        if name.lower() in seen:
            raise ConfigError(f"{where}: column '{name}' is listed twice (names ignore case)")
        seen.add(name.lower())
        columns.append((str(name), dtype))
    accepted: dict[str, str] = {}
    for name, _ in columns:
        if name in dict(computed):
            continue
        for source in dict(sources).get(name, (name,)):
            if source.lower() in accepted:
                raise ConfigError(
                    f"{where}: source column '{source}' is claimed by {accepted[source.lower()]}"
                    f" and {name}"
                )
            accepted[source.lower()] = name
    return tuple(columns), tuple(sources), tuple(computed)


def _parse_silver(raw: Any, where: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must be a mapping with derived and/or joins")
    derived = raw.get("derived") or {}
    if not isinstance(derived, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in derived.items()
    ):
        raise ConfigError(f"{where}.derived must map column names to SQL expressions")
    joins = []
    for i, join in enumerate(raw.get("joins") or []):
        at = f"{where}.joins[{i}]"
        if not isinstance(join, dict):
            raise ConfigError(f"{at} must be a mapping")
        for key in ("lookup", "key", "column", "columns"):
            _require(join, key, at)
        columns = join["columns"]
        if not isinstance(columns, dict) or not columns:
            raise ConfigError(f"{at}.columns must map lookup columns to Silver column names")
        joins.append(
            SilverJoin(join["lookup"], join["key"], join["column"], tuple(columns.items()))
        )
    return {"silver_derived": tuple(derived.items()), "silver_joins": tuple(joins)}


def _parse_dataset(name: str, raw: dict[str, Any]) -> Dataset:
    where = f"datasets.{name}"
    lookups = []
    for lookup_name, lookup in (raw.get("lookups") or {}).items():
        at = f"{where}.lookups.{lookup_name}"
        if not isinstance(lookup, dict):
            raise ConfigError(f"{at} must be a mapping")
        files = lookup.get("files")
        if files is not None and not (
            isinstance(files, list) and files and all(isinstance(f, str) for f in files)
        ):
            raise ConfigError(f"{at}.files must be a list of file names")
        if files is None and "file" not in lookup:
            raise ConfigError(f"missing required key 'file' in {at}")
        rename = lookup.get("rename") or {}
        if not isinstance(rename, dict):
            raise ConfigError(f"{at}.rename must map column names to new names")
        lookups.append(
            Lookup(
                name=lookup_name,
                url=lookup.get("url"),
                file=lookup.get("file") or files[0],
                files=tuple(files or ()),
                rename=tuple((str(a), str(b)) for a, b in rename.items()),
            )
        )
    lookups = tuple(lookups)
    archives = []
    for archive_name, raw_archive in (raw.get("archives") or {}).items():
        at = f"{where}.archives.{archive_name}"
        if not isinstance(raw_archive, dict):
            raise ConfigError(f"{at} must be a mapping")
        kind = str(_require(raw_archive, "kind", at))
        if kind not in ARCHIVE_KINDS:
            raise ConfigError(f"{at}: kind must be one of {', '.join(ARCHIVE_KINDS)}")
        archives.append(
            Archive(
                name=archive_name,
                url=str(_require(raw_archive, "url", at)),
                file=str(_require(raw_archive, "file", at)),
                kind=kind,
                members=raw_archive.get("members"),
            )
        )
    source = None
    if raw.get("source") is not None:
        at = f"{where}.source"
        raw_source = raw["source"]
        if not isinstance(raw_source, dict):
            raise ConfigError(f"{at} must be a mapping")
        kind = str(_require(raw_source, "kind", at))
        if kind not in SOURCE_KINDS:
            raise ConfigError(f"{at}: kind must be one of {', '.join(SOURCE_KINDS)}")
        nulls = raw_source.get("null_values") or []
        if not isinstance(nulls, list):
            raise ConfigError(f"{at}.null_values must be a list")
        source = Source(
            kind=kind,
            files=str(_require(raw_source, "files", at)),
            period_sql=str(_require(raw_source, "period_sql", at)),
            null_values=tuple(str(v) for v in nulls),
            url=raw_source.get("url"),
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
        archives=tuple(archives),
        source=source,
        **dict(
            zip(
                ("columns", "column_sources", "computed"),
                _parse_columns(raw.get("columns"), f"{where}.columns"),
                strict=True,
            )
        ),
        **_parse_silver(raw.get("silver"), f"{where}.silver"),
    )
    standard = {c for c, _ in dataset.columns}
    lookup_names = {lookup.name for lookup in dataset.lookups}
    added = [c for c, _ in dataset.silver_derived]
    for join in dataset.silver_joins:
        if join.lookup not in lookup_names:
            raise ConfigError(f"{where}.silver: unknown lookup '{join.lookup}'")
        if standard and join.column not in standard:
            raise ConfigError(
                f"{where}.silver: join column '{join.column}' is not a standard column"
            )
        added += [target for _, target in join.columns]
    clash = sorted({c for c in added if added.count(c) > 1} | (set(added) & standard))
    if clash:
        raise ConfigError(f"{where}.silver: column(s) defined twice: {', '.join(clash)}")
    if dataset.enabled:
        # a dataset with a source block is prepared locally: it has no per-period URL
        required = (
            ("file_template", "timestamp_column")
            if dataset.source
            else ("url_template", "file_template", "timestamp_column")
        )
        for key in required:
            if not getattr(dataset, key):
                raise ConfigError(f"missing required key '{key}' in {where} (dataset is enabled)")
        if not dataset.periods:
            raise ConfigError(f"missing required key 'periods' in {where} (dataset is enabled)")
    if dataset.timestamp_column and standard and dataset.timestamp_column not in standard:
        raise ConfigError(
            f"{where}: timestamp_column '{dataset.timestamp_column}' is not a standard column"
        )
    return dataset


def load_datasets_config(path: Path | None = None) -> DatasetsConfig:
    path = Path(path) if path else config_dir() / "datasets.yml"
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
    path = Path(path) if path else config_dir() / "quality_rules.yml"
    data = load_yaml(path)
    defaults = _require(data, "defaults", path.name)
    for section in ("baseline", "volume", "drift", "incidents"):
        _require(defaults, section, f"{path.name}: defaults")
    drift = defaults["drift"]
    if not 0 < drift["psi_medium"] < drift["psi_high"]:
        raise ConfigError(f"{path.name}: drift thresholds need 0 < psi_medium < psi_high")
    for dataset, columns in (drift.get("bins") or {}).items():
        if not isinstance(columns, dict):
            raise ConfigError(
                f"{path.name}: defaults.drift.bins.{dataset} must map columns to bin edges"
            )
        for column, edges in columns.items():
            where = f"{path.name}: defaults.drift.bins.{dataset}.{column}"
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
    row_rules = (defaults.get("incidents") or {}).get("row_rules") or {}
    where = f"{path.name}: defaults.incidents.row_rules"
    for key in ("relative_increase", "min_increase"):
        value = row_rules.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int | float) or value < 0
        ):
            raise ConfigError(f"{where}.{key} must be a number >= 0")
    for action, value in (row_rules.get("cold_start_max_rate") or {}).items():
        if action not in ("quarantine", "warn"):
            raise ConfigError(f"{where}.cold_start_max_rate: unknown action '{action}'")
        if isinstance(value, bool) or not isinstance(value, int | float) or not 0 < value <= 1:
            raise ConfigError(f"{where}.cold_start_max_rate.{action} must be between 0 and 1")
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
    path = Path(path) if path else config_dir() / "severity_weights.yml"
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
    scores = data.get("check_scores")
    if scores is not None:
        expected = {"pass", *SEVERITIES}
        if not isinstance(scores, dict) or set(scores) != expected:
            raise ConfigError(
                f"{path.name}: check_scores needs exactly {', '.join(sorted(expected))}"
            )
        for result, value in scores.items():
            if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= 1:
                raise ConfigError(f"{path.name}: check_scores.{result} must be between 0 and 1")
    health = data.get("health")
    if health is not None:
        limits = [
            health.get(k) if isinstance(health, dict) else None
            for k in ("critical_below", "warning_below")
        ]
        if any(isinstance(v, bool) or not isinstance(v, int | float) for v in limits) or not (
            0 < limits[0] <= limits[1] <= 100
        ):
            raise ConfigError(
                f"{path.name}: health needs 0 < critical_below <= warning_below <= 100"
            )
    return data
