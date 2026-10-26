"""Features of the anomaly model, computed per Silver row in DuckDB.

    python src/ml/features.py --period 2023-01       # summary of every feature in a month

The features are SQL expressions in configs/ml.yml, over the standard columns and Silver's
derived columns (trip_duration_min, pickup_hour). Each Silver row gives one feature row keyed
by bronze_row_id, so a flag can always be traced back to its source row. A feature that cannot
be computed (a division by zero) is NULL here; training fills NULLs with the training medians.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

if __package__ in (None, ""):
    # Allow running this file directly: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import db  # noqa: E402
from common.config import (  # noqa: E402
    PERIOD_PATTERN,
    REPO_ROOT,
    SEVERITIES,
    ConfigError,
    _parse_periods,
    _require,
    config_dir,
    load_datasets_config,
    load_yaml,
)
from ingestion.bronze_loader import read_bronze_sql  # noqa: E402

NAME = re.compile(r"^[a-z][a-z0-9_]*$")
KEY = "bronze_row_id"
MODELS_DIR_ENV = "DATAGUARD_MODELS_DIR"


@dataclass(frozen=True)
class MLConfig:
    dataset: str
    features: dict[str, str]  # name -> SQL expression
    train_months: tuple[str, ...]
    calibrate_month: str
    rows_per_month: int
    seed: int
    model: dict[str, Any]
    target_flag_rate: float
    top_features: int
    incidents: dict[str, Any]
    max_age_months: int
    models_dir: Path


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{where} must be a whole number above 0")
    return value


def _rate(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0 < value < 1:
        raise ConfigError(f"{where} must be a number above 0 and below 1")
    return float(value)


def load_ml_config(path: Path | None = None) -> MLConfig:
    path = Path(path) if path else config_dir() / "ml.yml"
    data = load_yaml(path)
    where = path.name

    dataset = load_datasets_config().dataset(str(_require(data, "dataset", where)))
    features = _require(data, "features", where)
    if not isinstance(features, dict) or not features:
        raise ConfigError(f"{where}: features must be a non-empty mapping of name: SQL")
    for name, expression in features.items():
        if not NAME.match(str(name)) or name == KEY:
            raise ConfigError(f"{where}: invalid feature name '{name}'")
        if not isinstance(expression, str) or not expression.strip():
            raise ConfigError(f"{where}: feature '{name}' needs a SQL expression")

    months = _require(data, "months", where)
    train = _parse_periods(_require(months, "train", f"{where} months"), f"{where} months.train")
    calibrate = str(_require(months, "calibrate", f"{where} months"))
    if not PERIOD_PATTERN.match(calibrate):
        raise ConfigError(f"{where}: months.calibrate '{calibrate}' must be YYYY-MM")
    if not train or calibrate <= train[-1]:
        raise ConfigError(f"{where}: months.calibrate must come after the train months")

    training = _require(data, "training", where)
    seed = _require(training, "seed", f"{where} training")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError(f"{where}: training.seed must be a whole number of at least 0")
    model = _require(data, "model", where)
    if not isinstance(model, dict):
        raise ConfigError(f"{where}: model must be a mapping of IsolationForest settings")
    _positive_int(_require(model, "n_estimators", f"{where} model"), f"{where} model.n_estimators")

    incidents = _require(data, "incidents", where)
    for key in ("min_increase", "cold_start_max_rate"):
        _rate(_require(incidents, key, f"{where} incidents"), f"{where} incidents.{key}")
    relative = _require(incidents, "relative_increase", f"{where} incidents")
    if isinstance(relative, bool) or not isinstance(relative, int | float) or relative < 0:
        raise ConfigError(f"{where}: incidents.relative_increase must be at least 0")
    if _require(incidents, "severity", f"{where} incidents") not in SEVERITIES:
        raise ConfigError(f"{where}: incidents.severity must be one of {', '.join(SEVERITIES)}")

    top = _positive_int(
        _require(_require(data, "explain", where), "top_features", f"{where} explain"),
        f"{where} explain.top_features",
    )
    if top > len(features):
        raise ConfigError(f"{where}: explain.top_features is more than the number of features")
    models_dir = Path(str(_require(data, "models_dir", where)))
    return MLConfig(
        dataset=dataset.name,
        features={str(k): " ".join(str(v).split()) for k, v in features.items()},
        train_months=train,
        calibrate_month=calibrate,
        rows_per_month=_positive_int(
            _require(training, "rows_per_month", f"{where} training"),
            f"{where} training.rows_per_month",
        ),
        seed=seed,
        model=dict(model),
        target_flag_rate=_rate(
            _require(_require(data, "threshold", where), "target_flag_rate", f"{where} threshold"),
            f"{where} threshold.target_flag_rate",
        ),
        top_features=top,
        incidents=dict(incidents),
        max_age_months=_positive_int(
            _require(_require(data, "retrain", where), "max_age_months", f"{where} retrain"),
            f"{where} retrain.max_age_months",
        ),
        models_dir=_models_dir(models_dir),
    )


def _models_dir(configured: Path) -> Path:
    """An absolute models_dir is used as is; a relative one is under the repository, or under
    DATAGUARD_MODELS_DIR when set (the test suite points it at a temporary folder, so tests
    never pick up a model trained on the real data)."""
    if configured.is_absolute():
        return configured
    override = os.environ.get(MODELS_DIR_ENV)
    return Path(override) if override else REPO_ROOT / configured


def feature_sql(config: MLConfig, source_sql: str) -> str:
    """SQL with bronze_row_id and one DOUBLE column per feature, in config order."""
    columns = ", ".join(
        f'CAST(({expression}) AS DOUBLE) AS "{name}"'
        for name, expression in config.features.items()
    )
    return f"SELECT {KEY}, {columns} FROM {source_sql} ORDER BY {KEY}"


def feature_frame(
    con: duckdb.DuckDBPyConnection, config: MLConfig, silver_file: Path
) -> pd.DataFrame:
    """The features of every row of one Silver file."""
    try:
        return con.execute(feature_sql(config, read_bronze_sql(Path(silver_file)))).df()
    except duckdb.Error as exc:
        raise ConfigError(f"a feature in ml.yml cannot be computed on Silver: {exc}") from exc


def summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Per feature: share of NULLs and the quantiles that show its range."""
    features = frame.drop(columns=[KEY])
    stats = features.quantile([0.01, 0.5, 0.99]).T
    stats.columns = ["p1", "median", "p99"]
    stats.insert(0, "null_pct", (100 * features.isna().mean()).round(2))
    stats["min"], stats["max"] = features.min(), features.max()
    return stats.round(3)


def main(argv: list[str] | None = None) -> int:
    from silver.process_silver import current_silver

    parser = argparse.ArgumentParser(description="Show the ML features of a Silver month.")
    parser.add_argument("--period", required=True, help="month, YYYY-MM")
    parser.add_argument("--config", type=Path, help="ML config (default: configs/ml.yml)")
    parser.add_argument("--database", type=Path, help="DuckDB file (default: paths.duckdb)")
    args = parser.parse_args(argv)
    try:
        config = load_ml_config(args.config)
        with db.connect(args.database, read_only=True) as con:
            files = current_silver(con, config.dataset).get(args.period)
            if files is None:
                print(f"no Silver for {args.period} (run process_silver.py)")
                return 1
            frame = feature_frame(con, config, files[0])
    except (ConfigError, FileNotFoundError) as exc:
        print(f"error: {exc}")
        return 2
    print(f"{args.period}: {len(frame):,} Silver rows, {len(config.features)} features")
    with pd.option_context("display.width", 200):
        print(summary(frame).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
