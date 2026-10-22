"""configs/ml.yml and the anomaly model's features (Phase 7.1)."""

import copy
from datetime import datetime

import duckdb
import pandas as pd
import pytest
import yaml
from pipeline_fixture import run_pipeline

from common.config import REPO_ROOT, ConfigError
from ml import features as mf

RAW = yaml.safe_load((REPO_ROOT / "configs" / "ml.yml").read_text())
FEATURES = [
    "trip_duration_min", "trip_distance", "speed_mph", "fare_per_mile", "fare_per_minute",
    "total_gap", "pickup_hour", "pickup_weekday", "passenger_count",
]  # fmt: skip


def write_config(tmp_path, change):
    data = copy.deepcopy(RAW)
    change(data)
    path = tmp_path / "ml.yml"
    path.write_text(yaml.safe_dump(data))
    return path


def test_config_loads():
    config = mf.load_ml_config()
    assert list(config.features) == FEATURES
    assert config.train_months == ("2023-01", "2023-02", "2023-03")
    assert config.calibrate_month == "2023-04"
    assert (config.rows_per_month, config.seed, config.target_flag_rate) == (100000, 42, 0.01)
    assert config.model["n_estimators"] == 200 and config.top_features == 3
    assert "\n" not in config.features["total_gap"]  # folded YAML becomes one line of SQL


def test_models_dir(monkeypatch, tmp_path):
    # the test suite points DATAGUARD_MODELS_DIR at a temporary folder (tests/conftest.py)
    assert mf.load_ml_config().models_dir != REPO_ROOT / "models"
    monkeypatch.delenv(mf.MODELS_DIR_ENV, raising=False)
    assert mf.load_ml_config().models_dir == REPO_ROOT / "models"
    monkeypatch.setenv(mf.MODELS_DIR_ENV, str(tmp_path / "elsewhere"))
    assert mf.load_ml_config().models_dir == tmp_path / "elsewhere"
    absolute = write_config(tmp_path, lambda d: d.update(models_dir=str(tmp_path / "abs")))
    assert mf.load_ml_config(absolute).models_dir == tmp_path / "abs"  # absolute wins


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.update(features={}), "non-empty mapping"),
        (lambda d: d["features"].update({"Speed": "1"}), "invalid feature name"),
        (lambda d: d["features"].update({"bronze_row_id": "1"}), "invalid feature name"),
        (lambda d: d["features"].update(speed_mph=""), "needs a SQL expression"),
        (lambda d: d["months"].update(calibrate="2023-02"), "after the train months"),
        (lambda d: d["months"].update(calibrate="April"), "must be YYYY-MM"),
        (lambda d: d["training"].update(rows_per_month=0), "rows_per_month"),
        (lambda d: d["training"].update(seed=-1), "seed"),
        (lambda d: d["model"].update(n_estimators=0), "n_estimators"),
        (lambda d: d["threshold"].update(target_flag_rate=1.5), "target_flag_rate"),
        (lambda d: d["threshold"].update(target_flag_rate=0), "target_flag_rate"),
        (lambda d: d["incidents"].update(min_increase=2), "min_increase"),
        (lambda d: d["incidents"].update(relative_increase=-1), "relative_increase"),
        (lambda d: d["incidents"].update(severity="urgent"), "severity must be one of"),
        (lambda d: d["explain"].update(top_features=20), "more than the number of features"),
        (lambda d: d["retrain"].update(max_age_months=0), "max_age_months"),
        (lambda d: d.update(dataset="green_taxi"), "green_taxi"),
    ],
)
def test_config_is_validated(tmp_path, change, message):
    with pytest.raises(ConfigError, match=message):
        mf.load_ml_config(write_config(tmp_path, change))


def silver_rows(tmp_path):
    """Four hand-made Silver rows with known answers."""
    rows = pd.DataFrame(
        {
            "bronze_row_id": [3, 1, 2, 4],  # stored out of order on purpose
            "tpep_pickup_datetime": [datetime(2023, 1, 2, 8, 0)] * 4,  # a Monday
            "trip_duration_min": [30.0, 0.0, 15.0, 60.0],
            "pickup_hour": [8, 8, 8, 8],
            "trip_distance": [10.0, 2.0, 0.0, 3.0],
            "fare_amount": [30.0, 10.0, 7.0, 90.0],
            "extra": [1.0, 0.0, 0.0, 0.0],
            "mta_tax": [0.5, 0.5, 0.5, 0.5],
            "tip_amount": [5.0, 0.0, 0.0, 0.0],
            "tolls_amount": [0.0, 0.0, 0.0, 0.0],
            "improvement_surcharge": [1.0, 1.0, 1.0, 1.0],
            "congestion_surcharge": [2.5, 2.5, None, 2.5],
            "airport_fee": [0.0, 0.0, 0.0, 0.0],
            "total_amount": [40.0, 14.0, 8.5, 99.0],
            "passenger_count": [1, 2, None, 1],
        }
    )
    path = tmp_path / "silver.parquet"
    rows.to_parquet(path, index=False)
    return path


def test_feature_values(tmp_path):
    config = mf.load_ml_config()
    with duckdb.connect() as con:
        frame = mf.feature_frame(con, config, silver_rows(tmp_path)).set_index("bronze_row_id")
    assert list(frame.index) == [1, 2, 3, 4]  # one row per Silver row, by bronze_row_id
    assert list(frame.columns) == FEATURES and (frame.dtypes == "float64").all()
    trip = frame.loc[3]  # 10 miles in 30 minutes for $30
    assert (trip.speed_mph, trip.fare_per_mile, trip.fare_per_minute) == (20.0, 3.0, 1.0)
    assert trip.total_gap == 0.0  # 30 + 1 + 0.5 + 5 + 1 + 2.5 = 40
    assert (trip.pickup_hour, trip.pickup_weekday) == (8.0, 1.0)
    zero_time = frame.loc[1]  # 0 minutes: no speed, no fare per minute
    assert pd.isna(zero_time.speed_mph) and pd.isna(zero_time.fare_per_minute)
    assert zero_time.fare_per_mile == 5.0
    zero_distance = frame.loc[2]  # 0 miles: no fare per mile; NULL surcharge counts as 0
    assert pd.isna(zero_distance.fare_per_mile) and zero_distance.speed_mph == 0.0
    assert zero_distance.total_gap == 0.0 and pd.isna(zero_distance.passenger_count)
    odd = frame.loc[4]  # $90 for 3 miles: in range, but a fare per mile of 30
    assert odd.fare_per_mile == 30.0 and odd.total_gap == pytest.approx(5.0)


def test_a_broken_feature_is_a_config_error(tmp_path):
    path = write_config(tmp_path, lambda d: d["features"].update(speed_mph="no_such_column / 2"))
    config = mf.load_ml_config(path)
    with duckdb.connect() as con, pytest.raises(ConfigError, match="cannot be computed"):
        mf.feature_frame(con, config, silver_rows(tmp_path))


def test_summary(tmp_path):
    config = mf.load_ml_config()
    with duckdb.connect() as con:
        stats = mf.summary(mf.feature_frame(con, config, silver_rows(tmp_path)))
    assert list(stats.index) == FEATURES
    assert stats.loc["fare_per_mile", "null_pct"] == 25.0
    assert stats.loc["fare_per_mile", "max"] == 30.0


@pytest.fixture(scope="module")
def database(tmp_path_factory):
    return run_pipeline(tmp_path_factory.mktemp("ml_features"))


def test_main(database, capsys):
    assert mf.main(["--period", "2023-01", "--database", str(database)]) == 0
    out = capsys.readouterr().out
    assert "2023-01: 189 Silver rows, 9 features" in out and "fare_per_mile" in out
    assert mf.main(["--period", "2024-01", "--database", str(database)]) == 1
    assert "no Silver for 2024-01" in capsys.readouterr().out
