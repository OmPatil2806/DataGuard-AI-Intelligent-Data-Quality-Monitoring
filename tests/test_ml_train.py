"""Training the anomaly model (Phase 7.2) on clean test months."""

import copy
import json

import duckdb
import numpy as np
import pytest
import yaml
from evaluation_fixture import make_sources

from common import db
from common.config import REPO_ROOT
from common.runs import get_run
from gold import rebuild
from ingestion import ingest_data
from ml import train as mt
from ml.features import feature_frame, load_ml_config

RAW = yaml.safe_load((REPO_ROOT / "configs" / "ml.yml").read_text())


def config_at(tmp_path, **changes):
    data = copy.deepcopy(RAW) | {"models_dir": str(tmp_path / "models")}
    for key, value in changes.items():
        data[key] = data[key] | value if isinstance(value, dict) else value
    path = tmp_path / f"ml-{len(list(tmp_path.glob('ml-*.yml')))}.yml"
    path.write_text(yaml.safe_dump(data))
    return path, load_ml_config(path)


@pytest.fixture(scope="module")
def database(tmp_path_factory):
    """Clean months 2023-01 to 2023-04 (the fixture's trips moved into each month) in Silver."""
    root = tmp_path_factory.mktemp("ml_train")
    source = make_sources(root / "source")
    database = root / "dataguard.duckdb"
    common = ["--database", str(database)]
    argv = ["--months", "2023-01:2023-04", "--source", str(source), "--bronze", str(root / "b")]
    assert ingest_data.main([*argv, *common]) == 0
    silver = ["--silver", str(root / "s"), "--quarantine", str(root / "q")]
    assert rebuild.main(["--lookups", str(source), *silver, *common]) == 0
    return database


def train(database, config):
    with db.connect(database) as con:
        db.init_schema(con)
        return mt.train(con, config)


def test_training_follows_the_time_split(database, tmp_path):
    _, config = config_at(tmp_path)
    trained = train(database, config)
    m = trained.metadata
    assert m["train_months"] == ["2023-01", "2023-02", "2023-03"]
    assert m["calibrate_month"] == "2023-04"
    with db.connect(database, read_only=True) as con:
        silver_rows = dict(
            con.execute(
                "SELECT period, silver_rows FROM silver_log WHERE status = 'processed'"
            ).fetchall()
        )
        versions = dict(con.execute("SELECT period, sha256 FROM gold_months").fetchall())
    assert m["train_rows"] == sum(silver_rows[p] for p in m["train_months"])  # months are small
    assert m["calibrate_rows"] == silver_rows["2023-04"]
    assert m["silver_versions"] == {
        p: versions[p] for p in ["2023-01", "2023-02", "2023-03", "2023-04"]
    }
    assert m["model"]["max_samples"] == 256 and m["model"]["n_estimators"] == 200


def test_threshold_flags_the_target_share_of_the_calibration_month(database, tmp_path):
    _, config = config_at(tmp_path)
    trained = train(database, config)
    m = trained.metadata
    with db.connect(database, read_only=True) as con:
        from silver.process_silver import current_silver

        april = feature_frame(con, config, current_silver(con, config.dataset)["2023-04"][0])
    scores = trained.scores(april)
    assert m["threshold"] == pytest.approx(np.quantile(scores, 0.99))
    assert m["calibrate_flag_rate"] == pytest.approx((scores > m["threshold"]).mean())
    assert 0.005 <= m["calibrate_flag_rate"] <= 0.02  # about the 1% target


def test_missing_values_are_filled_with_training_medians(database, tmp_path):
    _, config = config_at(tmp_path)
    trained = train(database, config)
    frame = feature_frame_for(database, config, "2023-01")
    assert frame["fare_per_mile"].isna().any()  # zero-distance trips
    matrix = trained.matrix(frame)
    assert not np.isnan(matrix).any()
    column = trained.features.index("fare_per_mile")
    filled = matrix[frame["fare_per_mile"].isna().to_numpy(), column]
    assert (filled == trained.metadata["medians"]["fare_per_mile"]).all()


def feature_frame_for(database, config, period):
    from silver.process_silver import current_silver

    with db.connect(database, read_only=True) as con:
        return feature_frame(con, config, current_silver(con, config.dataset)[period][0])


def test_same_data_same_model(database, tmp_path):
    _, config = config_at(tmp_path)
    a, b = train(database, config), train(database, config)
    assert a.metadata["model_id"] == b.metadata["model_id"]
    frame = feature_frame_for(database, config, "2023-02")
    assert np.array_equal(a.scores(frame), b.scores(frame))
    _, other = config_at(tmp_path, training={"seed": 7})
    assert train(database, other).metadata["model_id"] != a.metadata["model_id"]


def test_rows_per_month_samples_reproducibly(database, tmp_path):
    _, config = config_at(tmp_path, training={"rows_per_month": 50})
    a, b = train(database, config), train(database, config)
    assert a.metadata["train_rows"] == 150 and a.metadata["calibrate_rows"] == 50
    assert a.metadata["model"]["max_samples"] == 150  # never more than the training rows
    assert a.metadata["threshold"] == b.metadata["threshold"]  # the same sample both times
    with db.connect(database, read_only=True) as con:
        from silver.process_silver import current_silver

        path = current_silver(con, config.dataset)["2023-01"][0]
        first = mt.sample_features(con, config, path, 50, 1)
        again = mt.sample_features(con, config, path, 50, 1)
        other = mt.sample_features(con, config, path, 50, 2)
    assert first.equals(again) and not first.equals(other)
    assert first["bronze_row_id"].is_monotonic_increasing and len(first) == 50


def test_save_and_load(database, tmp_path):
    path, config = config_at(tmp_path)
    assert mt.main(["--config", str(path), "--database", str(database)]) == 0
    loaded = mt.load_model(config)
    model_id = loaded.metadata["model_id"]
    folder = config.models_dir / "yellow_taxi" / model_id
    assert (folder / mt.MODEL_FILE).is_file()
    assert (
        json.loads((folder / mt.METADATA_FILE).read_text())["threshold"]
        == loaded.metadata["threshold"]
    )
    assert mt.load_model(config, model_id).metadata == loaded.metadata
    with duckdb.connect(str(database), read_only=True) as con:
        row = con.execute(
            "SELECT run_id, train_months, threshold, path FROM ml_models WHERE model_id = ?",
            [model_id],
        ).fetchone()
        run = get_run(con, row[0])
    assert row[1] == "2023-01,2023-02,2023-03" and row[3] == folder.as_posix()
    assert (run["status"], run["command"]) == ("SUCCESS", "ml train")
    frame = feature_frame_for(database, config, "2023-03")
    assert np.array_equal(loaded.scores(frame), train(database, config).scores(frame))


def test_errors(database, tmp_path, capsys):
    path, config = config_at(tmp_path, months={"calibrate": "2023-05"})
    assert mt.main(["--config", str(path), "--database", str(database)]) == 1
    assert "no Silver for 2023-05" in capsys.readouterr().out
    with pytest.raises(FileNotFoundError, match="no trained model"):
        mt.load_model(config)
    with pytest.raises(FileNotFoundError, match="not found"):
        mt.load_model(config, "000000000000")
    bad = tmp_path / "bad.yml"
    bad.write_text(yaml.safe_dump(copy.deepcopy(RAW) | {"features": {}}))
    assert mt.main(["--config", str(bad), "--database", str(database)]) == 2
