"""Train the anomaly model: an Isolation Forest on clean Silver months, threshold on a later one.

    python src/ml/train.py

Time-based split (configs/ml.yml, months):

    train      a random sample of rows from each train month (training.rows_per_month)
    calibrate  a sample of the month after: the threshold is the anomaly score above which
               threshold.target_flag_rate of its rows lie, so a clean month flags about that
               share and a rise means something

Missing feature values (a division by zero) are filled with the training medians. The model
learns from the current Silver version of each month; its id is a hash of the settings and of
those versions, so training again on the same data gives the same model id and the same model.

Saved to models/<dataset>/<model_id>/: model.joblib and metadata.json (features, months and
Silver versions, medians and quartiles for the explanations, threshold, library version). The
latest model is named in models/<dataset>/current.json and registered in ml_models.
Exit codes: 0 trained, 1 a month has no Silver, 2 invalid config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import IsolationForest

if __package__ in (None, ""):
    # Allow running this file directly: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import db  # noqa: E402
from common.config import ConfigError  # noqa: E402
from common.runs import track_run, utc_now  # noqa: E402
from ingestion.bronze_loader import current_versions, read_bronze_sql  # noqa: E402
from ml.features import KEY, MLConfig, feature_sql, load_ml_config  # noqa: E402
from silver.process_silver import current_silver  # noqa: E402

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
CURRENT = "current.json"


class TrainingError(RuntimeError):
    """Raised when the model cannot be trained (a month without Silver, no rows)."""


@dataclass
class TrainedModel:
    model: IsolationForest
    metadata: dict[str, Any]

    @property
    def features(self) -> list[str]:
        return list(self.metadata["features"])

    def matrix(self, frame: pd.DataFrame) -> np.ndarray:
        """Feature values with missing ones filled by the training medians."""
        medians = self.metadata["medians"]
        return frame[self.features].fillna(medians).to_numpy(dtype=float)

    def scores(self, frame: pd.DataFrame) -> np.ndarray:
        """Anomaly score per row: higher is more unusual (the negated score_samples)."""
        return -self.model.score_samples(self.matrix(frame))


def sample_features(
    con: duckdb.DuckDBPyConnection, config: MLConfig, silver_file: Path, rows: int, seed: int
) -> pd.DataFrame:
    """Features of `rows` random Silver rows (all rows if the month has fewer), by row id."""
    source = read_bronze_sql(Path(silver_file))
    ids = np.sort(con.execute(f"SELECT {KEY} FROM {source}").fetchnumpy()[KEY])
    if len(ids) > rows:
        ids = np.sort(np.random.default_rng(seed).choice(ids, size=rows, replace=False))
    chosen = pd.DataFrame({KEY: ids})
    con.register("ml_chosen_rows", chosen)
    try:
        return con.execute(
            feature_sql(
                config,
                f"(SELECT * FROM {source} WHERE {KEY} IN (SELECT {KEY} FROM ml_chosen_rows))",
            )
        ).df()
    finally:
        con.unregister("ml_chosen_rows")


def _month_seed(config: MLConfig, period: str) -> int:
    return config.seed * 1_000_003 + int(period.replace("-", ""))


def model_id(config: MLConfig, versions: dict[str, str]) -> str:
    """Hash of what the model depends on: settings and the Silver versions it learns from."""
    parts = {
        "dataset": config.dataset,
        "features": config.features,
        "train": list(config.train_months),
        "calibrate": config.calibrate_month,
        "rows_per_month": config.rows_per_month,
        "seed": config.seed,
        "model": config.model,
        "target_flag_rate": config.target_flag_rate,
        "versions": versions,
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:12]


def train(con: duckdb.DuckDBPyConnection, config: MLConfig) -> TrainedModel:
    silver = current_silver(con, config.dataset)
    months = [*config.train_months, config.calibrate_month]
    missing = [m for m in months if m not in silver]
    if missing:
        raise TrainingError(f"no Silver for {', '.join(missing)} (run process_silver.py)")
    versions = {
        m: sha for m, (_, sha) in current_versions(con, config.dataset).items() if m in months
    }

    frames = [
        sample_features(con, config, silver[m][0], config.rows_per_month, _month_seed(config, m))
        for m in config.train_months
    ]
    train_frame = pd.concat(frames, ignore_index=True)
    if train_frame.empty:
        raise TrainingError("the train months have no Silver rows")
    features = list(config.features)
    medians = train_frame[features].median().fillna(0.0)
    q25, q75 = train_frame[features].quantile(0.25), train_frame[features].quantile(0.75)

    settings = {k: v for k, v in config.model.items()}
    settings["max_samples"] = min(int(settings.get("max_samples", 256)), len(train_frame))
    model = IsolationForest(**settings, random_state=config.seed)
    metadata: dict[str, Any] = {
        "features": config.features,
        "medians": {f: float(medians[f]) for f in features},
    }
    trained = TrainedModel(model, metadata)
    model.fit(trained.matrix(train_frame))

    calibration = sample_features(
        con,
        config,
        silver[config.calibrate_month][0],
        config.rows_per_month,
        _month_seed(config, config.calibrate_month),
    )
    calibration_scores = trained.scores(calibration)
    threshold = float(np.quantile(calibration_scores, 1 - config.target_flag_rate))
    train_scores = trained.scores(train_frame)

    metadata |= {
        "model_id": model_id(config, versions),
        "dataset": config.dataset,
        "train_months": list(config.train_months),
        "calibrate_month": config.calibrate_month,
        "silver_versions": versions,
        "train_rows": len(train_frame),
        "calibrate_rows": len(calibration),
        "rows_per_month": config.rows_per_month,
        "seed": config.seed,
        "model": settings,
        "quartiles": {f: [float(q25[f]), float(q75[f])] for f in features},
        "target_flag_rate": config.target_flag_rate,
        "threshold": threshold,
        "calibrate_flag_rate": float((calibration_scores > threshold).mean()),
        "train_flag_rate": float((train_scores > threshold).mean()),
        "sklearn_version": sklearn.__version__,
    }
    return trained


def save(
    con: duckdb.DuckDBPyConnection, config: MLConfig, trained: TrainedModel, run_id: str
) -> Path:
    meta = trained.metadata
    folder = config.models_dir / config.dataset / meta["model_id"]
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained.model, folder / MODEL_FILE)
    (folder / METADATA_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    pointer = config.models_dir / config.dataset / CURRENT
    pointer.write_text(json.dumps({"model_id": meta["model_id"]}) + "\n", encoding="utf-8")
    con.execute("DELETE FROM ml_models WHERE model_id = ?", [meta["model_id"]])
    con.execute(
        "INSERT INTO ml_models VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            meta["model_id"], config.dataset, run_id, utc_now(),
            ",".join(meta["train_months"]), meta["calibrate_month"], meta["train_rows"],
            meta["threshold"], meta["target_flag_rate"], folder.as_posix(), json.dumps(meta),
        ],
    )  # fmt: skip
    return folder


def load_model(config: MLConfig, model: str | None = None) -> TrainedModel:
    """The current model (or the one with this id) of the config's dataset."""
    base = config.models_dir / config.dataset
    if model is None:
        pointer = base / CURRENT
        if not pointer.is_file():
            raise FileNotFoundError(f"no trained model in {base} (run python src/ml/train.py)")
        model = json.loads(pointer.read_text(encoding="utf-8"))["model_id"]
    folder = base / model
    if not (folder / MODEL_FILE).is_file():
        raise FileNotFoundError(f"model {model} not found in {base}")
    metadata = json.loads((folder / METADATA_FILE).read_text(encoding="utf-8"))
    return TrainedModel(joblib.load(folder / MODEL_FILE), metadata)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the anomaly model on Silver.")
    parser.add_argument("--config", type=Path, help="ML config (default: configs/ml.yml)")
    parser.add_argument("--database", type=Path, help="DuckDB file (default: paths.duckdb)")
    args = parser.parse_args(argv)
    try:
        config = load_ml_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}")
        return 2
    start = time.perf_counter()
    with db.connect(args.database) as con:
        db.init_schema(con)
        try:
            with track_run(con, config.dataset, command="ml train") as run:
                trained = train(con, config)
                folder = save(con, config, trained, run.run_id)
        except TrainingError as exc:
            print(f"error: {exc}")
            return 1
    m = trained.metadata
    print(
        f"model {m['model_id']} trained in {time.perf_counter() - start:.1f}s -> {folder}\n"
        f"  train      {', '.join(m['train_months'])}: {m['train_rows']:,} rows "
        f"(flag rate {m['train_flag_rate']:.2%})\n"
        f"  calibrate  {m['calibrate_month']}: {m['calibrate_rows']:,} rows, threshold "
        f"{m['threshold']:.4f} flags {m['calibrate_flag_rate']:.2%} "
        f"(target {m['target_flag_rate']:.2%})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
