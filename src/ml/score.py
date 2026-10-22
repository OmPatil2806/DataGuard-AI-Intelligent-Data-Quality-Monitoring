"""Score Silver months with the anomaly model and explain every flag.

    python src/ml/score.py                       # every month with Silver, current model
    python src/ml/score.py --months 2023-06      # some months
    python src/ml/score.py --model 1479b0fa9f3d  # a specific model

Every row of a month's current Silver version gets an anomaly score; rows above the model's
threshold are flagged. A flag is a warning: the row stays in Silver. For each flagged row the
features furthest from their training median, measured in the feature's typical spread, are
kept as the reason ("fare_per_mile 30 (typical 5.9)").

    ml_month_summary   per month, file version and model: rows scored, flagged, flag rate
    ml_anomalies       per flagged row: bronze_row_id, score, most deviating features, reasons

Scoring a month again with the same model replaces its rows. Warnings (stored with the month
and making the run WARNING): the model is older than retrain.max_age_months before the month,
or a month it learnt from has a newer Silver version than the one it was trained on.
With --if-trained (used by gold/rebuild.py) a missing model is not an error: scoring is
skipped. Exit codes: 0 scored or skipped, 1 no model or no Silver, 2 invalid config or
arguments.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    # Allow running this file directly: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import db  # noqa: E402
from common.config import ConfigError  # noqa: E402
from common.runs import track_run, utc_now  # noqa: E402
from data_download.download_data import parse_months  # noqa: E402
from ingestion.bronze_loader import current_versions  # noqa: E402
from ml.features import KEY, MLConfig, feature_frame, load_ml_config  # noqa: E402
from ml.train import TrainedModel, load_model  # noqa: E402
from quality.baselines import month_number  # noqa: E402
from silver.process_silver import current_silver  # noqa: E402


@dataclass
class MonthResult:
    period: str
    sha256: str
    rows_scored: int
    flagged: pd.DataFrame  # bronze_row_id, score, top_features, reasons
    mean_score: float | None
    warnings: list[str] = field(default_factory=list)

    @property
    def flag_rate(self) -> float:
        return len(self.flagged) / self.rows_scored if self.rows_scored else 0.0


def _value(v: float) -> str:
    return f"{v:.3g}" if abs(v) < 1e4 else f"{v:,.0f}"


def explain(trained: TrainedModel, rows: pd.DataFrame, top: int) -> tuple[list, list]:
    """Per row: the `top` features furthest from their training median, and the reason text."""
    names = trained.features
    medians = np.array([trained.metadata["medians"][f] for f in names])
    scales = np.array([trained.metadata["scales"][f] for f in names])
    values = rows[names].to_numpy(dtype=float)
    distance = np.abs(values - medians) / scales
    distance = np.where(np.isnan(distance), -1.0, distance)  # a missing value never leads
    order = np.argsort(-distance, axis=1, kind="stable")[:, :top]
    features, reasons = [], []
    for row, picks in enumerate(order):
        chosen = [i for i in picks if distance[row, i] >= 0]
        features.append([names[i] for i in chosen])
        reasons.append(
            "; ".join(
                f"{names[i]} {_value(values[row, i])} (typical {_value(medians[i])})"
                for i in chosen
            )
        )
    return features, reasons


def model_warnings(
    trained: TrainedModel, config: MLConfig, period: str, versions: dict[str, str]
) -> list[str]:
    meta = trained.metadata
    warnings = []
    age = month_number(period) - month_number(meta["train_months"][-1])
    if age > config.max_age_months:
        warnings.append(
            f"model learnt from months up to {meta['train_months'][-1]}, {age} months before "
            f"{period} (more than {config.max_age_months}): retrain"
        )
    changed = sorted(m for m, sha in meta["silver_versions"].items() if versions.get(m, sha) != sha)
    if changed:
        warnings.append(f"Silver of {', '.join(changed)} changed since training: retrain")
    return warnings


def score_month(
    con: duckdb.DuckDBPyConnection,
    config: MLConfig,
    trained: TrainedModel,
    period: str,
    silver_file: Path,
    sha256: str,
    versions: dict[str, str],
) -> MonthResult:
    frame = feature_frame(con, config, silver_file)
    warnings = model_warnings(trained, config, period, versions)
    if frame.empty:
        empty = pd.DataFrame(columns=[KEY, "score", "top_features", "reasons"])
        return MonthResult(period, sha256, 0, empty, None, warnings)
    scores = trained.scores(frame)
    mask = scores > trained.metadata["threshold"]
    flagged = frame.loc[mask].reset_index(drop=True)
    features, reasons = explain(trained, flagged, config.top_features)
    result = pd.DataFrame(
        {
            KEY: flagged[KEY].astype("int64"),
            "score": scores[mask],
            "top_features": features,
            "reasons": reasons,
        }
    )
    return MonthResult(period, sha256, len(frame), result, float(scores.mean()), warnings)


def store(
    con: duckdb.DuckDBPyConnection,
    config: MLConfig,
    trained: TrainedModel,
    result: MonthResult,
    run_id: str,
) -> None:
    key = [config.dataset, result.period, result.sha256, trained.metadata["model_id"]]
    where = "dataset = ? AND period = ? AND sha256 = ? AND model_id = ?"
    con.execute(f"DELETE FROM ml_anomalies WHERE {where}", key)
    con.execute(f"DELETE FROM ml_month_summary WHERE {where}", key)
    con.execute(
        "INSERT INTO ml_month_summary VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            *key, run_id, result.rows_scored, len(result.flagged), result.flag_rate,
            result.mean_score, "; ".join(result.warnings) or None, utc_now(),
        ],
    )  # fmt: skip
    if len(result.flagged):
        rows = result.flagged.assign(dataset=key[0], period=key[1], sha256=key[2], model_id=key[3])[
            ["dataset", "period", "sha256", "model_id", KEY, "score", "top_features", "reasons"]
        ]
        con.register("ml_flagged_rows", rows)
        try:
            con.execute("INSERT INTO ml_anomalies SELECT * FROM ml_flagged_rows")
        finally:
            con.unregister("ml_flagged_rows")


def score_all(
    con: duckdb.DuckDBPyConnection,
    config: MLConfig,
    trained: TrainedModel,
    months: list[str] | None,
    run_id: str,
) -> list[MonthResult]:
    silver = current_silver(con, config.dataset)
    versions = {m: sha for m, (_, sha) in current_versions(con, config.dataset).items()}
    results = []
    for period in months or sorted(silver):
        if period not in silver:
            continue
        result = score_month(
            con, config, trained, period, silver[period][0], versions[period], versions
        )
        store(con, config, trained, result, run_id)
        results.append(result)
    return results


def overview(results: list[MonthResult]) -> pd.DataFrame:
    records = []
    for r in results:
        leading = r.flagged["top_features"].map(lambda f: f[0] if len(f) else None)
        common = leading.value_counts()
        records.append(
            {
                "period": r.period,
                "rows": r.rows_scored,
                "flagged": len(r.flagged),
                "flag_rate": f"{r.flag_rate:.2%}",
                "most_common_reason": common.index[0] if len(common) else "",
                "warning": "yes" if r.warnings else "",
            }
        )
    return pd.DataFrame(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score Silver months with the anomaly model.")
    parser.add_argument("--months", help="e.g. 2023-06 or 2023-01:2023-03 (default: all)")
    parser.add_argument("--model", help="model id (default: the current model)")
    parser.add_argument("--config", type=Path, help="ML config (default: configs/ml.yml)")
    parser.add_argument("--database", type=Path, help="DuckDB file (default: paths.duckdb)")
    parser.add_argument(
        "--if-trained", action="store_true", help="skip (exit 0) when no model is trained yet"
    )
    args = parser.parse_args(argv)
    try:
        config = load_ml_config(args.config)
        months = parse_months(args.months, ()) if args.months else None
    except ConfigError as exc:
        print(f"error: {exc}")
        return 2
    try:
        trained = load_model(config, args.model)
    except FileNotFoundError as exc:
        if args.if_trained and args.model is None:
            print(f"scoring skipped: {exc}")
            return 0
        print(f"error: {exc}")
        return 1
    with db.connect(args.database) as con:
        db.init_schema(con)
        with track_run(con, config.dataset, command="ml score") as run:
            results = score_all(con, config, trained, months, run.run_id)
            for warning in sorted({w for r in results for w in r.warnings}):
                run.warn(warning)
    if not results:
        print("no Silver to score (run process_silver.py)")
        return 1
    print(f"model {trained.metadata['model_id']}, threshold {trained.metadata['threshold']:.4f}")
    print(overview(results).to_string(index=False))
    for warning in sorted({w for r in results for w in r.warnings}):
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
