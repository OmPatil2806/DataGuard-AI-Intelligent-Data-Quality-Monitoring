"""Scoring Silver with the anomaly model (Phase 7.3).

Clean test months 2023-01 to 2023-05; in May, five short trips get ten times their fare: every
value stays plausible on its own, but fare per mile and per minute do not fit the trip.
"""

import copy

import duckdb
import pandas as pd
import pytest
import yaml
from evaluation_fixture import make_sources

from common import db
from common.config import REPO_ROOT
from common.runs import get_run
from gold import rebuild
from ingestion import ingest_data
from ml import score as ms
from ml import train as mt
from ml.features import load_ml_config

RAW = yaml.safe_load((REPO_ROOT / "configs" / "ml.yml").read_text())
PLANTED = 5


def write_config(folder, **changes):
    data = copy.deepcopy(RAW) | {"models_dir": str(folder / "models")}
    for key, value in changes.items():
        data[key] = data[key] | value
    path = folder / f"ml-{len(list(folder.glob('ml-*.yml')))}.yml"
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("ml_score")
    source = make_sources(root / "source")
    may = source / "yellow_tripdata_2023-05.parquet"
    trips = duckdb.sql(f"SELECT * FROM read_parquet('{may.as_posix()}')").df()
    short = trips.index[(trips["trip_distance"].between(0.5, 3)) & (trips["fare_amount"] > 0)]
    planted = list(short[:PLANTED])
    trips.loc[planted, "fare_amount"] *= 10
    trips.to_parquet(may, index=False)
    database = root / "dataguard.duckdb"
    common = ["--database", str(database)]
    argv = ["--months", "2023-01:2023-05", "--source", str(source), "--bronze", str(root / "b")]
    assert ingest_data.main([*argv, *common]) == 0
    silver = ["--silver", str(root / "s"), "--quarantine", str(root / "q")]
    assert rebuild.main(["--lookups", str(source), *silver, *common]) == 0
    config_file = write_config(root)
    assert mt.main(["--config", str(config_file), "--database", str(database)]) == 0
    assert ms.main(["--config", str(config_file), "--database", str(database)]) == 0
    return {
        "root": root,
        "db": database,
        "config_file": config_file,
        "planted": [p + 1 for p in planted],  # bronze_row_id is the 1-based position
    }


def query(env, sql, params=None):
    with duckdb.connect(str(env["db"]), read_only=True) as con:
        return con.execute(sql, params or []).df()


def test_every_month_is_scored(env):
    summary = query(env, "SELECT * FROM ml_month_summary ORDER BY period")
    assert summary["period"].tolist() == ["2023-01", "2023-02", "2023-03", "2023-04", "2023-05"]
    silver = query(env, "SELECT period, silver_rows FROM silver_log WHERE status = 'processed'")
    assert summary.set_index("period")["rows_scored"].to_dict() == dict(
        zip(silver["period"], silver["silver_rows"], strict=True)
    )
    flagged = query(env, "SELECT period, count(*) AS n FROM ml_anomalies GROUP BY 1")
    assert summary.set_index("period")["rows_flagged"].to_dict() == dict(
        zip(flagged["period"], flagged["n"], strict=True)
    )
    assert summary["warnings"].isna().all()


def test_clean_months_flag_about_the_target_rate(env):
    summary = query(env, "SELECT period, flag_rate FROM ml_month_summary").set_index("period")
    for period in ("2023-01", "2023-02", "2023-03", "2023-04"):
        assert summary.loc[period, "flag_rate"] <= 0.02, period


def test_planted_fares_rank_high_and_most_are_flagged(env):
    """An Isolation Forest does not extrapolate: a value far beyond the training range scores
    about the same as one just beyond it, so the planted rows compete with the month's natural
    oddities (totals that do not add up) for the flag budget. They must rank near the top."""
    config = load_ml_config(env["config_file"])
    trained = mt.load_model(config)
    with db.connect(env["db"], read_only=True) as con:
        from ml.features import feature_frame
        from silver.process_silver import current_silver

        frame = feature_frame(con, config, current_silver(con, config.dataset)["2023-05"][0])
    frame["score"] = trained.scores(frame)
    top = frame.nlargest(10, "score")["bronze_row_id"].tolist()  # top 5% of 189 rows
    assert set(env["planted"]) <= set(top), (env["planted"], top)

    may = query(env, "SELECT * FROM ml_anomalies WHERE period = '2023-05'").set_index(
        "bronze_row_id"
    )
    caught = [r for r in env["planted"] if r in may.index]
    assert len(caught) >= 3, (env["planted"], sorted(may.index))
    for row in caught:
        top_features = may.loc[row, "top_features"]
        assert {"fare_per_mile", "fare_per_minute", "total_gap"} & set(top_features[:2])
        assert "(typical" in may.loc[row, "reasons"]
    rate = query(env, "SELECT flag_rate FROM ml_month_summary WHERE period = '2023-05'")
    assert rate.iloc[0, 0] > 0.02  # the planted rows lift May above the clean months


def test_scores_are_above_the_threshold(env):
    threshold = query(env, "SELECT threshold FROM ml_models").iloc[0, 0]
    scores = query(env, "SELECT score, top_features FROM ml_anomalies")
    assert (scores["score"] > threshold).all()
    assert scores["top_features"].map(len).between(1, 3).all()


def test_explain_measures_in_typical_spread():
    meta = {
        "features": {"a": "a", "b": "b", "c": "c"},
        "medians": {"a": 0.0, "b": 10.0, "c": 5.0},
        "scales": {"a": 1.0, "b": 10.0, "c": 1.0},
    }
    trained = mt.TrainedModel(model=None, metadata=meta)
    rows = pd.DataFrame({"a": [3.0, None], "b": [50.0, 10.0], "c": [5.0, 9.0]})
    features, reasons = ms.explain(trained, rows, top=2)
    assert features[0] == ["b", "a"]  # b is 4 spreads away, a 3, c 0
    assert reasons[0] == "b 50 (typical 10); a 3 (typical 0)"
    assert features[1] == ["c", "b"]  # a missing value never leads


def test_rescoring_replaces_rows(env):
    before = query(env, "SELECT count(*) AS n FROM ml_anomalies").iloc[0, 0]
    assert ms.main(["--config", str(env["config_file"]), "--database", str(env["db"])]) == 0
    assert query(env, "SELECT count(*) AS n FROM ml_anomalies").iloc[0, 0] == before
    assert query(env, "SELECT count(*) AS n FROM ml_month_summary").iloc[0, 0] == 5


def test_warnings_for_an_old_model(env, capsys):
    old = write_config(env["root"], retrain={"max_age_months": 1})
    assert ms.main(["--config", str(old), "--database", str(env["db"]), "--months", "2023-05"]) == 0
    out = capsys.readouterr().out
    assert "warning: model learnt from months up to 2023-03, 2 months before 2023-05" in out
    with db.connect(env["db"], read_only=True) as con:
        run_id = con.execute("SELECT max(run_id) FROM pipeline_runs").fetchone()[0]
        assert get_run(con, run_id)["status"] == "WARNING"


def test_warning_when_training_data_changed(env):
    config = load_ml_config(env["config_file"])
    trained = mt.load_model(config)
    versions = dict(trained.metadata["silver_versions"]) | {"2023-02": "another-version"}
    warnings = ms.model_warnings(trained, config, "2023-04", versions)
    assert warnings == ["Silver of 2023-02 changed since training: retrain"]


def test_errors(env, tmp_path, capsys):
    no_model = write_config(tmp_path)
    assert ms.main(["--config", str(no_model), "--database", str(env["db"])]) == 1
    assert "no trained model" in capsys.readouterr().out
    argv = ["--config", str(no_model), "--database", str(env["db"]), "--if-trained"]
    assert ms.main(argv) == 0  # what gold/rebuild.py does before a model is trained
    assert "scoring skipped: no trained model" in capsys.readouterr().out
    assert ms.main(["--config", str(env["config_file"]), "--months", "May"]) == 2
    argv = ["--config", str(env["config_file"]), "--database", str(env["db"])]
    assert ms.main([*argv, "--months", "2024-01"]) == 1
    assert "no Silver to score" in capsys.readouterr().out
