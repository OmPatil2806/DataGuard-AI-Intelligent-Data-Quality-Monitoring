"""The anomaly model inside the pipeline (Phase 7.4): rebuild stage, incidents, score, Gold.

Clean test months 2023-01 to 2023-05; in May, ten short trips get ten times their fare, which
lifts May's flag rate clearly above the earlier months.
"""

import duckdb
import pytest
from evaluation_fixture import make_sources

from gold import rebuild
from incidents import incidents_cli
from incidents.severity_engine import MLSettings, ml_findings
from ingestion import ingest_data
from ml import train as mt
from quality.baselines import BaselineSettings

PLANTED = 10


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("ml_pipeline")
    source = make_sources(root / "source")
    may = source / "yellow_tripdata_2023-05.parquet"
    trips = duckdb.sql(f"SELECT * FROM read_parquet('{may.as_posix()}')").df()
    short = trips.index[(trips["trip_distance"].between(0.5, 3)) & (trips["fare_amount"] > 0)]
    trips.loc[list(short[:PLANTED]), "fare_amount"] *= 10
    trips.to_parquet(may, index=False)
    database = root / "dataguard.duckdb"
    common = ["--database", str(database)]
    argv = ["--months", "2023-01:2023-05", "--source", str(source), "--bronze", str(root / "b")]
    assert ingest_data.main([*argv, *common]) == 0
    rebuild_argv = ["--lookups", str(source), "--silver", str(root / "s"),
                    "--quarantine", str(root / "q"), *common]  # fmt: skip
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DATAGUARD_MODELS_DIR", str(root / "models"))  # this module's own models
        assert rebuild.main(rebuild_argv) == 0  # no model yet: the ml stage is skipped
        with duckdb.connect(str(database), read_only=True) as con:
            before = {
                "summaries": con.execute("SELECT count(*) FROM ml_month_summary").fetchone()[0],
                "ml_parts": con.execute(
                    "SELECT count(*) FROM quality_score_parts WHERE category = 'ml_anomaly'"
                ).fetchone()[0],
                "gold_rate": con.execute(
                    "SELECT count(ml_flag_rate) FROM gold_dataset_health"
                ).fetchone()[0],
            }
        assert mt.main(common) == 0
        assert rebuild.main(rebuild_argv) == 0  # now the ml stage scores every month
    return {"db": database, "before": before, "common": common}


def query(env, sql):
    with duckdb.connect(str(env["db"]), read_only=True) as con:
        return con.execute(sql).df()


def test_without_a_model_nothing_changes(env):
    assert env["before"] == {"summaries": 0, "ml_parts": 0, "gold_rate": 0}


def test_every_month_is_scored_by_the_rebuild(env):
    summary = query(env, "SELECT period, flag_rate FROM ml_month_summary ORDER BY period")
    assert summary["period"].tolist() == ["2023-01", "2023-02", "2023-03", "2023-04", "2023-05"]
    rates = summary.set_index("period")["flag_rate"]
    assert rates["2023-05"] > 2 * rates[["2023-01", "2023-02", "2023-03", "2023-04"]].max()


def test_may_opens_an_ml_incident(env):
    incidents = query(env, "SELECT * FROM incidents WHERE rule_id = 'ml_anomaly'")
    assert len(incidents) == 1
    incident = incidents.iloc[0]
    assert (incident.source, incident.first_seen_period, incident.severity) == (
        "dataset_check", "2023-05", "medium",
    )  # fmt: skip
    assert incident.title == "Unusual trips (ML)"
    assert "most often" in incident.latest_detail and "baseline" in incident.latest_expected


def test_incident_update_is_idempotent(env):
    before = query(env, "SELECT count(*) AS n FROM incident_events").iloc[0, 0]
    assert incidents_cli.main([*env["common"], "update"]) == 0
    assert query(env, "SELECT count(*) AS n FROM incident_events").iloc[0, 0] == before


def test_score_has_the_ml_category(env):
    parts = query(
        env,
        "SELECT p.period, p.pass_rate, p.weight, m.flag_rate FROM quality_score_parts p "
        "JOIN ml_month_summary m USING (dataset, period, sha256) "
        "WHERE p.category = 'ml_anomaly' ORDER BY p.period",
    )
    assert len(parts) == 5 and (parts["weight"] == 1).all()
    assert ((parts["pass_rate"] - (1 - parts["flag_rate"])).abs() < 1e-12).all()


def test_gold_shows_the_rates(env):
    health = query(
        env, "SELECT period, ml_flag_rate, ml_rows_flagged, ml_model_id FROM gold_dataset_health"
    )
    summary = query(env, "SELECT period, flag_rate, rows_flagged FROM ml_month_summary")
    merged = health.merge(summary, on="period")
    assert len(merged) == 5 and (merged["ml_flag_rate"] == merged["flag_rate"]).all()
    assert (merged["ml_rows_flagged"] == merged["rows_flagged"]).all()
    assert merged["ml_model_id"].nunique() == 1
    trends = query(env, "SELECT * FROM gold_trends WHERE metric = 'ml_flag_rate'")
    assert len(trends) == 5 and set(trends["unit"]) == {"share"}
    may = query(env, "SELECT incidents_found FROM gold_dataset_health WHERE period = '2023-05'")
    assert may.iloc[0, 0] >= 1


def test_ml_findings_cold_start(tmp_path):
    con = duckdb.connect()
    from common import db

    db.init_schema(con)
    for period, rate in (("2023-01", 0.01), ("2023-02", 0.05)):
        con.execute(
            "INSERT INTO ml_month_summary VALUES ('yellow_taxi', ?, 'x', 'm1', 'run_0001', 100, "
            "?, ?, 0.5, NULL, now())",
            [period, int(rate * 100), rate],
        )
    settings = MLSettings(min_increase=0.005, relative_increase=1.0, cold_start_max_rate=0.03,
                          severity="medium")  # fmt: skip
    limits = BaselineSettings(min_runs=3)
    months = ml_findings(con, "run_0001", "yellow_taxi", settings, limits)
    assert months["2023-01"][0] == []  # 1% is below the 3% cold-start limit
    (finding,) = months["2023-02"][0]  # 5% is above it
    assert finding.rule_id == "ml_anomaly" and "cold start" in finding.expected
    assert months["2023-02"][1] == {("yellow_taxi", "dataset_check", "ml_anomaly", "")}
    assert ml_findings(con, "run_9999", "yellow_taxi", settings, limits) == {}
