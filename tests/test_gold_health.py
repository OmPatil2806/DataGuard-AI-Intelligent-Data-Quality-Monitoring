"""gold_dataset_health on the five planted months, after the whole pipeline ran."""

from pathlib import Path

import pytest
import yaml
from pipeline_fixture import run_pipeline

from common import db
from common.config import ConfigError, load_severity_weights
from gold import monitoring_tables as gold
from gold.monitoring_tables import DATASET_HEALTH, GoldTable, dataset_health_sql


@pytest.fixture(scope="module")
def database(tmp_path_factory):
    database = run_pipeline(tmp_path_factory.mktemp("gold_health"))
    with db.connect(database) as con:
        gold.rebuild(con)
    return database


@pytest.fixture
def health(database):
    with db.connect(database, read_only=True) as con:
        frame = con.execute("SELECT * FROM gold_dataset_health").df()
    return frame.set_index("period")


def test_one_row_per_month_in_order(health):
    assert health.index.tolist() == ["2023-01", "2023-02", "2023-03", "2023-04", "2023-05"]
    assert (health["dataset"] == "yellow_taxi").all()


def test_rows_add_up(health):
    assert (health["silver_rows"] + health["quarantine_rows"] == health["bronze_rows"]).all()
    jan = health.loc["2023-01"]
    assert (jan.bronze_rows, jan.silver_rows, jan.quarantine_rows) == (200, 189, 11)
    assert jan.quarantine_pct == pytest.approx(5.5)
    assert health.loc["2023-04", "bronze_rows"] == 100


def test_score_matches_quality_scores(health, database):
    with db.connect(database, read_only=True) as con:
        scores = dict(con.execute("SELECT period, score FROM quality_scores").fetchall())
    assert health["score"].to_dict() == pytest.approx(scores)


def test_failed_checks_by_severity(health):
    assert health.loc["2023-02", ["failed_high", "failed_medium"]].tolist() == [
        1,
        1,
    ]  # schema, volume
    may = health.loc["2023-05"]
    assert (may.checks_failed, may.failed_high, may.failed_medium) == (3, 3, 0)  # NULLs, 2x drift
    assert (health["failed_critical"] == 0).all() and (health["failed_low"] == 0).all()
    assert (health["checks_run"] > 0).all()
    assert str(health["checks_failed"].dtype) == "int64"


def test_incidents_per_month(health):
    assert health.loc["2023-01", ["incidents_found", "incidents_opened"]].tolist() == [
        1,
        1,
    ]  # volume
    # February: volume seen again, schema and pickups outside the month opened
    assert health.loc["2023-02", ["incidents_found", "incidents_opened"]].tolist() == [3, 2]
    assert health.loc[
        "2023-05", ["incidents_found", "incidents_opened", "incidents_active"]
    ].tolist() == [3, 3, 3]
    # the schema incident was resolved later, so only the volume incident is still active
    assert health.loc["2023-02", "incidents_active"] == 1


def test_worst_severity_and_health(health):
    assert health["worst_severity"].to_dict() == {
        "2023-01": "medium",
        "2023-02": "high",
        "2023-03": "high",
        "2023-04": "medium",
        "2023-05": "high",
    }
    assert health["health"].to_dict() == {
        "2023-01": "warning",  # 91.2: below 95, medium volume finding
        "2023-02": "critical",  # 69.1: below 80
        "2023-03": "warning",
        "2023-04": "warning",
        "2023-05": "critical",  # 76.8: below 80
    }


@pytest.mark.parametrize(
    ("limits", "expected"),
    [
        ({"warning_below": 50, "critical_below": 40}, ["warning"] * 5),  # medium/high findings
        (
            {"warning_below": 100, "critical_below": 90},
            ["warning", "critical", "critical", "critical", "critical"],
        ),
    ],
)
def test_health_thresholds(database, limits, expected):
    table = GoldTable("gold_health_test", "", lambda: dataset_health_sql(limits), "period")
    with db.connect(database) as con:
        gold.build(con, "run_test", [gold.MONTHS, table])
        labels = [r[0] for r in con.execute("SELECT health FROM gold_health_test").fetchall()]
        con.execute("DROP TABLE gold_health_test")
        con.execute("DELETE FROM gold_build_log WHERE run_id = 'run_test'")
    assert labels == expected


def test_low_findings_alone_keep_a_month_healthy(database):
    table = GoldTable(
        "gold_health_test",
        "",
        lambda: dataset_health_sql({"warning_below": 50, "critical_below": 40}),
        "period",
    )
    with db.connect(database) as con:
        con.execute("CREATE TEMP TABLE saved AS SELECT * FROM dataset_check_results")
        con.execute("UPDATE dataset_check_results SET severity = 'low' WHERE status = 'fail'")
        con.execute("CREATE TEMP TABLE saved_events AS SELECT * FROM incident_events")
        con.execute("UPDATE incident_events SET severity = 'low'")
        gold.build(con, "run_test", [gold.MONTHS, table])
        labels = {r[0] for r in con.execute("SELECT health FROM gold_health_test").fetchall()}
        con.execute("DELETE FROM dataset_check_results")
        con.execute("INSERT INTO dataset_check_results SELECT * FROM saved")
        con.execute("DELETE FROM incident_events")
        con.execute("INSERT INTO incident_events SELECT * FROM saved_events")
        con.execute("DROP TABLE gold_health_test")
        con.execute("DELETE FROM gold_build_log WHERE run_id = 'run_test'")
    assert labels == {"healthy"}


def test_month_without_silver_is_not_scored(tmp_path):
    import shutil

    from test_monitoring_tables import FIXTURE

    from ingestion import ingest_data

    incoming = tmp_path / "incoming"
    incoming.mkdir()
    shutil.copy(FIXTURE, incoming / "yellow_tripdata_2023-01.parquet")
    argv = ["--months", "2023-01", "--source", str(incoming), "--bronze", str(tmp_path / "b")]
    assert ingest_data.main([*argv, "--database", str(tmp_path / "d.duckdb")]) == 0
    with db.connect(tmp_path / "d.duckdb") as con:
        gold.rebuild(con)
        row = con.execute(
            "SELECT health, score, silver_rows, checks_run, incidents_found, worst_severity "
            "FROM gold_dataset_health"
        ).fetchone()
    assert row == ("not_scored", None, None, 0, 0, None)


def test_rebuild_checksum_is_stable(database):
    with db.connect(database) as con:
        _, first = gold.rebuild(con)
        _, second = gold.rebuild(con)
    assert first == second
    assert DATASET_HEALTH.name in [b.name for b in first]


def test_health_config_is_validated(tmp_path):
    assert load_severity_weights()["health"] == {"warning_below": 95, "critical_below": 80}
    data = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "severity_weights.yml").read_text()
    )
    for bad in (
        {"warning_below": 80, "critical_below": 95},
        {"warning_below": 95},
        {"warning_below": 120, "critical_below": 80},
    ):
        data["health"] = bad
        path = tmp_path / "w.yml"
        path.write_text(yaml.safe_dump(data))
        with pytest.raises(ConfigError, match="health needs"):
            load_severity_weights(path)
