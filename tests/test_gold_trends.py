"""gold_trends on the five planted months, after the whole pipeline ran."""

import pytest
from pipeline_fixture import run_pipeline

from common import db
from gold import monitoring_tables as gold

PERIODS = ["2023-01", "2023-02", "2023-03", "2023-04", "2023-05"]


@pytest.fixture(scope="module")
def database(tmp_path_factory):
    database = run_pipeline(tmp_path_factory.mktemp("gold_trends"))
    with db.connect(database) as con:
        gold.rebuild(con)
    return database


@pytest.fixture(scope="module")
def trends(database):
    with db.connect(database, read_only=True) as con:
        return con.execute("SELECT * FROM gold_trends").df()


def series(trends, metric, item=""):
    rows = trends[(trends["metric"] == metric) & (trends["item"] == item)]
    return rows.set_index("period")


def test_one_row_per_month_metric_and_item(trends, database):
    assert not trends.duplicated(["dataset", "period", "metric", "item"]).any()
    with db.connect(database, read_only=True) as con:
        rules = con.execute("SELECT count(DISTINCT rule_id) FROM rule_run_log").fetchone()[0]
        columns = con.execute(
            "SELECT count(DISTINCT column_name) FROM dataset_metrics WHERE metric = 'null_rate'"
        ).fetchone()[0]
    counts = trends.groupby("metric").size().to_dict()
    assert counts == {
        "rows_per_day": 5,
        "quarantine_rate": 5,
        "warned_rate": 5,
        "score": 5,
        "null_rate": 5 * columns,
        "rule_failure_rate": 5 * rules,
        "psi": 6,  # three binned columns, April and May (drift is skipped during the cold start)
    }
    units = trends.groupby("metric")["unit"].unique().map(list).to_dict()
    assert units["rows_per_day"] == ["rows/day"] and units["psi"] == ["psi"]
    assert units["score"] == ["score"] and units["null_rate"] == ["share"]


def test_volume_against_its_baseline(trends):
    rows = series(trends, "rows_per_day")
    assert rows.index.tolist() == PERIODS
    apr = rows.loc["2023-04"]
    assert apr.baseline == pytest.approx(200 / 31)  # median of January to March
    assert bool(apr.baseline_trusted)
    assert apr.change == pytest.approx(100 / 30 - 200 / 31)
    assert (apr.check_status, apr.severity) == ("fail", "medium")
    assert rows.loc["2023-05", "check_status"] == "pass"
    # cold start: no trusted baseline, so no change
    assert rows.loc[["2023-01", "2023-02", "2023-03"], "change"].isna().all()


def test_null_spike(trends):
    fare = series(trends, "null_rate", "fare_amount").loc["2023-05"]
    assert fare.value == pytest.approx(0.685)
    assert fare.change == pytest.approx(0.685)
    assert (fare.check_status, fare.severity) == ("fail", "high")


def test_drift_values(trends):
    fare = series(trends, "psi", "fare_amount")
    assert fare.index.tolist() == ["2023-04", "2023-05"]
    assert fare.loc["2023-05", "value"] == pytest.approx(5.93)
    assert fare.loc["2023-05", "severity"] == "high"
    assert fare.loc["2023-05", "previous"] == fare.loc["2023-04", "value"]
    assert series(trends, "psi", "total_amount").loc["2023-05", "check_status"] == "pass"


def test_rates_and_score_match_the_health_table(trends, database):
    with db.connect(database, read_only=True) as con:
        health = con.execute("SELECT * FROM gold_dataset_health").df().set_index("period")
    quarantine = series(trends, "quarantine_rate")["value"]
    assert quarantine.to_dict() == pytest.approx(
        (health["quarantine_rows"] / health["bronze_rows"]).to_dict()
    )
    assert quarantine.loc["2023-01"] == pytest.approx(0.055)
    assert series(trends, "score")["value"].to_dict() == pytest.approx(health["score"].to_dict())


def test_rule_failure_rates(trends, database):
    with db.connect(database, read_only=True) as con:
        log = con.execute(
            "SELECT period, rule_id, rows_failed / rows_checked FROM rule_run_log"
        ).fetchall()
    rates = trends[trends["metric"] == "rule_failure_rate"]
    stored = {(r.period, r.item): r.value for r in rates.itertuples()}
    assert stored == pytest.approx({(p, r): v for p, r, v in log})
    pickup = series(trends, "rule_failure_rate", "pickup_in_file_month")
    assert pickup.loc["2023-01", "value"] == pytest.approx(0.01)
    assert pickup.loc["2023-02", "value"] == 1.0


def test_previous_is_the_month_before(trends):
    score = series(trends, "score")
    assert score["previous"].iloc[0] != score["previous"].iloc[0]  # NaN for the first month
    assert score["previous"].iloc[1:].tolist() == score["value"].iloc[:-1].tolist()


def test_rebuild_checksum_is_stable(database):
    with db.connect(database) as con:
        _, first = gold.rebuild(con)
        _, second = gold.rebuild(con)
    assert first == second
    assert "gold_trends" in [b.name for b in first]
