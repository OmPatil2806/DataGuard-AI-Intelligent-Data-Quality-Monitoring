"""Phase 10.2: the dashboard base and the Overview page, on the planted pipeline's database.

The five planted months (tests/pipeline_fixture.py), investigated, alerted and built into Gold:
2023-05 scores 76.8 (critical), four incidents are open (three high, one medium) and three
alerts went out.
"""

import hashlib
from pathlib import Path

import duckdb
import pytest
from pipeline_fixture import run_pipeline
from streamlit.testing.v1 import AppTest

from alerts import notifier
from common import db
from dashboard import data
from gold import monitoring_tables
from investigation import investigate

APP = Path(__file__).resolve().parents[1] / "src" / "dashboard" / "app.py"


@pytest.fixture(scope="module")
def database(tmp_path_factory):
    root = tmp_path_factory.mktemp("dashboard")
    database = run_pipeline(root)
    common = ["--database", str(database)]
    assert investigate.main([*common, "--no-reports"]) == 0
    assert notifier.main([*common, "--outbox", str(root / "outbox")]) == 0
    assert monitoring_tables.main(common) == 0
    return database


def gold(database, sql):
    with duckdb.connect(str(database), read_only=True) as con:
        return con.execute(sql).fetchall()


def app(monkeypatch, database) -> AppTest:
    monkeypatch.setenv(data.DATABASE_ENV, str(database))
    return AppTest.from_file(str(APP), default_timeout=60).run()


def test_overview_numbers_match_gold(database):
    assert data.datasets(database) == ["yellow_taxi"]
    numbers = data.overview(database, "yellow_taxi", "2023-05")
    scores = dict(gold(database, "SELECT period, score FROM gold_dataset_health"))
    score, before = scores["2023-05"], scores["2023-04"]
    assert numbers["score"] == pytest.approx(score)
    assert numbers["score_change"] == pytest.approx(score - before)
    assert (numbers["health"], numbers["bronze_rows"], numbers["months"]) == ("critical", 200, 5)
    assert numbers["open_incidents"] == {"critical": 0, "high": 3, "medium": 1, "low": 0}
    assert data.overview(database, "yellow_taxi", "2023-01")["score_change"] is None
    with pytest.raises(data.DashboardError, match="no month 2024-01"):
        data.overview(database, "yellow_taxi", "2024-01")


def test_top_incidents_alerts_and_runs(database):
    top = data.top_incidents(database, "yellow_taxi")
    assert top["incident_id"].tolist() == ["INC-0003", "INC-0004", "INC-0005", "INC-0001"]
    assert data.cause_text(top["headline"][1]).startswith("trip_distance looks like it is partly")
    alerts = data.recent_alerts(database, "yellow_taxi")
    assert sorted(alerts["incident_id"]) == ["INC-0003", "INC-0004", "INC-0005"]
    assert set(alerts["status"]) == {"sent"} and data.recent_alerts(database, "ved").empty
    runs = data.last_runs(database)
    assert runs["command"].tolist()[:3] == ["monitoring_tables", "alert", "investigate"]
    monthly = data.incident_monthly(database, "yellow_taxi")
    assert monthly["opened"].sum() == len(gold(database, "SELECT * FROM gold_incident_summary"))


def test_cause_text():
    assert data.cause_text("Schema change in 2023-02: most likely a rename (high confidence)") == (
        "a rename (high confidence)"
    )
    assert data.cause_text("Volume out of range in 2023-04: cause unknown (weak lead: x)") == (
        "cause unknown (weak lead: x)"
    )
    assert data.cause_text(None) == "not investigated yet"


def test_friendly_errors(tmp_path, monkeypatch):
    with pytest.raises(data.DashboardError, match="No database at"):
        data.check_ready(tmp_path / "none.duckdb")
    empty = tmp_path / "empty.duckdb"
    with db.connect(empty) as con:
        db.init_schema(con)
    with pytest.raises(data.DashboardError, match="Gold is not built yet"):
        data.check_ready(empty)

    def busy(*args, **kwargs):
        raise duckdb.IOException("Could not set lock on file")

    monkeypatch.setattr(db, "connect", busy)
    with pytest.raises(data.DashboardError, match="busy"):
        data.datasets(empty)


def test_the_overview_page(database, monkeypatch):
    at = app(monkeypatch, database)
    assert not at.exception
    assert at.title[0].value == "Overview: yellow_taxi"
    metrics = {m.label: m.value for m in at.metric}
    assert metrics == {"Quality score": "76.8", "Health": "Critical", "Open incidents": "4",
                       "Rows checked": "200", "Quarantined": "100.00%"}  # fmt: skip
    assert [s.value for s in at.subheader] == [
        "Quality score per month", "Incidents opened and resolved", "Most severe open incidents",
        "Latest alerts", "Latest pipeline runs",
    ]  # fmt: skip
    assert len(at.get("plotly_chart")) == 2
    assert at.dataframe[0].value["incident_id"].tolist()[0] == "INC-0003"
    assert len(at.dataframe[1].value) == 3  # the alerts
    month = at.sidebar.selectbox[1]
    assert month.options[0] == "2023-05"  # latest first
    month.select("2023-02").run()
    assert {m.label: m.value for m in at.metric}["Quality score"] == "69.1"
    assert not at.exception


def test_the_dashboard_never_writes(database, monkeypatch):
    before = hashlib.sha256(Path(database).read_bytes()).hexdigest()
    at = app(monkeypatch, database)
    at.sidebar.selectbox[1].select("2023-01").run()
    assert not at.exception
    assert hashlib.sha256(Path(database).read_bytes()).hexdigest() == before


def test_without_a_database_it_says_what_to_run(tmp_path, monkeypatch):
    at = app(monkeypatch, tmp_path / "missing.duckdb")
    assert not at.exception and not at.metric
    assert "python src/gold/rebuild.py" in at.info[0].value


# --- Dataset Health (Step 10.3) ---------------------------------------------------------------


def test_health_queries(database):
    parts = data.score_parts(database, "yellow_taxi", "2023-05")
    assert len(parts) == 10 and parts["category"].iloc[0] == "pattern"  # lowest pass rate first
    checks = data.checks(database, "yellow_taxi", "2023-05")
    (checks_run,) = gold(database, "SELECT checks_run FROM gold_dataset_health "
                                   "WHERE period = '2023-05'")[0]  # fmt: skip
    assert len(checks) == checks_run == 25
    assert checks["status"].tolist()[:4] == ["fail", "fail", "fail", "pass"]
    profile = data.column_profile(database, "yellow_taxi", "2023-05")
    assert profile["column_name"].tolist()[:2] == ["fare_amount", "trip_distance"]
    items = data.trend_items(database, "yellow_taxi")
    assert list(items)[:2] == ["rows_per_day", "quarantine_rate"] and items["rows_per_day"] == [""]
    assert "fare_amount" in items["null_rate"] and "trip_unique" in items["rule_failure_rate"]
    fares = data.trend(database, "yellow_taxi", "null_rate", "fare_amount")
    assert fares["period"].tolist()[-1] == "2023-05"
    assert (fares["value"].iloc[-1], fares["check_status"].iloc[-1]) == (0.685, "fail")
    changes = data.schema_changes(database, "yellow_taxi")
    assert changes[["column_name", "change"]].values.tolist() == [
        ["Airport_fee", "renamed"], ["VendorID", "type_changed"]
    ]  # fmt: skip


def test_distribution(database):
    shares = data.distribution(database, "yellow_taxi", "trip_distance", "2023-05")
    assert shares["period"].unique().tolist() == ["2023-05", "2023-04"]  # against the month before
    assert shares.groupby("period")["share"].sum().round(6).tolist() == [1.0, 1.0]
    assert shares["bin"].iloc[0] == "below 0" and shares["bin"].iloc[14] == "50 and more"
    first = data.distribution(database, "yellow_taxi", "trip_distance", "2023-01")
    assert first["period"].unique().tolist() == ["2023-01"]
    assert data.distribution(database, "yellow_taxi", "VendorID", "2023-05").empty  # not binned
    assert data.bin_labels([0, 1.5]) == ["below 0", "0 to 1.5", "1.5 and more"]


def test_the_health_page(database, monkeypatch):
    at = app(monkeypatch, database)
    at.switch_page("views/health.py").run()
    assert not at.exception
    assert at.title[0].value == "Dataset Health: yellow_taxi"
    assert [s.value for s in at.subheader] == [
        "Where the rows went", "Score by category in 2023-05", "Dataset checks in 2023-05",
        "Columns in 2023-05", "One measure over time", "Distribution in 2023-05", "Schema changes",
    ]  # fmt: skip
    assert len(at.get("plotly_chart")) == 4
    months, checks, columns, schema = (d.value for d in at.dataframe)
    assert (len(months), len(checks), len(columns), len(schema)) == (5, 3, 19, 2)  # failed checks
    at.toggle(key="only_failed").set_value(False).run()
    assert len(at.dataframe[1].value) == 25
    at.selectbox(key="metric").select("null_rate").run()
    at.selectbox(key="item:null_rate").select("fare_amount").run()
    at.selectbox(key="binned:yellow_taxi").select("fare_amount").run()
    assert not at.exception
