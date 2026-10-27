"""Phase 10 'done when': a broken file triggers an alert and appears on the dashboard.

The planted months of tests/test_run_checks.py arrive one delivery at a time, as on a real
pipeline: January to April first, then May, which is broken (fares empty in 68.5% of the
trips, distances in kilometres). Each delivery is ingested and rebuilt with the alert stage on,
and the dashboard is read as a person would see it.
"""

import json
import shutil

import duckdb
import pytest
from streamlit.testing.v1 import AppTest
from test_dashboard import APP
from test_run_checks import FIXTURE, MONTHS

from alerts import notifier
from dashboard import data
from gold import rebuild
from ingestion import ingest_data

FIXTURES = FIXTURE.parent
MAY = {"NULL rate increase: fare_amount", "Distribution drift: trip_distance",
       "Distribution drift: fare_amount"}  # fmt: skip


def deliver(root, months):
    """Ingest the months' files, then rebuild everything, alerts included."""
    database = root / "dataguard.duckdb"
    incoming = root / "incoming"
    for period in months:
        target = incoming / f"yellow_tripdata_{period}.parquet"
        duckdb.sql(f"COPY ({MONTHS[period].format(src=FIXTURE)}) TO '{target}' (FORMAT parquet)")
    ingest = ["--months", f"{months[0]}:{months[-1]}", "--source", str(incoming),
              "--bronze", str(root / "bronze"), "--database", str(database)]  # fmt: skip
    assert ingest_data.main(ingest) == 0
    argv = ["--lookups", str(incoming), "--silver", str(root / "silver"),
            "--quarantine", str(root / "quarantine"), "--reports", str(root / "reports"),
            "--database", str(database)]  # fmt: skip
    results = rebuild.run(rebuild.parse_args(argv))
    assert {r.name: r.exit_code for r in results}["alert"] == 0
    return database


def outbox(folder):
    path = folder / notifier.OUTBOX
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.fixture(scope="module")
def delivered(tmp_path_factory):
    root = tmp_path_factory.mktemp("proof")
    alerts = root / "alerts"
    (root / "incoming").mkdir()
    shutil.copy(
        FIXTURES / "taxi_zone_lookup_fixture.csv", root / "incoming" / "taxi_zone_lookup.csv"
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(notifier.ALERTS_DIR_ENV, str(alerts))
        database = deliver(root, ["2023-01", "2023-02", "2023-03", "2023-04"])
        before = outbox(alerts)
        deliver(root, ["2023-05"])
        after = outbox(alerts)[len(before) :]
    return {"db": database, "before": before, "after": after}


def test_the_broken_file_is_alerted_once(delivered):
    assert [a["title"] for a in delivered["before"]] == [
        "Rule failure rate up: pickup_in_file_month"
    ]
    assert not any(a["period"] == "2023-05" for a in delivered["before"])
    new = delivered["after"]
    assert {a["title"] for a in new} == MAY  # only May's problems, nothing earlier again
    assert {(a["severity"], a["period"]) for a in new} == {("high", "2023-05")}
    assert all("most likely" in a["text"] and "dashboard: http" in a["text"] for a in new)
    km = next(a for a in new if a["title"] == "Distribution drift: trip_distance")
    assert "kilometres instead of miles" in km["text"]
    with duckdb.connect(str(delivered["db"]), read_only=True) as con:
        logged = con.execute("SELECT a.status, count(*) FROM alert_log a JOIN incidents i "
                             "USING (incident_id) WHERE i.last_seen_period = '2023-05' "
                             "GROUP BY 1").fetchall()  # fmt: skip
    assert logged == [("sent", 3)]


def test_it_appears_on_the_dashboard(delivered, monkeypatch):
    monkeypatch.setenv(data.DATABASE_ENV, str(delivered["db"]))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert at.sidebar.selectbox[1].value == "2023-05"  # the new month comes first
    tiles = {m.label: m.value for m in at.metric}
    assert tiles["Health"] == "Critical" and int(tiles["Open incidents"]) >= 3
    top = at.dataframe[0].value
    assert MAY <= set(top["title"])
    alerts = at.dataframe[1].value
    assert len(alerts) >= 3 and alerts["summary"].str.contains("fare_amount").any()

    at.switch_page("views/incidents.py").run()
    assert not at.exception
    listed = at.dataframe[0].value
    assert MAY <= set(listed[listed["last_seen_period"] == "2023-05"]["title"])

    at.switch_page("views/health.py").run()
    assert not at.exception
    failed = at.dataframe[1].value  # the month's failed checks
    assert {"null_rate", "drift"} <= set(failed["check"])
