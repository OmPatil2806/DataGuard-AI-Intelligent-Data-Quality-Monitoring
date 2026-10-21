from datetime import datetime, timedelta

import duckdb
import pytest

from common import db
from incidents import incident_manager as im
from incidents.incident_manager import Finding, IncidentError

T0 = datetime(2023, 3, 1, 6, 0)


@pytest.fixture
def con(tmp_path):
    with db.connect(tmp_path / "d.duckdb") as connection:
        db.init_schema(connection)
        yield connection


def finding(**changes):
    values = {
        "dataset": "yellow_taxi",
        "source": "dataset_check",
        "rule_id": "volume",
        "severity": "medium",
        "period": "2023-04",
        "run_id": "run_0005",
        "title": "Volume drop",
        "observed": "57,000 rows/day",
        "expected": "109,000 rows/day ± 30%",
        "detail": "volume drop: -48%",
    }
    return Finding(**{**values, **changes})


def test_ids_count_up(con):
    assert im.next_incident_id(con) == "INC-0001"
    first = im.open_incident(con, finding(), now=T0)
    second = im.open_incident(con, finding(rule_id="schema", severity="high"), now=T0)
    assert (first.incident_id, second.incident_id) == ("INC-0001", "INC-0002")


def test_open_stores_the_finding(con):
    i = im.open_incident(con, finding(), now=T0)
    assert (i.status, i.severity, i.occurrences, i.clean_streak) == ("OPEN", "medium", 1, 0)
    assert (i.first_seen_period, i.last_seen_period) == ("2023-04", "2023-04")
    assert (i.first_seen_run, i.last_seen_run) == ("run_0005", "run_0005")
    assert i.latest_detail == "volume drop: -48%"
    assert i.opened_at == i.updated_at == T0 and i.resolved_at is None
    assert i.active and i.recurrence_of is None


def test_open_writes_the_first_event(con):
    i = im.open_incident(con, finding(), now=T0)
    [event] = im.events(con, i.incident_id)
    assert (event.seq, event.event, event.status_after, event.period) == (
        1,
        "opened",
        "OPEN",
        "2023-04",
    )
    assert event.detail == "volume drop: -48%"


def test_only_one_active_incident_per_problem(con):
    im.open_incident(con, finding(), now=T0)
    with pytest.raises(IncidentError, match="still active"):
        im.open_incident(con, finding(period="2023-05"))
    # Same check on another column, or another dataset, is a different problem.
    im.open_incident(con, finding(rule_id="null_rate", column="fare_amount"))
    im.open_incident(con, finding(dataset="ved_signals"))
    assert len(im.list_incidents(con)) == 3


def test_find_active(con):
    assert im.find_active(con, "yellow_taxi", "dataset_check", "volume") is None
    opened = im.open_incident(con, finding())
    assert im.find_active(con, "yellow_taxi", "dataset_check", "volume") == opened


def test_status_flow_and_history(con):
    i = im.open_incident(con, finding(), now=T0)
    im.set_status(
        con, i.incident_id, "INVESTIGATING", note="checking with TLC", now=T0 + timedelta(hours=1)
    )
    im.set_status(
        con, i.incident_id, "OPEN", note="not a source issue", now=T0 + timedelta(hours=2)
    )
    im.add_note(con, i.incident_id, "re-export requested", now=T0 + timedelta(hours=3))
    resolved = im.set_status(
        con,
        i.incident_id,
        "RESOLVED",
        note="fixed",
        period="2023-06",
        run_id="run_0009",
        now=T0 + timedelta(days=2),
    )
    assert (resolved.status, resolved.resolved_period) == ("RESOLVED", "2023-06")
    assert resolved.resolved_at == T0 + timedelta(days=2) and not resolved.active
    history = [(e.seq, e.event, e.status_after, e.detail) for e in im.events(con, i.incident_id)]
    assert history == [
        (1, "opened", "OPEN", "volume drop: -48%"),
        (2, "investigating", "INVESTIGATING", "checking with TLC"),
        (3, "reopened", "OPEN", "not a source issue"),
        (4, "note", "OPEN", "re-export requested"),
        (5, "resolved", "RESOLVED", "fixed"),
    ]


def test_invalid_status_changes(con):
    i = im.open_incident(con, finding())
    with pytest.raises(IncidentError, match="already OPEN"):
        im.set_status(con, i.incident_id, "OPEN")
    with pytest.raises(IncidentError, match="status must be one of"):
        im.set_status(con, i.incident_id, "CLOSED")
    im.set_status(con, i.incident_id, "RESOLVED")
    with pytest.raises(IncidentError, match="is resolved; a recurrence opens a new incident"):
        im.set_status(con, i.incident_id, "OPEN")
    with pytest.raises(IncidentError, match="unknown incident"):
        im.set_status(con, "INC-9999", "RESOLVED")


def test_recurrence_opens_a_new_incident_linked_to_the_old_one(con):
    first = im.open_incident(con, finding(), now=T0)
    im.set_status(con, first.incident_id, "RESOLVED", now=T0 + timedelta(days=1))
    again = im.open_incident(con, finding(period="2023-09"), now=T0 + timedelta(days=30))
    assert again.incident_id == "INC-0002" and again.recurrence_of == "INC-0001"
    assert im.events(con, again.incident_id)[0].detail.startswith("recurrence of INC-0001")


def test_list_orders_by_severity_then_newest(con):
    im.open_incident(con, finding(rule_id="volume", severity="medium"))
    im.open_incident(con, finding(rule_id="schema", severity="high"))
    im.open_incident(con, finding(rule_id="drift", column="fare_amount", severity="medium"))
    low = im.open_incident(con, finding(rule_id="freshness", severity="low"))
    im.set_status(con, low.incident_id, "RESOLVED")
    ids = [i.incident_id for i in im.list_incidents(con)]
    assert ids == ["INC-0002", "INC-0003", "INC-0001", "INC-0004"]
    assert [i.incident_id for i in im.list_incidents(con, active_only=True)] == ids[:3]
    assert [i.incident_id for i in im.list_incidents(con, status="RESOLVED")] == ["INC-0004"]


def test_finding_is_validated():
    with pytest.raises(IncidentError, match="source must be one of"):
        finding(source="manual")
    with pytest.raises(IncidentError, match="severity must be one of"):
        finding(severity="urgent")


def test_database_enforces_consistency(con):
    with pytest.raises(duckdb.ConstraintException):
        con.execute(
            "INSERT INTO incidents (incident_id, dataset, source, rule_id, title, severity, "
            "status, first_seen_period, last_seen_period, first_seen_run, last_seen_run, "
            "opened_at, updated_at) VALUES ('INC-0001', 'd', 'dataset_check', 'x', 't', 'medium', "
            "'RESOLVED', "
            "'2023-01', '2023-01', 'r', 'r', now(), now())"
        )  # RESOLVED without resolved_at
    with pytest.raises(duckdb.ConstraintException):
        con.execute(
            "INSERT INTO incident_events VALUES ('INC-0001', 1, 'closed', NULL, NULL, NULL, "
            "'OPEN', NULL, now())"
        )
