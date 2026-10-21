"""gold_incident_summary and gold_incident_monthly on the five planted months.

The pipeline opens six incidents: INC-0001 volume (2023-01, still open), INC-0002 schema
(2023-02, resolved 2023-04), INC-0003 to INC-0005 NULL spike and drift (2023-05, open) and
INC-0006 pickups outside the month (row rule, 2023-02, resolved 2023-05).
"""

import shutil

import pandas as pd
import pytest
from pipeline_fixture import run_pipeline

from common import db
from gold import monitoring_tables as gold
from incidents import incident_manager as im


@pytest.fixture(scope="module")
def pipeline_db(tmp_path_factory):
    database = run_pipeline(tmp_path_factory.mktemp("gold_incidents"))
    with db.connect(database) as con:
        gold.rebuild(con)
    return database


@pytest.fixture
def database(pipeline_db, tmp_path):
    """A copy for tests that change incidents."""
    copy = tmp_path / "copy.duckdb"
    shutil.copy(pipeline_db, copy)
    return copy


def summary(path):
    with db.connect(path, read_only=True) as con:
        return con.execute("SELECT * FROM gold_incident_summary").df().set_index("incident_id")


def monthly(path):
    with db.connect(path, read_only=True) as con:
        return con.execute("SELECT * FROM gold_incident_monthly").df().set_index("period")


def test_one_row_per_incident(pipeline_db):
    s = summary(pipeline_db)
    assert s.index.tolist() == [f"INC-000{i}" for i in range(1, 7)]
    with db.connect(pipeline_db, read_only=True) as con:
        status = dict(con.execute("SELECT incident_id, status FROM incidents").fetchall())
    assert s["status"].to_dict() == status


def test_durations(pipeline_db):
    s = summary(pipeline_db)
    schema = s.loc["INC-0002"]  # seen in February, clean in March and April
    assert (schema.resolved_period, schema.resolved_by) == ("2023-04", "lifecycle")
    assert (schema.months_seen, schema.months_active, schema.months_to_resolve) == (1, 3, 2)
    volume = s.loc["INC-0001"]  # open from January to the latest month, May
    assert volume.resolved_period is None and volume.resolved_by is None
    assert volume.months_active == 5 and volume.months_seen == 4
    assert pd.isna(volume.months_to_resolve)
    pickup = s.loc["INC-0006"]
    assert (pickup.source, pickup.months_to_resolve, pickup.months_active) == ("row_rule", 3, 4)


def test_history_counts(pipeline_db):
    s = summary(pipeline_db)
    assert s.loc["INC-0001", "events"] == 5  # opened, 3x seen again, clean
    assert not s["investigated"].any() and (s["notes"] == 0).all()
    assert (s["chain_root"] == s.index).all() and (s["chain_position"] == 1).all()


def test_monthly_counts(pipeline_db):
    m = monthly(pipeline_db)
    assert m["opened"].tolist() == [1, 2, 0, 0, 3]
    assert m["seen"].tolist() == [1, 3, 2, 1, 3]
    assert m["resolved"].tolist() == [0, 0, 0, 1, 1]
    assert m["active_end"].tolist() == [1, 3, 3, 2, 4]
    assert m.loc["2023-05", ["active_high", "active_medium"]].tolist() == [3, 1]
    assert (m["active_critical"] == 0).all() and (m["recurrences"] == 0).all()
    total = m[["active_critical", "active_high", "active_medium", "active_low"]].sum(axis=1)
    assert (total == m["active_end"]).all()


def test_active_at_the_end_matches_the_incidents_today(pipeline_db):
    with db.connect(pipeline_db, read_only=True) as con:
        active = con.execute("SELECT count(*) FROM incidents WHERE status <> 'RESOLVED'").fetchone()
    assert monthly(pipeline_db).loc["2023-05", "active_end"] == active[0]


def test_manual_actions_and_recurrence(database):
    with db.connect(database) as con:
        im.set_status(con, "INC-0001", "INVESTIGATING", note="asked the source")
        im.add_note(con, "INC-0001", "waiting")
        im.set_status(con, "INC-0001", "RESOLVED", note="fixed upstream")  # by hand: no month
        recurrence = im.open_incident(
            con,
            im.Finding("yellow_taxi", "dataset_check", "schema", "medium", "2023-05", "run_x", "t"),
        )
        gold.rebuild(con)
    s = summary(database)
    volume = s.loc["INC-0001"]
    assert bool(volume.investigated) and volume.notes == 1
    # resolved by hand: counted as resolved in the month of its last lifecycle event (May)
    assert (volume.resolved_by, volume.resolved_period, volume.months_to_resolve) == (
        "hand",
        "2023-05",
        4,
    )
    new = s.loc[recurrence.incident_id]
    assert new.recurrence_of == "INC-0002"
    assert (new.chain_root, new.chain_position) == ("INC-0002", 2)

    m = monthly(database)
    assert m.loc["2023-05", "recurrences"] == 1
    assert m.loc["2023-05", "resolved"] == 2  # INC-0006 by the lifecycle, INC-0001 by hand
    assert m.loc["2023-04", "active_end"] == 2  # unchanged before May
    assert m.loc["2023-05", "active_end"] == 4  # three May incidents and the recurrence


def test_rebuild_checksums_are_stable(pipeline_db):
    with db.connect(pipeline_db) as con:
        _, first = gold.rebuild(con)
        _, second = gold.rebuild(con)
    assert first == second
    names = [b.name for b in first]
    assert "gold_incident_summary" in names and "gold_incident_monthly" in names


def test_no_incidents(tmp_path):
    with db.connect(tmp_path / "e.duckdb") as con:
        db.init_schema(con)
        gold.rebuild(con)
        rows = [
            con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            for t in ("gold_incident_summary", "gold_incident_monthly")
        ]
    assert rows == [0, 0]
