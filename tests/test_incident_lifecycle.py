from dataclasses import replace
from pathlib import Path

import duckdb
import pytest

from common import db
from common.config import load_datasets_config
from incidents import incident_manager as im
from incidents import incidents_cli, lifecycle
from incidents.incident_manager import Finding
from ingestion import ingest_data
from quality import baselines, run_checks

DATASET = "yellow_taxi"
VOLUME = (DATASET, "dataset_check", "volume", "")
SCHEMA = (DATASET, "dataset_check", "schema", "")


@pytest.fixture
def con(tmp_path):
    with db.connect(tmp_path / "d.duckdb") as connection:
        db.init_schema(connection)
        yield connection


def volume(period, severity="medium", detail="volume drop"):
    return Finding(
        DATASET,
        "dataset_check",
        "volume",
        severity,
        period,
        f"run_{period}",
        "Volume out of range",
        detail=detail,
    )


def month(con, period, findings=(), checked=(VOLUME, SCHEMA)):
    return lifecycle.apply_period(
        con, DATASET, period, f"run_{period}", list(findings), set(checked), resolve_after=2
    )


def only(con):
    [incident] = im.list_incidents(con)
    return incident


def test_open_deduplicate_and_resolve(con):
    r = month(con, "2023-04", [volume("2023-04")])
    assert r.opened == ["INC-0001"]
    r = month(con, "2023-05", [volume("2023-05")])  # same problem again: no new incident
    assert (r.opened, r.updated) == ([], ["INC-0001"])
    i = only(con)
    assert (i.occurrences, i.first_seen_period, i.last_seen_period) == (2, "2023-04", "2023-05")
    month(con, "2023-06")  # clean 1
    assert (only(con).status, only(con).clean_streak) == ("OPEN", 1)
    r = month(con, "2023-07")  # clean 2 -> resolved
    assert r.resolved == ["INC-0001"]
    i = only(con)
    assert (i.status, i.resolved_period) == ("RESOLVED", "2023-07")
    events = [(e.event, e.period) for e in im.events(con, "INC-0001")]
    assert events == [
        ("opened", "2023-04"),
        ("seen_again", "2023-05"),
        ("clean", "2023-06"),
        ("clean", "2023-07"),
        ("resolved", "2023-07"),
    ]


def test_a_failure_resets_the_clean_streak(con):
    month(con, "2023-04", [volume("2023-04")])
    month(con, "2023-05")
    month(con, "2023-06", [volume("2023-06")])
    month(con, "2023-07")
    i = only(con)
    assert (i.status, i.clean_streak, i.occurrences) == ("OPEN", 1, 2)


def test_recurrence_after_resolution(con):
    month(con, "2023-04", [volume("2023-04")])
    month(con, "2023-05")
    month(con, "2023-06")
    r = month(con, "2023-09", [volume("2023-09")])
    assert r.opened == ["INC-0002"]
    assert im.get_incident(con, "INC-0002").recurrence_of == "INC-0001"


def test_unchecked_month_does_not_count_as_clean(con):
    month(con, "2023-04", [volume("2023-04")])
    month(con, "2023-05", checked=[SCHEMA])  # volume not checked (e.g. cold start, skipped)
    month(con, "2023-06", checked=[])  # month missing
    assert only(con).clean_streak == 0


def test_rerunning_a_month_changes_nothing(con):
    month(con, "2023-04", [volume("2023-04")])
    month(con, "2023-05", [volume("2023-05")])
    month(con, "2023-05", [volume("2023-05")])  # same month again
    month(con, "2023-04", [volume("2023-04")])  # an earlier month
    assert only(con).occurrences == 2
    month(con, "2023-06")
    month(con, "2023-06")
    assert only(con).clean_streak == 1


def test_severity_is_raised_but_never_lowered(con):
    month(con, "2023-04", [volume("2023-04", "medium")])
    month(con, "2023-05", [volume("2023-05", "high", "drop of 70%")])
    assert only(con).severity == "high"
    assert "severity raised medium -> high" in im.events(con, "INC-0001")[-1].detail
    month(con, "2023-06", [volume("2023-06", "low")])
    assert only(con).severity == "high"


def test_duplicate_findings_in_one_month_keep_the_worst(con):
    r = month(con, "2023-04", [volume("2023-04", "low"), volume("2023-04", "high")])
    assert r.opened == ["INC-0001"] and only(con).severity == "high"


def test_investigating_incidents_resolve_too(con):
    month(con, "2023-04", [volume("2023-04")])
    im.set_status(con, "INC-0001", "INVESTIGATING", note="asked the source")
    month(con, "2023-05")
    month(con, "2023-06")
    assert only(con).status == "RESOLVED"


def test_wrong_month_or_dataset_is_refused(con):
    with pytest.raises(im.IncidentError, match="applied to"):
        month(con, "2023-05", [volume("2023-04")])


# --- end to end with dataset check results ------------------------------------------------


@pytest.fixture(scope="module")
def checked_db(tmp_path_factory):
    """Five fixture months with planted problems (see tests/test_run_checks.py)."""
    from test_run_checks import FIXTURE, MONTHS

    tmp = tmp_path_factory.mktemp("lifecycle")
    incoming = tmp / "incoming"
    incoming.mkdir()
    for period, query in MONTHS.items():
        target = incoming / f"yellow_tripdata_{period}.parquet"
        duckdb.sql(f"COPY ({query.format(src=FIXTURE)}) TO '{target}' (FORMAT parquet)")
    database = tmp / "d.duckdb"
    assert (
        ingest_data.main(
            [
                "--months",
                "2023-01:2023-05",
                "--source",
                str(incoming),
                "--bronze",
                str(tmp / "bronze"),
                "--database",
                str(database),
            ]
        )
        == 0
    )
    taxi = load_datasets_config().dataset(DATASET)
    limits = replace(baselines.settings_for(DATASET), rows_per_day_min=1, rows_per_day_max=100)
    with db.connect(database) as c:
        run_id = run_checks.run_checks(c, taxi, limits=limits)
    return database, run_id


def test_check_run_becomes_incidents(checked_db, tmp_path):
    database, run_id = checked_db
    copy = tmp_path / "copy.duckdb"
    copy.write_bytes(Path(database).read_bytes())
    with db.connect(copy) as c:
        r = lifecycle.apply_check_run(c, run_id, DATASET, resolve_after=2)
        incidents = {(i.rule_id, i.column_name): i for i in im.list_incidents(c)}
    # Feb schema change: clean in March and April -> resolved in April.
    assert incidents[("schema", "")].status == "RESOLVED"
    assert incidents[("schema", "")].resolved_period == "2023-04"
    # April volume drop: clean once in May -> still open.
    assert (incidents[("volume", "")].status, incidents[("volume", "")].clean_streak) == ("OPEN", 1)
    # May: NULL spike and two drifts.
    assert incidents[("null_rate", "fare_amount")].severity == "high"
    assert {k for k, i in incidents.items() if k[0] == "drift"} == {
        ("drift", "fare_amount"),
        ("drift", "trip_distance"),
    }
    assert len(r.opened) == 5 and r.resolved == [incidents[("schema", "")].incident_id]


def test_cli(checked_db, tmp_path, capsys):
    database, _ = checked_db
    copy = tmp_path / "copy.duckdb"
    copy.write_bytes(Path(database).read_bytes())
    base = ["--database", str(copy)]
    assert incidents_cli.main([*base, "update"]) == 0
    assert "5 opened" in capsys.readouterr().out
    assert incidents_cli.main([*base, "update"]) == 0  # applying again changes nothing
    assert "0 opened, 0 seen again, 0 clean, 0 resolved" in capsys.readouterr().out

    assert incidents_cli.main([*base, "list"]) == 0
    listed = capsys.readouterr().out
    assert "Schema change" not in listed and "NULL rate increase: fare_amount" in listed
    assert incidents_cli.main([*base, "list", "--all"]) == 0
    assert "Schema change" in capsys.readouterr().out

    with db.connect(copy, read_only=True) as c:
        target = next(i.incident_id for i in im.list_incidents(c, active_only=True))
    assert incidents_cli.main([*base, "investigate", target, "--note", "asked TLC"]) == 0
    assert "INVESTIGATING" in capsys.readouterr().out
    assert incidents_cli.main([*base, "note", target, "--note", "waiting"]) == 0
    assert incidents_cli.main([*base, "resolve", target, "--note", "re-export loaded"]) == 0
    shown = capsys.readouterr().out
    assert "investigating" in shown and "waiting" in shown and "re-export loaded" in shown
    assert incidents_cli.main([*base, "resolve", target]) == 2  # already resolved
    assert incidents_cli.main([*base, "show", "INC-9999"]) == 2


def test_cli_without_a_check_run(tmp_path, capsys):
    assert incidents_cli.main(["--database", str(tmp_path / "e.duckdb"), "update"]) == 1


def test_rerunning_an_old_month_after_resolution_opens_nothing(con):
    month(con, "2023-02", [volume("2023-02")])
    month(con, "2023-03")
    month(con, "2023-04")  # resolved here
    r = month(con, "2023-02", [volume("2023-02")])  # the whole history applied again
    assert r.opened == [] and len(im.list_incidents(con)) == 1
    r = month(con, "2023-05", [volume("2023-05")])  # a genuinely new month still recurs
    assert r.opened == ["INC-0002"]
