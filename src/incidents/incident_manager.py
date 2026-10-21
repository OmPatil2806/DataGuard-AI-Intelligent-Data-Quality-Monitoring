"""Incidents: one record per data quality problem, with its full history.

An incident is identified by (dataset, source, rule_id, column_name):

    source     row_rule (a row rule failing too often) or dataset_check (schema, volume, ...)
    rule_id    the rule id or the check name
    column     the affected column, '' for checks on a whole month

At most one incident per identity is not RESOLVED at a time. Every change is written to
incident_events, so the history of an incident can always be replayed.

Statuses: OPEN -> INVESTIGATING (by hand) -> OPEN again or RESOLVED. The automatic lifecycle
(de-duplication and resolving after clean runs) is in Step 4.4.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any

import duckdb

from common.db import INCIDENT_SOURCES, INCIDENT_STATUSES, SEVERITY_LEVELS
from common.runs import utc_now

ID_PREFIX = "INC-"


class IncidentError(ValueError):
    """Raised for an invalid incident operation, such as resolving twice."""


@dataclass(frozen=True)
class Finding:
    """A failed check or rule, the input from which incidents are opened or updated."""

    dataset: str
    source: str
    rule_id: str
    severity: str
    period: str
    run_id: str
    title: str
    column: str = ""
    observed: str | None = None
    expected: str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if self.source not in INCIDENT_SOURCES:
            raise IncidentError(f"source must be one of {', '.join(INCIDENT_SOURCES)}")
        if self.severity not in SEVERITY_LEVELS:
            raise IncidentError(f"severity must be one of {', '.join(SEVERITY_LEVELS)}")

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.dataset, self.source, self.rule_id, self.column)


@dataclass(frozen=True)
class Incident:
    incident_id: str
    dataset: str
    source: str
    rule_id: str
    column_name: str
    title: str
    severity: str
    status: str
    first_seen_period: str
    last_seen_period: str
    first_seen_run: str
    last_seen_run: str
    occurrences: int
    clean_streak: int
    latest_observed: str | None
    latest_expected: str | None
    latest_detail: str | None
    recurrence_of: str | None
    opened_at: datetime
    updated_at: datetime
    resolved_at: datetime | None
    resolved_period: str | None

    @property
    def active(self) -> bool:
        return self.status != "RESOLVED"


@dataclass(frozen=True)
class Event:
    incident_id: str
    seq: int
    event: str
    period: str | None
    run_id: str | None
    severity: str | None
    status_after: str
    detail: str | None
    created_at: datetime


def _rows(cursor: duckdb.DuckDBPyConnection, cls: type) -> list[Any]:
    names = [d[0] for d in cursor.description]
    wanted = {f.name for f in fields(cls)}
    return [
        cls(**{k: v for k, v in zip(names, row, strict=True) if k in wanted})
        for row in cursor.fetchall()
    ]


def next_incident_id(con: duckdb.DuckDBPyConnection) -> str:
    last = con.execute(
        "SELECT max(CAST(substr(incident_id, 5) AS INTEGER)) FROM incidents "
        "WHERE regexp_matches(incident_id, '^INC-[0-9]+$')"
    ).fetchone()[0]
    return f"{ID_PREFIX}{(last or 0) + 1:04d}"


def get_incident(con: duckdb.DuckDBPyConnection, incident_id: str) -> Incident:
    found = _rows(
        con.execute("SELECT * FROM incidents WHERE incident_id = ?", [incident_id]), Incident
    )
    if not found:
        raise IncidentError(f"unknown incident '{incident_id}'")
    return found[0]


def list_incidents(
    con: duckdb.DuckDBPyConnection,
    dataset: str | None = None,
    status: str | None = None,
    active_only: bool = False,
) -> list[Incident]:
    """Incidents, most severe first, then newest first."""
    where, params = ["TRUE"], []
    if dataset:
        where.append("dataset = ?")
        params.append(dataset)
    if status:
        where.append("status = ?")
        params.append(status)
    if active_only:
        where.append("status <> 'RESOLVED'")
    order = (
        "CASE severity "
        + " ".join(f"WHEN '{s}' THEN {i}" for i, s in enumerate(SEVERITY_LEVELS))
        + " END"
    )
    sql = f"SELECT * FROM incidents WHERE {' AND '.join(where)} ORDER BY {order}, incident_id DESC"
    return _rows(con.execute(sql, params), Incident)


def find_active(
    con: duckdb.DuckDBPyConnection, dataset: str, source: str, rule_id: str, column: str = ""
) -> Incident | None:
    """The incident for this problem that is not resolved yet, if there is one."""
    found = _rows(
        con.execute(
            "SELECT * FROM incidents WHERE dataset = ? AND source = ? AND rule_id = ? "
            "AND column_name = ? AND status <> 'RESOLVED'",
            [dataset, source, rule_id, column],
        ),
        Incident,
    )
    if len(found) > 1:  # pragma: no cover - guarded by open_incident
        raise IncidentError(f"more than one active incident for {source} {rule_id} {column}")
    return found[0] if found else None


def latest_resolved(
    con: duckdb.DuckDBPyConnection, dataset: str, source: str, rule_id: str, column: str = ""
) -> Incident | None:
    found = _rows(
        con.execute(
            "SELECT * FROM incidents WHERE dataset = ? AND source = ? AND rule_id = ? "
            "AND column_name = ? AND status = 'RESOLVED' ORDER BY resolved_at DESC LIMIT 1",
            [dataset, source, rule_id, column],
        ),
        Incident,
    )
    return found[0] if found else None


def events(con: duckdb.DuckDBPyConnection, incident_id: str) -> list[Event]:
    return _rows(
        con.execute(
            "SELECT * FROM incident_events WHERE incident_id = ? ORDER BY seq", [incident_id]
        ),
        Event,
    )


def add_event(
    con: duckdb.DuckDBPyConnection,
    incident_id: str,
    event: str,
    status_after: str,
    period: str | None = None,
    run_id: str | None = None,
    severity: str | None = None,
    detail: str | None = None,
    now: datetime | None = None,
) -> None:
    seq = con.execute(
        "SELECT coalesce(max(seq), 0) + 1 FROM incident_events WHERE incident_id = ?",
        [incident_id],
    ).fetchone()[0]
    con.execute(
        "INSERT INTO incident_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [incident_id, seq, event, period, run_id, severity, status_after, detail, now or utc_now()],
    )


def open_incident(
    con: duckdb.DuckDBPyConnection, finding: Finding, now: datetime | None = None
) -> Incident:
    """Open a new incident for a finding. Refused while one is still active for it."""
    now = now or utc_now()
    if find_active(con, *finding.key):
        raise IncidentError(
            f"an incident for {finding.source} {finding.rule_id} {finding.column} is still active"
        )
    previous = latest_resolved(con, *finding.key)
    incident_id = next_incident_id(con)
    con.execute(
        """
        INSERT INTO incidents (incident_id, dataset, source, rule_id, column_name, title,
            severity, status, first_seen_period, last_seen_period, first_seen_run,
            last_seen_run, occurrences, clean_streak, latest_observed, latest_expected,
            latest_detail, recurrence_of, opened_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?, 1, 0, ?, ?, ?, ?, ?, ?)
        """,
        [
            incident_id,
            finding.dataset,
            finding.source,
            finding.rule_id,
            finding.column,
            finding.title,
            finding.severity,
            finding.period,
            finding.period,
            finding.run_id,
            finding.run_id,
            finding.observed,
            finding.expected,
            finding.detail,
            previous.incident_id if previous else None,
            now,
            now,
        ],
    )
    note = finding.detail or finding.title
    if previous:
        note = f"recurrence of {previous.incident_id}: {note}"
    add_event(
        con,
        incident_id,
        "opened",
        "OPEN",
        finding.period,
        finding.run_id,
        finding.severity,
        note,
        now,
    )
    return get_incident(con, incident_id)


def set_status(
    con: duckdb.DuckDBPyConnection,
    incident_id: str,
    status: str,
    note: str | None = None,
    period: str | None = None,
    run_id: str | None = None,
    now: datetime | None = None,
) -> Incident:
    """Change the status by hand or from the lifecycle.

    Allowed: OPEN -> INVESTIGATING, INVESTIGATING -> OPEN, OPEN/INVESTIGATING -> RESOLVED.
    A resolved incident stays resolved; a recurrence opens a new incident.
    """
    if status not in INCIDENT_STATUSES:
        raise IncidentError(f"status must be one of {', '.join(INCIDENT_STATUSES)}")
    incident = get_incident(con, incident_id)
    if incident.status == "RESOLVED":
        raise IncidentError(f"{incident_id} is resolved; a recurrence opens a new incident")
    if incident.status == status:
        raise IncidentError(f"{incident_id} is already {status}")
    now = now or utc_now()
    if status == "RESOLVED":
        con.execute(
            "UPDATE incidents SET status = 'RESOLVED', resolved_at = ?, resolved_period = ?, "
            "updated_at = ? WHERE incident_id = ?",
            [now, period, now, incident_id],
        )
        event = "resolved"
    else:
        con.execute(
            "UPDATE incidents SET status = ?, updated_at = ? WHERE incident_id = ?",
            [status, now, incident_id],
        )
        event = "investigating" if status == "INVESTIGATING" else "reopened"
    add_event(con, incident_id, event, status, period, run_id, None, note, now)
    return get_incident(con, incident_id)


def add_note(
    con: duckdb.DuckDBPyConnection, incident_id: str, note: str, now: datetime | None = None
) -> None:
    incident = get_incident(con, incident_id)
    add_event(con, incident_id, "note", incident.status, detail=note, now=now)
