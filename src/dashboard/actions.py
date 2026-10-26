"""The dashboard's only writes: what a person decides on the Incidents page (Step 10.4).

Each one opens the database for writing, makes one change through the same code as the command
line (incidents_cli.py, investigate.py --decide) and closes it again. Nothing is run: approving
an action records the decision, as on the command line. Every change names the person, so the
incident's history says who did what.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from common import db
from dashboard.data import DashboardError
from incidents import incident_manager as im
from investigation import investigate

DECISIONS = investigate.DECISIONS  # approved, rejected, done
STATUS_CHANGES = {  # button: (from statuses, to status)
    "Start investigating": (("OPEN",), "INVESTIGATING"),
    "Back to open": (("INVESTIGATING",), "OPEN"),
    "Resolve": (("OPEN", "INVESTIGATING"), "RESOLVED"),
}


def _write(database: Path, change, *args: Any) -> None:
    name = args[-1]
    if not name or not str(name).strip():
        raise DashboardError("Enter your name first: every decision records who took it.")
    if not Path(database).exists():
        raise DashboardError(f"No database at {database}.")
    try:
        with db.connect(database) as con:
            change(con, *args)
    except duckdb.IOException as exc:
        raise DashboardError(
            "The database is busy (the pipeline is writing to it). Try again in a moment."
        ) from exc
    except ValueError as exc:  # IncidentError and a wrong decision
        raise DashboardError(str(exc)) from exc


def _signed(note: str | None, by: str) -> str:
    note = (note or "").strip()
    return f"{note} (by {by.strip()})" if note else f"by {by.strip()}"


def set_status(database: Path, incident_id: str, status: str, note: str | None, by: str) -> None:
    """OPEN <-> INVESTIGATING, or resolve by hand (incident_manager.set_status)."""

    def change(con, incident_id, status, note, by):
        im.set_status(con, incident_id, status, note=_signed(note, by))

    _write(database, change, incident_id, status, note, by)


def add_note(database: Path, incident_id: str, note: str, by: str) -> None:
    if not note or not note.strip():
        raise DashboardError("The note is empty.")

    def change(con, incident_id, note, by):
        im.add_note(con, incident_id, _signed(note, by))

    _write(database, change, incident_id, note, by)


def decide(database: Path, incident_id: str, action_id: str, decision: str, by: str) -> None:
    """Approve, reject or mark done a suggested action (investigate.decide)."""

    def change(con, incident_id, action_id, decision, by):
        investigate.decide(con, incident_id, action_id, decision, by.strip())

    _write(database, change, incident_id, action_id, decision, by)
