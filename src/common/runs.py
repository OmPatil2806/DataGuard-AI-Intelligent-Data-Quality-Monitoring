"""Track pipeline runs in the pipeline_runs control table.

    with track_run(con, "yellow_taxi", command="ingest --months 2023-01") as run:
        ...                                  # do the work
        run.add(files_loaded=1, rows_loaded=3_066_766)
        if something_odd:
            run.warn("2023-01 arrived late")

The run is stored as RUNNING when it starts. Leaving the block normally marks it
SUCCESS (or WARNING after run.warn); an exception marks it FAILED with the error
message and is then re-raised. Times are stored in UTC.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import duckdb

FINAL_STATUSES = ("SUCCESS", "WARNING", "FAILED")
COUNTERS = ("files_total", "files_loaded", "files_skipped", "rows_loaded")
MAX_ERROR_LENGTH = 2000
INTERRUPTED = "interrupted: the process stopped before the run finished"


class RunError(RuntimeError):
    """Raised for an invalid operation on a run, such as finishing it twice."""


def utc_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def next_run_id(con: duckdb.DuckDBPyConnection) -> str:
    """run_0001, run_0002, ... (the number keeps growing past 9999)."""
    last = con.execute(
        "SELECT max(CAST(substr(run_id, 5) AS INTEGER)) FROM pipeline_runs "
        "WHERE regexp_matches(run_id, '^run_[0-9]+$')"
    ).fetchone()[0]
    return f"run_{(last or 0) + 1:04d}"


def get_run(con: duckdb.DuckDBPyConnection, run_id: str) -> dict[str, Any]:
    cursor = con.execute("SELECT * FROM pipeline_runs WHERE run_id = ?", [run_id])
    row = cursor.fetchone()
    if row is None:
        raise RunError(f"unknown run '{run_id}'")
    return dict(zip([d[0] for d in cursor.description], row, strict=True))


def fail_interrupted_runs(
    con: duckdb.DuckDBPyConnection, dataset: str, now: datetime | None = None
) -> list[str]:
    """Mark runs still RUNNING for a dataset as FAILED (their process was killed)."""
    stale = [
        run_id
        for (run_id,) in con.execute(
            "SELECT run_id FROM pipeline_runs WHERE dataset = ? AND status = 'RUNNING' "
            "ORDER BY run_id",
            [dataset],
        ).fetchall()
    ]
    for run_id in stale:
        finish_run(con, run_id, "FAILED", error=INTERRUPTED, now=now)
    return stale


def start_run(
    con: duckdb.DuckDBPyConnection,
    dataset: str,
    command: str | None = None,
    now: datetime | None = None,
) -> str:
    now = now or utc_now()
    fail_interrupted_runs(con, dataset, now)
    run_id = next_run_id(con)
    con.execute(
        "INSERT INTO pipeline_runs (run_id, dataset, command, status, started_at) "
        "VALUES (?, ?, ?, 'RUNNING', ?)",
        [run_id, dataset, command, now],
    )
    return run_id


def add_counts(con: duckdb.DuckDBPyConnection, run_id: str, **counts: int) -> None:
    """Increase a running run's counters, e.g. add_counts(con, run_id, files_loaded=1)."""
    unknown = set(counts) - set(COUNTERS)
    if unknown:
        raise RunError(f"unknown counter(s): {', '.join(sorted(unknown))}")
    if any(value < 0 for value in counts.values()):
        raise RunError("counters can only increase")
    if get_run(con, run_id)["status"] != "RUNNING":
        raise RunError(f"run '{run_id}' is already finished")
    if counts:
        assignments = ", ".join(f"{name} = {name} + ?" for name in counts)
        con.execute(
            f"UPDATE pipeline_runs SET {assignments} WHERE run_id = ?",
            [*counts.values(), run_id],
        )


def finish_run(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    status: str,
    error: str | None = None,
    now: datetime | None = None,
) -> None:
    if status not in FINAL_STATUSES:
        raise RunError(f"status must be one of {', '.join(FINAL_STATUSES)}, got '{status}'")
    if status == "FAILED" and not error:
        raise RunError("a FAILED run needs an error message")
    if get_run(con, run_id)["status"] != "RUNNING":
        raise RunError(f"run '{run_id}' is already finished")
    if error and len(error) > MAX_ERROR_LENGTH:
        error = error[: MAX_ERROR_LENGTH - 3] + "..."
    con.execute(
        "UPDATE pipeline_runs SET status = ?, finished_at = ?, error_message = ? WHERE run_id = ?",
        [status, now or utc_now(), error, run_id],
    )


@dataclass
class Run:
    con: duckdb.DuckDBPyConnection
    run_id: str
    warnings: list[str] = field(default_factory=list)

    def add(self, **counts: int) -> None:
        add_counts(self.con, self.run_id, **counts)

    def warn(self, message: str) -> None:
        """Finish as WARNING instead of SUCCESS; the messages are stored with the run."""
        self.warnings.append(message)


@contextmanager
def track_run(
    con: duckdb.DuckDBPyConnection,
    dataset: str,
    command: str | None = None,
    clock: Callable[[], datetime] = utc_now,
) -> Iterator[Run]:
    run = Run(con, start_run(con, dataset, command, now=clock()))
    try:
        yield run
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        finish_run(con, run.run_id, "FAILED", error=error, now=clock())
        raise
    if run.warnings:
        finish_run(con, run.run_id, "WARNING", error="; ".join(run.warnings), now=clock())
    else:
        finish_run(con, run.run_id, "SUCCESS", now=clock())
