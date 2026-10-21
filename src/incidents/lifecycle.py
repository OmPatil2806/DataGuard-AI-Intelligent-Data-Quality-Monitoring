"""Incident lifecycle: turn the findings of a month into opened, updated and resolved incidents.

For every month, in month order:

    a finding without an active incident      -> open a new one (linked to an earlier,
                                                 resolved incident of the same problem)
    a finding with an active incident         -> de-duplicate: occurrences + 1, last seen,
                                                 clean streak back to 0, severity raised
                                                 if the new finding is worse
    an active incident whose check ran clean  -> clean streak + 1; resolved after
                                                 `resolve_after` clean months in a row
    an active incident whose check did not run (month missing, check skipped) -> unchanged

Each incident remembers the last month it was updated for, so applying the same month
again (a re-run) or an earlier month changes nothing: results are never counted twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import duckdb

from common.config import SEVERITIES, load_quality_rules
from common.runs import utc_now
from incidents import incident_manager as im
from incidents.incident_manager import Finding, Incident
from incidents.severity_engine import rule_findings

Key = tuple[str, str, str, str]  # dataset, source, rule_id, column


@dataclass
class LifecycleResult:
    opened: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    clean: list[str] = field(default_factory=list)
    resolved: list[str] = field(default_factory=list)


def resolve_after_clean_runs() -> int:
    return load_quality_rules()["defaults"]["incidents"]["resolve_after_clean_runs"]


def _more_severe(a: str, b: str) -> bool:
    return SEVERITIES.index(a) < SEVERITIES.index(b)


def last_handled_period(con: duckdb.DuckDBPyConnection, incident_id: str) -> str | None:
    return con.execute(
        "SELECT max(period) FROM incident_events WHERE incident_id = ? AND period IS NOT NULL "
        "AND event IN ('opened', 'seen_again', 'clean', 'resolved')",
        [incident_id],
    ).fetchone()[0]


def _seen_again(
    con: duckdb.DuckDBPyConnection, incident: Incident, f: Finding, now: datetime
) -> None:
    severity = f.severity if _more_severe(f.severity, incident.severity) else incident.severity
    con.execute(
        """
        UPDATE incidents SET last_seen_period = ?, last_seen_run = ?,
            occurrences = occurrences + 1, clean_streak = 0, severity = ?,
            latest_observed = ?, latest_expected = ?, latest_detail = ?, updated_at = ?
        WHERE incident_id = ?
        """,
        [f.period, f.run_id, severity, f.observed, f.expected, f.detail, now, incident.incident_id],
    )
    note = f.detail or f.title
    if severity != incident.severity:
        note = f"severity raised {incident.severity} -> {severity}; {note}"
    im.add_event(
        con,
        incident.incident_id,
        "seen_again",
        incident.status,
        f.period,
        f.run_id,
        severity,
        note,
        now,
    )


def apply_period(
    con: duckdb.DuckDBPyConnection,
    dataset: str,
    period: str,
    run_id: str,
    findings: list[Finding],
    checked: set[Key],
    resolve_after: int = 2,
    now: datetime | None = None,
) -> LifecycleResult:
    """Apply one month's findings. `checked` lists every problem identity whose check or
    rule actually ran for the month (passed or failed); only those can count as clean."""
    now = now or utc_now()
    result = LifecycleResult()
    by_key: dict[Key, Finding] = {}
    for f in findings:
        if f.dataset != dataset or f.period != period:
            raise im.IncidentError(
                f"finding for {f.dataset} {f.period} applied to {dataset} {period}"
            )
        current = by_key.get(f.key)
        if current is None or _more_severe(f.severity, current.severity):
            by_key[f.key] = f

    for key, f in by_key.items():
        active = im.find_active(con, *key)
        if active is None:
            # A resolved incident may already cover this month: re-applying an old month
            # must not open a duplicate "recurrence".
            previous = im.latest_resolved(con, *key)
            handled = last_handled_period(con, previous.incident_id) if previous else None
            if handled is not None and period <= handled:
                continue
            result.opened.append(im.open_incident(con, f, now).incident_id)
            continue
        handled = last_handled_period(con, active.incident_id)
        if handled is not None and period <= handled:
            continue  # this month (or a later one) was already applied
        _seen_again(con, active, f, now)
        result.updated.append(active.incident_id)

    for incident in im.list_incidents(con, dataset=dataset, active_only=True):
        key = (incident.dataset, incident.source, incident.rule_id, incident.column_name)
        if key in by_key or key not in checked:
            continue
        handled = last_handled_period(con, incident.incident_id)
        if handled is not None and period <= handled:
            continue
        streak = incident.clean_streak + 1
        con.execute(
            "UPDATE incidents SET clean_streak = ?, updated_at = ? WHERE incident_id = ?",
            [streak, now, incident.incident_id],
        )
        im.add_event(
            con,
            incident.incident_id,
            "clean",
            incident.status,
            period,
            run_id,
            None,
            f"clean month {streak} of {resolve_after}",
            now,
        )
        result.clean.append(incident.incident_id)
        if streak >= resolve_after:
            im.set_status(
                con,
                incident.incident_id,
                "RESOLVED",
                note=f"{streak} clean month(s) in a row",
                period=period,
                run_id=run_id,
                now=now,
            )
            result.resolved.append(incident.incident_id)
    return result


# --- findings from the dataset checks ----------------------------------------------------

CHECK_TITLES = {
    "schema": "Schema change",
    "volume": "Volume out of range",
    "freshness": "Late delivery",
    "null_rate": "NULL rate increase",
    "drift": "Distribution drift",
}


def check_findings(
    con: duckdb.DuckDBPyConnection, run_id: str, dataset: str
) -> dict[str, tuple[list[Finding], set[Key]]]:
    """Per month of a check run: the failed checks as findings, and every identity checked."""
    rows = con.execute(
        'SELECT period, "check", column_name, status, severity, observed, expected, detail '
        "FROM dataset_check_results WHERE run_id = ? AND dataset = ? ORDER BY period",
        [run_id, dataset],
    ).fetchall()
    months: dict[str, tuple[list[Finding], set[Key]]] = {}
    for period, check, column, status, severity, observed, expected, detail in rows:
        findings, checked = months.setdefault(period, ([], set()))
        if status == "skipped":
            continue
        checked.add((dataset, "dataset_check", check, column))
        if status == "fail":
            title = CHECK_TITLES.get(check, check)
            if column:
                title += f": {column}"
            findings.append(
                Finding(
                    dataset,
                    "dataset_check",
                    check,
                    severity,
                    period,
                    run_id,
                    title,
                    column,
                    observed,
                    expected,
                    detail,
                )
            )
    return months


def apply_check_run(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    dataset: str,
    resolve_after: int | None = None,
    now: datetime | None = None,
) -> LifecycleResult:
    """Apply every month of a dataset check run, oldest month first."""
    return _apply_months(
        con, run_id, dataset, check_findings(con, run_id, dataset), resolve_after, now
    )


def apply_rule_run(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    dataset: str,
    resolve_after: int | None = None,
    now: datetime | None = None,
) -> LifecycleResult:
    """Apply every month of a row rule run (run_rules.py), oldest month first."""
    return _apply_months(
        con, run_id, dataset, rule_findings(con, run_id, dataset), resolve_after, now
    )


def _apply_months(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    dataset: str,
    months: dict[str, tuple[list[Finding], set[Key]]],
    resolve_after: int | None,
    now: datetime | None,
) -> LifecycleResult:
    resolve_after = resolve_after or resolve_after_clean_runs()
    total = LifecycleResult()
    for period, (findings, checked) in sorted(months.items()):
        r = apply_period(con, dataset, period, run_id, findings, checked, resolve_after, now)
        for name in ("opened", "updated", "clean", "resolved"):
            getattr(total, name).extend(getattr(r, name))
    return total
