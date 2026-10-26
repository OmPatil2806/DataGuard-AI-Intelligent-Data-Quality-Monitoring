"""What the dashboard shows, read from the Gold tables (Phase 10).

No Streamlit here: every function takes the DuckDB file and returns a DataFrame or a dict, so
the queries are tested without a browser. The database is opened read-only and closed after
each query: the dashboard can never change it, and it holds DuckDB's file lock only for the
milliseconds a query takes, so the pipeline can write in between.

The dashboard reads Gold, plus the control tables Gold does not copy: pipeline_runs (when the
pipeline last ran), alert_log (what was alerted), and for one month the checks of its latest
run (dataset_check_results) and its score by category (quality_score_parts).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from common import db
from common.config import SEVERITIES, load_severity_weights
from quality.metrics import drift_bins

DATABASE_ENV = "DATAGUARD_DATABASE"  # another DuckDB file (as for the notebooks)
GOLD_TABLES = (
    "gold_dataset_health",
    "gold_incident_summary",
    "gold_incident_monthly",
    "gold_investigations",
    "gold_trends",
    "gold_column_profile",
    "gold_schema_history",
)
HEALTH = ("healthy", "warning", "critical", "not_scored")


class DashboardError(Exception):
    """The database cannot be shown yet; the message says what to run."""


@dataclass(frozen=True)
class HealthLimits:
    warning_below: float
    critical_below: float


def database_path() -> Path:
    if os.environ.get(DATABASE_ENV):
        return Path(os.environ[DATABASE_ENV])
    return db.database_path()


def query(database: Path, sql: str, params: list[Any] | None = None) -> pd.DataFrame:
    """Run one query on a read-only connection that is closed straight after."""
    try:
        with db.connect(database, read_only=True) as con:
            return con.execute(sql, params or []).df()
    except FileNotFoundError as exc:
        raise DashboardError(
            f"No database at {database}. Run the pipeline first: python src/gold/rebuild.py"
        ) from exc
    except duckdb.IOException as exc:  # the pipeline holds the write lock right now
        raise DashboardError(
            "The database is busy (the pipeline is writing to it). Refresh in a moment."
        ) from exc


def tables(database: Path) -> set[str]:
    return set(query(database, "SELECT table_name FROM information_schema.tables")["table_name"])


def check_ready(database: Path) -> None:
    """Raise DashboardError unless Gold has been built and holds a month."""
    missing = [t for t in GOLD_TABLES if t not in tables(database)]
    if missing:
        raise DashboardError(
            f"Gold is not built yet ({', '.join(missing)} missing). Run: python src/gold/rebuild.py"
        )
    if not len(datasets(database)):
        raise DashboardError("Gold holds no month yet. Ingest data, then run rebuild.py.")


def datasets(database: Path) -> list[str]:
    frame = query(database, "SELECT DISTINCT dataset FROM gold_dataset_health ORDER BY 1")
    return frame["dataset"].tolist()


def health_limits() -> HealthLimits:
    health = load_severity_weights().get("health") or {}
    return HealthLimits(
        float(health.get("warning_below", 95)), float(health.get("critical_below", 80))
    )


def health(database: Path, dataset: str) -> pd.DataFrame:
    """One row per month: rows, score, findings and the health label (gold_dataset_health)."""
    return query(
        database,
        """
        SELECT period, health, score, bronze_rows, silver_rows, quarantine_rows, warned_rows,
               quarantine_pct, checks_run, checks_failed, incidents_found, incidents_opened,
               worst_severity
        FROM gold_dataset_health WHERE dataset = ? ORDER BY period
        """,
        [dataset],
    )


def cause_text(headline: str | None) -> str:
    """The cause part of an investigation headline: "<title> in <month>: most likely X (...)"
    or "<title> in <month>: cause unknown (...)"."""
    if headline is None or pd.isna(headline):
        return "not investigated yet"
    for marker, keep in ((": most likely ", ""), (": cause unknown", "cause unknown")):
        if marker in headline:
            return keep + headline.split(marker, 1)[1]
    return headline


def _severity_order(column: str) -> str:
    cases = " ".join(f"WHEN '{s}' THEN {i}" for i, s in enumerate(SEVERITIES))
    return f"CASE {column} {cases} ELSE {len(SEVERITIES)} END"


def open_incidents(database: Path, dataset: str) -> dict[str, int]:
    """Active (not resolved) incidents now, per severity, every severity present."""
    frame = query(
        database,
        "SELECT severity, count(*) AS n FROM gold_incident_summary "
        "WHERE dataset = ? AND status != 'RESOLVED' GROUP BY 1",
        [dataset],
    )
    counts = dict(zip(frame["severity"], frame["n"].astype(int), strict=True))
    return {s: counts.get(s, 0) for s in SEVERITIES}


def overview(database: Path, dataset: str, period: str) -> dict[str, Any]:
    """The headline numbers of one month, with the month before for the change."""
    months = health(database, dataset)
    if period not in set(months["period"]):
        raise DashboardError(f"{dataset} has no month {period} in Gold.")
    position = months.index[months["period"] == period][0]
    row = months.loc[position]
    previous = months.loc[position - 1] if position > months.index[0] else None

    def value(series: pd.Series | None, name: str) -> float | None:
        if series is None or pd.isna(series[name]):
            return None
        return float(series[name])

    score, before = value(row, "score"), value(previous, "score")
    return {
        "period": period,
        "health": row["health"],
        "score": score,
        "score_change": score - before if score is not None and before is not None else None,
        "bronze_rows": int(row["bronze_rows"]),
        "quarantine_pct": value(row, "quarantine_pct"),
        "checks_failed": int(row["checks_failed"] or 0),
        "checks_run": int(row["checks_run"] or 0),
        "incidents_opened": int(row["incidents_opened"] or 0),
        "open_incidents": open_incidents(database, dataset),
        "months": len(months),
    }


def incident_monthly(database: Path, dataset: str) -> pd.DataFrame:
    return query(
        database,
        "SELECT period, opened, resolved, active_end FROM gold_incident_monthly "
        "WHERE dataset = ? ORDER BY period",
        [dataset],
    )


def top_incidents(database: Path, dataset: str, limit: int = 5) -> pd.DataFrame:
    """The most severe active incidents, newest first within a severity, with their cause."""
    return query(
        database,
        f"""
        SELECT s.incident_id, s.severity, s.title, s.first_seen_period, s.last_seen_period,
               s.months_seen, v.headline, v.top_cause, v.confidence, v.explained,
               v.actions_pending
        FROM gold_incident_summary s
        LEFT JOIN gold_investigations v USING (incident_id)
        WHERE s.dataset = ? AND s.status != 'RESOLVED'
        ORDER BY {_severity_order("s.severity")}, s.last_seen_period DESC, s.incident_id
        LIMIT ?
        """,
        [dataset, limit],
    )


def recent_alerts(database: Path, dataset: str, limit: int = 10) -> pd.DataFrame:
    """The latest alerts of the dataset (alert_log; empty for a database from before 10.1)."""
    columns = ["created_at", "incident_id", "severity", "status", "message", "error"]
    if "alert_log" not in tables(database):
        return pd.DataFrame(columns=columns)
    return query(
        database,
        f"SELECT {', '.join(columns)} FROM alert_log WHERE dataset = ? "
        "ORDER BY created_at DESC, incident_id LIMIT ?",
        [dataset, limit],
    )


def last_runs(database: Path, limit: int = 5) -> pd.DataFrame:
    """The latest pipeline runs of any dataset (pipeline_runs)."""
    return query(
        database,
        "SELECT run_id, dataset, split_part(command, ' ', 1) AS command, status, started_at, "
        "finished_at, error_message FROM pipeline_runs "
        "ORDER BY started_at DESC, run_id DESC LIMIT ?",
        [limit],
    )


# --- Dataset Health (Step 10.3) ---------------------------------------------------------------

TREND_METRICS = {  # gold_trends metric: what it is called on the page
    "rows_per_day": "Rows per day",
    "quarantine_rate": "Quarantine rate",
    "warned_rate": "Warned rate",
    "null_rate": "NULL rate of a column",
    "psi": "Drift (PSI) of a column",
    "rule_failure_rate": "Failure rate of a rule",
    "ml_flag_rate": "Rows flagged by the model",
    "score": "Quality score",
}


def score_parts(database: Path, dataset: str, period: str) -> pd.DataFrame:
    """The month's score by category: weight, pass rate and where it comes from."""
    return query(
        database,
        """
        SELECT p.category, p.weight, p.pass_rate, p.source
        FROM quality_score_parts p
        JOIN gold_dataset_health h USING (dataset, period, sha256)
        WHERE p.dataset = ? AND p.period = ?
        ORDER BY p.pass_rate, p.weight DESC, p.category
        """,
        [dataset, period],
    )


def checks(database: Path, dataset: str, period: str) -> pd.DataFrame:
    """Every dataset check of the month's latest check run, failures first."""
    return query(
        database,
        f"""
        SELECT r."check", r.column_name, r.status, r.severity, r.observed, r.expected, r.detail
        FROM dataset_check_results r
        JOIN gold_dataset_health h
          ON h.dataset = r.dataset AND h.period = r.period AND h.check_run_id = r.run_id
        WHERE r.dataset = ? AND r.period = ?
        ORDER BY r.status = 'pass', r.status = 'skipped', {_severity_order("r.severity")},
                 r."check", r.column_name
        """,
        [dataset, period],
    )


def column_profile(database: Path, dataset: str, period: str) -> pd.DataFrame:
    """Each column's NULL rate and drift in the month, failures first."""
    return query(
        database,
        """
        SELECT column_name, null_rate, null_rate_baseline, null_rate_status, null_rate_severity,
               mean, median, p99, psi, drift_status, drift_severity
        FROM gold_column_profile WHERE dataset = ? AND period = ?
        ORDER BY coalesce(null_rate_status = 'fail' OR drift_status = 'fail', false) DESC,
                 column_name
        """,
        [dataset, period],
    )


def trend_items(database: Path, dataset: str) -> dict[str, list[str]]:
    """The measures of the dataset in gold_trends: metric -> its items ('' for none)."""
    frame = query(
        database,
        "SELECT DISTINCT metric, coalesce(item, '') AS item FROM gold_trends WHERE dataset = ? "
        "ORDER BY 1, 2",
        [dataset],
    )
    items: dict[str, list[str]] = {}
    for metric, item in frame.itertuples(index=False):
        items.setdefault(metric, []).append(item)
    return {m: items[m] for m in TREND_METRICS if m in items}


def trend(database: Path, dataset: str, metric: str, item: str = "") -> pd.DataFrame:
    """One measure month by month, with its baseline and check result where it has one."""
    return query(
        database,
        """
        SELECT period, unit, value, baseline, baseline_trusted, check_status, severity
        FROM gold_trends
        WHERE dataset = ? AND metric = ? AND coalesce(item, '') = ?
        ORDER BY period
        """,
        [dataset, metric, item],
    )


def bin_labels(edges: list[float]) -> list[str]:
    """Labels of the drift bins: below the first edge, between two edges, from the last edge."""

    def number(x: float) -> str:
        return f"{x:g}"

    inner = [f"{number(a)} to {number(b)}" for a, b in zip(edges, edges[1:], strict=False)]
    return [f"below {number(edges[0])}", *inner, f"{number(edges[-1])} and more"]


def distribution(database: Path, dataset: str, column: str, period: str) -> pd.DataFrame:
    """Share of the rows per drift bin in the month and in the month before (if it has one)."""
    edges = drift_bins(dataset).get(column)
    if not edges:
        return pd.DataFrame(columns=["bin", "period", "rows", "share"])
    frame = query(
        database,
        """
        SELECT period, histogram FROM gold_column_profile
        WHERE dataset = ? AND column_name = ? AND histogram IS NOT NULL AND period <= ?
        ORDER BY period DESC LIMIT 2
        """,
        [dataset, column, period],
    )
    rows = []
    for month, counts in frame.itertuples(index=False):
        labels = bin_labels(edges)
        if len(labels) != len(counts):  # the bins were changed after Gold was built
            labels = [f"bin {i}" for i in range(len(counts))]
        total = sum(counts) or 1
        rows += [(label, month, n, n / total) for label, n in zip(labels, counts, strict=True)]
    return pd.DataFrame(rows, columns=["bin", "period", "rows", "share"])


def binned_columns(dataset: str) -> list[str]:
    return list(drift_bins(dataset))


def schema_changes(database: Path, dataset: str) -> pd.DataFrame:
    """Every column change the schema check saw, newest month first."""
    return query(
        database,
        f"""
        SELECT period, column_name, change, severity, previous_period, detail
        FROM gold_schema_history WHERE dataset = ? AND change != 'unchanged'
        ORDER BY period DESC, {_severity_order("severity")}, column_key
        """,
        [dataset],
    )
