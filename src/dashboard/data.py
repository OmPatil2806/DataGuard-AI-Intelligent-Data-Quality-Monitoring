"""What the dashboard shows, read from the Gold tables (Phase 10).

No Streamlit here: every function takes the DuckDB file and returns a DataFrame or a dict, so
the queries are tested without a browser. The database is opened read-only and closed after
each query: these reads never change it, and they hold DuckDB's file lock only for the
milliseconds a query takes, so the pipeline can write in between. The dashboard's only writes,
the decisions people take on the Incidents page, are in dashboard/actions.py.

The dashboard reads Gold, plus the control tables Gold does not copy: pipeline_runs (when the
pipeline last ran), the incident tables (so a decision on the Incidents page shows at once),
alert_log (what was alerted), and for one month the checks of its latest run
(dataset_check_results) and its score by category (quality_score_parts). The Quarantine
Explorer and the row trace read the month's Silver and Quarantine files themselves.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from common import db
from common.config import SEVERITIES, load_severity_weights
from common.db import INCIDENT_STATUSES
from gold import lineage as lineage_module
from ingestion.bronze_loader import read_bronze_sql
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
    "gold_lineage",
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
        "SELECT severity, count(*) AS n FROM incidents "
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
    return incidents(database, dataset, ("OPEN", "INVESTIGATING")).head(limit)


def incidents(database: Path, dataset: str, statuses: tuple[str, ...] = INCIDENT_STATUSES
              ) -> pd.DataFrame:  # fmt: skip
    """The dataset's incidents with these statuses, most severe and newest first.

    Read from the incident tables, not Gold, so a decision taken on the Incidents page shows at
    once (Gold catches up at the next rebuild).
    """
    if not statuses:
        return query(database, "SELECT * FROM incidents WHERE false")
    marks = ", ".join("?" * len(statuses))
    return query(
        database,
        f"""
        SELECT i.incident_id, i.severity, i.status, i.title, i.source, i.rule_id, i.column_name,
               i.first_seen_period, i.last_seen_period, i.occurrences AS months_seen,
               i.resolved_period, v.headline, v.top_cause, v.confidence, v.explained,
               (SELECT count(*) FROM investigation_actions a WHERE a.incident_id = i.incident_id
                AND a.status = 'pending_approval') AS actions_pending
        FROM incidents i LEFT JOIN investigations v USING (incident_id)
        WHERE i.dataset = ? AND i.status IN ({marks})
        ORDER BY i.status = 'RESOLVED', {_severity_order("i.severity")},
                 i.last_seen_period DESC, i.incident_id
        """,
        [dataset, *statuses],
    )


def incident_detail(database: Path, incident_id: str) -> dict[str, Any]:
    """One incident: its row, its investigation (causes, where/when), actions and history."""
    rows = query(database, "SELECT * FROM incidents WHERE incident_id = ?", [incident_id])
    if rows.empty:
        raise DashboardError(f"No incident {incident_id}.")
    found = query(
        database,
        "SELECT headline, explained, confidence, where_verdict, explanation, report_path "
        "FROM investigations WHERE incident_id = ?",
        [incident_id],
    )
    investigation = None
    if not found.empty:
        row = found.iloc[0]
        explanation = json.loads(row["explanation"]) if row["explanation"] else {}
        investigation = {
            "headline": row["headline"],
            "explained": bool(row["explained"]),
            "confidence": row["confidence"],
            "where": row["where_verdict"],
            "causes": explanation.get("causes") or [],
            "report_path": row["report_path"],
        }
    return {
        "incident": rows.iloc[0].to_dict(),
        "investigation": investigation,
        "actions": query(
            database,
            "SELECT action_id, kind, text, for_cause, status, decided_by, decided_at "
            "FROM investigation_actions WHERE incident_id = ? ORDER BY created_at, action_id",
            [incident_id],
        ),
        "events": query(
            database,
            "SELECT seq, event, period, severity, status_after, detail, created_at "
            "FROM incident_events WHERE incident_id = ? ORDER BY seq DESC",
            [incident_id],
        ),
    }


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


# --- Quarantine Explorer and Lineage (Step 10.5) ----------------------------------------------

VIEWS = {  # which problem rows: what the page calls them
    "quarantine": "Quarantined rows (kept out of Silver)",
    "warned": "Warned rows (kept in Silver with a warning)",
    "ml": "Rows flagged by the anomaly model (in Silver)",
}
# Bookkeeping columns of the Silver and Quarantine files, left out of the sample rows.
HIDDEN = ("source_file", "ingestion_timestamp", "pipeline_run_id", "silver_run_id", "period",
          "version", "quarantine_rules")  # fmt: skip


def _file(path: str | None) -> str | None:
    """SQL reading one Silver or Quarantine file, or None when there is none on disk."""
    absolute = lineage_module._absolute(path)
    return read_bronze_sql(absolute) if absolute is not None and absolute.exists() else None


def month_files(database: Path, dataset: str, period: str) -> dict[str, str | None]:
    """The SQL sources of the month's current Silver and Quarantine files (gold_lineage)."""
    found = query(
        database,
        "SELECT silver_path, quarantine_path FROM gold_lineage WHERE dataset = ? AND period = ?",
        [dataset, period],
    )
    if found.empty:
        return {"silver": None, "quarantine": None}
    row = found.iloc[0]
    return {"silver": _file(row["silver_path"]), "quarantine": _file(row["quarantine_path"])}


def _count(database: Path, sql: str, params: list[Any] | None = None) -> int:
    return int(query(database, f"SELECT count(*) AS n {sql}", params)["n"][0])


# The anomaly model's flags of the month's current file version.
ML_FLAGS = (
    "ml_anomalies a JOIN gold_lineage l USING (dataset, period, sha256) "
    "WHERE a.dataset = ? AND a.period = ?"
)


def problem_counts(database: Path, dataset: str, period: str) -> dict[str, int | None]:
    """Rows of the month per view; None when the file is not there (not processed)."""
    files = month_files(database, dataset, period)
    quarantine, silver = files["quarantine"], files["silver"]
    return {
        "quarantine": _count(database, f"FROM {quarantine}") if quarantine else None,
        "warned": _count(database, f"FROM {silver} WHERE len(failed_rules) > 0")
        if silver
        else None,
        "ml": _count(database, f"FROM {ML_FLAGS}", [dataset, period]),
    }


def rule_counts(database: Path, dataset: str, period: str) -> pd.DataFrame:
    """Rows failing each rule in the month: in Quarantine, and warned in Silver.

    A row can fail several rules, so the counts can add up to more than the rows."""
    files = month_files(database, dataset, period)
    parts = [
        f"SELECT unnest(failed_rules) AS rule_id, '{layer}' AS layer FROM {source}"
        for layer, source in (("quarantine", files["quarantine"]), ("warned", files["silver"]))
        if source
    ]
    if not parts:
        return pd.DataFrame(columns=["rule_id", "quarantine", "warned"])
    return query(
        database,
        f"""
        SELECT rule_id, count(*) FILTER (layer = 'quarantine') AS quarantine,
               count(*) FILTER (layer = 'warned') AS warned
        FROM ({" UNION ALL ".join(parts)}) GROUP BY rule_id
        ORDER BY quarantine + warned DESC, rule_id
        """,
    )


def problem_rows(
    database: Path, dataset: str, period: str, view: str, rule: str | None = None,
    limit: int = 200,
) -> tuple[pd.DataFrame, int]:  # fmt: skip
    """Up to `limit` rows of one view (optionally failing one rule), and how many there are."""
    if view not in VIEWS:
        raise DashboardError(f"unknown view {view}")
    files = month_files(database, dataset, period)
    source = files["quarantine" if view == "quarantine" else "silver"]
    if source is None:
        return pd.DataFrame(), 0
    where, params = [], []
    if view == "warned":
        where.append("len(f.failed_rules) > 0")
    if rule:
        where.append("list_contains(f.failed_rules, ?)")
        params.append(rule)
    if view == "ml":
        sql = (
            f"FROM {source} f JOIN (SELECT a.bronze_row_id, a.score AS anomaly_score, "
            f"a.reasons AS model_reasons FROM {ML_FLAGS}) m USING (bronze_row_id)"
        )
        params = [dataset, period, *params]
        order = "m.anomaly_score DESC, bronze_row_id"
    else:
        sql, order = f"FROM {source} f", "bronze_row_id"
    condition = f" WHERE {' AND '.join(where)}" if where else ""
    total = _count(database, f"{sql}{condition}", params)
    rows = query(database, f"SELECT * {sql}{condition} ORDER BY {order} LIMIT ?", [*params, limit])
    rows = rows.drop(columns=[c for c in HIDDEN if c in rows.columns])
    for column in ("failed_rules", "reasons"):
        rows[column] = rows[column].map(lambda v: "; ".join(v) if v is not None else "")
    first = [c for c in ("bronze_row_id", "anomaly_score", "model_reasons", "reasons",
                         "failed_rules") if c in rows.columns]  # fmt: skip
    return rows[first + [c for c in rows.columns if c not in first]], total


def lineage(database: Path, dataset: str) -> pd.DataFrame:
    """Every month: source file, version, Bronze/Silver/Quarantine counts and their check."""
    return query(
        database,
        """
        SELECT period, file_name, source_url, sha256, size_bytes, versions_registered,
               manifest_rows, bronze_rows, silver_rows, quarantine_rows, logged_silver_rows,
               logged_quarantine_rows, status, detail, bronze_path, silver_path,
               quarantine_path, loaded_run_id, silver_run_id, rule_run_id, check_run_id, scored
        FROM gold_lineage WHERE dataset = ? ORDER BY period
        """,
        [dataset],
    )


def trace_row(database: Path, dataset: str, period: str, row_id: int) -> dict[str, Any]:
    """One row from its source file to Silver or Quarantine (gold/lineage.py trace)."""
    try:
        with db.connect(database, read_only=True) as con:
            return lineage_module.trace(con, period, int(row_id), dataset)
    except lineage_module.TraceError as exc:
        raise DashboardError(str(exc)) from exc
    except FileNotFoundError as exc:
        raise DashboardError(f"No database at {database}.") from exc
    except duckdb.IOException as exc:
        raise DashboardError(
            "The database is busy (the pipeline is writing to it). Refresh in a moment."
        ) from exc
