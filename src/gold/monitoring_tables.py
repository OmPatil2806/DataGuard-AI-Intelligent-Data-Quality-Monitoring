"""Gold monitoring tables: small summary tables in the DuckDB file, read by the dashboard.

    python src/gold/monitoring_tables.py           # rebuild every Gold table in one tracked run
    python src/gold/monitoring_tables.py --list    # the Gold tables and what they hold

Gold is always rebuilt in full from the control tables (file_manifest, silver_log, checks,
incidents, scores), never updated piece by piece, so it cannot drift away from them. All
tables of a build are replaced in one transaction: a reader sees either the previous build or
the new one, and a failed build leaves the previous one untouched.

Every build is recorded in gold_build_log with the row count and a checksum per table. The
checksum leaves out the table's volatile columns (run ids, timestamps), so building twice from
the same data gives the same checksums.

Exit codes: 0 success, 1 build failed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

if __package__ in (None, ""):
    # Allow running this file directly: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import db  # noqa: E402
from common.config import SEVERITIES, load_datasets_config, load_severity_weights  # noqa: E402
from common.runs import track_run, utc_now  # noqa: E402
from gold import lineage  # noqa: E402
from quality import baselines  # noqa: E402
from quality.dataset_checks import schema_changes, worst  # noqa: E402

RUN_DATASET = "gold"  # pipeline_runs.dataset of a Gold build (Gold covers every dataset)
NAME = re.compile(r"^gold_[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class GoldTable:
    name: str
    description: str
    # A SELECT over the control tables, or a function called with the connection at build time
    # that returns one (it may register a DataFrame computed in Python and select from it).
    sql: str | Callable[[duckdb.DuckDBPyConnection], str]
    order_by: str  # stored order, so the table reads the same after every build
    volatile: tuple[str, ...] = ()  # columns left out of the checksum

    def __post_init__(self) -> None:
        if not NAME.match(self.name) or self.name == "gold_build_log":
            raise ValueError(f"invalid Gold table name '{self.name}'")

    def query(self, con: duckdb.DuckDBPyConnection) -> str:
        return self.sql(con) if callable(self.sql) else self.sql


@dataclass(frozen=True)
class BuiltTable:
    name: str
    rows: int
    checksum: str


# The spine of Gold: one row per dataset and month, for the current Bronze version.
MONTHS = GoldTable(
    name="gold_months",
    description="One row per dataset and month: the current Bronze file version",
    sql="""
        SELECT dataset, period, sha256, file_name, source_url, row_count AS bronze_rows,
               size_bytes, loaded_run_id, registered_at
        FROM file_manifest
        WHERE status = 'LOADED' AND bronze_path IS NOT NULL
        QUALIFY row_number() OVER (PARTITION BY dataset, period ORDER BY registered_at DESC) = 1
    """,
    order_by="dataset, period",
    volatile=("loaded_run_id", "registered_at"),
)


def _rank(column: str) -> str:
    """critical 4 ... low 1, else 0 (the order of SEVERITIES)."""
    cases = " ".join(f"WHEN '{s}' THEN {len(SEVERITIES) - i}" for i, s in enumerate(SEVERITIES))
    return f"CASE {column} {cases} ELSE 0 END"


def dataset_health_sql(health: dict | None = None) -> str:
    """One row per month of gold_months: rows, score, findings and a health label."""
    health = health or load_severity_weights().get("health") or {}
    warning_below = float(health.get("warning_below", 95))
    critical_below = float(health.get("critical_below", 80))
    worst = "greatest(coalesce(c.worst_check, 0), coalesce(f.worst_incident, 0))"
    names = " ".join(f"WHEN {len(SEVERITIES) - i} THEN '{s}'" for i, s in enumerate(SEVERITIES))
    return f"""
        WITH silver AS (
            SELECT dataset, period, sha256, run_id, silver_rows, quarantine_rows, warned_rows
            FROM silver_log WHERE status = 'processed'
            QUALIFY row_number() OVER (
                PARTITION BY dataset, period, sha256 ORDER BY logged_at DESC, run_id DESC) = 1
        ),
        check_run AS (
            SELECT dataset, period, sha256, max(run_id) AS run_id
            FROM dataset_check_results GROUP BY ALL
        ),
        checks AS (
            SELECT r.dataset, r.period, r.sha256, r.run_id,
                   count_if(r.status <> 'skipped') AS checks_run,
                   count_if(r.status = 'fail') AS checks_failed,
                   {", ".join(f"count_if(r.severity = '{s}') AS failed_{s}" for s in SEVERITIES)},
                   max({_rank("r.severity")}) AS worst_check
            FROM dataset_check_results r JOIN check_run USING (dataset, period, sha256, run_id)
            GROUP BY ALL
        ),
        found AS (
            SELECT i.dataset, e.period,
                   count(DISTINCT e.incident_id) AS incidents_found,
                   count_if(e.event = 'opened') AS incidents_opened,
                   count(DISTINCT e.incident_id) FILTER (WHERE i.status <> 'RESOLVED')
                       AS incidents_active,
                   max({_rank("e.severity")}) AS worst_incident
            FROM incident_events e JOIN incidents i USING (incident_id)
            WHERE e.event IN ('opened', 'seen_again')
            GROUP BY ALL
        ),
        ml AS (  -- the latest anomaly scoring of the version (Phase 7)
            SELECT dataset, period, sha256, model_id, rows_flagged, flag_rate
            FROM ml_month_summary
            QUALIFY row_number() OVER (
                PARTITION BY dataset, period, sha256 ORDER BY scored_at DESC, model_id) = 1
        )
        SELECT m.dataset, m.period, m.sha256, m.bronze_rows,
               s.silver_rows, s.quarantine_rows, s.warned_rows,
               round(100.0 * s.quarantine_rows / nullif(m.bronze_rows, 0), 4) AS quarantine_pct,
               q.score,
               ml.rows_flagged AS ml_rows_flagged, ml.flag_rate AS ml_flag_rate,
               ml.model_id AS ml_model_id,
               coalesce(c.checks_run, 0)::BIGINT AS checks_run,
               coalesce(c.checks_failed, 0)::BIGINT AS checks_failed,
               {", ".join(f"coalesce(c.failed_{s}, 0)::BIGINT AS failed_{s}" for s in SEVERITIES)},
               coalesce(f.incidents_found, 0)::BIGINT AS incidents_found,
               coalesce(f.incidents_opened, 0)::BIGINT AS incidents_opened,
               coalesce(f.incidents_active, 0)::BIGINT AS incidents_active,
               CASE {worst} {names} END AS worst_severity,
               CASE
                   WHEN q.score IS NULL THEN 'not_scored'
                   WHEN q.score < {critical_below} OR {worst} = {len(SEVERITIES)} THEN 'critical'
                   WHEN q.score < {warning_below} OR {worst} >= {len(SEVERITIES) - 2}
                       THEN 'warning'
                   ELSE 'healthy'
               END AS health,
               s.run_id AS silver_run_id, c.run_id AS check_run_id
        FROM gold_months m
        LEFT JOIN silver s USING (dataset, period, sha256)
        LEFT JOIN quality_scores q USING (dataset, period, sha256)
        LEFT JOIN checks c USING (dataset, period, sha256)
        LEFT JOIN found f USING (dataset, period)
        LEFT JOIN ml USING (dataset, period, sha256)
    """


DATASET_HEALTH = GoldTable(
    name="gold_dataset_health",
    description="One row per dataset and month: rows, score, findings and a health label",
    sql=lambda con: dataset_health_sql(),
    order_by="dataset, period",
    volatile=("silver_run_id", "check_run_id"),
)

# Long format: one row per dataset, month, metric and item (a column, a rule id, or '').
# baseline and change only where a dataset check compared the month with one; change is
# value - baseline and only filled for a trusted baseline (as the checks use it). previous is
# the value of the month before in the table (months can be far apart, as 2019-01 and 2023-01).
TRENDS = GoldTable(
    name="gold_trends",
    description="Metrics per month next to their baseline: rows, rates, score, NULLs, PSI, rules",
    sql="""
        WITH h AS (SELECT * FROM gold_dataset_health),
        metric AS (
            SELECT h.dataset, h.period, d.metric, d.column_name AS item, d.value
            FROM h JOIN dataset_metrics d USING (dataset, period, sha256)
            WHERE d.metric IN ('rows_per_day', 'null_rate')
        ),
        base AS (
            SELECT b.dataset, b.period, b.metric, b.column_name AS item, b.value, b.trusted
            FROM h JOIN baselines b
              ON b.run_id = h.check_run_id AND b.dataset = h.dataset AND b.period = h.period
            WHERE b.metric IN ('rows_per_day', 'null_rate')
        ),
        check_of AS (
            SELECT h.dataset, h.period, r."check", r.column_name, r.status, r.severity, r.observed
            FROM h JOIN dataset_check_results r
              ON r.run_id = h.check_run_id AND r.dataset = h.dataset AND r.period = h.period
              AND r.sha256 = h.sha256
        ),
        checked AS (
            SELECT m.dataset, m.period, m.metric, m.item, m.value,
                   b.value AS baseline, b.trusted AS baseline_trusted,
                   c.status AS check_status, c.severity
            FROM metric m
            LEFT JOIN base b USING (dataset, period, metric, item)
            LEFT JOIN check_of c
              ON c.dataset = m.dataset AND c.period = m.period AND c.column_name = m.item
              AND c."check" = CASE m.metric WHEN 'rows_per_day' THEN 'volume' ELSE 'null_rate' END
            UNION ALL
            SELECT dataset, period, 'psi', column_name,
                   CAST(regexp_extract(observed, '^PSI ([0-9.]+)', 1) AS DOUBLE),
                   NULL, NULL, status, severity
            FROM check_of WHERE "check" = 'drift' AND status <> 'skipped'
        ),
        rule_run AS (
            SELECT l.* FROM h JOIN rule_run_log l USING (dataset, period, sha256)
            QUALIFY dense_rank() OVER (PARTITION BY l.dataset, l.period ORDER BY l.run_id DESC) = 1
        ),
        derived AS (
            SELECT dataset, period, 'quarantine_rate' AS metric, '' AS item,
                   quarantine_rows / nullif(bronze_rows, 0) AS value
            FROM h WHERE silver_rows IS NOT NULL
            UNION ALL
            SELECT dataset, period, 'warned_rate', '', warned_rows / nullif(bronze_rows, 0)
            FROM h WHERE silver_rows IS NOT NULL
            UNION ALL
            SELECT dataset, period, 'score', '', score FROM h WHERE score IS NOT NULL
            UNION ALL
            SELECT dataset, period, 'ml_flag_rate', '', ml_flag_rate
            FROM h WHERE ml_flag_rate IS NOT NULL
            UNION ALL
            SELECT dataset, period, 'rule_failure_rate', rule_id,
                   rows_failed / nullif(rows_checked, 0)
            FROM rule_run
        ),
        rows AS (
            SELECT * FROM checked
            UNION ALL BY NAME
            SELECT *, NULL::DOUBLE AS baseline, NULL::BOOLEAN AS baseline_trusted,
                   NULL::VARCHAR AS check_status, NULL::VARCHAR AS severity
            FROM derived
        )
        SELECT dataset, period, metric, item,
               CASE metric WHEN 'rows_per_day' THEN 'rows/day' WHEN 'score' THEN 'score'
                           WHEN 'psi' THEN 'psi' ELSE 'share' END AS unit,
               value, baseline, baseline_trusted,
               CASE WHEN baseline_trusted THEN value - baseline END AS change,
               lag(value) OVER (PARTITION BY dataset, metric, item ORDER BY period) AS previous,
               check_status, severity
        FROM rows
    """,
    order_by="dataset, metric, item, period",
)

SCHEMA_HISTORY_COLUMNS = {
    "dataset": "string",
    "period": "string",
    "column_key": "string",  # lower-case name: the same column across months
    "column_name": "string",  # as delivered; the standard or last name when absent
    "position": "Int64",  # 1-based position in the file; empty when absent
    "dtype": "string",
    "in_standard": "boolean",
    "previous_period": "string",
    "previous_name": "string",
    "previous_dtype": "string",
    "change": "string",  # unchanged, or the schema check's kinds joined by ', '
    "severity": "string",
    "detail": "string",
}


def schema_history_frame(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Every column per month of gold_months: delivered, missing or removed, with the change
    against the previous month as the schema check sees it (same rules, same previous month)."""
    months = con.execute(
        "SELECT m.dataset, m.period, d.text_value FROM gold_months m JOIN dataset_metrics d "
        "USING (dataset, period, sha256) WHERE d.metric = 'raw_schema' ORDER BY 1, 2"
    ).fetchall()
    by_dataset: dict[str, dict[str, list[tuple[str, str]]]] = {}
    for dataset, period, schema in months:
        by_dataset.setdefault(dataset, {})[period] = [tuple(c) for c in json.loads(schema)]
    config = load_datasets_config() if by_dataset else None

    records = []
    for name, schemas in by_dataset.items():
        dataset = config.dataset(name)
        settings = baselines.settings_for(name)
        # source names of the standard columns (as in the schema check)
        standard = {s.lower(): s for _, names in dataset.raw_columns() for s in names}
        for period, current in schemas.items():
            earlier = baselines.earlier_months(list(schemas), period, settings)
            previous_period = earlier[-1] if earlier else None
            previous = schemas[previous_period] if previous_period else None
            changes: dict[str, list] = {}
            for change in schema_changes(current, previous, dataset):
                changes.setdefault(change.column.lower(), []).append(change)
            now = {c.lower(): (i, c, t) for i, (c, t) in enumerate(current, start=1)}
            before = {c.lower(): (c, t) for c, t in previous or []}
            absent = [k for k in dict.fromkeys([*standard, *before]) if k not in now]
            for key in [*now, *absent]:
                found = changes.get(key, [])
                if key in now:
                    position, column, dtype = now[key]
                elif found:  # missing standard column or removed extra column
                    position, column, dtype = None, before.get(key, (standard.get(key),))[0], None
                else:
                    continue
                old = before.get(key, (None, None))
                records.append(
                    {
                        "dataset": name,
                        "period": period,
                        "column_key": key,
                        "column_name": column,
                        "position": position,
                        "dtype": dtype,
                        "in_standard": key in standard,
                        "previous_period": previous_period,
                        "previous_name": old[0],
                        "previous_dtype": old[1],
                        "change": ", ".join(c.kind for c in found) or "unchanged",
                        "severity": worst([c.severity for c in found]) if found else None,
                        "detail": "; ".join(c.detail for c in found) or None,
                    }
                )
    frame = pd.DataFrame(records, columns=list(SCHEMA_HISTORY_COLUMNS))
    return frame.astype(SCHEMA_HISTORY_COLUMNS)


def _schema_history_sql(con: duckdb.DuckDBPyConnection) -> str:
    con.register("gold_schema_history_frame", schema_history_frame(con))
    return "SELECT * FROM gold_schema_history_frame"


SCHEMA_HISTORY = GoldTable(
    name="gold_schema_history",
    description="Per month and column: name and type as delivered, and the change to last month",
    sql=_schema_history_sql,
    order_by="dataset, period, position NULLS LAST, column_key",
)

# One row per dataset, month and column with a NULL-rate metric (every standard column);
# mean, median, p99, histogram (rows per fixed bin) and PSI only for the binned columns.
COLUMN_PROFILE = GoldTable(
    name="gold_column_profile",
    description="Per month and column: NULL share, mean, median, p99, histogram and PSI",
    sql="""
        WITH stats AS (
            SELECT m.dataset, m.period, d.column_name,
                   max(d.value) FILTER (WHERE d.metric = 'null_rate') AS null_rate,
                   max(d.value) FILTER (WHERE d.metric = 'mean') AS mean,
                   max(d.value) FILTER (WHERE d.metric = 'median') AS median,
                   max(d.value) FILTER (WHERE d.metric = 'p99') AS p99,
                   list(CAST(d.value AS BIGINT) ORDER BY TRY_CAST(d.bin AS INTEGER))
                       FILTER (WHERE d.metric = 'histogram') AS histogram
            FROM gold_months m JOIN dataset_metrics d USING (dataset, period, sha256)
            WHERE d.metric IN ('null_rate', 'mean', 'median', 'p99', 'histogram')
              AND d.column_name <> ''
            GROUP BY ALL
        ),
        nulls AS (
            SELECT dataset, period, item AS column_name, baseline AS null_rate_baseline,
                   check_status AS null_rate_status, severity AS null_rate_severity
            FROM gold_trends WHERE metric = 'null_rate'
        ),
        drift AS (
            SELECT dataset, period, item AS column_name, value AS psi,
                   check_status AS drift_status, severity AS drift_severity
            FROM gold_trends WHERE metric = 'psi'
        )
        SELECT s.dataset, s.period, s.column_name,
               s.null_rate, n.null_rate_baseline, n.null_rate_status, n.null_rate_severity,
               s.mean, s.median, s.p99,
               CASE WHEN len(s.histogram) > 0 THEN s.histogram END AS histogram,
               d.psi, d.drift_status, d.drift_severity
        FROM stats s
        LEFT JOIN nulls n USING (dataset, period, column_name)
        LEFT JOIN drift d USING (dataset, period, column_name)
        WHERE s.null_rate IS NOT NULL
    """,
    order_by="dataset, column_name, period",
)


def _month_number(column: str) -> str:
    """YYYY-MM as a month count, so months can be subtracted."""
    return (
        f"(CAST(substr({column}, 1, 4) AS INTEGER) * 12 + CAST(substr({column}, 6, 2) AS INTEGER))"
    )


# Shared by both incident tables: when an incident stopped being active. The lifecycle stores
# the month it resolved an incident in; an incident resolved by hand has no month, so it counts
# as resolved in the month of its last lifecycle event.
_RESOLVED = """
    last_event AS (
        SELECT incident_id, max(period) AS period FROM incident_events GROUP BY ALL
    ),
    resolved AS (
        SELECT i.incident_id,
               CASE WHEN i.status = 'RESOLVED'
                    THEN coalesce(i.resolved_period, l.period) END AS resolved_month,
               CASE WHEN i.status <> 'RESOLVED' THEN NULL
                    WHEN i.resolved_period IS NOT NULL THEN 'lifecycle' ELSE 'hand' END
                   AS resolved_by
        FROM incidents i LEFT JOIN last_event l USING (incident_id)
    )
"""

INCIDENT_SUMMARY = GoldTable(
    name="gold_incident_summary",
    description="One row per incident: status, how long it lasted, its history and recurrences",
    sql=lambda con: (
        f"""
        WITH RECURSIVE chain (incident_id, chain_root, chain_position) AS (
            SELECT incident_id, incident_id, 1 FROM incidents WHERE recurrence_of IS NULL
            UNION ALL
            SELECT i.incident_id, c.chain_root, c.chain_position + 1
            FROM incidents i JOIN chain c ON i.recurrence_of = c.incident_id
        ),
        {_RESOLVED},
        latest AS (SELECT dataset, max(period) AS period FROM gold_months GROUP BY ALL),
        history AS (
            SELECT incident_id,
                   count(*) AS events,
                   count_if(event = 'investigating') > 0 AS investigated,
                   count_if(event = 'note') AS notes
            FROM incident_events GROUP BY ALL
        )
        SELECT i.incident_id, i.dataset, i.source, i.rule_id, i.column_name, i.title,
               i.severity, i.status, i.first_seen_period, i.last_seen_period,
               r.resolved_month AS resolved_period, r.resolved_by,
               i.occurrences AS months_seen,
               {_month_number("coalesce(r.resolved_month, l.period, i.last_seen_period)")}
                   - {_month_number("i.first_seen_period")} + 1 AS months_active,
               {_month_number("r.resolved_month")} - {_month_number("i.first_seen_period")}
                   AS months_to_resolve,
               i.clean_streak, i.recurrence_of, c.chain_root, c.chain_position,
               coalesce(h.investigated, false) AS investigated,
               coalesce(h.notes, 0)::BIGINT AS notes, coalesce(h.events, 0)::BIGINT AS events,
               i.latest_observed, i.latest_expected, i.latest_detail
        FROM incidents i
        JOIN resolved r USING (incident_id)
        LEFT JOIN chain c USING (incident_id)
        LEFT JOIN latest l USING (dataset)
        LEFT JOIN history h USING (incident_id)
    """
    ),
    order_by="incident_id",
)

INCIDENT_MONTHLY = GoldTable(
    name="gold_incident_monthly",
    description="Per dataset and month: incidents opened, seen, resolved and active by severity",
    sql=lambda con: (
        f"""
        WITH {_RESOLVED},
        span AS (
            SELECT i.incident_id, i.dataset, i.first_seen_period, r.resolved_month
            FROM incidents i JOIN resolved r USING (incident_id)
        ),
        severity_by AS (  -- severity an incident had reached by each month (it only rises)
            SELECT m.dataset, m.period, s.incident_id,
                   max({_rank("e.severity")}) AS rank
            FROM gold_months m
            JOIN span s ON s.dataset = m.dataset AND s.first_seen_period <= m.period
                AND (s.resolved_month IS NULL OR s.resolved_month > m.period)
            JOIN incident_events e
              ON e.incident_id = s.incident_id AND e.period <= m.period
            GROUP BY ALL
        ),
        active AS (
            SELECT dataset, period, count(*) AS active_end,
                   {
            ", ".join(
                f"count_if(rank = {len(SEVERITIES) - i}) AS active_{s}"
                for i, s in enumerate(SEVERITIES)
            )
        }
            FROM severity_by GROUP BY ALL
        ),
        moved AS (
            SELECT i.dataset, e.period,
                   count_if(e.event = 'opened') AS opened,
                   count_if(e.event = 'opened' AND i.recurrence_of IS NOT NULL) AS recurrences,
                   count(DISTINCT e.incident_id)
                       FILTER (WHERE e.event IN ('opened', 'seen_again')) AS seen
            FROM incident_events e JOIN incidents i USING (incident_id)
            WHERE e.period IS NOT NULL
            GROUP BY ALL
        ),
        closed AS (
            SELECT dataset, resolved_month AS period, count(*) AS resolved
            FROM span WHERE resolved_month IS NOT NULL GROUP BY ALL
        )
        SELECT m.dataset, m.period,
               coalesce(o.opened, 0)::BIGINT AS opened,
               coalesce(o.recurrences, 0)::BIGINT AS recurrences,
               coalesce(o.seen, 0)::BIGINT AS seen,
               coalesce(c.resolved, 0)::BIGINT AS resolved,
               coalesce(a.active_end, 0)::BIGINT AS active_end,
               {", ".join(f"coalesce(a.active_{s}, 0)::BIGINT AS active_{s}" for s in SEVERITIES)}
        FROM gold_months m
        LEFT JOIN moved o USING (dataset, period)
        LEFT JOIN closed c USING (dataset, period)
        LEFT JOIN active a USING (dataset, period)
    """
    ),
    order_by="dataset, period",
)

LINEAGE = GoldTable(
    name="gold_lineage",
    description="Per dataset and month: source -> Bronze -> Silver/Quarantine, runs, row counts",
    sql=lineage.lineage_sql,
    order_by="dataset, period",
    volatile=lineage.VOLATILE,
)

INVESTIGATIONS = GoldTable(
    name="gold_investigations",
    description="One row per investigated incident: likely cause, where/when, actions, report",
    sql="""
        WITH acts AS (
            SELECT incident_id, count(*) AS actions,
                   count_if(status = 'pending_approval') AS actions_pending,
                   count_if(status = 'approved') AS actions_approved,
                   count_if(status = 'rejected') AS actions_rejected,
                   count_if(status = 'done') AS actions_done
            FROM investigation_actions GROUP BY ALL
        )
        SELECT v.incident_id, v.dataset, i.title, i.severity, i.status, v.period,
               v.playbook, v.explained, v.top_cause, v.confidence, v.headline, v.where_verdict,
               coalesce(a.actions, 0)::BIGINT AS actions,
               coalesce(a.actions_pending, 0)::BIGINT AS actions_pending,
               coalesce(a.actions_approved, 0)::BIGINT AS actions_approved,
               coalesce(a.actions_rejected, 0)::BIGINT AS actions_rejected,
               coalesce(a.actions_done, 0)::BIGINT AS actions_done,
               v.evidence_md5, v.report_path, v.run_id AS investigation_run_id
        FROM investigations v
        JOIN incidents i USING (incident_id)
        LEFT JOIN acts a USING (incident_id)
    """,
    order_by="incident_id",
    # the evidence hash covers delivery times, which differ between two loads of the same file
    volatile=("evidence_md5", "report_path", "investigation_run_id"),
)

# In build order: later tables may read earlier ones.
TABLES: list[GoldTable] = [
    MONTHS,
    DATASET_HEALTH,
    TRENDS,
    SCHEMA_HISTORY,
    COLUMN_PROFILE,
    INCIDENT_SUMMARY,
    INCIDENT_MONTHLY,
    LINEAGE,
    INVESTIGATIONS,
]


def checksum(con: duckdb.DuckDBPyConnection, table: str, volatile: tuple[str, ...] = ()) -> str:
    """MD5 over the table's rows as text, sorted, without the volatile columns."""
    columns = f"* EXCLUDE ({', '.join(volatile)})" if volatile else "*"
    return con.execute(
        "SELECT md5(coalesce(string_agg(CAST(t AS VARCHAR), chr(10) "
        f"ORDER BY CAST(t AS VARCHAR)), '')) FROM (SELECT {columns} FROM {table}) t"
    ).fetchone()[0]


def build(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    tables: list[GoldTable] | None = None,
    now: datetime | None = None,
) -> list[BuiltTable]:
    """Replace every Gold table and record the build, all in one transaction."""
    tables = TABLES if tables is None else tables
    names = [t.name for t in tables]
    if len(set(names)) != len(names):
        raise ValueError("Gold table names must be unique")
    now = now or utc_now()
    built = []
    con.execute("BEGIN TRANSACTION")
    try:
        for table in tables:
            con.execute(
                f"CREATE OR REPLACE TABLE {table.name} AS "
                f"SELECT * FROM ({table.query(con)}) ORDER BY {table.order_by}"
            )
            rows = con.execute(f"SELECT count(*) FROM {table.name}").fetchone()[0]
            built.append(BuiltTable(table.name, rows, checksum(con, table.name, table.volatile)))
        con.executemany(
            "INSERT INTO gold_build_log VALUES (?, ?, ?, ?, ?)",
            [[run_id, b.name, b.rows, b.checksum, now] for b in built],
        )
        con.execute("COMMIT")
    except BaseException:
        con.execute("ROLLBACK")
        raise
    return built


def rebuild(
    con: duckdb.DuckDBPyConnection, tables: list[GoldTable] | None = None
) -> tuple[str, list[BuiltTable]]:
    """Build Gold in a tracked run (pipeline_runs, dataset 'gold')."""
    with track_run(con, RUN_DATASET, command="monitoring_tables") as run:
        built = build(con, run.run_id, tables)
    return run.run_id, built


def latest_build(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        "SELECT * FROM gold_build_log WHERE run_id = (SELECT max(run_id) FROM gold_build_log) "
        "ORDER BY table_name"
    ).df()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild the Gold monitoring tables.")
    parser.add_argument("--database", type=Path, help="DuckDB file (default: paths.duckdb)")
    parser.add_argument("--list", action="store_true", help="list the Gold tables and exit")
    args = parser.parse_args(argv)
    if args.list:
        for table in TABLES:
            print(f"{table.name:<24} {table.description}")
        return 0
    with db.connect(args.database) as con:
        db.init_schema(con)
        try:
            run_id, built = rebuild(con)
        except duckdb.Error as exc:
            print(f"Gold build failed, previous Gold kept: {exc}")
            return 1
    print(f"{run_id}: {len(built)} Gold table(s) built")
    for b in built:
        print(f"  {b.name:<24} {b.rows:>8} row(s)  checksum {b.checksum[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
