"""Evidence pack: everything known about one incident, gathered in one place (Phase 8).

    python src/investigation/evidence_pack.py INC-0004       # print one incident's evidence
    python src/investigation/evidence_pack.py --all          # build and store every incident's

The pack answers the questions an investigator asks first, from the control tables and the
Silver and Quarantine files only (never Gold, so it can run before Gold is built):

    incident    what was found: rule or check, severity, status, first and last month, latest values
    definition  what the rule or check tests (configs/quality_rules.yml, the check, the ML features)
    history     the rule's or check's value in every month, so "since when" is visible
    events      the incident's own history (opened, seen again, clean, resolved)
    versions    the file versions of the months the incident was seen in
    impact      rows of the last month seen, how many are affected, the score and its drop
    similar     earlier or later incidents of the same problem, and others in the same month
    metrics     the column's NULL rate, mean, median and p99 per month (or rows, for volume)
    samples     up to five affected rows with their key values and reasons (row-level problems)
    concentration  where and when the affected rows sit (concentration.py), or "unknown"

Month-level findings (schema, volume, freshness, drift) have no affected rows, so no samples.
The JSON holds no build times, so building it twice from the same data gives the same text.
It is stored in investigation_evidence, one row per incident.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

if __package__ in (None, ""):
    # Allow running this file directly: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import db  # noqa: E402
from common.runs import utc_now  # noqa: E402
from incidents.incident_manager import Incident, IncidentError, get_incident  # noqa: E402
from investigation import concentration as conc  # noqa: E402
from quality.rules import load_rules  # noqa: E402
from silver.process_silver import current_silver  # noqa: E402

SAMPLE_ROWS = 5
SAMPLE_COLUMNS = (
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "VendorID",
    "PULocationID",
    "DOLocationID",
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "total_amount",
    "payment_type",
)
CHECK_DEFINITIONS = {
    "schema": "the month's columns and types against the expected schema (configs/datasets.yml)",
    "volume": "rows per day against the median of the earlier months",
    "freshness": "the delivery time against the month's expected arrival plus the grace period",
    "null_rate": "the column's share of NULL values against the earlier months",
    "drift": "the column's distribution against the earlier months (PSI over fixed bins)",
    "ml_anomaly": "the share of Silver rows the Isolation Forest flags as unusual",
}
ROW_LEVEL_CHECKS = ("null_rate", "ml_anomaly")


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _sql_path(path: Path) -> str:
    return "'" + path.as_posix().replace("'", "''") + "'"


def _plain(value: Any) -> Any:
    """A JSON-friendly value: timestamps and dates as ISO text, NaN, NaT and pd.NA as None."""
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat(sep=" ") if hasattr(value, "hour") else value.isoformat()
    if hasattr(value, "item"):  # numpy scalars
        return value.item()
    return value


def _json_default(value: Any) -> Any:
    plain = _plain(value)
    return str(value) if plain is value else plain


def is_row_level(incident: Incident) -> bool:
    """Whether the problem sits in individual rows (so they can be listed and sampled)."""
    return incident.source == "row_rule" or incident.rule_id in ROW_LEVEL_CHECKS


def latest_model(con: duckdb.DuckDBPyConnection, dataset: str, period: str) -> str | None:
    """The model of the latest scoring of a month."""
    row = con.execute(
        "SELECT model_id FROM ml_month_summary WHERE dataset = ? AND period = ? "
        "ORDER BY scored_at DESC LIMIT 1",
        [dataset, period],
    ).fetchone()
    return row[0] if row else None


def month_rows(
    con: duckdb.DuckDBPyConnection, incident: Incident, period: str
) -> tuple[str, str] | None:
    """The rows the incident's rule or check looked at in a month, as a SQL source, and the
    condition that marks a row as affected; None for a month-level finding or a month without
    Silver. Row rules and NULL rates look at Silver and Quarantine, the model at Silver only."""
    if not is_row_level(incident):
        return None
    files = current_silver(con, incident.dataset).get(period)
    if files is None:
        return None
    if incident.rule_id == "ml_anomaly":
        files = files[:1]
    source = (
        f"read_parquet([{', '.join(_sql_path(p) for p in files)}], union_by_name = true, "
        "hive_partitioning = false)"
    )
    if incident.source == "row_rule":
        rule = incident.rule_id.replace("'", "''")
        where = f"list_contains(failed_rules, '{rule}')"
    elif incident.rule_id == "null_rate":
        where = f"{_q(incident.column_name)} IS NULL"
    else:
        model = latest_model(con, incident.dataset, period)
        if model is None:
            return None
        sha = con.execute(
            "SELECT sha256 FROM ml_month_summary WHERE dataset = ? AND period = ? "
            "AND model_id = ? ORDER BY scored_at DESC LIMIT 1",
            [incident.dataset, period, model],
        ).fetchone()[0]
        where = (
            "bronze_row_id IN (SELECT bronze_row_id FROM ml_anomalies WHERE dataset = "
            f"'{incident.dataset}' AND period = '{period}' AND sha256 = '{sha}' "
            f"AND model_id = '{model}')"
        )
    return source, where


def affected_rows_sql(
    con: duckdb.DuckDBPyConnection, incident: Incident, period: str
) -> str | None:
    """SQL over the month's rows that show the incident's problem (see month_rows)."""
    rows = month_rows(con, incident, period)
    if rows is None:
        return None
    source, where = rows
    return f"SELECT * FROM {source} WHERE {where}"


# --- the sections -----------------------------------------------------------------------------


def incident_section(incident: Incident) -> dict[str, Any]:
    return {
        "incident_id": incident.incident_id,
        "dataset": incident.dataset,
        "source": incident.source,
        "rule_id": incident.rule_id,
        "column": incident.column_name,
        "title": incident.title,
        "severity": incident.severity,
        "status": incident.status,
        "first_seen": incident.first_seen_period,
        "last_seen": incident.last_seen_period,
        "resolved_in": incident.resolved_period,
        "occurrences": incident.occurrences,
        "recurrence_of": incident.recurrence_of,
        "observed": incident.latest_observed,
        "expected": incident.latest_expected,
        "detail": incident.latest_detail,
    }


def definition(con: duckdb.DuckDBPyConnection, incident: Incident) -> dict[str, Any]:
    """What the rule or check tests."""
    if incident.source == "row_rule":
        rules = {r.id: r for r in load_rules(incident.dataset, include_disabled=True)}
        rule = rules.get(incident.rule_id)
        if rule is None:
            return {"kind": "row_rule", "description": "rule no longer in quality_rules.yml"}
        return {
            "kind": "row_rule",
            "description": rule.description or rule.reason,
            "category": rule.category,
            "check": rule.check,
            "action": rule.action,
            "columns": rule.columns,
            "params": rule.params,
            "enabled": rule.enabled,
        }
    result: dict[str, Any] = {
        "kind": "ml" if incident.rule_id == "ml_anomaly" else "dataset_check",
        "description": CHECK_DEFINITIONS.get(incident.rule_id, incident.rule_id),
    }
    if incident.column_name:
        result["column"] = incident.column_name
    if incident.rule_id == "ml_anomaly":
        model = latest_model(con, incident.dataset, incident.last_seen_period)
        if model:
            meta = con.execute(
                "SELECT metadata FROM ml_models WHERE model_id = ?", [model]
            ).fetchone()
            result["model_id"] = model
            if meta:
                parsed = json.loads(meta[0])
                result["features"] = list(parsed.get("features", {}))
                result["threshold"] = parsed.get("threshold")
    return result


def _current_shas(con: duckdb.DuckDBPyConnection, dataset: str) -> dict[str, str]:
    return dict(
        con.execute(
            """
            SELECT period, sha256 FROM file_manifest
            WHERE dataset = ? AND status = 'LOADED' AND bronze_path IS NOT NULL
            QUALIFY row_number() OVER (PARTITION BY period ORDER BY registered_at DESC) = 1
            """,
            [dataset],
        ).fetchall()
    )


def history(con: duckdb.DuckDBPyConnection, incident: Incident) -> list[dict[str, Any]]:
    """The rule's or check's latest result in every month of the current file versions."""
    shas = _current_shas(con, incident.dataset)
    if incident.source == "row_rule":
        rows = con.execute(
            """
            SELECT period, sha256, rows_failed, rows_checked FROM rule_run_log
            WHERE dataset = ? AND rule_id = ?
            QUALIFY row_number() OVER (PARTITION BY period, sha256 ORDER BY run_id DESC) = 1
            ORDER BY period
            """,
            [incident.dataset, incident.rule_id],
        ).fetchall()
        return [
            {
                "period": p,
                "value": round(failed / checked, 6) if checked else None,
                "observed": f"{failed:,} of {checked:,} rows failed",
            }
            for p, sha, failed, checked in rows
            if shas.get(p) == sha
        ]
    if incident.rule_id == "ml_anomaly":
        model = latest_model(con, incident.dataset, incident.last_seen_period)
        rows = con.execute(
            """
            SELECT period, sha256, flag_rate, rows_flagged, rows_scored FROM ml_month_summary
            WHERE dataset = ? AND model_id = ?
            QUALIFY row_number() OVER (PARTITION BY period, sha256 ORDER BY scored_at DESC) = 1
            ORDER BY period
            """,
            [incident.dataset, model],
        ).fetchall()
        return [
            {
                "period": p,
                "value": round(rate, 6),
                "observed": f"{flagged:,} of {scored:,} rows flagged",
            }
            for p, sha, rate, flagged, scored in rows
            if shas.get(p) == sha
        ]
    rows = con.execute(
        """
        SELECT period, sha256, status, severity, observed, expected FROM dataset_check_results
        WHERE dataset = ? AND "check" = ? AND column_name = ?
        QUALIFY row_number() OVER (PARTITION BY period, sha256 ORDER BY run_id DESC) = 1
        ORDER BY period
        """,
        [incident.dataset, incident.rule_id, incident.column_name],
    ).fetchall()
    return [
        {
            "period": p,
            "status": status,
            "severity": severity,
            "observed": observed,
            "expected": expected,
        }
        for p, sha, status, severity, observed, expected in rows
        if shas.get(p) == sha
    ]


def events(con: duckdb.DuckDBPyConnection, incident_id: str) -> list[dict[str, Any]]:
    rows = con.execute(
        "SELECT event, period, run_id, severity, status_after, detail FROM incident_events "
        "WHERE incident_id = ? ORDER BY seq",
        [incident_id],
    ).fetchall()
    keys = ("event", "period", "run_id", "severity", "status", "detail")
    return [dict(zip(keys, row, strict=True)) for row in rows]


def versions(con: duckdb.DuckDBPyConnection, incident: Incident) -> list[dict[str, Any]]:
    """Every file version registered for the first and the last month the incident was seen."""
    periods = sorted({incident.first_seen_period, incident.last_seen_period})
    rows = con.execute(
        f"""
        SELECT period, file_name, sha256, size_bytes, row_count, status, source_url,
               loaded_run_id, delivered_at, expected_at
        FROM file_manifest WHERE dataset = ? AND period IN ({", ".join("?" * len(periods))})
        ORDER BY period, registered_at
        """,
        [incident.dataset, *periods],
    ).fetchall()
    keys = (
        "period", "file_name", "sha256", "size_bytes", "rows", "status", "source_url",
        "loaded_run_id", "delivered_at", "expected_at",
    )  # fmt: skip
    return [{k: _plain(v) for k, v in zip(keys, row, strict=True)} for row in rows]


def impact(con: duckdb.DuckDBPyConnection, incident: Incident) -> dict[str, Any]:
    """The size of the problem in the last month it was seen."""
    period = incident.last_seen_period
    result: dict[str, Any] = {"period": period}
    sha = _current_shas(con, incident.dataset).get(period)
    logged = con.execute(
        """
        SELECT rows, silver_rows, quarantine_rows FROM silver_log
        WHERE dataset = ? AND period = ? AND sha256 = ? ORDER BY logged_at DESC LIMIT 1
        """,
        [incident.dataset, period, sha],
    ).fetchone()
    if logged:
        result |= {"rows": logged[0], "silver_rows": logged[1], "quarantine_rows": logged[2]}
    sql = affected_rows_sql(con, incident, period)
    if sql is not None:
        affected = con.execute(f"SELECT count(*) FROM ({sql})").fetchone()[0]
        result["affected_rows"] = affected
        if logged and logged[0]:
            result["affected_share"] = round(affected / logged[0], 6)
    scores = con.execute(
        """
        SELECT period, score FROM quality_scores
        WHERE dataset = ? AND period <= ?
        QUALIFY row_number() OVER (PARTITION BY period ORDER BY computed_at DESC) = 1
        ORDER BY period DESC LIMIT 2
        """,
        [incident.dataset, period],
    ).fetchall()
    if scores and scores[0][0] == period:
        result["score"] = round(scores[0][1], 2)
        if len(scores) > 1:
            result["previous_score"] = {"period": scores[1][0], "score": round(scores[1][1], 2)}
            result["score_change"] = round(scores[0][1] - scores[1][1], 2)
    return result


def similar(con: duckdb.DuckDBPyConnection, incident: Incident) -> dict[str, Any]:
    """Incidents of the same problem (same rule or check and column), and the other incidents
    that were active in the month this one was first seen."""
    same = con.execute(
        """
        SELECT incident_id, severity, status, first_seen_period, last_seen_period
        FROM incidents WHERE dataset = ? AND source = ? AND rule_id = ? AND column_name = ?
        AND incident_id != ? ORDER BY incident_id
        """,
        [
            incident.dataset,
            incident.source,
            incident.rule_id,
            incident.column_name,
            incident.incident_id,
        ],
    ).fetchall()
    month = con.execute(
        """
        SELECT DISTINCT i.incident_id, i.title, i.severity, i.source, i.rule_id, i.column_name
        FROM incident_events e JOIN incidents i USING (incident_id)
        WHERE i.dataset = ? AND e.period IN (?, ?) AND e.event IN ('opened', 'seen_again')
        AND i.incident_id != ? ORDER BY i.incident_id
        """,
        [
            incident.dataset,
            incident.first_seen_period,
            incident.last_seen_period,
            incident.incident_id,
        ],
    ).fetchall()
    keys = ("incident_id", "severity", "status", "first_seen", "last_seen")
    month_keys = ("incident_id", "title", "severity", "source", "rule_id", "column")
    return {
        "same_problem": [dict(zip(keys, row, strict=True)) for row in same],
        "same_month": [dict(zip(month_keys, row, strict=True)) for row in month],
    }


COLUMN_METRICS = ("null_rate", "mean", "median", "p99")
VOLUME_METRICS = ("rows", "rows_per_day")


def metrics(con: duckdb.DuckDBPyConnection, incident: Incident) -> dict[str, list[dict]]:
    """The dataset metrics behind the finding, per month of the current file versions: the
    column's NULL rate, mean, median and 99th percentile, or the month's rows for volume."""
    if incident.column_name:
        names, column = COLUMN_METRICS, incident.column_name
    elif incident.rule_id == "volume":
        names, column = VOLUME_METRICS, ""
    else:
        return {}
    shas = _current_shas(con, incident.dataset)
    rows = con.execute(
        f"""
        SELECT metric, period, sha256, value FROM dataset_metrics
        WHERE dataset = ? AND column_name = ? AND bin = ''
        AND metric IN ({", ".join("?" * len(names))}) ORDER BY metric, period
        """,
        [incident.dataset, column, *names],
    ).fetchall()
    result: dict[str, list[dict]] = {}
    for metric, period, sha, value in rows:
        if shas.get(period) == sha and value is not None:
            result.setdefault(metric, []).append({"period": period, "value": round(value, 6)})
    return {m: result[m] for m in names if m in result}


def samples(
    con: duckdb.DuckDBPyConnection, incident: Incident, limit: int = SAMPLE_ROWS
) -> list[dict[str, Any]]:
    """Up to `limit` affected rows of the last month seen, in file order."""
    period = incident.last_seen_period
    sql = affected_rows_sql(con, incident, period)
    if sql is None:
        return []
    frame = con.execute(f"SELECT * FROM ({sql}) ORDER BY bronze_row_id LIMIT {int(limit)}").df()
    if frame.empty:
        return []
    ml_reasons: dict[int, str] = {}
    if incident.rule_id == "ml_anomaly":
        model = latest_model(con, incident.dataset, period)
        ml_reasons = dict(
            con.execute(
                "SELECT bronze_row_id, reasons FROM ml_anomalies WHERE dataset = ? AND "
                "period = ? AND model_id = ? AND list_contains(?, bronze_row_id)",
                [incident.dataset, period, model, frame["bronze_row_id"].tolist()],
            ).fetchall()
        )
    columns = [c for c in SAMPLE_COLUMNS if c in frame.columns]
    result = []
    for _, row in frame.iterrows():
        row_id = int(row["bronze_row_id"])
        reasons = list(row["reasons"]) if row.get("reasons") is not None else []
        if row_id in ml_reasons:
            reasons = [ml_reasons[row_id], *reasons]
        result.append(
            {
                "period": period,
                "bronze_row_id": row_id,
                "values": {c: _plain(row[c]) for c in columns},
                "failed_rules": list(row["failed_rules"]),
                "reasons": reasons,
            }
        )
    return result


def concentration(
    con: duckdb.DuckDBPyConnection,
    incident: Incident,
    settings: conc.ConcentrationSettings | None = None,
) -> dict[str, Any]:
    """Where and when the problem sits in the last month seen."""
    settings = settings or conc.load_settings()
    period = incident.last_seen_period
    if incident.rule_id == "volume":
        files = current_silver(con, incident.dataset).get(period)
        if files is None:
            return {"analysed": False, "reason": "no Silver for the month", "period": period}
        source = (
            f"read_parquet([{', '.join(_sql_path(p) for p in files)}], union_by_name = true, "
            "hive_partitioning = false)"
        )
        days = conc.low_days(con, source, period, settings)
        if days["analysed"]:
            days["verdict"] = (
                f"when: {len(days['days'])} day(s) with fewer than "
                f"{100 * settings.low_day_share:.0f}% of the median day's rows"
                if days["days"]
                else "unknown: no day is far below the median, rows are missing evenly"
            )
        else:
            days["verdict"] = f"not analysed: {days['reason']}"
        return {"period": period, **days}
    rows = month_rows(con, incident, period)
    if rows is None:
        reason = "month-level finding" if not is_row_level(incident) else "no rows to compare"
        return {"analysed": False, "reason": reason, "period": period, "findings": [],
                "verdict": f"not analysed: {reason}"}  # fmt: skip
    found = conc.segments(con, *rows, settings)
    return {"period": period, **found, "verdict": conc.verdict(found)}


def build_evidence(con: duckdb.DuckDBPyConnection, incident_id: str) -> dict[str, Any]:
    """The evidence pack of one incident."""
    incident = get_incident(con, incident_id)
    return {
        "incident": incident_section(incident),
        "definition": definition(con, incident),
        "history": history(con, incident),
        "events": events(con, incident_id),
        "versions": versions(con, incident),
        "impact": impact(con, incident),
        "similar": similar(con, incident),
        "metrics": metrics(con, incident),
        "row_level": is_row_level(incident),
        "samples": samples(con, incident),
        "concentration": concentration(con, incident),
    }


def to_json(evidence: dict[str, Any]) -> str:
    return json.dumps(evidence, indent=2, default=_json_default, ensure_ascii=False)


def store(con: duckdb.DuckDBPyConnection, evidence: dict[str, Any]) -> None:
    """Upsert one incident's pack into investigation_evidence."""
    con.execute(
        "INSERT OR REPLACE INTO investigation_evidence VALUES (?, ?, ?, ?)",
        [
            evidence["incident"]["incident_id"],
            evidence["incident"]["dataset"],
            to_json(evidence),
            utc_now(),
        ],
    )


def build_all(con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    """Build and store the pack of every incident, in incident order."""
    ids = [r[0] for r in con.execute("SELECT incident_id FROM incidents ORDER BY 1").fetchall()]
    packs = [build_evidence(con, i) for i in ids]
    for pack in packs:
        store(con, pack)
    return packs


def stored(con: duckdb.DuckDBPyConnection, incident_id: str) -> dict[str, Any] | None:
    row = con.execute(
        "SELECT evidence FROM investigation_evidence WHERE incident_id = ?", [incident_id]
    ).fetchone()
    return json.loads(row[0]) if row else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gather the evidence of an incident.")
    parser.add_argument("incident", nargs="?", help="incident id, e.g. INC-0004")
    parser.add_argument("--all", action="store_true", help="build and store every incident")
    parser.add_argument("--database", type=Path, help="DuckDB file (default from config)")
    args = parser.parse_args(argv)
    if bool(args.incident) == args.all:
        parser.error("give one incident id or --all")
    con = db.connect(args.database)
    try:
        db.init_schema(con)
        if args.all:
            packs = build_all(con)
            for pack in packs:
                i = pack["incident"]
                extra = f", {len(pack['samples'])} sample row(s)" if pack["row_level"] else ""
                print(f"{i['incident_id']}  {i['title']} ({i['severity']}, {i['status']}){extra}")
            print(f"{len(packs)} evidence pack(s) stored in investigation_evidence")
            return 0
        try:
            pack = build_evidence(con, args.incident)
        except IncidentError as exc:
            print(exc)
            return 2
        store(con, pack)
        print(to_json(pack))
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())
