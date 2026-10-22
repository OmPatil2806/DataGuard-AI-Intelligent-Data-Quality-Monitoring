"""Decide when a row rule becomes an incident.

Rows that fail a rule are quarantined or warned every month regardless; this module only
decides when a rule's failure *rate* is unusual enough to raise an incident. Real data has
steady, known problems (about 1.2% negative fares every month), which must not become
permanent incidents, while a jump must.

A rule's failure rate in a month is compared with its own baseline: the median rate of the
earlier months, chosen exactly like the dataset baselines (by month, at most 12 months back,
up to six months, trusted from three). It becomes a finding when

    trusted baseline:  rate - baseline > max(min_increase, relative_increase x baseline)
    cold start:        rate > cold_start_max_rate[action]

The finding's severity is the rule's severity from configs/quality_rules.yml.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path

import duckdb

from common.config import load_quality_rules
from incidents.incident_manager import Finding
from quality import baselines
from quality.baselines import BaselineSettings

Key = tuple[str, str, str, str]


@dataclass(frozen=True)
class RowRuleSettings:
    relative_increase: float = 1.0
    min_increase: float = 0.001
    cold_start_max_rate: dict[str, float] | None = None

    def cold_start_limit(self, action: str) -> float | None:
        return (self.cold_start_max_rate or {}).get(action)


def row_rule_settings(path: Path | None = None) -> RowRuleSettings:
    config = (load_quality_rules(path)["defaults"].get("incidents") or {}).get("row_rules") or {}
    return RowRuleSettings(
        relative_increase=config.get("relative_increase", 1.0),
        min_increase=config.get("min_increase", 0.001),
        cold_start_max_rate=config.get("cold_start_max_rate") or {},
    )


@dataclass(frozen=True)
class RuleRate:
    period: str
    rule_id: str
    action: str
    severity: str
    rows_checked: int
    rows_failed: int
    run_id: str

    @property
    def rate(self) -> float:
        return self.rows_failed / self.rows_checked if self.rows_checked else 0.0


def rule_rates(con: duckdb.DuckDBPyConnection, dataset: str) -> dict[str, dict[str, RuleRate]]:
    """The latest measured rate of every rule in every month: {period: {rule_id: RuleRate}}."""
    rows = con.execute(
        """
        SELECT period, rule_id, action, severity, rows_checked, rows_failed, run_id
        FROM rule_run_log WHERE dataset = ?
        QUALIFY row_number() OVER (PARTITION BY period, rule_id ORDER BY run_id DESC) = 1
        """,
        [dataset],
    ).fetchall()
    rates: dict[str, dict[str, RuleRate]] = {}
    for row in rows:
        rate = RuleRate(*row)
        rates.setdefault(rate.period, {})[rate.rule_id] = rate
    return rates


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def judge(
    current: RuleRate,
    earlier: list[RuleRate],
    settings: RowRuleSettings,
    trusted: bool,
) -> tuple[bool, str, str]:
    """(is a finding, expected text, detail text) for one rule in one month."""
    observed = f"{_pct(current.rate)} of rows ({current.rows_failed:,})"
    if trusted:
        usual = statistics.median(r.rate for r in earlier)
        allowed = max(settings.min_increase, settings.relative_increase * usual)
        expected = f"at most {_pct(usual + allowed)} (baseline {_pct(usual)})"
        increase = current.rate - usual
        months = ", ".join(r.period for r in earlier)
        detail = f"{observed}: {100 * increase:+.2f} points against the median of {months}"
        return increase > allowed, expected, detail
    limit = settings.cold_start_limit(current.action)
    if limit is None:
        return False, "no baseline and no fixed limit", observed
    expected = f"at most {_pct(limit)} (cold-start limit for {current.action} rules)"
    return current.rate > limit, expected, f"{observed}, cold start"


def rule_findings(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    dataset: str,
    settings: RowRuleSettings | None = None,
    limits: BaselineSettings | None = None,
) -> dict[str, tuple[list[Finding], set[Key]]]:
    """Per month of a rule run: the rules whose failure rate is unusual, and every rule checked."""
    settings = settings or row_rule_settings()
    limits = limits or baselines.settings_for(dataset)
    rates = rule_rates(con, dataset)
    periods_in_run = [
        p
        for (p,) in con.execute(
            "SELECT DISTINCT period FROM rule_run_log WHERE run_id = ? AND dataset = ? "
            "ORDER BY period",
            [run_id, dataset],
        ).fetchall()
    ]
    result: dict[str, tuple[list[Finding], set[Key]]] = {}
    for period in periods_in_run:
        findings: list[Finding] = []
        checked: set[Key] = set()
        earlier_periods = baselines.earlier_months(list(rates), period, limits)
        for rule_id, current in rates[period].items():
            checked.add((dataset, "row_rule", rule_id, ""))
            earlier = [rates[p][rule_id] for p in earlier_periods if rule_id in rates[p]]
            trusted = len(earlier) >= limits.min_runs
            is_finding, expected, detail = judge(current, earlier, settings, trusted)
            if is_finding:
                findings.append(
                    Finding(
                        dataset,
                        "row_rule",
                        rule_id,
                        current.severity,
                        period,
                        run_id,
                        f"Rule failure rate up: {rule_id}",
                        "",
                        _pct(current.rate),
                        expected,
                        detail,
                    )
                )
        result[period] = (findings, checked)
    return result


# --- findings from the anomaly model (Phase 7) -------------------------------------------


@dataclass(frozen=True)
class MLSettings:
    min_increase: float
    relative_increase: float
    cold_start_max_rate: float
    severity: str


def ml_settings(path: Path | None = None) -> MLSettings:
    from ml.features import load_ml_config

    config = load_ml_config(path).incidents
    return MLSettings(
        config["min_increase"],
        config["relative_increase"],
        config["cold_start_max_rate"],
        config["severity"],
    )


def ml_findings(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    dataset: str,
    settings: MLSettings | None = None,
    limits: BaselineSettings | None = None,
) -> dict[str, tuple[list[Finding], set[Key]]]:
    """Per month of a scoring run: an `ml_anomaly` finding when the share of flagged rows rises
    above the same model's flag rate in the earlier months (as for the row rules)."""
    scored = con.execute(
        "SELECT period, model_id, flag_rate, rows_flagged FROM ml_month_summary "
        "WHERE run_id = ? AND dataset = ? ORDER BY period",
        [run_id, dataset],
    ).fetchall()
    if not scored:
        return {}
    settings = settings or ml_settings()
    limits = limits or baselines.settings_for(dataset)
    result: dict[str, tuple[list[Finding], set[Key]]] = {}
    key = (dataset, "dataset_check", "ml_anomaly", "")
    for period, model, rate, flagged in scored:
        rates = dict(
            con.execute(
                """
                SELECT period, flag_rate FROM ml_month_summary WHERE dataset = ? AND model_id = ?
                QUALIFY row_number() OVER (PARTITION BY period ORDER BY scored_at DESC) = 1
                """,
                [dataset, model],
            ).fetchall()
        )
        earlier = [rates[p] for p in baselines.earlier_months(list(rates), period, limits)]
        observed = f"{_pct(rate)} of rows flagged ({flagged:,})"
        if len(earlier) >= limits.min_runs:
            usual = statistics.median(earlier)
            allowed = max(settings.min_increase, settings.relative_increase * usual)
            expected = f"at most {_pct(usual + allowed)} (baseline {_pct(usual)})"
            is_finding = rate - usual > allowed
            detail = f"{observed}: {100 * (rate - usual):+.2f} points against the earlier months"
        else:
            expected = f"at most {_pct(settings.cold_start_max_rate)} (cold start)"
            is_finding = rate > settings.cold_start_max_rate
            detail = f"{observed}, cold start"
        reasons = con.execute(
            "SELECT top_features[1] AS f, count(*) FROM ml_anomalies WHERE dataset = ? "
            "AND period = ? AND model_id = ? GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 3",
            [dataset, period, model],
        ).fetchall()
        if reasons:
            detail += "; most often " + ", ".join(f"{f} ({n})" for f, n in reasons)
        findings = (
            [
                Finding(
                    dataset,
                    "dataset_check",
                    "ml_anomaly",
                    settings.severity,
                    period,
                    run_id,
                    "Unusual trips (ML)",
                    "",
                    observed,
                    expected,
                    detail,
                )
            ]  # fmt: skip
            if is_finding
            else []
        )
        result[period] = (findings, {key})
    return result
