"""Dataset-level checks: judge a whole month against the standard schema and its baseline.

Each check returns CheckResult objects with a status:

    pass      nothing to report
    fail      a problem, with a severity (critical, high, medium, low)
    skipped   not enough history to judge (cold start without a fixed limit)

Results are stored per run in Step 3.6; Phase 4 turns failures into incidents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from common.config import SEVERITIES, Dataset, load_quality_rules
from ingestion.manifest import ManifestEntry
from quality.baselines import Baseline, BaselineSettings
from quality.metrics import MonthMetrics

STATUSES = ("pass", "fail", "skipped")


@dataclass(frozen=True)
class Change:
    """One difference found by the schema check."""

    kind: str  # missing | unexpected | renamed | type_changed | removed | added
    column: str
    severity: str
    detail: str


@dataclass(frozen=True)
class CheckResult:
    check: str
    period: str
    status: str
    severity: str | None = None
    column: str = ""
    observed: str = ""
    expected: str = ""
    detail: str = ""
    baseline: str = ""
    changes: tuple[Change, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {', '.join(STATUSES)}")
        if (self.status == "fail") != (self.severity is not None):
            raise ValueError("a failed check needs a severity, other statuses must not have one")


@dataclass(frozen=True)
class CheckSettings:
    volume_tolerance: float = 0.30
    freshness_mode: str = "replay"
    grace_days: float = 3


def check_settings(path: Path | None = None) -> CheckSettings:
    defaults = load_quality_rules(path)["defaults"]
    freshness = defaults.get("freshness") or {}
    return CheckSettings(
        volume_tolerance=defaults["volume"]["tolerance"],
        freshness_mode=freshness.get("mode", "replay"),
        grace_days=freshness.get("grace_days", 3),
    )


def worst(severities: list[str]) -> str:
    """The most severe of the given severities (critical > high > medium > low)."""
    return min(severities, key=SEVERITIES.index)


# --- schema --------------------------------------------------------------------------------


def schema_changes(
    current: list[tuple[str, str]],
    previous: list[tuple[str, str]] | None,
    dataset: Dataset,
) -> list[Change]:
    """Differences of a raw schema to the standard schema and to the previous month.

    Against the standard schema (names only; standardization handles types):
        missing      a standard column is absent                          critical
        unexpected   a column outside the standard schema (it is dropped)  low
    Against the previous month (when there is one):
        renamed      same column, name differs only by case               high
        type_changed same column, different type                          medium
        removed      a non-standard column of last month is gone          medium
        added        a standard column that last month did not have       low
    """
    standard = {name.lower(): name for name, _ in dataset.columns}
    now = {name.lower(): (name, dtype) for name, dtype in current}
    changes: list[Change] = []

    for key, name in standard.items():
        if key not in now:
            changes.append(
                Change("missing", name, "critical", f"required column {name} is missing")
            )
    for key, (name, dtype) in now.items():
        if key not in standard:
            changes.append(
                Change(
                    "unexpected",
                    name,
                    "low",
                    f"new column {name} ({dtype}) is not in the standard schema",
                )
            )

    if previous is None:
        return changes

    before = {name.lower(): (name, dtype) for name, dtype in previous}
    for key, (name, dtype) in now.items():
        if key not in before:
            if key in standard:
                changes.append(Change("added", name, "low", f"column {name} is back"))
            continue
        old_name, old_type = before[key]
        if old_name != name:
            changes.append(Change("renamed", name, "high", f"renamed {old_name} -> {name}"))
        if old_type != dtype:
            changes.append(
                Change("type_changed", name, "medium", f"{name} type {old_type} -> {dtype}")
            )
    for key, (name, _) in before.items():
        if key not in now and key not in standard:  # standard columns are reported as missing
            changes.append(Change("removed", name, "medium", f"column {name} was removed"))
    return changes


def check_schema(m: MonthMetrics, baseline: Baseline, dataset: Dataset) -> CheckResult:
    """One result per month listing every schema change, with the worst severity."""
    changes = schema_changes(m.raw_schema, baseline.previous_schema, dataset)
    compared = (
        f"compared with {baseline.previous_period}"
        if baseline.previous_period
        else "no earlier month to compare; checked against the standard schema only"
    )
    observed = f"{len(m.raw_schema)} columns"
    expected = (
        f"{len(baseline.previous_schema)} columns as in {baseline.previous_period}"
        if baseline.previous_schema is not None
        else f"{len(dataset.columns)} standard columns"
    )
    if not changes:
        return CheckResult(
            "schema",
            m.period,
            "pass",
            observed=observed,
            expected=expected,
            detail="schema unchanged",
            baseline=compared,
        )
    return CheckResult(
        "schema",
        m.period,
        "fail",
        severity=worst([c.severity for c in changes]),
        observed=observed,
        expected=expected,
        detail="; ".join(c.detail for c in changes),
        baseline=compared,
        changes=tuple(changes),
    )


# --- volume --------------------------------------------------------------------------------


def check_volume(
    m: MonthMetrics,
    baseline: Baseline,
    settings: CheckSettings,
    limits: BaselineSettings,
) -> CheckResult:
    """Rows per day against the baseline median, or against fixed limits during a cold start.

    With a baseline, a deviation beyond the tolerance (default 30%) fails: medium up to twice
    the tolerance, high beyond that. During a cold start the month must lie within the
    configured rows_per_day limits (medium otherwise); without limits the check is skipped.
    """
    observed = f"{m.rows_per_day:,.0f} rows/day"
    if baseline.trusted and baseline.rows_per_day:
        change = m.rows_per_day / baseline.rows_per_day - 1
        expected = f"{baseline.rows_per_day:,.0f} rows/day ± {settings.volume_tolerance:.0%}"
        detail = f"{change:+.1%} against the median of {', '.join(baseline.source_periods)}"
        if abs(change) <= settings.volume_tolerance:
            return CheckResult(
                "volume",
                m.period,
                "pass",
                observed=observed,
                expected=expected,
                detail=detail,
                baseline=baseline.note,
            )
        severity = "high" if abs(change) > 2 * settings.volume_tolerance else "medium"
        direction = "drop" if change < 0 else "rise"
        return CheckResult(
            "volume",
            m.period,
            "fail",
            severity=severity,
            observed=observed,
            expected=expected,
            detail=f"volume {direction}: {detail}",
            baseline=baseline.note,
        )

    low, high = limits.rows_per_day_min, limits.rows_per_day_max
    if low is None and high is None:
        return CheckResult(
            "volume",
            m.period,
            "skipped",
            observed=observed,
            detail="no baseline and no fixed limits",
            baseline=baseline.note,
        )
    low = low if low is not None else 0
    high = high if high is not None else float("inf")
    expected = f"{low:,.0f} to {high:,.0f} rows/day (cold-start limits)"
    if low <= m.rows_per_day <= high:
        return CheckResult(
            "volume",
            m.period,
            "pass",
            observed=observed,
            expected=expected,
            detail="within the cold-start limits",
            baseline=baseline.note,
        )
    side = "below" if m.rows_per_day < low else "above"
    return CheckResult(
        "volume",
        m.period,
        "fail",
        severity="medium",
        observed=observed,
        expected=expected,
        detail=f"{side} the cold-start limits",
        baseline=baseline.note,
    )


# --- freshness -----------------------------------------------------------------------------


def delivery_time(entry: ManifestEntry, settings: CheckSettings) -> datetime | None:
    """The delivery time the check judges: simulated, or real in live mode."""
    if entry.simulated_delivered_at is not None:
        return entry.simulated_delivered_at
    if settings.freshness_mode == "live":
        return entry.delivered_at
    return entry.expected_at  # replay: historical files count as delivered on time


def check_freshness(entry: ManifestEntry, settings: CheckSettings) -> CheckResult:
    """Arrival against expected_at; late beyond the grace period fails.

    Late by up to 7 days (after the grace period): low; up to 30 days: medium; more: high.
    """
    if entry.expected_at is None:
        return CheckResult(
            "freshness",
            entry.period,
            "skipped",
            detail="no expected arrival time configured (expected_lag_days)",
        )
    delivered = delivery_time(entry, settings)
    source = (
        "simulated delivery"
        if entry.simulated_delivered_at is not None
        else f"{settings.freshness_mode} mode"
    )
    late_days = (delivered - entry.expected_at).total_seconds() / 86400
    observed = f"delivered {delivered:%Y-%m-%d %H:%M} ({source})"
    expected = f"by {entry.expected_at:%Y-%m-%d} + {settings.grace_days:g} day(s) grace"
    if late_days <= settings.grace_days:
        detail = "on time" if late_days <= 0 else f"{late_days:.1f} day(s) late, within grace"
        return CheckResult(
            "freshness", entry.period, "pass", observed=observed, expected=expected, detail=detail
        )
    overdue = late_days - settings.grace_days
    severity = "low" if overdue <= 7 else "medium" if overdue <= 30 else "high"
    return CheckResult(
        "freshness",
        entry.period,
        "fail",
        severity=severity,
        observed=observed,
        expected=expected,
        detail=f"{late_days:.1f} day(s) late",
    )
