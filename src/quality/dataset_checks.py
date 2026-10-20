"""Dataset-level checks: judge a whole month against the standard schema and its baseline.

Each check returns CheckResult objects with a status:

    pass      nothing to report
    fail      a problem, with a severity (critical, high, medium, low)
    skipped   not enough history to judge (cold start without a fixed limit)

Results are stored per run in Step 3.6; Phase 4 turns failures into incidents.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from common.config import SEVERITIES, Dataset
from quality.baselines import Baseline
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
