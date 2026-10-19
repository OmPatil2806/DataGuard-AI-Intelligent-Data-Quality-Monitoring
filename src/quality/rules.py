"""Load and validate the row-level rules in configs/quality_rules.yml.

Every rule is checked against the dataset's standard schema (configs/datasets.yml), so a
misspelt column, an unknown check type or a missing parameter is reported with the rule's
id before any data is read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common.config import (
    CONFIG_DIR,
    SEVERITIES,
    ConfigError,
    Dataset,
    load_datasets_config,
    load_quality_rules,
)

ROW_CATEGORIES = ("completeness", "uniqueness", "validity", "pattern", "referential", "consistency")
ACTIONS = ("quarantine", "warn")
OPERATORS = ("<", "<=", ">", ">=", "=", "!=")
RULE_ID = re.compile(r"^[a-z][a-z0-9_]*$")
PLACEHOLDER = re.compile(r"\{([^{}]+)\}")
TIMESTAMP_TYPES = ("TIMESTAMP", "DATE")
NUMERIC_TYPES = ("INTEGER", "BIGINT", "DOUBLE")

COMMON_KEYS = {
    "id",
    "category",
    "check",
    "action",
    "severity",
    "reason",
    "when",
    "enabled",
    "description",
}

# check type -> (required parameters, optional parameters)
CHECKS: dict[str, tuple[set[str], set[str]]] = {
    "not_null": ({"columns"}, {"mode"}),
    "range": ({"column"}, {"min", "max", "exclusive_min", "exclusive_max"}),
    "allowed_values": ({"column", "values"}, set()),
    "compare": ({"left", "op", "right"}, set()),
    "duration": ({"start", "end"}, {"min_seconds", "max_seconds"}),
    "within_period": ({"column"}, set()),
    "foreign_key": ({"column", "lookup", "lookup_column"}, set()),
    "unique": ({"columns"}, set()),
    "sum_equals": ({"total", "parts"}, {"tolerance", "alternatives"}),
}


@dataclass(frozen=True)
class Rule:
    id: str
    dataset: str
    category: str
    check: str
    action: str
    severity: str
    reason: str
    params: dict[str, Any] = field(default_factory=dict)
    when: str | None = None
    enabled: bool = True
    description: str | None = None

    @property
    def columns(self) -> list[str]:
        """Every standard column the rule reads."""
        p = self.params
        names: list[str] = []
        for key in ("column", "left", "right", "start", "end", "total"):
            if key in p:
                names.append(p[key])
        for key in ("columns", "parts"):
            names.extend(p.get(key, []))
        for alternative in p.get("alternatives", []):
            names.extend(alternative)
        return list(dict.fromkeys(names))


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _validate(rule: Rule, dataset: Dataset) -> None:
    where = f"rules.{dataset.name}.{rule.id}"
    p = rule.params
    types = {name: dtype for name, dtype in dataset.columns}

    def fail(message: str) -> None:
        raise ConfigError(f"{where}: {message}")

    for name in rule.columns:
        if not isinstance(name, str) or name not in types:
            fail(f"unknown column '{name}' (not in the standard schema of '{dataset.name}')")

    def column_list(key: str) -> None:
        value = p[key]
        if not isinstance(value, list) or not value:
            fail(f"'{key}' must be a non-empty list of columns")
        if len(set(value)) != len(value):
            fail(f"'{key}' lists a column twice")

    def number(key: str) -> None:
        if key in p and not _is_number(p[key]):
            fail(f"'{key}' must be a number")

    def flag(key: str) -> None:
        if key in p and not isinstance(p[key], bool):
            fail(f"'{key}' must be true or false")

    check = rule.check
    if check == "not_null":
        column_list("columns")
        if p.get("mode", "any") not in ("any", "all"):
            fail("'mode' must be 'any' or 'all'")
    elif check == "range":
        if "min" not in p and "max" not in p:
            fail("range needs 'min' and/or 'max'")
        number("min")
        number("max")
        flag("exclusive_min")
        flag("exclusive_max")
        if "min" in p and "max" in p and p["min"] > p["max"]:
            fail("'min' is greater than 'max'")
        if types[p["column"]] not in NUMERIC_TYPES:
            fail(f"range needs a numeric column, '{p['column']}' is {types[p['column']]}")
    elif check == "allowed_values":
        if not isinstance(p["values"], list) or not p["values"]:
            fail("'values' must be a non-empty list")
    elif check == "compare":
        if p["op"] not in OPERATORS:
            fail(f"'op' must be one of {' '.join(OPERATORS)}")
    elif check == "duration":
        if "min_seconds" not in p and "max_seconds" not in p:
            fail("duration needs 'min_seconds' and/or 'max_seconds'")
        number("min_seconds")
        number("max_seconds")
        for key in ("start", "end"):
            if types[p[key]] not in TIMESTAMP_TYPES:
                fail(f"'{key}' must be a timestamp column, '{p[key]}' is {types[p[key]]}")
    elif check == "within_period":
        if types[p["column"]] not in TIMESTAMP_TYPES:
            fail(f"'{p['column']}' must be a timestamp column")
    elif check == "foreign_key":
        if p["lookup"] not in {lookup.name for lookup in dataset.lookups}:
            known = ", ".join(lookup.name for lookup in dataset.lookups) or "none"
            fail(f"unknown lookup '{p['lookup']}' (known: {known})")
        if not isinstance(p["lookup_column"], str) or not p["lookup_column"]:
            fail("'lookup_column' must be a column name of the lookup")
    elif check == "unique":
        column_list("columns")
    elif check == "sum_equals":
        column_list("parts")
        number("tolerance")
        if p.get("tolerance", 0) < 0:
            fail("'tolerance' must be 0 or more")
        alternatives = p.get("alternatives", [])
        if not isinstance(alternatives, list) or not all(
            isinstance(a, list) and a for a in alternatives
        ):
            fail("'alternatives' must be a list of non-empty column lists")

    for placeholder in PLACEHOLDER.findall(rule.reason):
        if placeholder not in types:
            fail(f"reason uses '{{{placeholder}}}', which is not a column")


def _parse_rule(raw: Any, index: int, dataset: Dataset) -> Rule:
    where = f"rules.{dataset.name}[{index}]"
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: each rule must be a mapping")
    rule_id = raw.get("id")
    if not isinstance(rule_id, str) or not RULE_ID.match(rule_id):
        raise ConfigError(f"{where}: 'id' must be snake_case, e.g. fare_not_negative")
    where = f"rules.{dataset.name}.{rule_id}"

    for key in ("category", "check", "action", "severity", "reason"):
        if not raw.get(key):
            raise ConfigError(f"{where}: missing required key '{key}'")
    check = raw["check"]
    if check not in CHECKS:
        raise ConfigError(f"{where}: unknown check '{check}' (use {', '.join(CHECKS)})")
    if raw["category"] not in ROW_CATEGORIES:
        raise ConfigError(
            f"{where}: category must be one of {', '.join(ROW_CATEGORIES)}, got '{raw['category']}'"
        )
    if raw["action"] not in ACTIONS:
        raise ConfigError(f"{where}: action must be quarantine or warn, got '{raw['action']}'")
    if raw["severity"] not in SEVERITIES:
        raise ConfigError(
            f"{where}: severity must be one of {', '.join(SEVERITIES)}, got '{raw['severity']}'"
        )
    if "enabled" in raw and not isinstance(raw["enabled"], bool):
        raise ConfigError(f"{where}: 'enabled' must be true or false")
    if raw.get("when") is not None and not isinstance(raw["when"], str):
        raise ConfigError(f"{where}: 'when' must be a SQL condition in quotes")

    required, optional = CHECKS[check]
    params = {k: v for k, v in raw.items() if k not in COMMON_KEYS}
    missing = sorted(required - set(params))
    if missing:
        raise ConfigError(f"{where}: check '{check}' needs {', '.join(missing)}")
    unknown = sorted(set(params) - required - optional)
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) for check '{check}': {', '.join(unknown)}")

    rule = Rule(
        id=rule_id,
        dataset=dataset.name,
        category=raw["category"],
        check=check,
        action=raw["action"],
        severity=raw["severity"],
        reason=str(raw["reason"]),
        params=params,
        when=raw.get("when"),
        enabled=raw.get("enabled", True),
        description=raw.get("description"),
    )
    _validate(rule, dataset)
    return rule


def load_rules(
    dataset: Dataset | str,
    path: Path | None = None,
    include_disabled: bool = False,
) -> list[Rule]:
    """The validated row-level rules of one dataset, in file order."""
    if isinstance(dataset, str):
        dataset = load_datasets_config().dataset(dataset)
    path = Path(path) if path else CONFIG_DIR / "quality_rules.yml"
    config = load_quality_rules(path)
    raw_rules = config["rules"].get(dataset.name) if isinstance(config["rules"], dict) else None
    if raw_rules is None:
        return []
    if not isinstance(raw_rules, list):
        raise ConfigError(f"rules.{dataset.name} must be a list of rules")

    rules = [_parse_rule(raw, i, dataset) for i, raw in enumerate(raw_rules)]
    seen: set[str] = set()
    for rule in rules:
        if rule.id in seen:
            raise ConfigError(f"rules.{dataset.name}: rule id '{rule.id}' is used twice")
        seen.add(rule.id)
    return rules if include_disabled else [r for r in rules if r.enabled]
