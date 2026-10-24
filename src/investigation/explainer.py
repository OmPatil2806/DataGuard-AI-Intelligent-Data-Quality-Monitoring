"""TemplateExplainer: likely causes and next steps for an incident, from its evidence (Phase 8).

    python src/investigation/explainer.py INC-0004          # explain one incident
    python src/investigation/explainer.py --all             # explain every incident

The evidence pack (evidence_pack.py) is turned into flat facts (FACTS below), the first
playbook of configs/playbooks.yml whose `match` fits the incident is chosen, and each of its
causes is scored: its prior plus the weights of the signals that hold. The best causes are
written out from templates, with the evidence that supports them and the suggested actions.

No language model is involved: the same evidence always gives the same explanation, and every
sentence can be traced to a fact. When no cause reaches medium confidence the explanation says
"unknown" instead of guessing. Actions are suggestions: each is `pending_approval`.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    # Allow running this file directly: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import db  # noqa: E402
from common.config import CONFIG_DIR, ConfigError, load_yaml  # noqa: E402
from incidents.incident_manager import IncidentError  # noqa: E402
from investigation import evidence_pack as ep  # noqa: E402

FACTS = {
    "incident_id": "the incident",
    "title": "the incident's title",
    "period": "the last month the incident was seen",
    "source": "row_rule or dataset_check",
    "rule_id": "row rule id or check name",
    "column": "the column of a column check ('' otherwise)",
    "category": "the row rule's category (definition)",
    "severity": "the incident's severity",
    "occurrences": "months the incident was seen",
    "recurring": "an earlier or later incident of the same problem exists",
    "detail": "the latest detail text of the finding",
    "observed": "the latest observed value",
    "expected": "the latest expected value",
    "row_level": "the problem sits in individual rows",
    "affected_share": "share of the month's rows affected (row-level problems)",
    "concentrated": "a where/when concentration was found (None when not analysed)",
    "where_kind": "where or when, of the strongest concentration",
    "where_dimension": "its dimension, e.g. vendor",
    "where_value": "its value, e.g. 2",
    "where_share": "its share of the affected rows",
    "low_days": "days far below the month's median day (volume incidents)",
    "late_days": "days the file arrived late (freshness incidents)",
    "shared_file": "one file version (SHA-256) is registered for more than one month",
    "same_month": "rule ids of the other incidents of the same month",
    "same_month_columns": "the same, as rule_id:column",
    "incidents_same_month": "how many other incidents the month has",
    "ml_top_reason": "the feature that most often leads the model's flags",
    "<metric>_ratio": "the metric in the month / its median over the earlier months "
    "(null_rate, mean, median, p99, rows, rows_per_day)",
}
OPERATORS = ("is", "in", "contains", "contains_any", "not_contains", "gt", "lt", "between",
             "exists")  # fmt: skip
ACTION_KINDS = ("investigate", "contact_provider", "reprocess", "change_config", "change_rule",
                "accept")  # fmt: skip
MAX_CAUSES = 3


# --- templates ---------------------------------------------------------------------------------


class _Formatter(string.Formatter):
    def get_value(self, key, args, kwargs):
        value = kwargs.get(key) if isinstance(key, str) else None
        return "?" if value is None else value

    def format_field(self, value, format_spec):
        try:
            return super().format_field(value, format_spec)
        except (TypeError, ValueError):
            return str(value)


_FORMATTER = _Formatter()


def render(template: str, facts: dict[str, Any]) -> str:
    """Fill {fact} and {fact:spec} from the facts; a missing fact prints as '?'."""
    return _FORMATTER.vformat(template, (), facts)


# --- playbooks ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Condition:
    fact: str
    op: str
    value: Any
    weight: float = 0.0
    say: str = ""


@dataclass(frozen=True)
class Cause:
    id: str
    text: str
    prior: float
    requires: tuple[Condition, ...] = ()
    signals: tuple[Condition, ...] = ()
    actions: tuple[str, ...] = ()


@dataclass(frozen=True)
class Playbook:
    id: str
    match: dict[str, tuple[str, ...]]
    causes: tuple[Cause, ...]


@dataclass(frozen=True)
class Playbooks:
    playbooks: tuple[Playbook, ...]
    actions: dict[str, dict[str, str]]
    high: float = 0.7
    medium: float = 0.4

    def for_facts(self, facts: dict[str, Any]) -> Playbook | None:
        for book in self.playbooks:
            if all(facts.get(key) in values for key, values in book.match.items()):
                return book
        return None


def _condition(raw: Any, where: str, signal: bool) -> Condition:
    if not isinstance(raw, dict) or "fact" not in raw:
        raise ConfigError(f"{where}: a condition needs 'fact'")
    ops = [k for k in raw if k in OPERATORS]
    if len(ops) != 1:
        raise ConfigError(f"{where}: a condition needs exactly one of {', '.join(OPERATORS)}")
    op = ops[0]
    value = raw[op]
    if op == "between" and not (isinstance(value, list) and len(value) == 2):
        raise ConfigError(f"{where}: between needs [low, high]")
    if op in ("in", "contains_any") and not isinstance(value, list):
        raise ConfigError(f"{where}: {op} needs a list")
    weight = raw.get("weight", 0.0)
    if signal and not isinstance(weight, int | float):
        raise ConfigError(f"{where}: weight must be a number")
    return Condition(raw["fact"], op, value, float(weight), str(raw.get("say", "")))


def load_playbooks(path: Path | None = None) -> Playbooks:
    path = Path(path) if path else CONFIG_DIR / "playbooks.yml"
    raw = load_yaml(path)
    actions = raw.get("actions") or {}
    if not isinstance(actions, dict):
        raise ConfigError(f"{path}: actions must be a mapping")
    for name, spec in actions.items():
        if not isinstance(spec, dict) or "text" not in spec:
            raise ConfigError(f"{path}: action '{name}' needs text")
        if spec.get("kind") not in ACTION_KINDS:
            raise ConfigError(f"{path}: action '{name}': kind must be one of {ACTION_KINDS}")
    books = []
    seen: set[str] = set()
    for i, book in enumerate(raw.get("playbooks") or []):
        where = f"{path}: playbooks[{i}]"
        if not isinstance(book, dict) or "id" not in book:
            raise ConfigError(f"{where} needs an id")
        match = {
            k: tuple(v if isinstance(v, list) else [v])
            for k, v in (book.get("match") or {}).items()
        }
        causes = []
        for j, c in enumerate(book.get("causes") or []):
            cw = f"{where}.causes[{j}]"
            if not isinstance(c, dict) or not {"id", "text", "prior"} <= set(c):
                raise ConfigError(f"{cw}: a cause needs id, text and prior")
            if c["id"] in seen:
                raise ConfigError(f"{cw}: cause id '{c['id']}' is used twice")
            seen.add(c["id"])
            unknown = [a for a in c.get("actions") or [] if a not in actions]
            if unknown:
                raise ConfigError(f"{cw}: unknown action(s) {', '.join(unknown)}")
            causes.append(
                Cause(
                    c["id"],
                    c["text"],
                    float(c["prior"]),
                    tuple(_condition(r, cw, False) for r in c.get("requires") or []),
                    tuple(_condition(s, cw, True) for s in c.get("signals") or []),
                    tuple(c.get("actions") or []),
                )
            )
        books.append(Playbook(book["id"], match, tuple(causes)))
    if not books:
        raise ConfigError(f"{path}: no playbooks")
    thresholds = raw.get("thresholds") or {}
    return Playbooks(
        tuple(books),
        actions,
        float(thresholds.get("high", 0.7)),
        float(thresholds.get("medium", 0.4)),
    )


# --- facts -------------------------------------------------------------------------------------


def _ratio(series: list[dict[str, Any]], period: str) -> float | None:
    now = [s["value"] for s in series if s["period"] == period]
    earlier = [s["value"] for s in series if s["period"] < period]
    if not now or not earlier:
        return None
    usual = statistics.median(earlier)
    if usual == 0:
        return None if now[0] == 0 else float("inf")
    return round(now[0] / usual, 4)


def facts(evidence: dict[str, Any]) -> dict[str, Any]:
    """The flat facts the playbooks test, from one evidence pack."""
    inc = evidence["incident"]
    period = inc["last_seen"]
    f: dict[str, Any] = {
        "incident_id": inc["incident_id"],
        "title": inc["title"],
        "period": period,
        "source": inc["source"],
        "rule_id": inc["rule_id"],
        "column": inc["column"] or "",
        "category": evidence["definition"].get("category"),
        "severity": inc["severity"],
        "occurrences": inc["occurrences"],
        "recurring": bool(evidence["similar"]["same_problem"]),
        "detail": inc["detail"] or "",
        "observed": inc["observed"] or "",
        "expected": inc["expected"] or "",
        "row_level": evidence["row_level"],
        "affected_share": evidence["impact"].get("affected_share"),
    }
    where = evidence.get("concentration") or {}
    findings = where.get("findings") or []
    f["concentrated"] = bool(findings) if where.get("analysed") else None
    if findings:
        top = findings[0]
        f |= {
            "where_kind": top["kind"],
            "where_dimension": top["dimension"],
            "where_value": top["value"],
            "where_share": top["share"],
        }
    if inc["rule_id"] == "volume":
        f["low_days"] = len(where.get("days") or []) if where.get("analysed") else None
    if inc["rule_id"] == "freshness":
        late = re.search(r"([0-9.]+) day\(s\) late", f["detail"])
        f["late_days"] = float(late.group(1)) if late else None
    periods: dict[str, set[str]] = {}
    for v in evidence["versions"]:
        periods.setdefault(v["sha256"], set()).add(v["period"])
    f["shared_file"] = any(len(p) > 1 for p in periods.values())
    month = evidence["similar"]["same_month"]
    f["same_month"] = sorted({m["rule_id"] for m in month})
    f["same_month_columns"] = sorted({f"{m['rule_id']}:{m['column']}" for m in month})
    f["incidents_same_month"] = len(month)
    reason = re.search(r"most often ([a-z_]+)", f["detail"])
    f["ml_top_reason"] = reason.group(1) if reason else None
    for metric, series in (evidence.get("metrics") or {}).items():
        f[f"{metric}_ratio"] = _ratio(series, period)
    return f


# --- scoring -----------------------------------------------------------------------------------


def holds(condition: Condition, f: dict[str, Any]) -> bool:
    value = f.get(condition.fact)
    target = condition.value
    if isinstance(target, str):
        target = render(target, f)
    op = condition.op
    if op == "exists":
        return (value is not None) == bool(target)
    if value is None:
        return False
    if op == "is":
        return value == target
    if op == "in":
        return value in target
    if op in ("contains", "not_contains"):
        found = target in value if isinstance(value, str | list | tuple | set) else False
        return found if op == "contains" else not found
    if op == "contains_any":
        return any(t in value for t in target)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    if op == "gt":
        return value > target
    if op == "lt":
        return value < target
    low, high = target
    return low <= value <= high


def confidence(score: float, books: Playbooks) -> str:
    return "high" if score >= books.high else "medium" if score >= books.medium else "low"


def explain(evidence: dict[str, Any], books: Playbooks | None = None) -> dict[str, Any]:
    """The explanation of one incident: ranked causes, the evidence for each, the actions."""
    books = books or load_playbooks()
    f = facts(evidence)
    book = books.for_facts(f)
    candidates = []
    for order, cause in enumerate(book.causes if book else ()):
        if not all(holds(r, f) for r in cause.requires):
            continue
        matched = [s for s in cause.signals if holds(s, f)]
        score = round(min(1.0, cause.prior + sum(s.weight for s in matched)), 4)
        candidates.append((-score, order, cause, matched, score))
    candidates.sort(key=lambda c: (c[0], c[1]))
    causes = [
        {
            "cause_id": cause.id,
            "text": render(cause.text, f),
            "score": score,
            "confidence": confidence(score, books),
            "evidence": [render(s.say, f) for s in matched if s.say],
        }
        for _, _, cause, matched, score in candidates[:MAX_CAUSES]
    ]
    explained = bool(causes) and causes[0]["confidence"] != "low"
    actions = []
    if explained:
        top = candidates[0][2]
        for name in top.actions:
            spec = books.actions[name]
            actions.append(
                {
                    "action_id": name,
                    "kind": spec["kind"],
                    "text": render(spec["text"], f),
                    "for_cause": top.id,
                    "status": "pending_approval",
                }
            )
    else:
        for name in ("inspect_samples",) if f["row_level"] else ("contact_provider",):
            spec = books.actions.get(name)
            if spec:
                actions.append(
                    {
                        "action_id": name,
                        "kind": spec["kind"],
                        "text": render(spec["text"], f),
                        "for_cause": "unknown",
                        "status": "pending_approval",
                    }
                )
    where = (evidence.get("concentration") or {}).get("verdict")
    if explained:
        headline = (
            f"{f['title']} in {f['period']}: most likely {causes[0]['text'][0].lower()}"
            f"{causes[0]['text'][1:]} ({causes[0]['confidence']} confidence)"
        )
    else:
        headline = f"{f['title']} in {f['period']}: cause unknown"
        if causes:
            headline += f" (weak lead: {causes[0]['text'][0].lower()}{causes[0]['text'][1:]})"
    return {
        "incident_id": f["incident_id"],
        "playbook": book.id if book else None,
        "headline": headline,
        "explained": explained,
        "causes": causes,
        "where": where,
        "actions": actions,
        "facts": {k: v for k, v in f.items() if v is not None and v != [] and v != ""},
    }


def format_explanation(e: dict[str, Any]) -> str:
    lines = [f"{e['incident_id']}  {e['headline']}"]
    if e["where"]:
        lines.append(f"  where/when: {e['where']}")
    for i, cause in enumerate(e["causes"], 1):
        lines.append(f"  {i}. [{cause['confidence']} {cause['score']:.2f}] {cause['text']}")
        lines += [f"       - {item}" for item in cause["evidence"]]
    for action in e["actions"]:
        lines.append(f"  -> {action['text']} [{action['kind']}, {action['status']}]")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explain an incident from its evidence.")
    parser.add_argument("incident", nargs="?", help="incident id, e.g. INC-0004")
    parser.add_argument("--all", action="store_true", help="explain every incident")
    parser.add_argument("--json", action="store_true", help="print JSON")
    parser.add_argument("--database", type=Path, help="DuckDB file (default from config)")
    args = parser.parse_args(argv)
    if bool(args.incident) == args.all:
        parser.error("give one incident id or --all")
    books = load_playbooks()
    con = db.connect(args.database, read_only=True)
    try:
        ids = (
            [r[0] for r in con.execute("SELECT incident_id FROM incidents ORDER BY 1").fetchall()]
            if args.all
            else [args.incident]
        )
        try:
            results = [explain(ep.build_evidence(con, i), books) for i in ids]
        except IncidentError as exc:
            print(exc)
            return 2
    finally:
        con.close()
    if args.json:
        print(json.dumps(results if args.all else results[0], indent=2, default=str))
    else:
        print("\n\n".join(format_explanation(e) for e in results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
