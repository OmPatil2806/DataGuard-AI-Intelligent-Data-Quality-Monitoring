"""Investigate every incident: evidence, explanation, actions and a report (Phase 8).

    python src/investigation/investigate.py                        # every incident
    python src/investigation/investigate.py --incident INC-0004    # one incident
    python src/investigation/investigate.py --decide INC-0004 contact_provider approved --by om

For each incident, in one tracked run:

    1 evidence    evidence_pack.py    stored in investigation_evidence
    2 explain     explainer.py        stored in investigations (headline, causes, where/when)
    3 actions     the suggestions     stored in investigation_actions, pending_approval
    4 report      Markdown            <paths.reports>/investigations/<dataset>/<incident>.md

Everything is rebuilt each time from the control tables, so it is also the `investigate`
stage of rebuild.py (after the score, before Gold). A suggested action keeps its decision
(approved, rejected, done) while the explanation still suggests it; a pending action that is no
longer suggested is removed. Deciding is the only change made by hand, and nothing runs an
action: the decision is a record for whoever does the work.

Reports hold no times, so the same data gives the same report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import duckdb

if __package__ in (None, ""):
    # Allow running this file directly: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import db  # noqa: E402
from common.config import REPO_ROOT, ConfigError, load_datasets_config  # noqa: E402
from common.runs import track_run, utc_now  # noqa: E402
from incidents.incident_manager import IncidentError, get_incident  # noqa: E402
from investigation import evidence_pack as ep  # noqa: E402
from investigation import explainer as ex  # noqa: E402

DECISIONS = tuple(s for s in db.ACTION_STATUSES if s != "pending_approval")
REPORTS_DIR_ENV = "DATAGUARD_REPORTS_DIR"  # overrides paths.reports (the tests use their own)


def reports_dir() -> Path:
    if os.environ.get(REPORTS_DIR_ENV):
        return Path(os.environ[REPORTS_DIR_ENV])
    paths = load_datasets_config().paths
    return paths.get("reports", paths["data_dir"] / "reports")


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


# --- actions -----------------------------------------------------------------------------------


def store_actions(
    con: duckdb.DuckDBPyConnection, incident_id: str, actions: list[dict[str, Any]]
) -> None:
    """Insert new suggestions as pending; keep the decision of the ones still suggested; drop
    pending ones no longer suggested."""
    now = utc_now()
    wanted = {a["action_id"] for a in actions}
    existing = dict(
        con.execute(
            "SELECT action_id, status FROM investigation_actions WHERE incident_id = ?",
            [incident_id],
        ).fetchall()
    )
    for action_id, status in existing.items():
        if action_id not in wanted and status == "pending_approval":
            con.execute(
                "DELETE FROM investigation_actions WHERE incident_id = ? AND action_id = ?",
                [incident_id, action_id],
            )
    for a in actions:
        if a["action_id"] in existing:
            con.execute(
                "UPDATE investigation_actions SET kind = ?, text = ?, for_cause = ? "
                "WHERE incident_id = ? AND action_id = ?",
                [a["kind"], a["text"], a["for_cause"], incident_id, a["action_id"]],
            )
        else:
            con.execute(
                "INSERT INTO investigation_actions VALUES (?, ?, ?, ?, ?, 'pending_approval', ?, "
                "NULL, NULL)",
                [incident_id, a["action_id"], a["kind"], a["text"], a["for_cause"], now],
            )


def decide(
    con: duckdb.DuckDBPyConnection, incident_id: str, action_id: str, status: str, by: str
) -> None:
    """Record a person's decision on a suggested action."""
    if status not in DECISIONS:
        raise ValueError(f"decision must be one of {', '.join(DECISIONS)}")
    found = con.execute(
        "SELECT count(*) FROM investigation_actions WHERE incident_id = ? AND action_id = ?",
        [incident_id, action_id],
    ).fetchone()[0]
    if not found:
        raise IncidentError(f"{incident_id} has no suggested action '{action_id}'")
    con.execute(
        "UPDATE investigation_actions SET status = ?, decided_at = ?, decided_by = ? "
        "WHERE incident_id = ? AND action_id = ?",
        [status, utc_now(), by, incident_id, action_id],
    )


def stored_actions(con: duckdb.DuckDBPyConnection, incident_id: str) -> list[dict[str, Any]]:
    rows = con.execute(
        "SELECT action_id, kind, text, for_cause, status, decided_by FROM investigation_actions "
        "WHERE incident_id = ? ORDER BY created_at, action_id",
        [incident_id],
    ).fetchall()
    keys = ("action_id", "kind", "text", "for_cause", "status", "decided_by")
    return [dict(zip(keys, r, strict=True)) for r in rows]


# --- the report --------------------------------------------------------------------------------


def _cell(value: Any) -> str:
    if value is None or value == "":
        return "–"
    if isinstance(value, float):
        value = f"{value:.4g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(header: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    return lines


def render_report(
    evidence: dict[str, Any], explanation: dict[str, Any], actions: list[dict[str, Any]]
) -> str:
    """The incident's investigation as Markdown."""
    inc = evidence["incident"]
    out = [f"# {inc['incident_id']} · {inc['title']}", "", f"**{explanation['headline']}**", ""]
    out += _table(
        ["Status", "Severity", "First seen", "Last seen", "Months", "Observed", "Expected"],
        [[inc["status"], inc["severity"], inc["first_seen"], inc["last_seen"],
          inc["occurrences"], inc["observed"], inc["expected"]]],
    )  # fmt: skip
    if inc["detail"]:
        out += ["", f"Detail: {inc['detail']}"]

    out += ["", "## Likely cause", ""]
    if not explanation["explained"]:
        out += ["Cause unknown: no playbook cause reaches medium confidence.", ""]
    for i, cause in enumerate(explanation["causes"], 1):
        out.append(f"{i}. **{cause['text']}** ({cause['confidence']}, score {cause['score']:.2f})")
        out += [f"   - {item}" for item in cause["evidence"]]
    if explanation["causes"]:
        out.append("")
    out += [f"Playbook: `{explanation['playbook']}`."]

    out += ["", "## Where and when", "", explanation["where"] or "not analysed"]
    where = evidence.get("concentration") or {}
    if where.get("findings"):
        out += [""] + _table(
            ["Dimension", "Value", "Affected", "Rows", "Rate", "Rate elsewhere", "Share", "z"],
            [[f["dimension"], f["value"], f["affected_rows"], f["rows"], f"{f['rate']:.1%}",
              f"{f['rate_outside']:.1%}", f"{f['share']:.0%}", f["z"]]
             for f in where["findings"]],
        )  # fmt: skip
    if where.get("days"):
        out += [""] + _table(
            ["Low day", "Rows", "Of the median day"],
            [[d["day"], d["rows"], f"{d['share_of_median']:.0%}"] for d in where["days"]],
        )

    out += ["", "## Suggested actions", ""]
    if actions:
        out += ["Suggestions only: nothing runs them. Decide with "
                "`investigate.py --decide <incident> <action> approved|rejected|done --by <name>`.",
                ""]  # fmt: skip
        out += _table(
            ["Action", "Kind", "Status", "What"],
            [[a["action_id"], a["kind"], a["status"], a["text"]] for a in actions],
        )
    else:
        out.append("none")

    out += ["", "## Since when", ""]
    history = evidence["history"]
    if history and "value" in history[0]:
        out += _table(["Month", "Rate", "Observed"],
                      [[h["period"], h["value"], h["observed"]] for h in history])  # fmt: skip
    elif history:
        out += _table(
            ["Month", "Status", "Severity", "Observed", "Expected"],
            [[h["period"], h["status"], h["severity"], h["observed"], h["expected"]]
             for h in history],
        )  # fmt: skip
    else:
        out.append("no results")

    impact = evidence["impact"]
    out += ["", "## Impact", ""]
    rows = [["Month", impact["period"]]]
    labels = {
        "rows": "Rows",
        "silver_rows": "Silver rows",
        "quarantine_rows": "Quarantine rows",
        "affected_rows": "Affected rows",
        "score": "Quality score",
        "score_change": "Change from the month before",
    }
    for key, label in labels.items():
        if key in impact:
            rows.append([label, impact[key]])
    if "affected_share" in impact:
        rows.append(["Affected share", f"{impact['affected_share']:.2%}"])
    out += _table(["", ""], rows)

    metrics = evidence.get("metrics") or {}
    if metrics:
        periods = sorted({s["period"] for series in metrics.values() for s in series})
        values = {(m, s["period"]): s["value"] for m, series in metrics.items() for s in series}
        out += ["", "## Metrics", ""]
        out += _table(["Month", *metrics], [[p, *(values.get((m, p)) for m in metrics)]
                                            for p in periods])  # fmt: skip

    if evidence["samples"]:
        columns = list(evidence["samples"][0]["values"])
        out += ["", "## Sample rows", ""]
        out += _table(
            ["Row", *columns, "Reasons"],
            [[s["bronze_row_id"], *(s["values"][c] for c in columns), "; ".join(s["reasons"])]
             for s in evidence["samples"]],
        )  # fmt: skip

    out += ["", "## File versions", ""]
    out += _table(
        ["Month", "File", "SHA-256", "Rows", "Status", "Delivered", "Expected by"],
        [[v["period"], v["file_name"], v["sha256"][:12], v["rows"], v["status"],
          v["delivered_at"], v["expected_at"]] for v in evidence["versions"]],
    )  # fmt: skip

    similar = evidence["similar"]
    out += ["", "## Related incidents", ""]
    related = [[s["incident_id"], "same problem", s["severity"], f"{s['first_seen']} to "
                f"{s['last_seen']} ({s['status']})"] for s in similar["same_problem"]]  # fmt: skip
    related += [[s["incident_id"], "same month", s["severity"], s["title"]]
                for s in similar["same_month"]]  # fmt: skip
    out += _table(["Incident", "Relation", "Severity", ""], related) if related else ["none"]

    definition = evidence["definition"]
    out += ["", "## What is tested", "", definition.get("description") or "–"]
    if definition.get("kind") == "row_rule":
        out += ["", f"Rule `{inc['rule_id']}`: {definition.get('category')}, "
                f"{definition.get('check')}, action {definition.get('action')}, "
                f"params `{json.dumps(definition.get('params'), sort_keys=True)}`."]  # fmt: skip
    out += ["", "---", "", "Built from the incident's evidence pack by the template explainer "
            "(configs/playbooks.yml); no language model is involved.", ""]  # fmt: skip
    return "\n".join(out)


def write_report(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# --- one incident and all of them --------------------------------------------------------------


def investigate_incident(
    con: duckdb.DuckDBPyConnection,
    incident_id: str,
    run_id: str,
    books: ex.Playbooks,
    reports: Path | None,
) -> dict[str, Any]:
    evidence = ep.build_evidence(con, incident_id)
    ep.store(con, evidence)
    explanation = ex.explain(evidence, books)
    store_actions(con, incident_id, explanation["actions"])
    actions = stored_actions(con, incident_id)
    report_path = None
    inc = evidence["incident"]
    if reports is not None:
        path = reports / "investigations" / inc["dataset"] / f"{incident_id}.md"
        write_report(path, render_report(evidence, explanation, actions))
        report_path = _relative(path)
    top = explanation["causes"][0] if explanation["causes"] else None
    con.execute(
        "INSERT OR REPLACE INTO investigations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            incident_id,
            inc["dataset"],
            run_id,
            inc["last_seen"],
            explanation["playbook"],
            explanation["explained"],
            top["cause_id"] if top else None,
            top["confidence"] if top else None,
            explanation["headline"],
            explanation["where"],
            json.dumps(explanation, sort_keys=True, default=str),
            hashlib.md5(ep.to_json(evidence).encode()).hexdigest(),
            report_path,
            utc_now(),
        ],
    )
    return explanation


def investigate_all(
    con: duckdb.DuckDBPyConnection,
    dataset: str,
    incident_ids: list[str] | None = None,
    reports: Path | None = None,
    books: ex.Playbooks | None = None,
    command: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Investigate the incidents of a dataset (all, or the ones given) in one tracked run."""
    books = books or ex.load_playbooks()
    if incident_ids is None:
        incident_ids = [
            r[0]
            for r in con.execute(
                "SELECT incident_id FROM incidents WHERE dataset = ? ORDER BY 1", [dataset]
            ).fetchall()
        ]
    else:
        for i in incident_ids:
            get_incident(con, i)  # unknown ids fail before the run starts
    with track_run(con, dataset, command or "investigate") as run:
        results = [investigate_incident(con, i, run.run_id, books, reports) for i in incident_ids]
        unknown = sum(not e["explained"] for e in results)
        if unknown:
            run.warn(f"{unknown} incident(s) with cause unknown")
    return run.run_id, results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Investigate incidents and write reports.")
    parser.add_argument("--dataset", default="yellow_taxi", help="dataset name in datasets.yml")
    parser.add_argument("--incident", action="append", help="incident id (repeatable)")
    parser.add_argument(
        "--decide", nargs=3, metavar=("INCIDENT", "ACTION", "DECISION"),
        help=f"record a decision on a suggested action: {', '.join(DECISIONS)}",
    )  # fmt: skip
    parser.add_argument("--by", help="who decides (with --decide)")
    parser.add_argument("--reports", type=Path, help="reports folder (default: paths.reports)")
    parser.add_argument("--no-reports", action="store_true", help="do not write Markdown")
    parser.add_argument("--database", type=Path, help="DuckDB file (default: paths.duckdb)")
    args = parser.parse_args(argv)
    try:
        books = ex.load_playbooks()
        reports = None if args.no_reports else (args.reports or reports_dir())
    except ConfigError as exc:
        print(exc)
        return 2
    with db.connect(args.database) as con:
        db.init_schema(con)
        if args.decide:
            if not args.by:
                print("--decide needs --by <name>")
                return 2
            try:
                decide(con, *args.decide, args.by)
            except (ValueError, IncidentError) as exc:
                print(exc)
                return 2
            print(f"{args.decide[0]} {args.decide[1]}: {args.decide[2]} by {args.by}")
            return 0
        try:
            run_id, results = investigate_all(con, args.dataset, args.incident, reports, books)
        except IncidentError as exc:
            print(exc)
            return 2
        pending = con.execute(
            "SELECT count(*) FROM investigation_actions WHERE status = 'pending_approval'"
        ).fetchone()[0]
    if not results:
        print(f"no incidents for '{args.dataset}'")
        return 0
    for e in results:
        print(f"{e['incident_id']}  {e['headline']}")
    unknown = sum(not e["explained"] for e in results)
    print(
        f"\n{run_id}: {len(results)} incident(s) investigated, {unknown} with cause unknown, "
        f"{pending} action(s) pending approval"
    )
    if reports is not None:
        print(f"reports in {_relative(reports / 'investigations' / args.dataset)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
