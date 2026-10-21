"""Work with incidents from the command line.

python src/incidents/incidents_cli.py update                     # apply the latest check run
python src/incidents/incidents_cli.py list                       # active incidents
python src/incidents/incidents_cli.py list --all
python src/incidents/incidents_cli.py show INC-0001              # details and history
python src/incidents/incidents_cli.py investigate INC-0001 --note "asked TLC"
python src/incidents/incidents_cli.py reopen INC-0001 --note "not a source problem"
python src/incidents/incidents_cli.py resolve INC-0001 --note "re-export loaded"
python src/incidents/incidents_cli.py note INC-0001 --note "waiting for a reply"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    # Allow running this file directly: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import db  # noqa: E402
from incidents import incident_manager as im  # noqa: E402
from incidents import lifecycle  # noqa: E402


def _table(incidents: list[im.Incident]) -> str:
    if not incidents:
        return "no incidents"
    frame = pd.DataFrame(
        [
            {
                "incident": i.incident_id,
                "severity": i.severity,
                "status": i.status,
                "title": i.title,
                "first_seen": i.first_seen_period,
                "last_seen": i.last_seen_period,
                "seen": i.occurrences,
                "clean": i.clean_streak,
            }
            for i in incidents
        ]
    )
    return frame.to_string(index=False)


def _show(con, incident_id: str) -> str:
    i = im.get_incident(con, incident_id)
    lines = [
        f"{i.incident_id}  {i.severity.upper()}  {i.status}  {i.title}",
        f"  {i.dataset} / {i.source} / {i.rule_id}"
        + (f" / {i.column_name}" if i.column_name else ""),
        f"  seen {i.occurrences}x, {i.first_seen_period} to {i.last_seen_period}; "
        f"clean streak {i.clean_streak}",
        f"  observed: {i.latest_observed}",
        f"  expected: {i.latest_expected}",
        f"  detail:   {i.latest_detail}",
    ]
    if i.recurrence_of:
        lines.append(f"  recurrence of {i.recurrence_of}")
    lines.append("  history:")
    for e in im.events(con, incident_id):
        where = f" {e.period}" if e.period else ""
        lines.append(
            f"    {e.seq:>3} {e.created_at:%Y-%m-%d %H:%M} {e.event:<13}{where}  {e.detail or ''}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Work with incidents.")
    parser.add_argument("--database", type=Path, help="DuckDB file (default: paths.duckdb)")
    parser.add_argument("--dataset", default="yellow_taxi", help="dataset name")
    sub = parser.add_subparsers(dest="command", required=True)
    update = sub.add_parser(
        "update", help="apply the latest dataset check run and the latest row rule run"
    )
    update.add_argument("--check-run", help="run id of a run_checks.py run (default: latest)")
    update.add_argument("--rule-run", help="run id of a run_rules.py run (default: latest)")
    listing = sub.add_parser("list", help="list incidents")
    listing.add_argument("--all", action="store_true", help="include resolved incidents")
    sub.add_parser("show", help="details and history").add_argument("incident")
    for name in ("investigate", "reopen", "resolve", "note"):
        action = sub.add_parser(name)
        action.add_argument("incident")
        action.add_argument("--note", required=name == "note", help="text for the history")
    args = parser.parse_args(argv)

    with db.connect(args.database) as con:
        db.init_schema(con)
        try:
            if args.command == "update":
                sources = [
                    (
                        "dataset_check_results",
                        args.check_run,
                        lifecycle.apply_check_run,
                        "checks",
                        "run_checks.py",
                    ),
                    (
                        "rule_run_log",
                        args.rule_run,
                        lifecycle.apply_rule_run,
                        "rules",
                        "run_rules.py",
                    ),
                ]
                applied = 0
                for table, chosen, apply, label, command in sources:
                    run_id = (
                        chosen
                        or con.execute(
                            f"SELECT max(run_id) FROM {table} WHERE dataset = ?", [args.dataset]
                        ).fetchone()[0]
                    )
                    if not run_id:
                        print(f"{label}: no run found (run python src/quality/{command})")
                        continue
                    r = apply(con, run_id, args.dataset)
                    applied += 1
                    print(
                        f"{label} {run_id}: {len(r.opened)} opened, {len(r.updated)} seen again, "
                        f"{len(r.clean)} clean, {len(r.resolved)} resolved"
                    )
                if not applied:
                    return 1
                print(_table(im.list_incidents(con, dataset=args.dataset, active_only=True)))
            elif args.command == "list":
                print(
                    _table(im.list_incidents(con, dataset=args.dataset, active_only=not args.all))
                )
            elif args.command == "show":
                print(_show(con, args.incident))
            elif args.command == "note":
                im.add_note(con, args.incident, args.note)
                print(_show(con, args.incident))
            else:
                status = {"investigate": "INVESTIGATING", "reopen": "OPEN", "resolve": "RESOLVED"}[
                    args.command
                ]
                im.set_status(con, args.incident, status, note=args.note)
                print(_show(con, args.incident))
        except im.IncidentError as exc:
            print(f"error: {exc}")
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
