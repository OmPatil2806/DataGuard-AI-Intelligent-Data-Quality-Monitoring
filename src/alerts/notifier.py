"""Alert on new critical and high incidents: outbox file and Slack (Phase 10).

    python src/alerts/notifier.py                    # alert what is new, every dataset
    python src/alerts/notifier.py --dataset ved_signals
    python src/alerts/notifier.py --silent           # record as alerted without sending

Run after the investigation (the `alert` stage of rebuild.py), so a message carries the likely
cause. An active incident whose severity is listed in configs/alerts.yml is alerted once per
channel and severity: again only when it escalates, never when it is merely seen again. Every
alert is recorded in alert_log; a send that failed is retried by the next run.

    outbox  one JSON line per alert in <data_dir>/alerts/outbox.jsonl (DATAGUARD_ALERTS_DIR)
    slack   POST to the incoming webhook in $DATAGUARD_SLACK_WEBHOOK (skipped when unset);
            the URL is a secret and is never stored or printed

Exit code: 0, or 1 when a send failed (recorded; retried next time), 2 for a bad config.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import requests

if __package__ in (None, ""):
    # Allow running this file directly: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import db  # noqa: E402
from common.config import (  # noqa: E402
    SEVERITIES,
    ConfigError,
    config_dir,
    load_datasets_config,
    load_yaml,
)
from common.runs import track_run, utc_now  # noqa: E402

CHANNELS = ("outbox", "slack")
WEBHOOK_ENV = "DATAGUARD_SLACK_WEBHOOK"
ALERTS_DIR_ENV = "DATAGUARD_ALERTS_DIR"
OUTBOX = "outbox.jsonl"
TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class AlertSettings:
    severities: tuple[str, ...]
    channels: tuple[str, ...]
    dashboard_url: str | None = None


@dataclass(frozen=True)
class Alert:
    incident_id: str
    dataset: str
    severity: str
    title: str
    status: str
    first_seen: str
    last_seen: str
    escalated: bool  # alerted before at another severity
    headline: str | None
    where: str | None
    actions_pending: int
    report_path: str | None

    def text(self, dashboard_url: str | None) -> str:
        tag = "escalated to" if self.escalated else "new"
        lines = [
            f"[{self.severity.upper()}] {self.incident_id} {self.title} "
            f"({self.dataset}, {self.last_seen}; {tag} {self.severity})",
            self.headline or "not investigated yet",
        ]
        if self.where and not self.where.startswith("not analysed"):
            lines.append(f"where/when: {self.where}")
        if self.actions_pending:
            lines.append(f"{self.actions_pending} suggested action(s) waiting for approval")
        if self.report_path:
            lines.append(f"report: {self.report_path}")
        if dashboard_url:
            lines.append(f"dashboard: {dashboard_url}")
        return "\n".join(lines)

    def record(self, text: str) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "dataset": self.dataset,
            "severity": self.severity,
            "escalated": self.escalated,
            "title": self.title,
            "period": self.last_seen,
            "text": text,
        }


def load_settings(path: Path | None = None) -> AlertSettings:
    path = Path(path) if path else config_dir() / "alerts.yml"
    raw = load_yaml(path)
    severities = raw.get("severities") or []
    if not isinstance(severities, list) or not set(severities) <= set(SEVERITIES):
        raise ConfigError(f"{path.name}: severities must be a list of {', '.join(SEVERITIES)}")
    channels = raw.get("channels") or []
    if not isinstance(channels, list) or not set(channels) <= set(CHANNELS):
        raise ConfigError(f"{path.name}: channels must be a list of {', '.join(CHANNELS)}")
    return AlertSettings(tuple(severities), tuple(channels), raw.get("dashboard_url"))


def alerts_dir() -> Path:
    if os.environ.get(ALERTS_DIR_ENV):
        return Path(os.environ[ALERTS_DIR_ENV])
    return load_datasets_config().paths["data_dir"] / "alerts"


def due(
    con: duckdb.DuckDBPyConnection,
    settings: AlertSettings,
    channel: str,
    dataset: str | None = None,
) -> list[Alert]:
    """Active incidents at an alerted severity not yet sent (or skipped) on this channel."""
    if not settings.severities:
        return []
    marks = ", ".join("?" * len(settings.severities))
    rows = con.execute(
        f"""
        SELECT i.incident_id, i.dataset, i.severity, i.title, i.status, i.first_seen_period,
               i.last_seen_period,
               EXISTS (SELECT 1 FROM alert_log a WHERE a.incident_id = i.incident_id
                       AND a.channel = ? AND a.severity != i.severity
                       AND a.status IN ('sent', 'skipped')) AS escalated,
               v.headline, v.where_verdict,
               (SELECT count(*) FROM investigation_actions x
                WHERE x.incident_id = i.incident_id AND x.status = 'pending_approval'),
               v.report_path
        FROM incidents i LEFT JOIN investigations v USING (incident_id)
        WHERE i.status != 'RESOLVED' AND i.severity IN ({marks})
          AND (? IS NULL OR i.dataset = ?)
          AND NOT EXISTS (SELECT 1 FROM alert_log a WHERE a.incident_id = i.incident_id
                          AND a.severity = i.severity AND a.channel = ?
                          AND a.status IN ('sent', 'skipped'))
        ORDER BY i.incident_id
        """,
        [channel, *settings.severities, dataset, dataset, channel],
    ).fetchall()
    return [Alert(*row) for row in rows]


def send_slack(text: str, session: requests.Session | None = None) -> str | None:
    """POST to the webhook; None when sent, else the error (without the secret URL)."""
    url = os.environ.get(WEBHOOK_ENV)
    if not url:
        return f"{WEBHOOK_ENV} is not set"
    try:
        response = (session or requests).post(url, json={"text": text}, timeout=TIMEOUT_SECONDS)
        if response.status_code >= 400:
            return f"Slack answered HTTP {response.status_code}"
    except requests.RequestException as exc:
        return f"{type(exc).__name__}: could not reach Slack"
    return None


def notify(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    settings: AlertSettings,
    dataset: str | None = None,
    outbox: Path | None = None,
    silent: bool = False,
    session: requests.Session | None = None,
) -> list[tuple[Alert, str, str, str | None]]:
    """Send what is due on every channel; returns (alert, channel, status, error) per send."""
    results = []
    folder = outbox or alerts_dir()
    for channel in settings.channels:
        if channel == "slack" and not silent and not os.environ.get(WEBHOOK_ENV):
            continue  # no webhook: Slack is off, nothing recorded, sent once it is set
        for alert in due(con, settings, channel, dataset):
            text = alert.text(settings.dashboard_url)
            error = None
            if silent:
                status = "skipped"
            elif channel == "outbox":
                folder.mkdir(parents=True, exist_ok=True)
                line = {"created_at": utc_now().isoformat(), "run_id": run_id, **alert.record(text)}
                with (folder / OUTBOX).open("a", encoding="utf-8") as f:
                    f.write(json.dumps(line) + "\n")
                status = "sent"
            else:
                error = send_slack(text, session)
                status = "failed" if error else "sent"
            con.execute(
                "INSERT OR REPLACE INTO alert_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [alert.incident_id, alert.severity, channel, alert.dataset, status, run_id, text,
                 error, utc_now()],
            )  # fmt: skip
            results.append((alert, channel, status, error))
    return results


def main(argv: list[str] | None = None, session: requests.Session | None = None) -> int:
    parser = argparse.ArgumentParser(description="Alert on new critical and high incidents.")
    parser.add_argument("--dataset", help="only this dataset (default: every dataset)")
    parser.add_argument("--outbox", type=Path, help="outbox folder (default: data/alerts)")
    parser.add_argument("--silent", action="store_true", help="record as alerted, send nothing")
    parser.add_argument("--database", type=Path, help="DuckDB file (default: paths.duckdb)")
    args = parser.parse_args(argv)
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(exc)
        return 2
    with db.connect(args.database) as con:
        db.init_schema(con)
        with track_run(con, args.dataset or "all", command="alert") as run:
            results = notify(con, run.run_id, settings, args.dataset, args.outbox, args.silent,
                             session)  # fmt: skip
            failed = [r for r in results if r[2] == "failed"]
            if failed:
                run.warn(f"{len(failed)} alert(s) not sent, retried next run")
    if "slack" in settings.channels and not os.environ.get(WEBHOOK_ENV) and not args.silent:
        print(f"slack: off ({WEBHOOK_ENV} not set)")
    for alert, channel, status, error in results:
        note = f": {error}" if error else ""
        print(
            f"{channel:<7} {status:<8} {alert.incident_id} [{alert.severity}] {alert.title}{note}"
        )
    sent = sum(1 for r in results if r[2] == "sent")
    print(f"{run.run_id}: {sent} alert(s) sent, {len(failed)} failed"
          + (" (recorded as alerted, nothing sent)" if args.silent else ""))  # fmt: skip
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
