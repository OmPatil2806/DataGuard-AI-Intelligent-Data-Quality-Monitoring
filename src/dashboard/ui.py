"""Streamlit parts every page shares: the chosen dataset and month, cached reads, labels.

Reads are cached per database file and its modification time, so a page redraw does not query
again, and a rebuild (which changes the file) shows on the next redraw by itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard import data

# Status colors (fixed, never used for series) and the series colors of the charts.
STATUS_COLORS = {
    "healthy": "#0ca30c",
    "warning": "#fab219",
    "critical": "#d03b3b",
    "not_scored": "#898781",
}
SEVERITY_COLORS = {"critical": "#d03b3b", "high": "#ec835a", "medium": "#fab219", "low": "#898781"}
SERIES = ("#2a78d6", "#eb6834")  # categorical slots 1 and 2
MUTED = "#898781"
HEALTH_LABELS = {
    "healthy": "✅ healthy",
    "warning": "⚠️ warning",
    "critical": "⛔ critical",
    "not_scored": "⏳ not scored",
}
HEALTH_WORDS = {"healthy": "Healthy", "warning": "Warning", "critical": "Critical",
                "not_scored": "Not scored"}  # fmt: skip
STATUS_WORDS = {"pass": "passed", "fail": "failed", "skipped": "skipped"}
STATUS_LABELS = {"pass": "✅ pass", "fail": "❌ fail", "skipped": "⏭️ skipped"}
ACTION_LABELS = {"pending_approval": "⏳ waiting", "approved": "👍 approved",
                 "rejected": "👎 rejected", "done": "✅ done"}  # fmt: skip
DECISION_BUTTONS = {"approved": "Approve", "rejected": "Reject", "done": "Mark done"}
SEVERITY_LABELS = {"critical": "⛔ critical", "high": "🔴 high", "medium": "🟠 medium",
                   "low": "⚪ low"}  # fmt: skip


@dataclass(frozen=True)
class Context:
    database: Path
    dataset: str
    period: str


def _stamp(database: Path) -> tuple[int, int]:
    """Modification times of the file and its write-ahead log (a write may sit in the log)."""

    def mtime(path: Path) -> int:
        try:
            return path.stat().st_mtime_ns
        except OSError:
            return 0

    return mtime(database), mtime(database.with_name(database.name + ".wal"))


@st.cache_data(ttl=300, show_spinner=False)
def _cached(name: str, database: str, stamp: tuple[int, int], *args):
    return getattr(data, name)(Path(database), *args)


def load(name: str, database: Path, *args):
    """data.<name>(database, *args), cached until the database file changes."""
    return _cached(name, str(database), _stamp(database), *args)


def stop_with(message: str) -> None:
    st.info(message, icon="ℹ️")
    st.stop()


def sidebar() -> Context:
    """Dataset and month pickers; stops the page with a hint when nothing can be shown yet."""
    database = data.database_path()
    st.sidebar.title("DataGuard AI")
    try:
        data.check_ready(database)
        datasets = load("datasets", database)
    except data.DashboardError as exc:
        stop_with(str(exc))
    dataset = st.sidebar.selectbox("Dataset", datasets, key="dataset")
    periods = load("health", database, dataset)["period"].tolist()
    period = st.sidebar.selectbox(
        "Month", list(reversed(periods)), key=f"period:{dataset}", help="latest first"
    )
    if st.sidebar.button("Refresh", help="read the database again"):
        st.cache_data.clear()
    st.sidebar.caption(f"Database: `{database}`")
    return Context(database, dataset, period)


def health_label(value: str | None) -> str:
    return HEALTH_LABELS.get(value or "", value or "")


def severity_label(value: str | None) -> str:
    return SEVERITY_LABELS.get(value or "", value or "")


def status_label(value: str | None) -> str:
    return STATUS_LABELS.get(value or "", value or "")


def format_time(value) -> str:
    return "" if value is None or pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")


def format_value(value) -> str:
    """A delivered value for display: empty as NULL, floats without binary noise."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return "NULL"
    if isinstance(value, float):
        return repr(round(value, 6))
    return str(value)
