"""DataGuard dashboard (Phase 10): the monitoring results in the browser.

    streamlit run src/dashboard/app.py
    DATAGUARD_DATABASE=other.duckdb streamlit run src/dashboard/app.py

Opens http://localhost:8501 (the link in every alert). Reads the DuckDB file read-only;
rebuild.py fills it. The only writes are the decisions people take on the Incidents page
(dashboard/actions.py). Pages in src/dashboard/views/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

if __package__ in (None, ""):
    # streamlit runs this file as a script: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard import data, ui  # noqa: E402

VIEWS = Path(__file__).resolve().parent / "views"


def coming(step: str):
    def page() -> None:
        st.title(page.title)
        st.info(f"This page comes in Phase {step}.", icon="🚧")

    return page


def pages() -> list:
    def soon(title: str, step: str, icon: str, url: str):
        page = coming(step)
        page.title = title
        return st.Page(page, title=title, icon=icon, url_path=url)

    return [
        st.Page(str(VIEWS / "overview.py"), title="Overview", icon="🏠", default=True),
        st.Page(str(VIEWS / "health.py"), title="Dataset Health", icon="📈", url_path="health"),
        st.Page(str(VIEWS / "incidents.py"), title="Incidents", icon="🚨", url_path="incidents"),
        soon("Quarantine Explorer", "10.5", "🧪", "quarantine"),
        soon("Lineage", "10.5", "🔗", "lineage"),
    ]


def main() -> None:
    st.set_page_config(page_title="DataGuard AI", page_icon="🛡️", layout="wide")
    page = st.navigation(pages())
    st.session_state["context"] = ui.sidebar()
    try:
        page.run()
    except data.DashboardError as exc:
        ui.stop_with(str(exc))


main()
