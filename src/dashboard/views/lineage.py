"""Lineage: where every month and every row came from, and whether the row counts add up."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import data, ui

ctx: ui.Context = st.session_state["context"]
LINEAGE_LABELS = {"ok": "✅ ok", "not_processed": "⏳ not processed",
                  "missing_file": "❌ missing file", "mismatch": "⚠️ mismatch"}  # fmt: skip

months = ui.load("lineage", ctx.database, ctx.dataset)
month = months[months["period"] == ctx.period]
st.title(f"Lineage: {ctx.dataset}")
if month.empty:
    ui.stop_with(f"{ctx.period} is not in the lineage table. Run: python src/gold/rebuild.py")
month = month.iloc[0]
st.caption(f"{ctx.period} · {month['file_name']} · {LINEAGE_LABELS.get(month['status'])}"
           + (f": {month['detail']}" if month["detail"] else ""))  # fmt: skip

# --- The month's flow -------------------------------------------------------------------------
st.subheader(f"Where the rows of {ctx.period} went")
if pd.isna(month["silver_rows"]):
    st.caption("Not processed into Silver yet.")
else:
    counts = ui.load("problem_counts", ctx.database, ctx.dataset, ctx.period)
    warned = counts["warned"] or 0
    bronze, silver = int(month["bronze_rows"]), int(month["silver_rows"])
    quarantine = int(month["quarantine_rows"])
    nodes = [f"Source file<br>{int(month['manifest_rows']):,}", f"Bronze<br>{bronze:,}",
             f"Silver, passed<br>{silver - warned:,}", f"Silver, warned<br>{warned:,}",
             f"Quarantine<br>{quarantine:,}"]  # fmt: skip
    figure = go.Figure(go.Sankey(
        arrangement="fixed", textfont={"size": 13, "shadow": "none"},
        node={"label": nodes, "pad": 24, "thickness": 16,
              "color": [ui.MUTED, ui.MUTED, ui.SERIES[0], "#1baf7a", ui.SERIES[1]],
              "hovertemplate": "%{label}<extra></extra>"},
        link={"source": [0, 1, 1, 1], "target": [1, 2, 3, 4],
              "value": [bronze, silver - warned, warned, quarantine],
              "color": ["rgba(137,135,129,0.35)", "rgba(42,120,214,0.35)",
                        "rgba(27,175,122,0.35)", "rgba(235,104,52,0.35)"],
              "hovertemplate": "%{value:,} rows<extra></extra>"},
    ))  # fmt: skip
    figure.update_layout(height=280, margin={"l": 0, "r": 0, "t": 10, "b": 10})
    st.plotly_chart(figure, key="flow")
    st.caption("Counted in the files themselves when Gold was built, and compared with the "
               "manifest and silver_log: the status says whether they agree.")  # fmt: skip

with st.expander(f"Files and runs of {ctx.period}"):
    st.markdown("\n".join([
        f"- **Source:** {month['source_url'] or '—'}",
        f"- **File:** `{month['file_name']}`, SHA-256 `{month['sha256']}`, "
        f"{int(month['size_bytes'] or 0):,} bytes, {int(month['versions_registered'])} "
        "version(s) registered",
        f"- **Bronze:** `{month['bronze_path']}` (run {month['loaded_run_id']})",
        f"- **Silver:** `{month['silver_path'] or '—'}` (run {month['silver_run_id'] or '—'})",
        f"- **Quarantine:** `{month['quarantine_path'] or '—'}`",
        f"- **Rules:** run {month['rule_run_id'] or '—'} · **checks:** run "
        f"{month['check_run_id'] or '—'} · **scored:** {'yes' if month['scored'] else 'no'}",
    ]))  # fmt: skip

# --- Every month ------------------------------------------------------------------------------
st.subheader("Every month")
st.dataframe(
    months.assign(status=months["status"].map(lambda s: LINEAGE_LABELS.get(s, s)),
                  sha256=months["sha256"].str[:12], detail=months["detail"].fillna("")),
    hide_index=True,
    column_order=["period", "status", "file_name", "sha256", "versions_registered",
                  "manifest_rows", "bronze_rows", "silver_rows", "quarantine_rows", "detail"],
    column_config={
        "period": "Month", "status": "Counts", "file_name": "File", "sha256": "Version",
        "versions_registered": "Versions",
        "manifest_rows": st.column_config.NumberColumn("Registered", format="localized"),
        "bronze_rows": st.column_config.NumberColumn("Bronze", format="localized"),
        "silver_rows": st.column_config.NumberColumn("Silver", format="localized"),
        "quarantine_rows": st.column_config.NumberColumn("Quarantine", format="localized"),
        "detail": st.column_config.TextColumn("Detail", width="large"),
    },
    key="lineage",
)  # fmt: skip

# --- Trace one row ----------------------------------------------------------------------------
st.subheader("Trace one row")
row_id = st.number_input(
    f"Row of {ctx.period} (bronze_row_id, its position in the source file)", min_value=1,
    max_value=max(1, int(month["bronze_rows"] or 1)), value=1, step=1, key=f"row:{ctx.period}",
)  # fmt: skip
try:
    trace = ui.load("trace_row", ctx.database, ctx.dataset, ctx.period, int(row_id))
except data.DashboardError as exc:
    st.warning(str(exc))
    st.stop()
if trace["layer"] is None:
    st.info("Not processed into Silver yet.", icon="⏳")
elif trace["layer"] == "Quarantine":
    st.error("In **Quarantine**: " + "; ".join(trace["reasons"]), icon="🧪")
elif trace["reasons"]:
    st.warning("In **Silver with a warning**: " + "; ".join(trace["reasons"]), icon="⚠️")
else:
    st.success("In **Silver**: passed every rule.", icon="✅")
steps = [
    ("Source", trace["source_url"] or "—"),
    ("File", f"{trace['source_file']} (SHA-256 {trace['sha256'][:12]})"),
    ("Bronze", f"{trace['bronze_path']} (run {trace['loaded_run_id']})"),
]
if trace["layer"]:
    steps.append((trace["layer"], f"{trace['layer_path']} (run {trace['silver_run_id']})"))
if trace["check_run_id"]:
    steps.append(("Checks", f"run {trace['check_run_id']}"))
if trace["month_score"] is not None:
    steps.append(("Score", f"{trace['month_score']:.1f} for the month"))
st.dataframe(pd.DataFrame(steps, columns=["Step", "Where"]), hide_index=True, key="trace")
values = {k: v for k, v in trace["bronze_row"].items() if k not in ("bronze_row_id", "source_file")}
st.dataframe(
    pd.DataFrame({"Column": list(values),
                  "Value as delivered": [ui.format_value(v) for v in values.values()]}),
    hide_index=True, key="values",
)  # fmt: skip
