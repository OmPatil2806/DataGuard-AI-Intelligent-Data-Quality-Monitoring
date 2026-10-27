"""Quarantine Explorer: the month's problem rows, why they failed, and the rows themselves."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from dashboard import data, ui

ctx: ui.Context = st.session_state["context"]
LIMIT = 200

st.title(f"Quarantine Explorer: {ctx.dataset}")
st.caption(f"{ctx.period} · the rows the rules kept out of Silver or warned about, and the rows "
           "the anomaly model flagged")  # fmt: skip

counts = ui.load("problem_counts", ctx.database, ctx.dataset, ctx.period)
if counts["quarantine"] is None and counts["warned"] is None:
    ui.stop_with(f"{ctx.period} has no Silver or Quarantine file yet. "
                 "Run: python src/gold/rebuild.py")  # fmt: skip

tiles = st.columns(3)
for tile, (view, label) in zip(tiles, (("quarantine", "Quarantined rows"),
                                       ("warned", "Warned rows"),
                                       ("ml", "Flagged by the model")), strict=True):  # fmt: skip
    value = counts[view]
    tile.metric(label, "—" if value is None else f"{value:,}", help=data.VIEWS[view], border=True)

# --- Rows per rule ----------------------------------------------------------------------------
rules = ui.load("rule_counts", ctx.database, ctx.dataset, ctx.period)
st.subheader("Rows failing each rule")
if rules.empty:
    st.caption("No row failed a rule this month.")
else:
    shown = rules.iloc[::-1]  # most rows on top
    figure = go.Figure()
    for column, name, color in (("quarantine", "Quarantined", ui.SERIES[1]),
                                ("warned", "Warned (kept in Silver)", ui.SERIES[0])):  # fmt: skip
        figure.add_bar(
            x=shown[column], y=shown["rule_id"], orientation="h", name=name, marker_color=color,
            marker={"line": {"width": 1, "color": "rgba(255,255,255,0.9)"}},
            hovertemplate="%{y}: %{x:,} row(s) " + name.lower() + "<extra></extra>",
        )  # fmt: skip
    figure.update_layout(
        barmode="stack", height=max(220, 28 * len(shown) + 80), legend_traceorder="normal",
        margin={"l": 0, "r": 0, "t": 10, "b": 0}, xaxis={"title": "rows"},
        legend={"orientation": "h", "y": -0.15},
    )  # fmt: skip
    st.plotly_chart(figure, key="rules_chart")
    st.caption("A row can fail several rules, so the bars can add up to more than the rows.")

# --- The rows ---------------------------------------------------------------------------------
st.subheader("The rows")
left, right = st.columns([3, 2])
view = left.radio("Show", list(data.VIEWS), format_func=data.VIEWS.get, horizontal=True,
                  key="view")  # fmt: skip
column = {"quarantine": "quarantine", "warned": "warned"}.get(view)
options = rules[rules[column] > 0]["rule_id"].tolist() if column else rules["rule_id"].tolist()
rule = right.selectbox("Failing rule", ["(any rule)", *options], key=f"rule:{view}")
rule = None if rule == "(any rule)" else rule

rows, total = ui.load("problem_rows", ctx.database, ctx.dataset, ctx.period, view, rule, LIMIT)
if not total:
    nothing = "No row flagged: the model has not scored this month, or found nothing."
    st.caption(nothing if view == "ml" else "No row.")
else:
    st.caption(f"{total:,} row(s)" + (f", the first {LIMIT} shown" if total > LIMIT else "")
               + ". bronze_row_id is the row's position in the source file: trace it on the "
               "Lineage page.")  # fmt: skip
    st.dataframe(
        rows,
        hide_index=True,
        column_config={
            "bronze_row_id": "Row",
            "anomaly_score": st.column_config.NumberColumn("Anomaly score", format="%.3f"),
            "model_reasons": st.column_config.TextColumn("Model's reasons", width="large"),
            "reasons": st.column_config.TextColumn("Why", width="large"),
            "failed_rules": "Rules failed",
        },
        key="rows",
    )  # fmt: skip
    st.download_button(
        "Download these rows (CSV)", rows.to_csv(index=False).encode("utf-8"),
        file_name=f"{ctx.dataset}_{ctx.period}_{view}.csv", mime="text/csv", key="download",
    )  # fmt: skip
