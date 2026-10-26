"""Dataset Health: every month of a dataset, and the chosen month's checks, columns and score."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import data, ui

ctx: ui.Context = st.session_state["context"]
months = ui.load("health", ctx.database, ctx.dataset)
month = months[months["period"] == ctx.period].iloc[0]

st.title(f"Dataset Health: {ctx.dataset}")
score = "not scored" if pd.isna(month["score"]) else f"score {month['score']:.1f}"
st.caption(f"{ctx.period} · {ui.health_label(month['health'])} · {score}")

LAYOUT = {"height": 320, "margin": {"l": 0, "r": 0, "t": 10, "b": 0},
          "legend": {"orientation": "h", "y": -0.18}}  # fmt: skip

# --- Every month ------------------------------------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Where the rows went")
    figure = go.Figure()
    for column, name, color in (("silver_rows", "Silver", ui.SERIES[0]),
                                ("quarantine_rows", "Quarantine", ui.SERIES[1])):  # fmt: skip
        figure.add_bar(
            x=months["period"], y=months[column], name=name, marker_color=color,
            marker={"line": {"width": 1, "color": "rgba(255,255,255,0.9)"}},
            customdata=months[["bronze_rows", "warned_rows"]],
            hovertemplate="%{x}: %{y:,} " + name + " of %{customdata[0]:,} rows"
            "<br>warned (kept in Silver) %{customdata[1]:,}<extra></extra>",
        )  # fmt: skip
    figure.update_layout(barmode="stack", bargap=0.3, yaxis={"title": "rows"},
                         xaxis={"type": "category"}, legend_traceorder="normal",
                         **LAYOUT)  # fmt: skip
    st.plotly_chart(figure, key="rows_chart")
with right:
    st.subheader(f"Score by category in {ctx.period}")
    parts = ui.load("score_parts", ctx.database, ctx.dataset, ctx.period)
    if parts.empty:
        st.caption("This month has no quality score yet.")
    else:
        parts = parts.iloc[::-1]  # lowest pass rate on top
        figure = go.Figure()
        figure.add_bar(
            x=parts["pass_rate"] * 100, y=parts["category"], orientation="h",
            marker_color=ui.SERIES[0], marker={"cornerradius": 4},
            customdata=parts[["weight", "source"]], texttemplate="%{x:.0f}%",
            textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:.1f}% pass<br>weight %{customdata[0]:g}"
            "<br>from %{customdata[1]}<extra></extra>",
        )  # fmt: skip
        figure.update_layout(xaxis={"title": "% passed", "range": [0, 112]}, showlegend=False,
                             **LAYOUT)  # fmt: skip
        st.plotly_chart(figure, key="score_parts")

st.dataframe(
    months.assign(health=months["health"].map(ui.health_label),
                  worst_severity=months["worst_severity"].map(ui.severity_label)),
    hide_index=True,
    column_order=["period", "health", "score", "bronze_rows", "silver_rows", "quarantine_rows",
                  "warned_rows", "quarantine_pct", "checks_failed", "checks_run",
                  "incidents_found", "worst_severity"],
    column_config={
        "period": "Month", "health": "Health",
        "score": st.column_config.NumberColumn("Score", format="%.1f"),
        "bronze_rows": st.column_config.NumberColumn("Rows", format="localized"),
        "silver_rows": st.column_config.NumberColumn("Silver", format="localized"),
        "quarantine_rows": st.column_config.NumberColumn("Quarantine", format="localized"),
        "warned_rows": st.column_config.NumberColumn("Warned", format="localized"),
        "quarantine_pct": st.column_config.NumberColumn("Quarantined", format="%.2f%%"),
        "checks_failed": "Checks failed", "checks_run": "Checks run",
        "incidents_found": "Incidents", "worst_severity": "Worst finding",
    },
    key="months",
)  # fmt: skip

# --- The chosen month: checks and columns -----------------------------------------------------
st.subheader(f"Dataset checks in {ctx.period}")
results = ui.load("checks", ctx.database, ctx.dataset, ctx.period)
failed = results[results["status"] == "fail"]
st.caption(f"{len(failed)} failed, {(results['status'] == 'skipped').sum()} skipped, "
           f"{(results['status'] == 'pass').sum()} passed")  # fmt: skip
if st.toggle("Only failed checks", value=len(failed) > 0, key="only_failed"):
    results = failed
st.dataframe(
    results.assign(status=results["status"].map(ui.status_label),
                   severity=results["severity"].map(ui.severity_label)),
    hide_index=True,
    column_config={"check": "Check", "column_name": "Column", "status": "Result",
                   "severity": "Severity", "observed": "Observed", "expected": "Expected",
                   "detail": st.column_config.TextColumn("Detail", width="large")},
    key="checks",
)  # fmt: skip

st.subheader(f"Columns in {ctx.period}")
profile = ui.load("column_profile", ctx.database, ctx.dataset, ctx.period)
st.dataframe(
    profile.assign(
        null_rate=profile["null_rate"] * 100,
        null_rate_baseline=profile["null_rate_baseline"] * 100,
        null_rate_status=profile["null_rate_status"].map(ui.status_label),
        drift_status=profile["drift_status"].map(ui.status_label),
    ),
    hide_index=True,
    column_order=["column_name", "null_rate", "null_rate_baseline", "null_rate_status", "psi",
                  "drift_status", "mean", "median", "p99"],
    column_config={
        "column_name": "Column",
        "null_rate": st.column_config.NumberColumn("NULL %", format="%.2f"),
        "null_rate_baseline": st.column_config.NumberColumn("Usual NULL %", format="%.2f"),
        "null_rate_status": "NULL check",
        "psi": st.column_config.NumberColumn("PSI", format="%.3f"),
        "drift_status": "Drift check",
        "mean": st.column_config.NumberColumn("Mean", format="%.2f"),
        "median": st.column_config.NumberColumn("Median", format="%.2f"),
        "p99": st.column_config.NumberColumn("p99", format="%.2f"),
    },
    key="columns",
)  # fmt: skip

# --- One measure over time --------------------------------------------------------------------
st.subheader("One measure over time")
items = ui.load("trend_items", ctx.database, ctx.dataset)
left, right = st.columns(2)
metric = left.selectbox("Measure", list(items), format_func=data.TREND_METRICS.get, key="metric")
item = ""
if items[metric] != [""]:
    item = right.selectbox("Column" if metric != "rule_failure_rate" else "Rule", items[metric],
                           key=f"item:{metric}")  # fmt: skip
series = ui.load("trend", ctx.database, ctx.dataset, metric, item)
scale = 100 if (series["unit"] == "share").all() else 1
unit = "%" if scale == 100 else series["unit"].iloc[0]
figure = go.Figure()
figure.add_scatter(x=series["period"], y=series["value"] * scale, name="value", mode="lines",
                   line={"color": ui.SERIES[0], "width": 2}, hoverinfo="skip",
                   showlegend=False)  # fmt: skip
checked = series["check_status"].where(series["check_status"].isin(["pass", "fail"]), "none")
for status, color in (("pass", ui.STATUS_COLORS["healthy"]), ("fail", ui.STATUS_COLORS["critical"]),
                      ("none", ui.SERIES[0])):  # fmt: skip
    part = series[checked == status]
    if part.empty:
        continue
    name = ui.STATUS_WORDS.get(status, "no check")
    figure.add_scatter(
        x=part["period"], y=part["value"] * scale, mode="markers", name=name,
        marker={"color": color, "size": 10, "line": {"width": 2, "color": "rgba(255,255,255,0.9)"}},
        hovertemplate="%{x}: %{y:.4g} " + unit + "<extra>" + name + "</extra>",
    )  # fmt: skip
if series["baseline"].notna().any():
    figure.add_scatter(
        x=series["period"], y=series["baseline"] * scale, name="baseline", mode="lines",
        line={"color": ui.MUTED, "width": 2, "dash": "dot"},
        hovertemplate="%{x}: baseline %{y:.4g} " + unit + "<extra></extra>",
    )  # fmt: skip
figure.update_layout(yaxis={"title": unit, "rangemode": "tozero"}, xaxis={"type": "category"},
                     **LAYOUT)  # fmt: skip
st.plotly_chart(figure, key="trend_chart")
st.caption(
    "Markers show the month's check: passed, failed, or no check for this measure. "
    "The dotted line is the baseline the check compared with (after the cold start)."
)

# --- Distribution -----------------------------------------------------------------------------
binned = data.binned_columns(ctx.dataset)
if binned:
    st.subheader(f"Distribution in {ctx.period}")
    column = st.selectbox("Column", binned, key=f"binned:{ctx.dataset}")
    shares = ui.load("distribution", ctx.database, ctx.dataset, column, ctx.period)
    if shares.empty:
        st.caption("No distribution for this month (not processed into Silver yet).")
    else:
        figure = go.Figure()
        for (period, part), color in zip(shares.groupby("period", sort=False), ui.SERIES,
                                         strict=False):  # fmt: skip
            figure.add_bar(
                x=part["bin"], y=part["share"] * 100, name=period, marker_color=color,
                marker={"cornerradius": 4}, customdata=part[["rows"]],
                hovertemplate="%{x}: %{y:.1f}% (%{customdata[0]:,} rows)<extra>" + period
                + "</extra>",
            )  # fmt: skip
        figure.update_layout(barmode="group", bargap=0.25, bargroupgap=0.1,
                             yaxis={"title": "% of rows"}, xaxis={"type": "category",
                             "title": column}, **LAYOUT)  # fmt: skip
        st.plotly_chart(figure, key="distribution")
        st.caption("Share of the month's Silver rows per drift bin, against the month before.")

# --- Schema -----------------------------------------------------------------------------------
st.subheader("Schema changes")
changes = ui.load("schema_changes", ctx.database, ctx.dataset)
if changes.empty:
    st.caption("No column was added, removed, renamed or retyped in any month.")
else:
    st.dataframe(
        changes.assign(severity=changes["severity"].map(ui.severity_label)),
        hide_index=True,
        column_config={"period": "Month", "column_name": "Column", "change": "Change",
                       "severity": "Severity", "previous_period": "Compared with",
                       "detail": st.column_config.TextColumn("Detail", width="large")},
        key="schema",
    )  # fmt: skip
