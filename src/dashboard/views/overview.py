"""Overview: the chosen month at a glance, its trend, the open incidents and the latest alerts."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import data, ui

ctx: ui.Context = st.session_state["context"]
numbers = ui.load("overview", ctx.database, ctx.dataset, ctx.period)
months = ui.load("health", ctx.database, ctx.dataset)
limits = data.health_limits()

st.title(f"Overview: {ctx.dataset}")
st.caption(f"{ctx.period} · {numbers['months']} month(s) in Gold")

# --- Headline numbers -------------------------------------------------------------------------
score, change = numbers["score"], numbers["score_change"]
open_now = numbers["open_incidents"]
tiles = st.columns(5)
tiles[0].metric(
    "Quality score",
    "—" if score is None else f"{score:.1f}",
    None if change is None else f"{change:+.1f} pts",
    help="0 to 100, from the rules and dataset checks (Phase 4); change against the month before",
    border=True,
)
tiles[1].metric(
    "Health",
    ui.HEALTH_WORDS.get(numbers["health"], numbers["health"]),
    help=f"critical: score below {limits.critical_below:g} or a critical finding · "
    f"warning: below {limits.warning_below:g} or a high or medium finding",
    border=True,
)
tiles[2].metric(
    "Open incidents",
    sum(open_now.values()),
    help="active now, every month of the dataset",
    border=True,
)
tiles[3].metric("Rows checked", f"{numbers['bronze_rows']:,}", help="rows in the file", border=True)
quarantined = numbers["quarantine_pct"]
tiles[4].metric(
    "Quarantined",
    "—" if quarantined is None else f"{quarantined:.2f}%",
    help="rows kept out of Silver by a rule",
    border=True,
)
st.caption(
    "Open now: "
    + " · ".join(f"{ui.severity_label(s)} **{n}**" for s, n in open_now.items())
    + f" · checks failed this month: **{numbers['checks_failed']}** of {numbers['checks_run']}"
)

# --- Trend ------------------------------------------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Quality score per month")
    figure = go.Figure()
    figure.add_scatter(
        x=months["period"], y=months["score"], mode="lines", line={"color": ui.SERIES[0],
        "width": 2}, showlegend=False, hoverinfo="skip",
    )  # fmt: skip
    for label in data.HEALTH:  # one marker trace per health label, for the legend
        part = months[months["health"] == label]
        if part.empty:
            continue
        figure.add_scatter(
            x=part["period"], y=part["score"], mode="markers", name=ui.HEALTH_WORDS[label],
            marker={"color": ui.STATUS_COLORS[label], "size": 10,
                    "line": {"width": 2, "color": "rgba(255,255,255,0.9)"}},
            customdata=part[["quarantine_pct", "checks_failed"]],
            hovertemplate="%{x}: score %{y:.1f}<br>quarantined %{customdata[0]:.2f}%"
            "<br>checks failed %{customdata[1]}<extra>" + ui.health_label(label) + "</extra>",
        )  # fmt: skip
    for value, name in ((limits.warning_below, "warning"), (limits.critical_below, "critical")):
        figure.add_hline(
            y=value, line={"dash": "dot", "width": 1, "color": ui.MUTED},
            annotation_text=f"{name} below {value:g}", annotation_position="top left",
            annotation_font_color=ui.MUTED,
        )  # fmt: skip
    low = months["score"].min()
    bottom = 0 if pd.isna(low) else max(0, low - 10)
    figure.update_layout(
        height=320, margin={"l": 0, "r": 0, "t": 10, "b": 0}, hovermode="closest",
        yaxis={"title": "score", "range": [bottom, 101]},
        xaxis={"type": "category", "title": None},
        legend={"orientation": "h", "y": -0.18},
    )  # fmt: skip
    st.plotly_chart(figure, key="score_trend")

with right:
    st.subheader("Incidents opened and resolved")
    monthly = ui.load("incident_monthly", ctx.database, ctx.dataset)
    figure = go.Figure()
    for column, color in zip(("opened", "resolved"), ui.SERIES, strict=True):
        figure.add_bar(
            x=monthly["period"], y=monthly[column], name=column, marker_color=color,
            marker={"cornerradius": 4}, customdata=monthly[["active_end"]],
            hovertemplate="%{x}: %{y} " + column
            + "<br>active at month end %{customdata[0]}<extra></extra>",
        )  # fmt: skip
    figure.update_layout(
        height=320, margin={"l": 0, "r": 0, "t": 10, "b": 0}, barmode="group", bargap=0.3,
        bargroupgap=0.1, yaxis={"title": "incidents", "rangemode": "tozero", "dtick": 1},
        xaxis={"type": "category", "title": None}, legend={"orientation": "h", "y": -0.18},
    )  # fmt: skip
    st.plotly_chart(figure, key="incident_trend")

# --- Open incidents ---------------------------------------------------------------------------
st.subheader("Most severe open incidents")
top = ui.load("top_incidents", ctx.database, ctx.dataset)
if top.empty:
    st.success("No open incident.", icon="✅")
else:
    table = top.assign(
        severity=top["severity"].map(ui.severity_label),
        cause=top["headline"].map(data.cause_text),
        since=top["first_seen_period"],
    )
    st.dataframe(
        table[["incident_id", "severity", "title", "since", "last_seen_period", "cause",
               "actions_pending"]],
        hide_index=True,
        column_config={
            "incident_id": "Incident", "severity": "Severity", "title": "What",
            "since": "Since", "last_seen_period": "Last seen",
            "cause": st.column_config.TextColumn("Likely cause", width="large"),
            "actions_pending": "Actions waiting",
        },
        key="top_incidents",
    )  # fmt: skip

# --- Alerts and runs --------------------------------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Latest alerts")
    alerts = ui.load("recent_alerts", ctx.database, ctx.dataset)
    if alerts.empty:
        st.caption("No alert yet: alerts go out for new critical and high incidents.")
    else:
        alerts = alerts.assign(
            when=alerts["created_at"].map(ui.format_time),
            severity=alerts["severity"].map(ui.severity_label),
            summary=alerts["message"].str.split("\n").str[0].str.split(" ", n=2).str[2],
        )
        st.dataframe(
            alerts[["when", "incident_id", "severity", "status", "summary"]],
            hide_index=True,
            column_config={"when": "Time (UTC)", "incident_id": "Incident",
                           "severity": "Severity", "status": "Status", "summary": "Alert"},
            key="alerts",
        )  # fmt: skip
        st.caption("Written to data/alerts/outbox.jsonl; a failed one is retried next run.")
with right:
    st.subheader("Latest pipeline runs")
    runs = ui.load("last_runs", ctx.database)
    runs = runs.assign(
        started=runs["started_at"].map(ui.format_time),
        seconds=(runs["finished_at"] - runs["started_at"]).dt.total_seconds().round(1),
    )
    st.dataframe(
        runs[["run_id", "command", "dataset", "status", "started", "seconds"]],
        hide_index=True,
        column_config={"run_id": "Run", "command": "Command", "dataset": "Dataset",
                       "status": "Status", "started": "Started (UTC)", "seconds": "Seconds"},
        key="runs",
    )  # fmt: skip
