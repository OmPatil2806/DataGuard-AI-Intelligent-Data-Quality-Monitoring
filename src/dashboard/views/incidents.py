"""Incidents: every incident of a dataset, one in depth, and the decisions people take on it.

The only page that writes: approving, rejecting or finishing a suggested action, changing the
status and adding a note, each recorded with the person's name (dashboard/actions.py).
"""

from __future__ import annotations

import streamlit as st

from dashboard import actions, data, ui

ctx: ui.Context = st.session_state["context"]

st.title(f"Incidents: {ctx.dataset}")
if "flash" in st.session_state:  # the result of the last decision, shown after the redraw
    st.success(st.session_state.pop("flash"), icon="✅")


def apply(change, *args, done: str) -> None:
    try:
        change(ctx.database, *args)
    except data.DashboardError as exc:
        st.error(str(exc), icon="⛔")
        return
    st.cache_data.clear()
    st.session_state["flash"] = done
    st.rerun()


# --- The list ---------------------------------------------------------------------------------
left, middle, right = st.columns([2, 2, 3])
statuses = left.multiselect("Status", list(data.INCIDENT_STATUSES),
                            default=["OPEN", "INVESTIGATING"], key="statuses")  # fmt: skip
severities = middle.multiselect("Severity", list(ui.SEVERITY_LABELS),
                                format_func=ui.severity_label, key="severities")  # fmt: skip
search = right.text_input("Search", placeholder="title, column or rule", key="search")

found = ui.load("incidents", ctx.database, ctx.dataset, tuple(statuses))
if severities:
    found = found[found["severity"].isin(severities)]
if search:
    text = (
        found[["title", "column_name", "rule_id", "incident_id"]].fillna("").agg(" ".join, axis=1)
    )
    found = found[text.str.contains(search, case=False, regex=False)]

st.caption(f"{len(found)} incident(s)")
if found.empty:
    st.info("No incident matches.", icon="ℹ️")
    st.stop()
st.dataframe(
    found.assign(severity=found["severity"].map(ui.severity_label),
                 cause=found["headline"].map(data.cause_text)),
    hide_index=True,
    column_order=["incident_id", "severity", "status", "title", "first_seen_period",
                  "last_seen_period", "months_seen", "cause", "actions_pending"],
    column_config={
        "incident_id": "Incident", "severity": "Severity", "status": "Status", "title": "What",
        "first_seen_period": "Since", "last_seen_period": "Last seen",
        "months_seen": "Months seen",
        "cause": st.column_config.TextColumn("Likely cause", width="large"),
        "actions_pending": "Actions waiting",
    },
    key="incident_list",
)  # fmt: skip

# --- One incident -----------------------------------------------------------------------------
labels = {r.incident_id: f"{r.incident_id} · {r.severity} · {r.title}" for r in found.itertuples()}
chosen = st.selectbox("Incident", list(labels), format_func=labels.get,
                      key=f"incident:{ctx.dataset}")  # fmt: skip
detail = ui.load("incident_detail", ctx.database, chosen)
incident, investigation = detail["incident"], detail["investigation"]

st.subheader(f"{chosen}: {incident['title']}")
tiles = st.columns([1, 1.6, 1, 1])
tiles[0].metric("Severity", incident["severity"].capitalize(), border=True)
tiles[1].metric("Status", incident["status"].capitalize(), border=True)
tiles[2].metric("Months seen", int(incident["occurrences"]), border=True)
tiles[3].metric("Last seen", incident["last_seen_period"], border=True)
st.caption(f"First seen {incident['first_seen_period']}")
finding = f"**Latest finding:** {incident['latest_observed'] or '—'}"
if incident["latest_expected"]:
    finding += f" (expected {incident['latest_expected']})"
if incident["latest_detail"]:
    finding += f": {incident['latest_detail']}"
st.markdown(finding)
if incident["recurrence_of"]:
    st.caption(f"Returns after {incident['recurrence_of']}.")

st.markdown("#### Likely cause")
if investigation is None:
    st.caption("Not investigated yet: python src/investigation/investigate.py")
else:
    (st.info if investigation["explained"] else st.warning)(investigation["headline"])
    causes = []
    for number, cause in enumerate(investigation["causes"], start=1):
        causes.append(f"{number}. **[{cause['confidence']} {cause['score']:.2f}]** {cause['text']}")
        causes += [f"    - {line}" for line in cause.get("evidence") or []]
    st.markdown("\n".join(causes))
    if investigation["where"]:
        st.caption(f"Where and when: {investigation['where']}")
    if investigation["report_path"]:
        st.caption(f"Report: `{investigation['report_path']}`")

# --- Decisions --------------------------------------------------------------------------------
st.markdown("#### Suggested actions")
by = st.text_input("Your name", key="decided_by",
                   help="recorded with every decision on this page")  # fmt: skip
st.caption("Approving records the decision; nothing is run automatically.")
suggested = detail["actions"]
if suggested.empty:
    st.caption("No suggested action.")
for row in suggested.itertuples():
    text, state, buttons = st.columns([6, 2, 3], vertical_alignment="center")
    text.markdown(f"{row.text}  \n`{row.action_id}` · {row.kind}")
    decided = f" by {row.decided_by}" if row.decided_by else ""
    state.markdown(f"{ui.ACTION_LABELS.get(row.status, row.status)}{decided}")
    choices = {"pending_approval": ("approved", "rejected"), "approved": ("done",)}
    for column, decision in zip(buttons.columns(2), choices.get(row.status, ()), strict=False):
        if column.button(ui.DECISION_BUTTONS[decision], key=f"{chosen}:{row.action_id}:{decision}"):
            apply(actions.decide, chosen, row.action_id, decision, by,
                  done=f"{row.action_id}: {decision}")  # fmt: skip

st.markdown("#### Status and notes")
note = st.text_area("Note", key=f"note:{chosen}", placeholder="optional with a status change")
buttons = st.columns(4)
changes = [(label, target) for label, (allowed, target) in actions.STATUS_CHANGES.items()
           if incident["status"] in allowed]  # fmt: skip
for column, (label, target) in zip(buttons, changes, strict=False):
    if column.button(label, key=f"{chosen}:status:{target}"):
        apply(actions.set_status, chosen, target, note, by, done=f"{chosen} is now {target}")
if buttons[3].button("Add note", key=f"{chosen}:note"):
    apply(actions.add_note, chosen, note, by, done=f"Note added to {chosen}")
if incident["status"] == "RESOLVED":
    st.caption("A resolved incident stays resolved; if the problem returns, a new one opens.")

st.markdown("#### History")
history = detail["events"]
st.dataframe(
    history.assign(created_at=history["created_at"].map(ui.format_time),
                   severity=history["severity"].map(ui.severity_label)),
    hide_index=True,
    column_config={"seq": "#", "event": "Event", "period": "Month", "severity": "Severity",
                   "status_after": "Status after", "created_at": "Time (UTC)",
                   "detail": st.column_config.TextColumn("Detail", width="large")},
    key="history",
)  # fmt: skip
