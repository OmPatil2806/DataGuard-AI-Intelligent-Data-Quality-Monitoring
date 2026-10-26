"""explainer.py (Phase 8.3): template explanations from evidence packs and playbooks."""

import json

import duckdb
import pytest
from pipeline_fixture import run_pipeline

from common.config import ConfigError
from investigation import evidence_pack as ep
from investigation import explainer as ex

BOOKS = ex.load_playbooks()


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    database = run_pipeline(tmp_path_factory.mktemp("explainer"))
    with duckdb.connect(str(database)) as con:
        ids = [r[0] for r in con.execute("SELECT incident_id FROM incidents ORDER BY 1").fetchall()]
        explained = {i: ex.explain(ep.build_evidence(con, i), BOOKS) for i in ids}
    return {"db": database, "explained": explained}


def evidence(rule_id, source="dataset_check", column="", detail="", category=None,
             affected_share=None, findings=None, analysed=True, same_month=(), metrics=None,
             versions=(), days=None, period="2023-06"):  # fmt: skip
    """A minimal evidence pack for one incident."""
    concentration = {"analysed": analysed, "findings": findings or [], "verdict": "v"}
    if days is not None:
        concentration["days"] = days
    return {
        "incident": {
            "incident_id": "INC-0100", "title": f"title {rule_id}", "source": source,
            "rule_id": rule_id, "column": column, "severity": "high", "status": "OPEN",
            "first_seen": period, "last_seen": period, "occurrences": 1, "detail": detail,
            "observed": "", "expected": "",
        },
        "definition": {"category": category},
        "similar": {
            "same_problem": [],
            "same_month": [{"rule_id": r, "column": c} for r, c in same_month],
        },
        "row_level": affected_share is not None,
        "impact": {} if affected_share is None else {"affected_share": affected_share},
        "concentration": concentration,
        "versions": list(versions),
        "metrics": metrics or {},
    }  # fmt: skip


def series(*values, period="2023-06"):
    months = [f"2023-0{i}" for i in range(1, len(values))] + [period]
    return [{"period": p, "value": v} for p, v in zip(months, values, strict=True)]


def finding(dimension, value, kind="where", share=1.0):
    return {"dimension": dimension, "value": value, "kind": kind, "share": share}


def top(e):
    return e["causes"][0]["cause_id"]


# --- the planted pipeline ----------------------------------------------------------------------


def test_pipeline_incidents(pipeline):
    got = {i: (top(e), e["causes"][0]["confidence"]) for i, e in pipeline["explained"].items()}
    assert got == {
        "INC-0001": ("rows_missing_evenly", "medium"),
        "INC-0002": ("column_renamed", "high"),
        "INC-0003": ("source_nulls", "medium"),
        "INC-0004": ("unit_change", "high"),
        "INC-0005": ("nulls_shift_distribution", "medium"),
        "INC-0006": ("wrong_file", "high"),
    }
    for e in pipeline["explained"].values():
        assert e["explained"] and e["actions"]
        assert {a["status"] for a in e["actions"]} == {"pending_approval"}
        assert all("?" not in a["text"] for a in e["actions"]), e["actions"]
        json.dumps(e)


def test_explanations_cite_their_evidence(pipeline):
    wrong_file = pipeline["explained"]["INC-0006"]["causes"][0]
    assert "the same file (SHA-256) is registered for more than one month" in wrong_file["evidence"]
    unit = pipeline["explained"]["INC-0004"]
    assert "kilometres" in unit["headline"] and "mean ×1.61" in unit["causes"][0]["text"]
    assert pipeline["explained"]["INC-0003"]["where"].startswith("unknown")


def test_main(pipeline, capsys):
    assert ex.main(["--all", "--database", str(pipeline["db"])]) == 0
    out = capsys.readouterr().out
    assert out.count("pending_approval") >= 6 and "INC-0006" in out
    assert ex.main(["INC-0002", "--json", "--database", str(pipeline["db"])]) == 0
    assert json.loads(capsys.readouterr().out)["playbook"] == "schema"
    assert ex.main(["INC-9999", "--database", str(pipeline["db"])]) == 2


# --- one playbook at a time --------------------------------------------------------------------


def test_missing_column_with_nulls():
    e = ex.explain(evidence("schema", detail="required column passenger_count is missing",
                            same_month=[("null_rate", "passenger_count")]), BOOKS)  # fmt: skip
    assert top(e) == "column_missing" and e["causes"][0]["confidence"] == "high"
    assert e["causes"][0]["evidence"] == ["the NULL rate rose in the same month"]
    null_rates = {"null_rate": series(0.01, 0.01, 0.01, 1.0)}
    nulls = ex.explain(
        evidence("null_rate", column="passenger_count", affected_share=1.0,
                 same_month=[("schema", "")], metrics=null_rates),
        BOOKS,
    )  # fmt: skip
    assert top(nulls) == "column_missing_upstream" and nulls["causes"][0]["confidence"] == "high"


def test_null_spike_in_one_vendor_names_the_vendor():
    e = ex.explain(
        evidence("null_rate", column="fare_amount", affected_share=0.05,
                 findings=[finding("vendor", "2", share=1.0)]),
        BOOKS,
    )  # fmt: skip
    assert top(e) == "segment_sends_nulls" and e["causes"][0]["confidence"] == "high"
    assert "vendor = 2" in e["headline"]
    assert any(a["action_id"] == "inspect_segment" and "100%" in a["text"] for a in e["actions"])
    spread = ex.explain(evidence("null_rate", column="fare_amount", affected_share=0.05), BOOKS)
    assert top(spread) == "source_nulls" and "vendor" not in spread["headline"]


def test_volume_and_freshness():
    days = [{"day": "2023-06-10"}, {"day": "2023-06-11"}]
    rows = {"rows_per_day": series(100, 100, 100, 90)}
    gaps = ex.explain(
        evidence("volume", detail="volume drop: -10%", days=days, metrics=rows), BOOKS
    )
    assert top(gaps) == "missing_days" and "2 day(s)" in gaps["causes"][0]["text"]
    half = {"rows_per_day": series(100, 100, 100, 50)}
    even = ex.explain(evidence("volume", detail="volume drop: -50%", days=[], metrics=half), BOOKS)
    assert top(even) == "rows_missing_evenly" and even["causes"][0]["confidence"] == "high"
    late = ex.explain(evidence("freshness", detail="20.0 day(s) late"), BOOKS)
    assert top(late) == "late_delivery" and "20.0 day(s)" in late["causes"][0]["text"]
    assert late["causes"][0]["confidence"] == "high"


def test_row_rule_playbooks():
    cases = {
        ("trip_unique", "uniqueness"): "duplicate_delivery",
        ("dropoff_not_before_pickup", "consistency"): "times_swapped",
        ("pickup_zone_known", "referential"): "unknown_location",
        ("fare_not_negative", "validity"): "out_of_range_values",
        ("pickup_in_file_month", "pattern"): "trips_from_other_months",
        ("trip_has_duration", "consistency"): "rule_failures_up",
    }
    for (rule, category), cause in cases.items():
        e = ex.explain(evidence(rule, "row_rule", category=category, affected_share=0.05), BOOKS)
        assert top(e) == cause, rule
        assert "5.0%" in e["causes"][0]["text"], rule


def test_ml_playbook():
    detail = "2.1% of rows flagged; most often fare_per_mile (5)"
    fare = ex.explain(evidence("ml_anomaly", affected_share=0.02, detail=detail), BOOKS)
    assert top(fare) == "fare_does_not_fit_trip" and fare["causes"][0]["confidence"] == "medium"
    gap = ex.explain(evidence("ml_anomaly", affected_share=0.02,
                              detail="most often total_gap (9)"), BOOKS)  # fmt: skip
    assert top(gap) == "totals_do_not_add_up"


def test_unknown_when_nothing_fits():
    e = ex.explain(evidence("drift", column="tip_amount", detail="PSI 0.3"), BOOKS)
    assert not e["explained"] and "cause unknown" in e["headline"] and "weak lead" in e["headline"]
    assert [a["for_cause"] for a in e["actions"]] == ["unknown"]
    nothing = ex.explain(evidence("schema", detail="something else"), BOOKS)
    assert nothing["causes"] == [] and nothing["headline"].endswith("cause unknown")


# --- the parts -----------------------------------------------------------------------------------


def test_render_and_conditions():
    assert ex.render("{a:.1%} of {b} {c:.2f}", {"a": 0.25, "b": "x"}) == "25.0% of x ?"
    f = {"n": 3, "flag": True, "items": ["a", "b"], "text": "volume drop", "col": "x"}
    cond = ex.Condition
    assert ex.holds(cond("n", "between", [1, 3]), f) and not ex.holds(cond("n", "gt", 3), f)
    assert ex.holds(cond("items", "contains", "a"), f)
    assert ex.holds(cond("items", "not_contains", "c"), f)
    assert ex.holds(cond("text", "contains_any", ["rise", "drop"]), f)
    assert ex.holds(cond("missing", "exists", False), f) and not ex.holds(
        cond("missing", "gt", 0), f
    )
    assert not ex.holds(cond("flag", "gt", 0), f)  # booleans are not numbers
    assert ex.holds(cond("items", "contains", "{col}"), {**f, "items": ["x"]})


def test_ratio():
    assert ex._ratio(series(1, 1, 3, 2), "2023-06") == 2.0
    assert ex._ratio(series(2), "2023-06") is None
    assert ex._ratio(series(0, 0, 1), "2023-06") == float("inf")


def test_playbook_validation(tmp_path):
    path = tmp_path / "playbooks.yml"

    def check(text, match):
        path.write_text(text)
        with pytest.raises(ConfigError, match=match):
            ex.load_playbooks(path)

    action = "actions:\n  a: {kind: investigate, text: t}\n"
    check("playbooks: []\n", "no playbooks")
    check("actions:\n  a: {kind: run_it, text: t}\nplaybooks: []\n", "kind")
    cause = "playbooks:\n  - id: p\n    causes:\n      - {id: c, text: t, prior: 0.5, %s}\n"
    check(action + cause % "actions: [b]", "unknown action")
    check(action + cause % "requires: [{fact: x}]", "exactly one")
    check(action + cause % "requires: [{fact: x, between: 1}]", "between")
    check(action + "playbooks:\n  - id: p\n    causes:\n      - {id: c, text: t}\n", "prior")
    path.write_text(action + cause % "actions: [a]")
    assert ex.load_playbooks(path).playbooks[0].causes[0].actions == ("a",)


def test_every_cause_id_is_unique_and_every_fact_is_documented():
    ids = [c.id for b in BOOKS.playbooks for c in b.causes]
    assert len(ids) == len(set(ids))
    assert BOOKS.playbooks[-1].match == {}  # the default comes last
    used = {c.fact for b in BOOKS.playbooks for cause in b.causes
            for c in (*cause.requires, *cause.signals)}  # fmt: skip
    documented = set(ex.FACTS) | {f"{m}_ratio" for m in ("null_rate", "mean", "median", "p99",
                                                         "rows", "rows_per_day")}  # fmt: skip
    assert used <= documented, used - documented


def test_every_playbook_and_cause_is_documented():
    from common.config import REPO_ROOT

    doc = (REPO_ROOT / "docs" / "investigation.md").read_text(encoding="utf-8")
    for book in BOOKS.playbooks:
        assert f"| `{book.id}` |" in doc, book.id
        for cause in book.causes:
            assert f"`{cause.id}`" in doc, cause.id
