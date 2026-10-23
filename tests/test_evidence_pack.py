"""evidence_pack.py (Phase 8.1): the evidence of each incident of the planted pipeline.

The incidents of tests/pipeline_fixture.py: INC-0001 volume (2023-01..04), INC-0002 schema
(2023-02, resolved), INC-0003 NULL fares in May, INC-0004/5 drift in May, INC-0006 the
pickup_in_file_month rule (2023-02..03, resolved).
"""

import json

import duckdb
import pytest
from pipeline_fixture import run_pipeline

from investigation import evidence_pack as ep


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    database = run_pipeline(tmp_path_factory.mktemp("evidence"))
    con = duckdb.connect(str(database))
    packs = {p["incident"]["incident_id"]: p for p in ep.build_all(con)}
    yield {"db": database, "con": con, "packs": packs}
    con.close()


def test_one_pack_per_incident_with_every_section(env):
    assert sorted(env["packs"]) == [f"INC-000{i}" for i in range(1, 7)]
    sections = {"incident", "definition", "history", "events", "versions", "impact", "similar",
                "row_level", "samples", "concentration"}  # fmt: skip
    for pack in env["packs"].values():
        assert set(pack) == sections
        json.dumps(pack)  # JSON-serialisable as built


def test_null_rate_incident(env):
    pack = env["packs"]["INC-0003"]
    assert pack["incident"]["column"] == "fare_amount" and pack["row_level"]
    assert "NULL" in pack["definition"]["description"]
    statuses = {h["period"]: h["status"] for h in pack["history"]}
    assert statuses == {"2023-01": "pass", "2023-02": "pass", "2023-03": "pass",
                        "2023-04": "pass", "2023-05": "fail"}  # fmt: skip
    impact = pack["impact"]
    assert (impact["period"], impact["rows"], impact["affected_rows"]) == ("2023-05", 200, 137)
    assert impact["affected_share"] == 0.685 and impact["score_change"] < 0
    assert impact["previous_score"]["period"] == "2023-04"
    assert len(pack["samples"]) == ep.SAMPLE_ROWS
    assert all(s["values"]["fare_amount"] is None for s in pack["samples"])
    same_month = {s["incident_id"] for s in pack["similar"]["same_month"]}
    assert same_month == {"INC-0004", "INC-0005"}  # the drift of the same month
    where = pack["concentration"]  # the NULLs are planted at random: no "where"
    assert where["analysed"] and where["findings"] == []
    assert where["verdict"].startswith("unknown") and where["affected_rows"] == 137


def test_row_rule_incident(env):
    pack = env["packs"]["INC-0006"]
    definition = pack["definition"]
    assert definition["kind"] == "row_rule" and definition["action"] == "quarantine"
    assert definition["category"] and definition["description"]
    values = {h["period"]: h["value"] for h in pack["history"]}
    assert values["2023-02"] == 1.0 and values["2023-01"] < 0.1
    assert pack["incident"]["status"] == "RESOLVED"
    assert [e["event"] for e in pack["events"]][0] == "opened"
    assert pack["events"][-1]["event"] == "resolved"
    assert {v["period"] for v in pack["versions"]} == {"2023-02", "2023-03"}
    assert pack["impact"]["affected_rows"] == 200
    for sample in pack["samples"]:
        assert "pickup_in_file_month" in sample["failed_rules"]
        assert any("outside the file's month" in r for r in sample["reasons"])


def test_month_level_findings_have_no_samples(env):
    for incident_id in ("INC-0001", "INC-0002", "INC-0004", "INC-0005"):
        pack = env["packs"][incident_id]
        assert not pack["row_level"] and pack["samples"] == [], incident_id
        assert "affected_rows" not in pack["impact"], incident_id
        assert not pack["concentration"]["analysed"], incident_id
    volume = env["packs"]["INC-0001"]
    assert len(volume["history"]) == 5 and volume["incident"]["occurrences"] == 4
    schema = env["packs"]["INC-0002"]
    assert schema["history"][1]["status"] == "fail" and "renamed" in schema["incident"]["detail"]


def test_stored_and_deterministic(env):
    con = env["con"]
    assert con.execute("SELECT count(*) FROM investigation_evidence").fetchone()[0] == 6
    stored = ep.stored(con, "INC-0004")
    assert stored == json.loads(ep.to_json(env["packs"]["INC-0004"]))
    again = ep.build_evidence(con, "INC-0004")
    assert ep.to_json(again) == ep.to_json(env["packs"]["INC-0004"])
    assert ep.stored(con, "INC-9999") is None


def test_main(env, capsys):
    env["con"].close()
    try:
        assert ep.main(["--all", "--database", str(env["db"])]) == 0
        assert "6 evidence pack(s) stored" in capsys.readouterr().out
        assert ep.main(["INC-0003", "--database", str(env["db"])]) == 0
        printed = json.loads(capsys.readouterr().out)
        assert printed["incident"]["incident_id"] == "INC-0003"
        assert ep.main(["INC-9999", "--database", str(env["db"])]) == 2
        assert "unknown incident" in capsys.readouterr().out
        with pytest.raises(SystemExit):
            ep.main(["--database", str(env["db"])])
    finally:
        env["con"] = duckdb.connect(str(env["db"]))
