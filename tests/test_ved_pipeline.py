"""Phase 9.5 'done when': VED is monitored with configuration only.

The VED fixture (tests/ved_fixture.py, the real files' shape) runs through ingest and the
whole rebuild with the real configs/: the weekly CSVs regrouped by month, VED's standard
schema, its 14 rules (two of them asking the vehicles lookup for the engine type), the lookup
stacked from the two static files, its drift bins and investigation dimensions. And no
monitoring code names VED: only the tools that download and profile it do.
"""

import ast
import re

import duckdb
import pytest
from ved_fixture import make_unpacked

from common.config import REPO_ROOT
from gold import rebuild
from ingestion import ingest_data
from quality.rules import load_rules

MONITORING = [
    "common",
    "ingestion",
    "silver",
    "quality",
    "incidents",
    "investigation",
    "gold",
    "ml",
]
NAMES = re.compile(r"\bved(_signals)?\b|vehid|daynum", re.IGNORECASE)  # whole words


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("ved_pipeline")
    incoming = root / "incoming"
    make_unpacked(incoming / "ved_signals")
    database = str(root / "dataguard.duckdb")
    common = ["--dataset", "ved_signals", "--database", database]
    months = ["--months", "2017-11:2017-12"]
    bronze = ["--source", str(incoming), "--bronze", str(root / "bronze")]
    ingested = ingest_data.main([*common, *months, *bronze])
    folders = ["--silver", str(root / "silver"), "--quarantine", str(root / "quarantine")]
    argv = [*common, "--lookups", str(incoming), *folders, "--reports", str(root / "reports")]
    stages = {r.name: r.exit_code for r in rebuild.run(rebuild.parse_args(argv))}
    return {"db": database, "ingested": ingested, "stages": stages}


def query(env, sql):
    with duckdb.connect(env["db"], read_only=True) as con:
        return con.execute(sql).fetchall()


def test_every_stage_runs(env):
    assert env["ingested"] == 0
    assert set(env["stages"].values()) == {0} and len(env["stages"]) == 9


def test_rules_from_the_profile(env):
    failed = dict(query(env, "SELECT rule_id, sum(rows_failed) FROM rule_run_log "
                             "WHERE dataset = 'ved_signals' GROUP BY 1"))  # fmt: skip
    assert len(failed) == len(load_rules("ved_signals")) == 14
    assert failed["trip_time_not_negative"] == 5  # the fixture's trip that starts at -5 s
    for rule in ("reading_identified", "speed_in_range", "latitude_in_area", "vehicle_known",
                 "reading_unique", "reading_in_file_month", "engine_signals_present_ice",
                 "battery_signals_present_ev", "battery_charge_in_range"):  # fmt: skip
        assert failed[rule] == 0, rule
    quarantined = query(env, "SELECT sum(quarantine_rows) FROM silver_log")[0][0]
    assert quarantined == 5


def test_the_engine_type_comes_from_both_static_files(env):
    path = query(env, "SELECT silver_path FROM silver_log WHERE period = '2017-11'")[0][0]
    types = query(env, f"SELECT engine_type, count(DISTINCT vehicle_id) FROM '{path}' "
                       "GROUP BY 1 ORDER BY 1")  # fmt: skip
    assert types == [("EV", 1), ("HEV", 1), ("ICE", 2), ("PHEV", 1)]
    columns = [r[0] for r in query(env, f"DESCRIBE SELECT * FROM '{path}'")]
    assert columns[:2] == ["day_num", "vehicle_id"] and "row_time" in columns
    assert {"engine_type", "origin_file", "origin_line"} <= set(columns)


def test_checks(env):
    results = dict(query(env, """SELECT "check", status FROM dataset_check_results
                                 WHERE period = '2017-11' AND column_name = ''"""))  # fmt: skip
    assert results["schema"] == "pass" and results["freshness"] == "skipped"
    drift_columns = {r[0] for r in query(env, "SELECT DISTINCT column_name FROM dataset_metrics "
                                              "WHERE metric = 'histogram'")}  # fmt: skip
    assert drift_columns == {"speed_kmh", "engine_rpm", "battery_soc_pct"}


def test_gold_and_investigation(env):
    health = query(env, "SELECT period FROM gold_dataset_health WHERE dataset = 'ved_signals' "
                        "ORDER BY 1")  # fmt: skip
    assert [p for (p,) in health] == ["2017-11", "2017-12"]
    assert query(env, "SELECT DISTINCT status FROM gold_lineage") == [("ok",)]
    assert query(env, "SELECT count(*) FROM incidents") == query(
        env, "SELECT count(*) FROM investigations"
    )  # every incident investigated (the fixture's few rows a day fail the volume limits)


def _code_strings(path):
    """String constants of a module that are not docstrings or comments."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.FunctionDef | ast.ClassDef | ast.AsyncFunctionDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and id(node) not in docstrings
    ]  # fmt: skip


def test_no_monitoring_code_names_ved():
    found = []
    for package in MONITORING:
        for path in sorted((REPO_ROOT / "src" / package).glob("*.py")):
            for text in _code_strings(path):
                if NAMES.search(text):
                    found.append(f"{path.relative_to(REPO_ROOT)}: {text[:60]!r}")
    assert not found, found
