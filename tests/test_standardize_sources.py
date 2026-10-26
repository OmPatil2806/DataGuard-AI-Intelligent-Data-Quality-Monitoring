"""Standardization with source names and computed columns (Phase 9.3), on VED's shape."""

from datetime import datetime

import duckdb
import pytest
import yaml
from ved_fixture import HEADER, make_unpacked

from common.config import REPO_ROOT, ConfigError, load_datasets_config
from ingestion import ingest_data
from ingestion.bronze_loader import PASS_THROUGH, read_bronze_sql
from ingestion.manifest import expected_at
from quality.dataset_checks import check_freshness, schema_changes
from quality.standardize import plan, standardize

VED = load_datasets_config().dataset("ved_signals")


@pytest.fixture(scope="module")
def bronze(tmp_path_factory):
    """The VED fixture through prepare, register and Bronze: the file standardization reads."""
    root = tmp_path_factory.mktemp("ved")
    make_unpacked(root / "incoming" / "ved_signals")
    database = root / "d.duckdb"
    argv = ["--dataset", "ved_signals", "--months", "2017-11:2017-12",
            "--source", str(root / "incoming"), "--bronze", str(root / "bronze"),
            "--database", str(database)]  # fmt: skip
    assert ingest_data.main(argv) == 0
    with duckdb.connect(str(database), read_only=True) as con:
        path = con.execute(
            "SELECT bronze_path FROM file_manifest WHERE period = '2017-11'"
        ).fetchone()[0]
        entry = con.execute("SELECT expected_at FROM file_manifest LIMIT 1").fetchone()[0]
    return {"source": read_bronze_sql(path), "expected_at": entry}


def test_source_names_are_mapped(bronze):
    con = duckdb.connect()
    sql, report = standardize(con, bronze["source"], VED)
    names = [r[0] for r in con.execute(f"DESCRIBE {sql}").fetchall()]
    assert names == [c for c, _ in VED.columns] + [c for c in PASS_THROUGH]
    assert ("Vehicle Speed[km/h]", "speed_kmh") in report.renamed
    assert ("OAT[DegC]", "outside_temp_c") in report.renamed  # the first accepted name present
    assert report.computed == ["row_time"] and not report.added and not report.dropped
    assert not report.lossy and "computed row_time" in report.summary()
    types = dict(con.execute(f"SELECT column_name, column_type FROM (DESCRIBE {sql})").fetchall())
    assert (types["vehicle_id"], types["row_time"], types["origin_line"]) == (
        "INTEGER", "TIMESTAMP", "BIGINT",
    )  # fmt: skip


def test_row_time_is_computed(bronze):
    con = duckdb.connect()
    sql, _ = standardize(con, bronze["source"], VED)
    first = con.execute(
        f"SELECT day_num, trip_ms, row_time FROM ({sql}) ORDER BY bronze_row_id LIMIT 2"
    ).fetchall()
    assert first[0][2] == datetime(2017, 11, 1, 9, 7, 12)  # DayNum 1.38
    assert (first[1][2] - first[0][2]).total_seconds() == 1.0  # one second later in the trip
    assert con.execute(
        f"SELECT count_if(strftime(row_time, '%Y-%m') <> '2017-11') FROM ({sql})"
    ).fetchone() == (0,)  # the month's file holds the month's rows (Phase 9.2)


def test_schema_check_uses_the_source_names(bronze):
    con = duckdb.connect()
    described = con.execute(f"DESCRIBE SELECT * FROM {bronze['source']}").fetchall()
    raw = [(n, t) for n, t, *_ in described if n not in PASS_THROUGH]
    assert [n for n, _ in raw] == HEADER
    assert schema_changes(raw, None, VED) == []  # every column known; row_time is not in it
    readme_name = [("Outside Air Temperature[DegC]" if n == "OAT[DegC]" else n, t) for n, t in raw]
    assert schema_changes(readme_name, None, VED) == []  # the README's name is accepted too
    without = [(n, t) for n, t in raw if n != "Vehicle Speed[km/h]"]
    (missing,) = schema_changes(without, None, VED)
    assert (missing.kind, missing.column, missing.severity) == (
        "missing", "Vehicle Speed[km/h]", "critical",
    )  # fmt: skip
    # month to month, a source name switching between two accepted names is not missing
    switched = schema_changes(readme_name, raw, VED)
    assert "missing" not in {c.kind for c in switched} and "unexpected" not in {
        c.kind for c in switched
    }


def test_no_freshness_without_a_schedule(bronze):
    assert VED.expected_lag_days is None and bronze["expected_at"] is None
    assert expected_at("2017-11", None) is None
    from ingestion.manifest import ManifestEntry

    entry = ManifestEntry("ved_signals", "2017-11", "f", "s", 1, 1, None, datetime(2020, 1, 1),
                          datetime(2020, 1, 1), "LOADED")  # fmt: skip
    from quality.dataset_checks import check_settings

    result = check_freshness(entry, check_settings())
    assert result.status == "skipped" and "expected_lag_days" in result.detail


def test_plan_prefers_the_first_source_name():
    _, report = plan({"Outside Air Temperature[DegC]": "DOUBLE", "OAT[DegC]": "DOUBLE"}, VED)
    assert ("OAT[DegC]", "outside_temp_c") in report.renamed
    assert "Outside Air Temperature[DegC]" in report.dropped


def config_with(tmp_path, change):
    data = yaml.safe_load((REPO_ROOT / "configs" / "datasets.yml").read_text())
    change(data["datasets"]["ved_signals"]["columns"], data["datasets"]["ved_signals"])
    path = tmp_path / "datasets.yml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return load_datasets_config(path).dataset("ved_signals")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda c, d: c.update(x={"type": "DOUBLE", "source": "a", "sql": "1"}), "not both"),
        (lambda c, d: c.update(x={"type": "DOUBLE", "unit": "km"}), "unknown key"),
        (lambda c, d: c.update(x={"source": "a"}), "type"),
        (lambda c, d: c.update(x={"type": "DOUBLE", "source": []}), "source must be"),
        (lambda c, d: c.update(x={"type": "DOUBLE", "sql": " "}), "sql must be"),
        (lambda c, d: c.update(x={"type": "DOUBLE", "source": "DayNum"}), "claimed by"),
        (lambda c, d: d.update(timestamp_column="pickup_time"), "not a standard column"),
    ],
)
def test_column_specs_are_validated(tmp_path, change, message):
    with pytest.raises(ConfigError, match=message):
        config_with(tmp_path, change)


def test_column_spec_forms(tmp_path):
    ved = config_with(tmp_path, lambda c, d: c.update(extra="double"))
    assert ("extra", "DOUBLE") in ved.columns and ved.source_names("extra") == ("extra",)
    assert ved.source_names("outside_temp_c") == ("OAT[DegC]", "Outside Air Temperature[DegC]")
    assert ved.computed_names == {"row_time"}
    assert "row_time" not in dict(ved.raw_columns()) and len(ved.raw_columns()) == 23
