"""prepare_source.py and the ingestion of a prepared source (Phase 9.2), on VED's shape."""

import dataclasses
import hashlib

import duckdb
import pytest
import yaml
from ved_fixture import HEADER, ROWS_PER_TRIP, make_unpacked

from common.config import REPO_ROOT, ConfigError, load_datasets_config
from ingestion import ingest_data
from ingestion import prepare_source as ps

VED = load_datasets_config().dataset("ved_signals")
ROWS = (5 * 3 * 3 + 2) * ROWS_PER_TRIP


@pytest.fixture
def incoming(tmp_path):
    make_unpacked(tmp_path / "incoming" / "ved_signals")
    return tmp_path / "incoming"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path, sql="count(*)"):
    return duckdb.sql(f"SELECT {sql} FROM read_parquet('{path.as_posix()}')").fetchall()


def test_rows_are_regrouped_into_months(incoming):
    result = ps.prepare(VED, incoming)
    assert sorted(result.files) == ["2017-11", "2017-12"]
    assert sum(result.files.values()) == ROWS  # every source row in exactly one month
    assert "2018-01" in result.empty and not (incoming / "ved_signals_2018-01.parquet").exists()
    november = incoming / "ved_signals_2017-11.parquet"
    columns = [
        r[0] for r in duckdb.sql(f"DESCRIBE SELECT * FROM '{november.as_posix()}'").fetchall()
    ]
    assert columns == [*HEADER, "origin_file", "origin_line"]  # source columns unchanged
    december = rows(incoming / "ved_signals_2017-12.parquet", "count(*), min(DayNum)")
    assert december == [(result.files["2017-12"], 31.3 + 0.08)]  # vehicle 8's trip on Dec 1


def test_every_row_traces_to_its_source_line(incoming):
    ps.prepare(VED, incoming)
    november = incoming / "ved_signals_2017-11.parquet"
    first = rows(november, 'origin_file, origin_line, VehId, Trip, "Timestamp(ms)"')[0]
    line = (incoming / "ved_signals" / first[0]).read_text().splitlines()[first[1]]
    assert line.split(",")[1:4] == [str(first[2]), str(first[3]), str(first[4])]
    ordered = rows(
        november, "bool_and(origin_line > 0), count(DISTINCT (origin_file, origin_line))"
    )
    assert ordered == [(True, rows(november)[0][0])]
    assert rows(november, 'count(*) FILTER (WHERE "OAT[DegC]" IS NULL)')[0][0] > 0  # NaN -> NULL


def test_deterministic_and_idempotent(incoming):
    ps.prepare(VED, incoming)
    november = incoming / "ved_signals_2017-11.parquet"
    before = sha(november)
    again = ps.prepare(VED, incoming)
    assert (
        again.staged == [] and again.files == {} and set(again.unchanged) == {"2017-11", "2017-12"}
    )
    november.unlink()
    assert ps.prepare(VED, incoming).files.keys() == {"2017-11"}
    assert sha(november) == before  # same CSVs, same file, same SHA-256
    forced = ps.prepare(VED, incoming, force=True)
    assert len(forced.staged) == 3 and sha(november) == before


def test_a_reader_that_loses_rows_stops_the_run(incoming, monkeypatch):
    monkeypatch.setattr(ps, "data_lines", lambda path: 10**6)
    with pytest.raises(ps.PrepareError, match="read 6.* rows, the file has 1,000,000"):
        ps.prepare(VED, incoming)


def test_line_count(tmp_path):
    path = tmp_path / "a.csv"
    path.write_bytes(b"h\r\n1\n2\n")
    assert ps.data_lines(path) == 2
    path.write_bytes(b"h\r\n1\n2")  # no newline after the last row
    assert ps.data_lines(path) == 2
    path.write_bytes(b"h\r\n")
    assert ps.data_lines(path) == 0


def test_rows_outside_the_periods_and_clashes(incoming):
    only_november = dataclasses.replace(VED, periods=("2017-11",))
    result = ps.prepare(only_november, incoming)
    assert list(result.files) == ["2017-11"] and result.rows_outside == {
        "2017-12": 5 * ROWS_PER_TRIP
    }
    week = incoming / "ved_signals" / "VED_171108_week.csv"
    text = week.read_text().splitlines()
    week.write_text("\n".join([text[0] + ",origin_line", *[t + ",1" for t in text[1:]]]) + "\n")
    with pytest.raises(ps.PrepareError, match="already has column"):
        ps.prepare(VED, incoming)


def test_no_source_or_no_files(tmp_path):
    with pytest.raises(ConfigError, match="no source block"):
        ps.prepare(load_datasets_config().dataset("yellow_taxi"), tmp_path)
    with pytest.raises(ConfigError, match="no files matching"):
        ps.prepare(VED, tmp_path)


def test_ingest_prepares_then_loads(incoming, tmp_path):
    db, bronze = tmp_path / "d.duckdb", tmp_path / "bronze"
    argv = ["--dataset", "ved_signals", "--months", "2017-11:2017-12", "--source", str(incoming),
            "--bronze", str(bronze), "--database", str(db)]  # fmt: skip
    assert ingest_data.main(argv) == 0
    with duckdb.connect(str(db), read_only=True) as con:
        loaded = con.execute(
            "SELECT period, row_count, source_url, status FROM file_manifest ORDER BY 1"
        ).fetchall()
        path = con.execute("SELECT bronze_path FROM file_manifest LIMIT 1").fetchone()[0]
        lineage = con.execute(f"SELECT bronze_row_id, origin_line FROM '{path}' LIMIT 1").fetchone()
    assert [r[0] for r in loaded] == ["2017-11", "2017-12"] and sum(r[1] for r in loaded) == ROWS
    assert {r[2] for r in loaded} == {"https://github.com/gsoh/VED"} and {r[3] for r in loaded} == {
        "LOADED"
    }
    assert lineage == (1, 1)
    assert ingest_data.main(argv) == 0  # again: nothing new
    with duckdb.connect(str(db), read_only=True) as con:
        last = con.execute(
            "SELECT files_loaded, files_skipped FROM pipeline_runs ORDER BY run_id DESC LIMIT 1"
        ).fetchone()
    assert last == (0, 2)
    # nothing to prepare there and no month files: every month missing (as for taxi files)
    assert ingest_data.main([*argv[:4], "--source", str(tmp_path / "none"), *argv[6:]]) == 1


def test_source_block_is_validated(tmp_path):
    data = yaml.safe_load((REPO_ROOT / "configs" / "datasets.yml").read_text())

    def load(change):
        copy = yaml.safe_load(yaml.safe_dump(data))
        change(copy["datasets"]["ved_signals"])
        path = tmp_path / "datasets.yml"
        path.write_text(yaml.safe_dump(copy))
        return load_datasets_config(path).dataset("ved_signals")

    with pytest.raises(ConfigError, match="kind must be one of"):
        load(lambda d: d["source"].update(kind="xml"))
    with pytest.raises(ConfigError, match="null_values must be a list"):
        load(lambda d: d["source"].update(null_values="NaN"))
    with pytest.raises(ConfigError, match="period_sql"):
        load(lambda d: d["source"].pop("period_sql"))
    enabled = load(lambda d: d.update(enabled=True, timestamp_column="row_time"))
    assert enabled.url_template is None and enabled.source.null_values == ("NaN",)
