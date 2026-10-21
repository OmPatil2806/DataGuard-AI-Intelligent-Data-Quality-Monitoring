"""gold_lineage and trace() on the five planted months, and on broken copies."""

import shutil
from pathlib import Path

import duckdb
import pandas as pd
import pytest
from pipeline_fixture import run_pipeline

from common import db
from gold import lineage
from gold import monitoring_tables as gold
from gold.lineage import TraceError, trace
from ingestion import ingest_data

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "yellow_tripdata_fixture.parquet"


@pytest.fixture(scope="module")
def pipeline_db(tmp_path_factory):
    database = run_pipeline(tmp_path_factory.mktemp("gold_lineage"))
    with db.connect(database) as con:
        gold.rebuild(con)
    return database


def lineage_table(path):
    with db.connect(path, read_only=True) as con:
        return con.execute("SELECT * FROM gold_lineage").df().set_index("period")


def rebuild(path):
    with db.connect(path) as con:
        gold.rebuild(con)
    return lineage_table(path)


@pytest.fixture
def broken(pipeline_db, tmp_path):
    """A copy of the pipeline (database and files) that a test may damage."""
    root = tmp_path / "copy"
    shutil.copytree(pipeline_db.parent, root)
    database = root / pipeline_db.name
    # Point the copy's control tables at the copied files. Paths are stored with forward
    # slashes (as_posix), also on Windows, so replace that form and the native one.
    source = pipeline_db.parent.resolve()
    with db.connect(database) as con:
        for old, new in {(source.as_posix(), root.resolve().as_posix()), (str(source), str(root))}:
            con.execute(
                "UPDATE file_manifest SET bronze_path = replace(bronze_path, ?, ?)", [old, new]
            )
            con.execute(
                "UPDATE silver_log SET silver_path = replace(silver_path, ?, ?), "
                "quarantine_path = replace(quarantine_path, ?, ?)",
                [old, new, old, new],
            )
        paths = con.execute(
            "SELECT bronze_path FROM file_manifest UNION ALL "
            "SELECT silver_path FROM silver_log UNION ALL SELECT quarantine_path FROM silver_log"
        ).fetchall()
    # never let a test damage the shared pipeline's files
    assert all(Path(p).resolve().is_relative_to(root.resolve()) for (p,) in paths), paths
    return database


def test_every_month_reconciles(pipeline_db):
    t = lineage_table(pipeline_db)
    assert t.index.tolist() == ["2023-01", "2023-02", "2023-03", "2023-04", "2023-05"]
    assert (t["status"] == "ok").all() and t["detail"].isna().all()
    assert (t["manifest_rows"] == t["bronze_rows"]).all()
    assert (t["silver_rows"] + t["quarantine_rows"] == t["bronze_rows"]).all()
    assert t.loc["2023-01", ["bronze_rows", "silver_rows", "quarantine_rows"]].tolist() == [
        200,
        189,
        11,
    ]
    assert (t["versions_registered"] == 1).all() and t["scored"].all()
    for column in ("loaded_run_id", "silver_run_id", "rule_run_id", "check_run_id"):
        assert t[column].notna().all(), column
    assert t.loc["2023-01", "source_url"].endswith("yellow_tripdata_2023-01.parquet")
    assert t.loc["2023-04", "silver_path"].endswith("part-0.parquet")


def test_missing_quarantine_file(broken):
    path = rebuild(broken).loc["2023-03", "quarantine_path"]
    Path(path).unlink()
    t = rebuild(broken)
    assert t.loc["2023-03", "status"] == "missing_file"
    assert "Quarantine file not found" in t.loc["2023-03", "detail"]
    assert pd.isna(t.loc["2023-03", "quarantine_rows"])
    assert (t.drop(index="2023-03")["status"] == "ok").all()


def test_silver_file_with_a_row_less(broken):
    path = rebuild(broken).loc["2023-01", "silver_path"]
    duckdb.sql(
        f"COPY (SELECT * FROM read_parquet('{path}') WHERE bronze_row_id <> 1) "
        f"TO '{path}.new' (FORMAT parquet)"
    )
    Path(f"{path}.new").replace(path)
    jan = rebuild(broken).loc["2023-01"]
    assert jan.status == "mismatch" and jan.silver_rows == 188
    assert "Silver + Quarantine = 199 rows, Bronze 200" in jan.detail
    assert "silver_log 189 + 11" in jan.detail


def test_manifest_count_differs(broken):
    with db.connect(broken) as con:
        con.execute("UPDATE file_manifest SET row_count = 201 WHERE period = '2023-02'")
    feb = rebuild(broken).loc["2023-02"]
    assert feb.status == "mismatch" and "manifest 201" in feb.detail


def test_months_without_silver_and_new_versions(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    shutil.copy(FIXTURE, incoming / "yellow_tripdata_2023-01.parquet")
    argv = ["--months", "2023-01", "--source", str(incoming), "--bronze", str(tmp_path / "b")]
    argv += ["--database", str(tmp_path / "d.duckdb")]
    assert ingest_data.main(argv) == 0
    trips = pd.read_parquet(incoming / "yellow_tripdata_2023-01.parquet")
    trips.loc[0, "fare_amount"] += 1.0  # a corrected re-export
    trips.to_parquet(incoming / "yellow_tripdata_2023-01.parquet", index=False)
    assert ingest_data.main(argv) == 0
    jan = rebuild(tmp_path / "d.duckdb").loc["2023-01"]
    assert (jan.status, jan.versions_registered, jan.bronze_rows) == ("not_processed", 2, 200)
    assert pd.isna(jan.silver_path) and not jan.scored
    with db.connect(tmp_path / "d.duckdb", read_only=True) as con:
        newest = con.execute(
            "SELECT sha256 FROM file_manifest ORDER BY registered_at DESC LIMIT 1"
        ).fetchone()[0]
        t = trace(con, "2023-01", 1)
    assert jan.sha256 == newest and t["sha256"] == newest
    assert t["layer"] is None and t["bronze_row"]["fare_amount"] == trips.loc[0, "fare_amount"]
    assert "not processed yet" in lineage.format_trace(t)


def test_rebuild_checksum_ignores_paths_and_runs(broken, pipeline_db):
    with db.connect(pipeline_db) as con:
        _, original = gold.rebuild(con)
    with db.connect(broken) as con:
        _, copied = gold.rebuild(con)
    by_name = {b.name: b.checksum for b in copied}
    assert {b.name: b.checksum for b in original} == by_name  # other folder, same content


# --- trace ----------------------------------------------------------------------------------


def test_trace_a_silver_row(pipeline_db):
    with db.connect(pipeline_db, read_only=True) as con:
        t = trace(con, "2023-01", 1)
    assert t["layer"] == "Silver" and t["failed_rules"] == [] and t["reasons"] == []
    assert t["source_file"] == "yellow_tripdata_2023-01.parquet"
    assert t["bronze_row"]["bronze_row_id"] == 1
    assert t["month_score"] == pytest.approx(91.17, abs=0.01)
    text = lineage.format_trace(t)
    assert "passed every rule" in text and "Silver" in text and "score      91.2" in text


def test_trace_a_quarantined_row(pipeline_db):
    with db.connect(pipeline_db, read_only=True) as con:
        t = trace(con, "2023-02", 5)
    assert t["layer"] == "Quarantine"
    assert "pickup_in_file_month" in t["failed_rules"]
    assert any("outside the file's month" in r for r in t["reasons"])
    assert "    - pickup" in lineage.format_trace(t)


def test_trace_follows_every_row_to_exactly_one_layer(pipeline_db):
    with db.connect(pipeline_db, read_only=True) as con:
        layers = [trace(con, "2023-01", i)["layer"] for i in range(1, 201)]
    assert layers.count("Silver") == 189 and layers.count("Quarantine") == 11


def test_trace_errors(pipeline_db):
    with db.connect(pipeline_db, read_only=True) as con:
        with pytest.raises(TraceError, match="no row 201"):
            trace(con, "2023-01", 201)
        with pytest.raises(TraceError, match="not in Bronze"):
            trace(con, "2024-01", 1)


def test_main(pipeline_db, capsys):
    common = ["--database", str(pipeline_db)]
    assert lineage.main([*common, "--period", "2023-02", "--row", "5"]) == 0
    assert "Quarantine" in capsys.readouterr().out
    assert lineage.main([*common, "--period", "2023-02", "--row", "0"]) == 1
    assert "error: 2023-02 has no row 0" in capsys.readouterr().out
