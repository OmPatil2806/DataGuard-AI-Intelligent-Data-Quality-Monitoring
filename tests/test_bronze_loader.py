import shutil
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq
import pytest

from common import db
from common.config import load_datasets_config
from ingestion import bronze_loader, manifest
from ingestion.bronze_loader import LINEAGE_COLUMNS, BronzeError

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "yellow_tripdata_fixture.parquet"
T0 = datetime(2023, 3, 5, 6, 0, 0)


@pytest.fixture
def taxi():
    return load_datasets_config().dataset("yellow_taxi")


@pytest.fixture
def con(tmp_path):
    with db.connect(tmp_path / "dataguard.duckdb") as connection:
        db.init_schema(connection)
        yield connection


@pytest.fixture
def jan(tmp_path):
    path = tmp_path / "incoming" / "yellow_tripdata_2023-01.parquet"
    path.parent.mkdir()
    shutil.copy(FIXTURE, path)
    return path


@pytest.fixture
def bronze(tmp_path):
    return tmp_path / "bronze"


def register(con, taxi, path):
    return manifest.register_file(con, taxi, path, now=T0).entry


def test_bronze_is_an_exact_copy_plus_lineage(con, taxi, jan, bronze):
    entry = register(con, taxi, jan)
    result = bronze_loader.load_file(con, entry, jan, bronze, "run_0001", now=T0)
    assert result.status == "loaded" and result.rows == 200
    assert (
        result.path
        == bronze
        / "yellow_taxi"
        / "period=2023-01"
        / f"version={entry.sha256[:12]}"
        / "part-0.parquet"
    )

    # Compare as Arrow tables: column names, Parquet types, values and row order.
    source = pq.read_table(jan).replace_schema_metadata(None)
    table = pq.read_table(result.path).replace_schema_metadata(None)
    assert table.column_names == source.column_names + list(LINEAGE_COLUMNS)
    assert table.select(source.column_names).equals(source)
    loaded = table.to_pandas()
    assert loaded["bronze_row_id"].tolist() == list(range(1, 201))
    assert set(loaded["source_file"]) == {"yellow_tripdata_2023-01.parquet"}
    assert set(loaded["pipeline_run_id"]) == {"run_0001"}
    assert set(loaded["ingestion_timestamp"]) == {pd.Timestamp(T0)}


def test_raw_schema_is_kept_for_other_versions(con, taxi, jan, bronze, tmp_path):
    # February-style file: Airport_fee renamed and IDs as INTEGER. Bronze must not "fix" it.
    feb = jan.with_name("yellow_tripdata_2023-02.parquet")
    duckdb.sql(
        f"COPY (SELECT * EXCLUDE (airport_fee, PULocationID), airport_fee AS Airport_fee, "
        f"CAST(PULocationID AS INTEGER) AS PULocationID FROM read_parquet('{jan}')) "
        f"TO '{feb}' (FORMAT parquet)"
    )
    result = bronze_loader.load_file(con, register(con, taxi, feb), feb, bronze, "run_0001")
    types = dict(
        duckdb.sql(f"DESCRIBE SELECT * FROM read_parquet('{result.path}')").fetchall()[i][:2]
        for i in range(23)
    )
    assert "Airport_fee" in types and "airport_fee" not in types
    assert types["PULocationID"] == "INTEGER"


def test_manifest_is_marked_loaded(con, taxi, jan, bronze):
    entry = register(con, taxi, jan)
    bronze_loader.load_file(con, entry, jan, bronze, "run_0007")
    stored = manifest.get_entries(con, "yellow_taxi")[0]
    assert stored.status == "LOADED"
    assert stored.loaded_run_id == "run_0007"
    assert stored.bronze_path.endswith("version=" + entry.sha256[:12] + "/part-0.parquet")


def test_loading_the_same_version_again_is_skipped(con, taxi, jan, bronze):
    entry = register(con, taxi, jan)
    first = bronze_loader.load_file(con, entry, jan, bronze, "run_0001", now=T0)
    second = bronze_loader.load_file(con, entry, jan, bronze, "run_0002")
    assert second.status == "skipped" and second.rows == 200
    assert set(pd.read_parquet(first.path)["pipeline_run_id"]) == {"run_0001"}  # untouched
    forced = bronze_loader.load_file(con, entry, jan, bronze, "run_0003", force=True)
    assert forced.status == "loaded"
    assert set(pd.read_parquet(forced.path)["pipeline_run_id"]) == {"run_0003"}


def test_new_version_is_kept_next_to_the_old_one(con, taxi, jan, bronze):
    old = register(con, taxi, jan)
    old_path = bronze_loader.load_file(con, old, jan, bronze, "run_0001").path
    trips = pd.read_parquet(jan)
    trips.loc[0, "fare_amount"] += 1.0
    trips.to_parquet(jan, index=False)
    new = manifest.register_file(con, taxi, jan, now=datetime(2023, 4, 1)).entry
    new_path = bronze_loader.load_file(con, new, jan, bronze, "run_0002").path
    assert old_path.exists() and new_path.exists() and old_path != new_path
    assert bronze_loader.current_files(con, taxi) == {"2023-01": new_path}


def test_file_changed_after_registration_is_refused(con, taxi, jan, bronze):
    entry = register(con, taxi, jan)
    trips = pd.read_parquet(jan)
    trips.loc[0, "fare_amount"] = 999.0
    trips.to_parquet(jan, index=False)
    with pytest.raises(BronzeError, match="changed since it was registered"):
        bronze_loader.load_file(con, entry, jan, bronze, "run_0001")
    assert not (bronze / "yellow_taxi").exists()


def test_lineage_column_clash_is_refused(con, taxi, jan, bronze):
    query = f"SELECT *, 'x' AS source_file FROM read_parquet('{FIXTURE}')"
    duckdb.sql(f"COPY ({query}) TO '{jan}' (FORMAT parquet)")
    entry = register(con, taxi, jan)
    with pytest.raises(BronzeError, match="already has lineage column"):
        bronze_loader.load_file(con, entry, jan, bronze, "run_0001")


def test_row_count_mismatch_leaves_nothing_behind(con, taxi, jan, bronze):
    entry = register(con, taxi, jan)
    wrong = manifest.ManifestEntry(**{**entry.__dict__, "row_count": 999})
    with pytest.raises(BronzeError, match="wrote 200 rows but the source has 999"):
        bronze_loader.load_file(con, wrong, jan, bronze, "run_0001")
    period_dir = bronze / "yellow_taxi" / "period=2023-01"
    assert list(period_dir.iterdir()) == []  # no partial or temporary folder
    assert manifest.get_entries(con, "yellow_taxi")[0].status == "REGISTERED"


def test_current_files_ignores_unloaded_versions(con, taxi, jan, bronze):
    register(con, taxi, jan)
    assert bronze_loader.current_files(con, "yellow_taxi") == {}
