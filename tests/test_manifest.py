import hashlib
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from common import db
from common.config import ConfigError, load_datasets_config
from ingestion import manifest

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "yellow_tripdata_fixture.parquet"
NOW = datetime(2023, 3, 5, 6, 0, 0)


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


def test_sha256_matches_hashlib(jan):
    assert manifest.sha256_of(jan) == hashlib.sha256(jan.read_bytes()).hexdigest()


def test_expected_at_is_month_end_plus_lag():
    assert manifest.expected_at("2023-01", 60) == datetime(2023, 4, 2)
    assert manifest.expected_at("2023-02", 0) == datetime(2023, 3, 1)
    assert manifest.expected_at("2023-12", 60) == datetime(2024, 3, 1)
    assert manifest.expected_at("2023-01", None) is None


def test_register_new_file(con, taxi, jan):
    result = manifest.register_file(con, taxi, jan, now=NOW)
    e = result.entry
    assert result.is_new and result.other_versions == ()
    assert (e.dataset, e.period, e.file_name, e.status) == (
        "yellow_taxi",
        "2023-01",
        "yellow_tripdata_2023-01.parquet",
        "REGISTERED",
    )
    assert e.row_count == 200
    assert e.size_bytes == jan.stat().st_size
    assert e.source_url.endswith("/yellow_tripdata_2023-01.parquet")
    assert e.expected_at == datetime(2023, 4, 2)
    assert e.registered_at == NOW


def test_delivered_at_is_file_modification_time(con, taxi, jan):
    stamp = datetime(2023, 3, 20, 12, 0, 0).timestamp()
    os.utime(jan, (stamp, stamp))
    entry = manifest.register_file(con, taxi, jan).entry
    assert entry.delivered_at == manifest.delivered_at(jan)


def test_same_file_twice_is_recognised(con, taxi, jan):
    first = manifest.register_file(con, taxi, jan, now=NOW)
    copy = jan.with_name("copy_of_yellow_tripdata_2023-01.parquet")
    shutil.copy(jan, copy)  # same content, other name: still the same version
    second = manifest.register_file(con, taxi, copy, period="2023-01")
    assert not second.is_new
    assert second.entry == first.entry
    assert len(manifest.get_entries(con, "yellow_taxi")) == 1


def test_corrected_file_is_a_new_version(con, taxi, jan):
    old = manifest.register_file(con, taxi, jan, now=NOW).entry
    # The source re-exports the month with one fare corrected: same name, new content.
    trips = pd.read_parquet(jan)
    trips.loc[0, "fare_amount"] += 1.0
    trips.to_parquet(jan, index=False)
    result = manifest.register_file(con, taxi, jan, now=NOW)
    assert result.is_new
    assert result.entry.sha256 != old.sha256
    assert result.other_versions == (old,)
    assert len(manifest.get_entries(con, "yellow_taxi", "2023-01")) == 2


def test_period_from_name_or_argument(con, taxi, tmp_path):
    unnamed = tmp_path / "trips.parquet"
    shutil.copy(FIXTURE, unnamed)
    with pytest.raises(ConfigError, match="cannot tell the period"):
        manifest.register_file(con, taxi, unnamed)
    assert manifest.register_file(con, taxi, unnamed, period="2023-05").entry.period == "2023-05"


def test_missing_file(con, taxi, tmp_path):
    with pytest.raises(FileNotFoundError):
        manifest.register_file(con, taxi, tmp_path / "yellow_tripdata_2023-01.parquet")


def test_set_status(con, taxi, jan):
    entry = manifest.register_file(con, taxi, jan).entry
    manifest.set_status(con, entry, "LOADED", bronze_path="data/bronze/x", run_id="run_0001")
    updated = manifest.get_entries(con, "yellow_taxi")[0]
    assert (updated.status, updated.bronze_path, updated.loaded_run_id) == (
        "LOADED",
        "data/bronze/x",
        "run_0001",
    )
    manifest.set_status(con, updated, "SKIPPED")  # keeps the earlier path and run
    assert manifest.get_entries(con, "yellow_taxi")[0].bronze_path == "data/bronze/x"
    with pytest.raises(ValueError, match="status must be one of"):
        manifest.set_status(con, entry, "DONE")


def test_main_registers_and_reports(jan, tmp_path, capsys):
    database = tmp_path / "dataguard.duckdb"
    argv = ["--months", "2023-01:2023-02", "--source", str(jan.parent), "--database", str(database)]
    assert manifest.main(argv) == 0  # 2023-02 is missing, 2023-01 is registered
    assert manifest.main(argv) == 0  # second run: already known, nothing added
    out = capsys.readouterr().out
    assert out.count("yellow_tripdata_2023-01.parquet") == 2
    with db.connect(database, read_only=True) as con:
        assert con.execute("SELECT count(*) FROM file_manifest").fetchone()[0] == 1


def test_main_fails_when_no_file_exists(tmp_path):
    argv = [
        "--months",
        "2023-01",
        "--source",
        str(tmp_path),
        "--database",
        str(tmp_path / "d.duckdb"),
    ]
    assert manifest.main(argv) == 1
    assert manifest.main(["--months", "2023-13"]) == 2


def test_simulated_delivery_can_be_set_and_cleared(con, taxi, jan):
    entry = manifest.register_file(con, taxi, jan).entry
    assert entry.simulated_delivered_at is None
    manifest.set_simulated_delivery(con, entry, datetime(2023, 4, 20))
    assert manifest.get_entries(con, "yellow_taxi")[0].simulated_delivered_at == datetime(
        2023, 4, 20
    )
    manifest.set_simulated_delivery(con, entry, None)
    assert manifest.get_entries(con, "yellow_taxi")[0].simulated_delivered_at is None
