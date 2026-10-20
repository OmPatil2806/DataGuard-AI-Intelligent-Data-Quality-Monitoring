"""Silver enrichment (derived and lookup columns) and the process_silver.py command."""

import json
import shutil
from pathlib import Path

import duckdb
import pandas as pd
import pytest
import yaml

from common import db
from common.config import ConfigError, load_datasets_config
from ingestion import ingest_data
from silver import process_silver
from silver.silver_processor import SilverError, enrich_sql

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TRIPS = FIXTURES / "yellow_tripdata_fixture.parquet"
ZONES = FIXTURES / "taxi_zone_lookup_fixture.csv"
ISSUES = json.loads((FIXTURES / "yellow_tripdata_fixture_issues.json").read_text())


@pytest.fixture
def paths(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    shutil.copy(TRIPS, incoming / "yellow_tripdata_2023-01.parquet")
    shutil.copy(ZONES, incoming / "taxi_zone_lookup.csv")
    duckdb.sql(
        f"COPY (SELECT * EXCLUDE (airport_fee), airport_fee AS Airport_fee FROM '{TRIPS}') "
        f"TO '{incoming / 'yellow_tripdata_2023-02.parquet'}' (FORMAT parquet)"
    )
    p = {
        "incoming": incoming,
        "db": tmp_path / "d.duckdb",
        "silver": tmp_path / "silver",
        "quarantine": tmp_path / "quarantine",
    }
    argv = [
        "--months",
        "2023-01:2023-02",
        "--source",
        str(incoming),
        "--bronze",
        str(tmp_path / "bronze"),
        "--database",
        str(p["db"]),
    ]
    assert ingest_data.main(argv) == 0
    return p


def run(paths, *extra):
    argv = [
        "--lookups",
        str(paths["incoming"]),
        "--database",
        str(paths["db"]),
        "--silver",
        str(paths["silver"]),
        "--quarantine",
        str(paths["quarantine"]),
        *extra,
    ]
    return process_silver.main(argv)


def query(paths, sql):
    with db.connect(paths["db"], read_only=True) as con:
        return con.execute(sql).fetchall()


def silver_files(paths):
    with db.connect(paths["db"], read_only=True) as con:
        return process_silver.current_silver(con, "yellow_taxi")


def both(paths, period="2023-01"):
    silver, quarantine = silver_files(paths)[period]
    return pd.concat([pd.read_parquet(silver), pd.read_parquet(quarantine)])


def test_command_processes_every_month(paths):
    assert run(paths) == 0
    runs = query(
        paths, "SELECT status, files_loaded, rows_loaded FROM pipeline_runs ORDER BY run_id"
    )
    assert runs[-1] == ("SUCCESS", 2, 400)
    log = query(
        paths,
        "SELECT period, status, rows, silver_rows + quarantine_rows FROM silver_log "
        "ORDER BY period",
    )
    assert log == [("2023-01", "processed", 200, 200), ("2023-02", "processed", 200, 200)]


def test_second_run_skips_and_force_rewrites(paths):
    run(paths)
    assert run(paths) == 0
    assert query(paths, "SELECT DISTINCT status FROM silver_log WHERE run_id = 'run_0003'") == [
        ("skipped",)
    ]
    assert run(paths, "--force", "--months", "2023-01") == 0
    assert query(paths, "SELECT status FROM silver_log WHERE run_id = 'run_0004'") == [
        ("processed",)
    ]


def test_zone_names_match_the_lookup(paths):
    run(paths)
    rows = both(paths)
    zones = pd.read_csv(ZONES, keep_default_na=False).set_index("LocationID")
    known = rows[rows["PULocationID"].isin(zones.index)]
    assert (
        known["pickup_borough"].to_numpy() == zones.loc[known["PULocationID"], "Borough"].to_numpy()
    ).all()
    assert (
        known["pickup_zone"].to_numpy() == zones.loc[known["PULocationID"], "Zone"].to_numpy()
    ).all()
    dropoff = rows[rows["DOLocationID"] == 265].iloc[0]
    assert (dropoff["dropoff_borough"], dropoff["dropoff_zone"]) == ("N/A", "Outside of NYC")


def test_unknown_zone_gets_no_name(paths):
    run(paths)
    rows = both(paths)
    orphans = rows[rows["PULocationID"] == 999]
    assert len(orphans) == len(ISSUES["unknown_location_id"])
    assert orphans["pickup_zone"].isna().all() and orphans["pickup_borough"].isna().all()


def test_derived_columns(paths):
    run(paths)
    rows = both(paths)
    minutes = (rows["tpep_dropoff_datetime"] - rows["tpep_pickup_datetime"]).dt.total_seconds() / 60
    assert (rows["trip_duration_min"] - minutes).abs().max() < 1e-9
    assert (
        pd.to_datetime(rows["pickup_date"]) == rows["tpep_pickup_datetime"].dt.normalize()
    ).all()
    assert (rows["pickup_hour"] == rows["tpep_pickup_datetime"].dt.hour).all()
    backwards = rows[rows["bronze_row_id"] - 1 == ISSUES["dropoff_before_pickup"][0]]
    assert backwards["trip_duration_min"].iloc[0] == pytest.approx(-5)  # kept, in Quarantine


def test_quarantine_rows_have_a_quarantine_rule(paths):
    run(paths)
    _, quarantine = silver_files(paths)["2023-01"]
    q = pd.read_parquet(quarantine)
    assert len(q) > 0 and all(len(rules) > 0 for rules in q["quarantine_rules"])


def test_month_missing_from_bronze_gives_warning(paths):
    assert run(paths, "--months", "2023-01:2023-03") == 0
    status, error = query(
        paths, "SELECT status, error_message FROM pipeline_runs ORDER BY run_id DESC LIMIT 1"
    )[0]
    assert status == "WARNING" and "2023-03 not in Bronze" in error


def test_unreadable_month_fails_run_but_others_are_written(paths):
    bronze = query(paths, "SELECT bronze_path FROM file_manifest WHERE period = '2023-02'")[0][0]
    Path(bronze).write_bytes(b"not parquet")
    assert run(paths) == 1
    assert list(silver_files(paths)) == ["2023-01"]


def test_current_silver_follows_the_bronze_version(paths):
    run(paths)
    path = paths["incoming"] / "yellow_tripdata_2023-01.parquet"
    trips = pd.read_parquet(path)
    trips.loc[0, "fare_amount"] += 1.0
    trips.to_parquet(path, index=False)
    ingest_argv = [
        "--months",
        "2023-01",
        "--source",
        str(paths["incoming"]),
        "--bronze",
        str(paths["db"].parent / "bronze"),
        "--database",
        str(paths["db"]),
    ]
    assert ingest_data.main(ingest_argv) == 0
    assert "2023-01" not in silver_files(paths)  # the new version has no Silver yet
    run(paths, "--months", "2023-01")
    assert "2023-01" in silver_files(paths)


def test_bad_arguments(paths, tmp_path):
    assert run(paths, "--months", "2023-13") == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    assert process_silver.main(["--lookups", str(empty), "--database", str(paths["db"])]) == 2


def test_duplicate_lookup_keys_are_refused():
    taxi = load_datasets_config().dataset("yellow_taxi")
    with duckdb.connect() as con:
        con.execute(
            "CREATE TABLE lookup_taxi_zone AS SELECT * FROM (VALUES (1, 'A', 'x'), "
            "(1, 'B', 'y')) t(LocationID, Borough, Zone)"
        )
        source = (
            "(SELECT 1 AS PULocationID, 1 AS DOLocationID, "
            "TIMESTAMP '2023-01-01 10:00' AS tpep_pickup_datetime, "
            "TIMESTAMP '2023-01-01 10:10' AS tpep_dropoff_datetime)"
        )
        with pytest.raises(SilverError, match="duplicate values in LocationID"):
            enrich_sql(con, taxi, source)


def write_config(tmp_path, change):
    data = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "datasets.yml").read_text()
    )
    change(data["datasets"]["yellow_taxi"]["silver"])
    path = tmp_path / "datasets.yml"
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda s: s["joins"][0].update(lookup="zones"), "unknown lookup 'zones'"),
        (
            lambda s: s["joins"][0].update(column="pickup_id"),
            "'pickup_id' is not a standard column",
        ),
        (lambda s: s["derived"].update(pickup_zone="'x'"), "defined twice: pickup_zone"),
        (lambda s: s["derived"].update(fare_amount="1"), "defined twice: fare_amount"),
        (lambda s: s.update(derived={"x": ""}), "must map column names to SQL expressions"),
        (lambda s: s["joins"][0].pop("key"), "missing required key 'key'"),
    ],
)
def test_silver_config_is_validated(tmp_path, change, message):
    with pytest.raises(ConfigError, match=message):
        load_datasets_config(write_config(tmp_path, change))
