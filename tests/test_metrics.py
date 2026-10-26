import shutil
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
import yaml

from common import db
from common.config import ConfigError, load_datasets_config, load_quality_rules
from ingestion import ingest_data
from quality import metrics

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TRIPS = FIXTURES / "yellow_tripdata_fixture.parquet"


@pytest.fixture
def taxi():
    return load_datasets_config().dataset("yellow_taxi")


@pytest.fixture
def bronze(tmp_path):
    """Bronze with January (2023-01 schema) and February (2023-02 schema, 150 rows)."""
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    shutil.copy(TRIPS, incoming / "yellow_tripdata_2023-01.parquet")
    duckdb.sql(
        f"COPY (SELECT * EXCLUDE (airport_fee), airport_fee AS Airport_fee "
        f"FROM read_parquet('{TRIPS}') LIMIT 150) "
        f"TO '{incoming / 'yellow_tripdata_2023-02.parquet'}' (FORMAT parquet)"
    )
    database = tmp_path / "d.duckdb"
    argv = [
        "--months",
        "2023-01:2023-02",
        "--source",
        str(incoming),
        "--bronze",
        str(tmp_path / "bronze"),
        "--database",
        str(database),
    ]
    assert ingest_data.main(argv) == 0
    return database


@pytest.fixture
def con(bronze):
    with db.connect(bronze) as connection:
        db.init_schema(connection)
        yield connection


def test_volume_and_schema(con, taxi):
    m = metrics.collect(con, taxi)
    jan, feb = m["2023-01"], m["2023-02"]
    assert (jan.rows, feb.rows) == (200, 150)
    assert jan.rows_per_day == pytest.approx(200 / 31)
    assert feb.rows_per_day == pytest.approx(150 / 28)
    assert ("airport_fee", "DOUBLE") in jan.raw_schema
    assert ("Airport_fee", "DOUBLE") in feb.raw_schema  # raw schema, not standardized
    assert not any(name == "bronze_row_id" for name, _ in jan.raw_schema)


def test_null_rates_match_pandas(con, taxi):
    jan = metrics.collect(con, taxi)["2023-01"]
    expected = pd.read_parquet(TRIPS).isna().mean()
    for column, _ in taxi.columns:
        assert jan.null_rates[column] == pytest.approx(expected[column]), column
    assert jan.null_rates["passenger_count"] == pytest.approx(0.02)  # 4 of 200


def test_histograms_match_numpy(con, taxi):
    bins = metrics.drift_bins("yellow_taxi")
    jan = metrics.collect(con, taxi)["2023-01"]
    trips = pd.read_parquet(TRIPS)
    for column, edges in bins.items():
        values = trips[column].dropna().to_numpy()
        # bin i: edges[i-1] <= x < edges[i]; bin 0 below the first edge; last bin from the last edge
        expected = np.bincount(
            np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1
        )
        assert jan.histograms[column] == expected.tolist(), column
        assert sum(jan.histograms[column]) == len(values)


def test_negative_values_land_in_the_first_bin(con, taxi):
    fare = metrics.collect(con, taxi)["2023-01"].histograms["fare_amount"]
    assert fare[0] == 3  # the three planted negative fares


def test_stats(con, taxi):
    jan = metrics.collect(con, taxi)["2023-01"]
    distance = pd.read_parquet(TRIPS)["trip_distance"]
    assert jan.stats["trip_distance"]["median"] == pytest.approx(distance.median())
    assert jan.stats["trip_distance"]["mean"] == pytest.approx(distance.mean())
    assert jan.stats["trip_distance"]["p99"] == pytest.approx(distance.quantile(0.99))


def test_metrics_are_stored_once_and_reloaded(con, taxi):
    first = metrics.collect(con, taxi)
    stored = con.execute("SELECT count(*), min(computed_at) FROM dataset_metrics").fetchone()
    again = metrics.collect(con, taxi)
    assert (
        con.execute("SELECT count(*), min(computed_at) FROM dataset_metrics").fetchone() == stored
    )
    assert again["2023-01"] == first["2023-01"]
    metrics.collect(con, taxi, force=True)
    assert con.execute("SELECT count(*) FROM dataset_metrics").fetchone()[0] == stored[0]


def test_only_requested_months(con, taxi):
    assert list(metrics.collect(con, taxi, months=["2023-02", "2023-07"])) == ["2023-02"]


def test_load_unknown_version(con):
    with pytest.raises(KeyError, match="no metrics"):
        metrics.load(con, "yellow_taxi", "2023-01", "0" * 64)


def test_bin_sql_boundaries():
    edges = [0, 10]
    sql = metrics.bin_sql("x", edges)
    rows = duckdb.sql(
        f"SELECT x, {sql} FROM (VALUES (-1.0), (0.0), (9.99), (10.0), (500.0)) t(x)"
    ).fetchall()
    assert [b for _, b in rows] == [0, 1, 1, 2, 2]


def test_unknown_bin_column(con, taxi):
    with pytest.raises(ConfigError, match="unknown column\\(s\\) fare"):
        metrics.collect(con, taxi, bins={"fare": [0, 1]})


@pytest.mark.parametrize(
    ("edges", "message"),
    [([5], "at least two"), ([0, "a"], "only numbers"), ([0, 5, 5], "strictly increasing")],
)
def test_bin_config_is_validated(tmp_path, edges, message):
    data = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "quality_rules.yml").read_text()
    )
    data["defaults"]["drift"]["bins"] = {"yellow_taxi": {"trip_distance": edges}}
    path = tmp_path / "quality_rules.yml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match=message):
        load_quality_rules(path)


def test_main_prints_overview(bronze, capsys):
    assert metrics.main(["--database", str(bronze)]) == 0
    out = capsys.readouterr().out
    assert "2023-01" in out and "2023-02" in out and "passenger_count 2.00%" in out


def test_main_without_bronze(tmp_path, capsys):
    assert metrics.main(["--database", str(tmp_path / "empty.duckdb")]) == 1
