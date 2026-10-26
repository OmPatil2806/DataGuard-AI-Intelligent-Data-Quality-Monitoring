from pathlib import Path

import duckdb
import pytest

from common.config import ConfigError, load_datasets_config
from quality.standardize import StandardizeError, plan, standardize

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "yellow_tripdata_fixture.parquet"
SOURCE = f"read_parquet('{FIXTURE}')"

# The three real schema versions found in profiling (docs/data_profile.md, section 2).
JAN_2023 = SOURCE  # the fixture follows the 2023-01 schema
FEB_2023 = f"""(SELECT * EXCLUDE (airport_fee, VendorID, PULocationID, DOLocationID,
                                  RatecodeID, passenger_count),
                       CAST(VendorID AS INTEGER) AS VendorID,
                       CAST(PULocationID AS INTEGER) AS PULocationID,
                       CAST(DOLocationID AS INTEGER) AS DOLocationID,
                       CAST(RatecodeID AS BIGINT) AS RatecodeID,
                       CAST(passenger_count AS BIGINT) AS passenger_count,
                       airport_fee AS Airport_fee
                FROM {SOURCE})"""
JAN_2019 = f"(SELECT * REPLACE (CAST(airport_fee AS INTEGER) AS airport_fee) FROM {SOURCE})"


@pytest.fixture
def taxi():
    return load_datasets_config().dataset("yellow_taxi")


@pytest.fixture
def con():
    with duckdb.connect() as connection:
        yield connection


def schema(con, sql):
    return [(n, t) for n, t, *_ in con.execute(f"DESCRIBE {sql}").fetchall()]


def test_config_has_standard_schema(taxi):
    assert len(taxi.columns) == 19
    assert ("airport_fee", "DOUBLE") in taxi.columns
    assert ("passenger_count", "INTEGER") in taxi.columns


@pytest.mark.parametrize(
    "source", [JAN_2023, FEB_2023, JAN_2019], ids=["2023-01", "2023-02", "2019-01"]
)
def test_every_version_gets_the_same_schema(con, taxi, source):
    sql, report = standardize(con, source, taxi)
    assert schema(con, sql) == list(taxi.columns)
    assert report.lossy == {} and report.added == [] and report.dropped == []


def test_february_changes_are_reported(con, taxi):
    _, report = standardize(con, FEB_2023, taxi)
    assert report.renamed == [("Airport_fee", "airport_fee")]
    casts = {column: (src, dst) for column, src, dst in report.cast}
    # payment_type is BIGINT in every real file, so it is cast in every version.
    assert casts == {
        "passenger_count": ("BIGINT", "INTEGER"),
        "RatecodeID": ("BIGINT", "INTEGER"),
        "payment_type": ("BIGINT", "INTEGER"),
    }


def test_january_casts_doubles_to_integers(con, taxi):
    _, report = standardize(con, JAN_2023, taxi)
    casts = {column: (src, dst) for column, src, dst in report.cast}
    assert casts["passenger_count"] == ("DOUBLE", "INTEGER")
    assert casts["VendorID"] == ("BIGINT", "INTEGER")
    assert report.renamed == []


def test_values_are_unchanged(con, taxi):
    """Every value equals the source value after the type change, NULLs included."""
    sql, _ = standardize(con, FEB_2023, taxi)
    std = con.execute(f"SELECT * FROM ({sql})").df()
    raw = con.execute(f"SELECT * FROM {SOURCE}").df()
    assert len(std) == len(raw) == 200
    for column, _ in taxi.columns:
        left, right = std[column], raw[column]
        assert ((left == right) | (left.isna() & right.isna())).all(), column


def test_lossy_casts_are_counted(con, taxi):
    source = (
        f"(SELECT * REPLACE (CASE WHEN fare_amount > 30 THEN 1.5 ELSE passenger_count END "
        f"AS passenger_count) FROM {SOURCE})"
    )
    expected = con.execute(f"SELECT count_if(fare_amount > 30) FROM {SOURCE}").fetchone()[0]
    assert expected > 0
    sql, report = standardize(con, source, taxi)
    assert report.lossy == {"passenger_count": expected}
    assert "LOSSY passenger_count" in report.summary()
    assert con.execute(f"SELECT count(*) FROM ({sql})").fetchone()[0] == 200  # rows are kept


def test_missing_and_unexpected_columns(con, taxi):
    source = f"(SELECT * EXCLUDE (airport_fee), 0.0 AS cbd_congestion_fee FROM {SOURCE})"
    sql, report = standardize(con, source, taxi)
    assert report.added == ["airport_fee"]
    assert report.dropped == ["cbd_congestion_fee"]
    assert schema(con, sql) == list(taxi.columns)
    assert con.execute(f"SELECT count(airport_fee) FROM ({sql})").fetchone()[0] == 0


def test_bronze_lineage_columns_pass_through(con, taxi):
    source = (
        f"(SELECT *, 1::BIGINT AS bronze_row_id, 'f.parquet' AS source_file, "
        f"now()::TIMESTAMP AS ingestion_timestamp, 'run_0001' AS pipeline_run_id FROM {SOURCE})"
    )
    sql, report = standardize(con, source, taxi)
    names = [n for n, _ in schema(con, sql)]
    assert names[-4:] == ["bronze_row_id", "source_file", "ingestion_timestamp", "pipeline_run_id"]
    assert report.dropped == []


def test_already_standard_source(con, taxi):
    sql, _ = standardize(con, JAN_2023, taxi)
    _, report = standardize(con, f"({sql})", taxi)
    assert not report.changed
    assert report.summary() == "already standard"


def test_names_differing_only_by_case_are_refused(taxi):
    with pytest.raises(StandardizeError, match="both 'airport_fee' and 'Airport_fee'"):
        plan({"airport_fee": "DOUBLE", "Airport_fee": "DOUBLE"}, taxi)


def test_dataset_without_schema(taxi):
    import dataclasses

    without = dataclasses.replace(taxi, columns=())
    with pytest.raises(StandardizeError, match="no standard schema"):
        plan({"a": "INTEGER"}, without)


def test_config_rejects_bad_column_types(tmp_path):
    text = (Path(__file__).resolve().parents[1] / "configs" / "datasets.yml").read_text()
    path = tmp_path / "datasets.yml"
    path.write_text(text.replace("VendorID: INTEGER", "VendorID: NUMBER"))
    with pytest.raises(ConfigError, match="unknown type 'NUMBER'"):
        load_datasets_config(path)
    path.write_text(text.replace("VendorID: INTEGER", "VendorID: INTEGER\n      vendorid: INTEGER"))
    with pytest.raises(ConfigError, match="listed twice"):
        load_datasets_config(path)
