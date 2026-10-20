import shutil
from pathlib import Path

import duckdb
import pytest

from common import db
from common.config import load_datasets_config
from ingestion import ingest_data
from quality import baselines, metrics
from quality.baselines import BaselineSettings
from quality.dataset_checks import CheckResult, check_schema, schema_changes, worst
from quality.metrics import MonthMetrics

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "yellow_tripdata_fixture.parquet"
SETTINGS = BaselineSettings()

# The real raw schemas found in profiling (docs/data_profile.md, section 2).
JAN_2023 = [(n, t) for n, t, *_ in duckdb.sql(f"DESCRIBE SELECT * FROM '{FIXTURE}'").fetchall()]
FEB_CHANGES = {
    "airport_fee": ("Airport_fee", "DOUBLE"),
    "VendorID": ("VendorID", "INTEGER"),
    "PULocationID": ("PULocationID", "INTEGER"),
    "DOLocationID": ("DOLocationID", "INTEGER"),
    "RatecodeID": ("RatecodeID", "BIGINT"),
    "passenger_count": ("passenger_count", "BIGINT"),
}
FEB_2023 = [FEB_CHANGES.get(n, (n, t)) for n, t in JAN_2023]


@pytest.fixture
def taxi():
    return load_datasets_config().dataset("yellow_taxi")


def month(period, schema):
    return MonthMetrics(period, period * 4, 100, 3.0, list(schema))


def run_check(taxi, schemas):
    months = {p: month(p, s) for p, s in schemas.items()}
    last = max(months)
    return check_schema(months[last], baselines.build(months, last, SETTINGS), taxi)


def test_unchanged_schema_passes(taxi):
    result = run_check(taxi, {"2023-02": FEB_2023, "2023-03": FEB_2023})
    assert (result.status, result.severity, result.detail) == ("pass", None, "schema unchanged")
    assert result.baseline == "compared with 2023-02"


def test_january_to_february_2023_is_one_finding(taxi):
    result = run_check(taxi, {"2023-01": JAN_2023, "2023-02": FEB_2023})
    assert result.status == "fail" and result.severity == "high"
    kinds = sorted((c.kind, c.column) for c in result.changes)
    assert kinds == [
        ("renamed", "Airport_fee"),
        ("type_changed", "DOLocationID"),
        ("type_changed", "PULocationID"),
        ("type_changed", "RatecodeID"),
        ("type_changed", "VendorID"),
        ("type_changed", "passenger_count"),
    ]
    assert "renamed airport_fee -> Airport_fee" in result.detail
    assert "passenger_count type DOUBLE -> BIGINT" in result.detail


def test_2019_and_2023_01_have_no_earlier_month(taxi):
    jan_2019 = [(n, "INTEGER" if n == "airport_fee" else t) for n, t in JAN_2023]
    result = run_check(taxi, {"2019-01": jan_2019, "2023-01": JAN_2023})
    assert result.status == "pass"
    assert result.baseline.startswith("no earlier month to compare")


def test_missing_required_column_is_critical(taxi):
    without = [(n, t) for n, t in FEB_2023 if n != "tpep_dropoff_datetime"]
    result = run_check(taxi, {"2023-02": FEB_2023, "2023-03": without})
    assert result.severity == "critical"
    assert [c.kind for c in result.changes] == ["missing"]


def test_missing_column_is_found_even_without_history(taxi):
    without = [(n, t) for n, t in JAN_2023 if n != "fare_amount"]
    result = run_check(taxi, {"2023-01": without})
    assert (result.status, result.severity) == ("fail", "critical")
    assert "required column fare_amount is missing" in result.detail


def test_new_and_removed_extra_columns(taxi):
    with_fee = [*FEB_2023, ("cbd_congestion_fee", "DOUBLE")]
    added = run_check(taxi, {"2023-02": FEB_2023, "2023-03": with_fee})
    assert (added.severity, [c.kind for c in added.changes]) == ("low", ["unexpected"])
    removed = run_check(taxi, {"2023-02": with_fee, "2023-03": FEB_2023})
    assert (removed.severity, [c.kind for c in removed.changes]) == ("medium", ["removed"])


def test_standard_column_coming_back(taxi):
    without = [(n, t) for n, t in FEB_2023 if n != "Airport_fee"]
    result = run_check(taxi, {"2023-02": without, "2023-03": FEB_2023})
    assert [(c.kind, c.severity) for c in result.changes] == [("added", "low")]


def test_schema_changes_without_previous(taxi):
    assert schema_changes(JAN_2023, None, taxi) == []


def test_worst_and_result_validation():
    assert worst(["low", "critical", "medium"]) == "critical"
    with pytest.raises(ValueError, match="needs a severity"):
        CheckResult("schema", "2023-01", "fail")
    with pytest.raises(ValueError, match="status must be"):
        CheckResult("schema", "2023-01", "broken")


def test_on_real_bronze(tmp_path, taxi):
    """End to end: Bronze -> metrics -> baseline -> schema check finds the February change."""
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    shutil.copy(FIXTURE, incoming / "yellow_tripdata_2023-01.parquet")
    changes = ", ".join(
        f"CAST({old} AS {new_type}) AS {new}" for old, (new, new_type) in FEB_CHANGES.items()
    )
    duckdb.sql(
        f"COPY (SELECT * EXCLUDE ({', '.join(FEB_CHANGES)}), {changes} FROM '{FIXTURE}') "
        f"TO '{incoming / 'yellow_tripdata_2023-02.parquet'}' (FORMAT parquet)"
    )
    database = tmp_path / "d.duckdb"
    assert (
        ingest_data.main(
            [
                "--months",
                "2023-01:2023-02",
                "--source",
                str(incoming),
                "--bronze",
                str(tmp_path / "bronze"),
                "--database",
                str(database),
            ]
        )
        == 0
    )
    with db.connect(database) as con:
        db.init_schema(con)
        months = metrics.collect(con, taxi)
    results = {
        p: check_schema(months[p], baselines.build(months, p, SETTINGS), taxi) for p in months
    }
    assert results["2023-01"].status == "pass"
    assert results["2023-02"].severity == "high"
    assert len(results["2023-02"].changes) == 6
