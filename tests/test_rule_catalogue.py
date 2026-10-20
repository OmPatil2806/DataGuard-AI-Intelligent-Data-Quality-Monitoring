import shutil
from pathlib import Path

import pytest

from common import db
from ingestion import ingest_data
from quality import rule_catalogue, run_rules
from quality.rules import load_rules

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_committed_catalogue_is_up_to_date():
    """Fails when configs/quality_rules.yml changed without regenerating the catalogue."""
    rules = load_rules("yellow_taxi", include_disabled=True)
    expected = rule_catalogue.render_catalogue(rules, "yellow_taxi")
    assert rule_catalogue.CATALOGUE.read_text(encoding="utf-8") == expected, (
        "docs/rule_catalogue.md is out of date: run python src/quality/rule_catalogue.py"
    )


def test_every_rule_is_documented_with_a_reason():
    rules = load_rules("yellow_taxi", include_disabled=True)
    text = rule_catalogue.CATALOGUE.read_text(encoding="utf-8")
    for rule in rules:
        assert f"`{rule.id}`" in text
        assert rule.description, f"{rule.id} has no description"
    assert "## Pending" in text and "`total_matches_parts`" in text


@pytest.mark.parametrize(
    ("rule_id", "expected"),
    [
        ("fare_not_negative", "`fare_amount` ≥ 0"),
        ("fare_not_zero", "`fare_amount` ≠ 0"),
        ("pickup_zone_not_unknown", "`PULocationID` is not 264, 265"),
        ("dropoff_not_before_pickup", "`tpep_dropoff_datetime` ≥ `tpep_pickup_datetime`"),
        ("trip_under_24_hours", "lasts at most 86,400 s"),
        ("pickup_in_file_month", "inside the file's month"),
        ("pickup_zone_known", "exists in lookup `taxi_zone.LocationID`"),
        ("trip_metadata_missing", "not all of `passenger_count`"),
        ("trip_times_present", "are filled"),
        ("trip_unique", "first row with this `VendorID`"),
        ("total_matches_parts", "(± 0.01)"),
    ],
)
def test_conditions_read_naturally(rule_id, expected):
    rule = next(r for r in load_rules("yellow_taxi", include_disabled=True) if r.id == rule_id)
    assert expected in rule_catalogue.condition(rule)


@pytest.fixture
def database(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    shutil.copy(
        FIXTURES / "yellow_tripdata_fixture.parquet", incoming / "yellow_tripdata_2023-01.parquet"
    )
    shutil.copy(FIXTURES / "taxi_zone_lookup_fixture.csv", incoming / "taxi_zone_lookup.csv")
    path = tmp_path / "d.duckdb"
    common = ["--database", str(path)]
    assert (
        ingest_data.main(
            [
                "--months",
                "2023-01",
                "--source",
                str(incoming),
                "--bronze",
                str(tmp_path / "bronze"),
                *common,
            ]
        )
        == 0
    )
    assert run_rules.main(["--lookups", str(incoming), *common]) == 0
    return path


def test_results_from_latest_run(database, tmp_path):
    out, results = tmp_path / "catalogue.md", tmp_path / "results.md"
    argv = [
        "--results",
        "--database",
        str(database),
        "--out",
        str(out),
        "--results-out",
        str(results),
    ]
    assert rule_catalogue.main(argv) == 0
    text = results.read_text()
    assert "Run `run_0002`" in text and "1 month(s), 200 rows" in text
    assert "| `fare_not_negative` | quarantine | 3 | 1.5000 | 1.5000 | 1 of 1 |" in text
    assert "| 2023-01 | 200 | 11 | 5.500 | 12 | 6.000 |" in text


def test_results_without_a_run(tmp_path):
    database = tmp_path / "empty.duckdb"
    with db.connect(database) as con:
        db.init_schema(con)
    argv = ["--results", "--database", str(database), "--out", str(tmp_path / "c.md")]
    assert rule_catalogue.main(argv) == 1
    assert (
        rule_catalogue.main(
            [
                *argv[:1],
                "--database",
                str(tmp_path / "missing.duckdb"),
                "--out",
                str(tmp_path / "c.md"),
            ]
        )
        == 1
    )
