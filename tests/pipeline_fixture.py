"""The whole pipeline on the five planted months of tests/test_run_checks.py.

Bronze -> Silver/Quarantine -> dataset checks -> row rules -> incidents -> score, all in one
DuckDB file, as a user would run it. Used by the notebook 04 test and the Gold tests.
"""

import shutil
from pathlib import Path

import duckdb
from test_run_checks import FIXTURE, MONTHS

from incidents import incidents_cli
from ingestion import ingest_data
from quality import run_checks, run_rules, score
from silver import process_silver

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def run_pipeline(root: Path) -> Path:
    """Run every stage into root/ and return the DuckDB file."""
    incoming = root / "incoming"
    incoming.mkdir(parents=True)
    for period, query in MONTHS.items():
        target = incoming / f"yellow_tripdata_{period}.parquet"
        duckdb.sql(f"COPY ({query.format(src=FIXTURE)}) TO '{target}' (FORMAT parquet)")
    shutil.copy(FIXTURES / "taxi_zone_lookup_fixture.csv", incoming / "taxi_zone_lookup.csv")
    database = root / "dataguard.duckdb"
    common = ["--database", str(database)]
    bronze = ["--source", str(incoming), "--bronze", str(root / "bronze")]
    assert ingest_data.main(["--months", "2023-01:2023-05", *bronze, *common]) == 0
    silver = ["--silver", str(root / "silver"), "--quarantine", str(root / "quarantine")]
    assert process_silver.main(["--lookups", str(incoming), *silver, *common]) == 0
    assert run_checks.main(common) == 0
    assert run_rules.main(["--lookups", str(incoming), *common]) == 0
    assert incidents_cli.main([*common, "update"]) == 0
    assert score.main(common) == 0
    return database
