import json
from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def fixture_trips() -> pd.DataFrame:
    return pd.read_parquet(FIXTURES / "yellow_tripdata_fixture.parquet")


@pytest.fixture
def fixture_issues() -> dict[str, list[int]]:
    return json.loads((FIXTURES / "yellow_tripdata_fixture_issues.json").read_text())


@pytest.fixture
def fixture_zones() -> pd.DataFrame:
    # keep_default_na=False: the zone lookup uses the literal string "N/A", not a missing value.
    return pd.read_csv(FIXTURES / "taxi_zone_lookup_fixture.csv", keep_default_na=False)
