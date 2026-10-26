import json
from pathlib import Path

import pandas as pd
import pytest

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(autouse=True, scope="session")
def _isolated_models_dir(tmp_path_factory):
    """Keep every test away from models trained on the real data (models/ in the repository)."""
    import os

    folder = tmp_path_factory.mktemp("models")
    previous = os.environ.get("DATAGUARD_MODELS_DIR")
    os.environ["DATAGUARD_MODELS_DIR"] = str(folder)
    yield folder
    if previous is None:
        os.environ.pop("DATAGUARD_MODELS_DIR", None)
    else:
        os.environ["DATAGUARD_MODELS_DIR"] = previous


@pytest.fixture(autouse=True, scope="session")
def _isolated_reports_dir(tmp_path_factory):
    """Keep investigation reports of the tests out of data/reports in the repository."""
    import os

    folder = tmp_path_factory.mktemp("reports")
    previous = os.environ.get("DATAGUARD_REPORTS_DIR")
    os.environ["DATAGUARD_REPORTS_DIR"] = str(folder)
    yield folder
    if previous is None:
        os.environ.pop("DATAGUARD_REPORTS_DIR", None)
    else:
        os.environ["DATAGUARD_REPORTS_DIR"] = previous


@pytest.fixture(autouse=True, scope="session")
def _isolated_alerts(tmp_path_factory):
    """Alerts of the tests go to their own outbox, and never to a real Slack webhook."""
    import os

    folder = tmp_path_factory.mktemp("alerts")
    previous = {k: os.environ.get(k) for k in ("DATAGUARD_ALERTS_DIR", "DATAGUARD_SLACK_WEBHOOK")}
    os.environ["DATAGUARD_ALERTS_DIR"] = str(folder)
    os.environ.pop("DATAGUARD_SLACK_WEBHOOK", None)
    yield folder
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


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
