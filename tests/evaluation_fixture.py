"""Source files and a config for running the detection experiments on test data.

The fixture's trips are January 2023; every other month is the same trips moved into that
month, so the history is steady and the target month is clean, like the real 2023 months.
"""

import copy
import shutil
from pathlib import Path

import duckdb
import yaml

from common.config import REPO_ROOT
from evaluation import fault_injector as fi

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURE = FIXTURES / "yellow_tripdata_fixture.parquet"
MONTHS = ["2023-01", "2023-02", "2023-03", "2023-04", "2023-05", "2023-06"]


def make_sources(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for offset, month in enumerate(MONTHS):
        target = folder / f"yellow_tripdata_{month}.parquet"
        duckdb.sql(
            f"COPY (SELECT * REPLACE ("
            f"tpep_pickup_datetime + INTERVAL {offset} MONTH AS tpep_pickup_datetime, "
            f"tpep_dropoff_datetime + INTERVAL {offset} MONTH AS tpep_dropoff_datetime) "
            f"FROM read_parquet('{FIXTURE.as_posix()}')) TO '{target.as_posix()}' (FORMAT parquet)"
        )
    shutil.copy(FIXTURES / "taxi_zone_lookup_fixture.csv", folder / "taxi_zone_lookup.csv")
    return folder


def make_config(folder: Path, workspace: Path, **changes) -> fi.FaultsConfig:
    data = copy.deepcopy(yaml.safe_load((REPO_ROOT / "configs" / "faults.yml").read_text()))
    data |= {"sample_fraction": 1.0, "workspace": str(workspace)} | changes
    path = folder / "faults.yml"
    path.write_text(yaml.safe_dump(data))
    return fi.load_faults_config(path)
