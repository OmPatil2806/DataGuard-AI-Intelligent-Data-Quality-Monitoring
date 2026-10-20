"""Run notebooks top to bottom on the test fixtures, so a broken cell fails CI."""

import shutil
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"


@pytest.fixture
def fixture_data_dir(tmp_path):
    shutil.copy(
        FIXTURES / "yellow_tripdata_fixture.parquet", tmp_path / "yellow_tripdata_2023-01.parquet"
    )
    shutil.copy(FIXTURES / "taxi_zone_lookup_fixture.csv", tmp_path / "taxi_zone_lookup.csv")
    return tmp_path


def test_profiling_notebook_runs(fixture_data_dir, monkeypatch):
    monkeypatch.setenv("DATAGUARD_PROFILE_DIR", str(fixture_data_dir))
    notebook = nbformat.read(REPO_ROOT / "notebooks" / "01_data_profiling.ipynb", as_version=4)
    client = NotebookClient(
        notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": REPO_ROOT}}
    )
    client.execute()
    outputs = [o for cell in notebook.cells if cell.cell_type == "code" for o in cell.outputs]
    assert not [o for o in outputs if o.output_type == "error"]
    text = "".join(str(o.get("text", "")) + str(o.get("data", {})) for o in outputs)
    assert "1 file(s)" in text
    assert "Total rows: 200" in text


def test_rule_experiments_notebook_runs(fixture_data_dir, tmp_path, monkeypatch):
    from ingestion import ingest_data
    from quality import run_rules

    database = tmp_path / "dataguard.duckdb"
    common = ["--database", str(database)]
    source = ["--source", str(fixture_data_dir), "--bronze", str(tmp_path / "bronze")]
    assert ingest_data.main(["--months", "2023-01", *source, *common]) == 0
    assert run_rules.main(["--lookups", str(fixture_data_dir), *common]) == 0

    monkeypatch.setenv("DATAGUARD_DATABASE", str(database))
    monkeypatch.setenv("DATAGUARD_LOOKUPS", str(fixture_data_dir))
    notebook = nbformat.read(REPO_ROOT / "notebooks" / "02_rule_experiments.ipynb", as_version=4)
    client = NotebookClient(
        notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": REPO_ROOT}}
    )
    client.execute()
    outputs = [o for cell in notebook.cells if cell.cell_type == "code" for o in cell.outputs]
    assert not [o for o in outputs if o.output_type == "error"]
    text = "".join(str(o.get("text", "")) + str(o.get("data", {})) for o in outputs)
    assert "latest rule run: run_0002" in text
    assert "fare_not_negative" in text
    assert "PULocationID 999 not in zone lookup" in text
