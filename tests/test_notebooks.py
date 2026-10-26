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


def test_baselines_and_drift_notebook_runs(tmp_path, monkeypatch):
    """Five fixture months with planted month-level problems (see tests/test_run_checks.py)."""
    import duckdb
    from test_run_checks import FIXTURE, MONTHS

    from ingestion import ingest_data
    from quality import run_checks

    incoming = tmp_path / "incoming"
    incoming.mkdir()
    for period, query in MONTHS.items():
        target = incoming / f"yellow_tripdata_{period}.parquet"
        duckdb.sql(f"COPY ({query.format(src=FIXTURE)}) TO '{target}' (FORMAT parquet)")
    database = tmp_path / "dataguard.duckdb"
    assert (
        ingest_data.main(
            [
                "--months",
                "2023-01:2023-05",
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
    assert run_checks.main(["--database", str(database)]) == 0

    monkeypatch.setenv("DATAGUARD_DATABASE", str(database))
    notebook = nbformat.read(REPO_ROOT / "notebooks" / "03_baselines_and_drift.ipynb", as_version=4)
    client = NotebookClient(
        notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": REPO_ROOT}}
    )
    client.execute()
    outputs = [o for cell in notebook.cells if cell.cell_type == "code" for o in cell.outputs]
    assert not [o for o in outputs if o.output_type == "error"]
    text = "".join(str(o.get("text", "")) + str(o.get("data", {})) for o in outputs)
    assert "latest check run: run_0002" in text
    assert '"expected_but_missing"' in text and "2023-02" in text


def test_quality_score_notebook_runs(tmp_path, monkeypatch):
    """The five planted months through Silver, checks, rules, incidents and the score."""
    from pipeline_fixture import run_pipeline

    run_pipeline(tmp_path)

    monkeypatch.setenv("DATAGUARD_DATABASE", str(tmp_path / "dataguard.duckdb"))
    path = REPO_ROOT / "notebooks" / "04_quality_score_calibration.ipynb"
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook, timeout=120, kernel_name="python3", resources={"metadata": {"path": REPO_ROOT}}
    )
    client.execute()
    outputs = [o for cell in notebook.cells if cell.cell_type == "code" for o in cell.outputs]
    assert not [o for o in outputs if o.output_type == "error"]
    text = "".join(str(o.get("text", "")) + str(o.get("data", {})) for o in outputs)
    assert "5 month(s) scored" in text
    assert '"lowest"' in text and '"2023-05"' in text
    assert '"active_incidents"' in text


def test_detection_evaluation_notebook_runs(tmp_path, monkeypatch):
    """A few experiments on clean test months, then the notebook evaluates them."""
    from evaluation_fixture import make_config, make_sources

    from evaluation import run_experiments

    config = make_config(tmp_path, tmp_path / "workspace")
    experiments = ["control", "missing_column", "out_of_range-1pct", "volume_drop-20pct",
                   "stale_file-20d"]  # fmt: skip
    run_experiments.run_all(config, make_sources(tmp_path / "source"), experiments, out=print)

    monkeypatch.setenv("DATAGUARD_FAULTS_CONFIG", str(tmp_path / "faults.yml"))
    path = REPO_ROOT / "notebooks" / "05_detection_evaluation.ipynb"
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": REPO_ROOT}}
    )
    client.execute()
    outputs = [o for cell in notebook.cells if cell.cell_type == "code" for o in cell.outputs]
    assert not [o for o in outputs if o.output_type == "error"]
    text = "".join(str(o.get("text", "")) + str(o.get("data", {})) for o in outputs)
    assert "4 experiment(s) evaluated of 36 planned" in text
    assert "control month raises: nothing" in text
    assert '"volume_drop-20pct"' in text and '"flagged_without_incident"' in text
    assert (tmp_path / "workspace" / "evaluation_results.csv").is_file()


def test_ml_anomaly_notebook_runs(tmp_path, monkeypatch):
    """A pipeline with a trained model (May has planted fares) and two clean experiments."""
    import duckdb
    from evaluation_fixture import make_config, make_sources

    from evaluation import run_experiments
    from gold import rebuild
    from ingestion import ingest_data
    from ml import train

    monkeypatch.setenv("DATAGUARD_MODELS_DIR", str(tmp_path / "models"))
    source = make_sources(tmp_path / "source")
    may = source / "yellow_tripdata_2023-05.parquet"
    trips = duckdb.sql(f"SELECT * FROM read_parquet('{may.as_posix()}')").df()
    short = trips.index[trips["trip_distance"].between(0.5, 3) & (trips["fare_amount"] > 0)]
    trips.loc[list(short[:10]), "fare_amount"] *= 10
    trips.to_parquet(may, index=False)
    common = ["--database", str(tmp_path / "dataguard.duckdb")]
    bronze = ["--source", str(source), "--bronze", str(tmp_path / "bronze")]
    assert ingest_data.main(["--months", "2023-01:2023-05", *bronze, *common]) == 0
    argv = ["--lookups", str(source), "--silver", str(tmp_path / "silver"),
            "--quarantine", str(tmp_path / "quarantine"), *common]  # fmt: skip
    assert rebuild.main(argv) == 0
    assert train.main(common) == 0
    assert rebuild.main(argv) == 0  # scores every month with the model

    clean = make_sources(tmp_path / "clean")  # the experiments start from clean months
    config = make_config(tmp_path, tmp_path / "workspace")
    run_experiments.run_all(config, clean, ["control", "plausible_but_wrong-20pct"], out=print)

    monkeypatch.setenv("DATAGUARD_DATABASE", str(tmp_path / "dataguard.duckdb"))
    monkeypatch.setenv("DATAGUARD_FAULTS_CONFIG", str(tmp_path / "faults.yml"))
    path = REPO_ROOT / "notebooks" / "06_ml_anomaly_detection.ipynb"
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": REPO_ROOT}}
    )
    client.execute()
    outputs = [o for cell in notebook.cells if cell.cell_type == "code" for o in cell.outputs]
    assert not [o for o in outputs if o.output_type == "error"]
    text = "".join(str(o.get("text", "")) + str(o.get("data", {})) for o in outputs)
    assert '"ml_incidents": [\n    "2023-05"\n  ]' in text
    assert '"ml_adds": [\n      "plausible_but_wrong-20pct"\n    ]' in text
    assert "1 ML incident(s)" in text


def test_investigation_notebook_runs(tmp_path, monkeypatch):
    """The planted pipeline, investigated, and three experiments (one concentrated fault)."""
    from evaluation_fixture import make_config, make_sources
    from pipeline_fixture import run_pipeline

    from evaluation import run_experiments
    from investigation import investigate

    database = run_pipeline(tmp_path / "pipeline")
    monkeypatch.setenv("DATAGUARD_REPORTS_DIR", str(tmp_path / "reports"))
    assert investigate.main(["--database", str(database)]) == 0
    config = make_config(tmp_path, tmp_path / "workspace")
    plan = ["control", "null_spike-20pct", "null_spike_vendor-20pct"]
    run_experiments.run_all(config, make_sources(tmp_path / "clean"), plan, out=print)

    monkeypatch.setenv("DATAGUARD_DATABASE", str(database))
    monkeypatch.setenv("DATAGUARD_FAULTS_CONFIG", str(tmp_path / "faults.yml"))
    path = REPO_ROOT / "notebooks" / "07_investigation.ipynb"
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": REPO_ROOT}}
    )
    client.execute()
    outputs = [o for cell in notebook.cells if cell.cell_type == "code" for o in cell.outputs]
    assert not [o for o in outputs if o.output_type == "error"]
    text = "".join(str(o.get("text", "")) + str(o.get("data", {})) for o in outputs)
    assert "6 incident(s) investigated" in text
    assert '"explained": 6' in text and '"INC-0004": "unit_change"' in text
    assert "# INC-0003 · NULL rate increase: fare_amount" in text  # most severe active, rendered
    assert '"top1": 2' in text and '"false_where": 0' in text and '"where_found": 1' in text


def test_ved_profiling_notebook_runs(tmp_path, monkeypatch):
    """The VED profile on the small fixture in the real files' shape."""
    from ved_fixture import make_unpacked

    monkeypatch.setenv("DATAGUARD_VED_DIR", str(make_unpacked(tmp_path / "incoming")))
    monkeypatch.setenv("DATAGUARD_VED_CACHE", str(tmp_path / "cache"))
    path = REPO_ROOT / "notebooks" / "08_ved_profiling.ipynb"
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook, timeout=300, kernel_name="python3", resources={"metadata": {"path": REPO_ROOT}}
    )
    client.execute()
    outputs = [o for cell in notebook.cells if cell.cell_type == "code" for o in cell.outputs]
    assert not [o for o in outputs if o.output_type == "error"]
    text = "".join(str(o.get("text", "")) + str(o.get("data", {})) for o in outputs)
    assert "2 weekly file(s), 1,280 rows" in text and '"rows_lost_by_reader": 0' in text
    assert '"negative_ms_rows": 5' in text and '"ICE": 2' in text
