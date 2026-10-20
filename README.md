# DataGuard AI: Intelligent Data Quality Monitoring

[![CI](https://github.com/OmPatil2806/DataGuard-AI-Intelligent-Data-Quality-Monitoring/actions/workflows/ci.yml/badge.svg)](https://github.com/OmPatil2806/DataGuard-AI-Intelligent-Data-Quality-Monitoring/actions/workflows/ci.yml)

DataGuard AI finds bad data that pipelines let through, explains what broke and since when,
and recommends what to do about it. Raw files land in **Bronze**, a rule engine sends valid rows
to **Silver** and invalid rows to **Quarantine** with a reason, dataset-level checks (schema,
freshness, volume, drift) raise incidents, and **Gold** monitoring tables feed a Streamlit dashboard.
Version 1 runs on real NYC TLC Yellow Taxi data with deliberately injected faults, so detection
quality is measured instead of assumed.

**Status:** Phase 3 complete (dataset-level checks). Pending a run on the full data: the
total-amount rule (open item O1), the measured rule results and the drift calibration check
(notebook 03). Next: Phase 4, Silver, Quarantine and incidents.

## Setup

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest
ruff check .
```

## Download the data

```bash
python src/data_download/download_data.py                    # every month in configs/datasets.yml
python src/data_download/download_data.py --months 2023-01   # a single month
```

Files land in `data/incoming/` (ignored by Git). Existing valid files are skipped.

For fast development, make a reproducible 10% sample in `data/sample/`:

```bash
python src/data_download/make_sample.py                    # every downloaded month
python src/data_download/make_sample.py --fraction 0.05    # 5% instead
```

Tests never download data: they use the synthetic fixtures in `tests/fixtures/`
(regenerate with `python tests/fixtures/make_fixtures.py`).

## Load the data into Bronze

```bash
python src/ingestion/ingest_data.py                        # every downloaded month
python src/ingestion/ingest_data.py --months 2023-01       # one month
python src/ingestion/verify_bronze.py                      # check Bronze against the source
python src/common/db.py                                    # control tables and latest runs
```

Each file is registered in `file_manifest` (with its SHA-256), written unchanged to
`data/bronze/yellow_taxi/period=YYYY-MM/version=<sha>/` with lineage columns, and counted in
`pipeline_runs`. Running the same command again skips files that are already in Bronze.
Details: [docs/bronze_layout.md](docs/bronze_layout.md).

## Check the data quality rules

```bash
python src/quality/standardize.py      # how each Bronze month is mapped to the standard schema
python src/quality/run_rules.py        # run the row rules on every month in Bronze
```

Rules live in `configs/quality_rules.yml`. Each row gets the list of rules it fails, with a
reason; results per rule and month are stored in `rule_run_log` and `rule_run_summary`.
`notebooks/02_rule_experiments.ipynb` shows the latest run in detail.
Every rule is documented in [docs/rule_catalogue.md](docs/rule_catalogue.md), generated from the
config with `python src/quality/rule_catalogue.py` (`--results` also writes `docs/rule_results.md`).

## Write Silver and Quarantine

```bash
python src/silver/process_silver.py    # every month in Bronze
```

Rows without a quarantine failure go to `data/silver/`, the others to `data/quarantine/` with every
column and the failed rules and reasons. Both get trip duration, pickup date and hour, and pickup
and dropoff zone and borough names (`configs/datasets.yml`, `silver:`). Counts per month are in
`silver_log`.

## Check whole months

```bash
python src/quality/metrics.py          # metrics per month (computed once per file version)
python src/quality/run_checks.py       # schema, volume, freshness, NULL rate and drift checks
```

Each month is compared with a baseline built from up to six earlier months (cold-start limits
apply until three exist). Results are stored in `dataset_check_results`, the baselines used in
`baselines`. Details and calibration: [docs/dataset_checks.md](docs/dataset_checks.md);
`notebooks/03_baselines_and_drift.ipynb` shows the latest run.

## Profile the data

Open `notebooks/01_data_profiling.ipynb` with the `.venv` kernel and run all cells. It reads
`data/incoming/` and measures volume, schema differences, NULLs, row-level problems, duplicates and
distributions per file. The findings on the full 2019-01 and 2023 data are in
[docs/data_profile.md](docs/data_profile.md).

## Repository layout

```
configs/      dataset list, quality rules, severity weights (YAML)
data/         raw and processed data, DuckDB file (ignored by Git)
docs/         design notes and data profiling findings
notebooks/    profiling, rule experiments, baselines, calibration, evaluation
src/          pipeline code, one package per stage
tests/        pytest suite
```

The full Version 1 plan is in [docs/project_plan.md](docs/project_plan.md).
