# DataGuard AI: Intelligent Data Quality Monitoring

[![CI](https://github.com/OmPatil2806/DataGuard-AI-Intelligent-Data-Quality-Monitoring/actions/workflows/ci.yml/badge.svg)](https://github.com/OmPatil2806/DataGuard-AI-Intelligent-Data-Quality-Monitoring/actions/workflows/ci.yml)

DataGuard AI finds bad data that pipelines let through, explains what broke and since when,
and recommends what to do about it. Raw files land in **Bronze**, a rule engine sends valid rows
to **Silver** and invalid rows to **Quarantine** with a reason, dataset-level checks (schema,
freshness, volume, drift) raise incidents, and **Gold** monitoring tables feed a Streamlit dashboard.
Version 1 runs on real NYC TLC Yellow Taxi data with deliberately injected faults, so detection
quality is measured instead of assumed.

**Status:** Phase 0 — setup and data profiling (in progress)

## Setup

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
pytest
ruff check .
```

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
