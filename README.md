# DataGuard AI: Intelligent Data Quality Monitoring

DataGuard AI finds bad data that pipelines let through, explains what broke and since when,
and recommends what to do about it. Raw files land in **Bronze**, a rule engine sends valid rows
to **Silver** and invalid rows to **Quarantine** with a reason, dataset-level checks (schema,
freshness, volume, drift) raise incidents, and **Gold** monitoring tables feed a Streamlit dashboard.
Version 1 runs on real NYC TLC Yellow Taxi data with deliberately injected faults, so detection
quality is measured instead of assumed.

**Status:** Phase 0 — setup and data profiling (in progress)

## Repository layout

```
configs/      dataset list, quality rules, severity weights (YAML)
data/         raw and processed data, DuckDB file (ignored by Git)
docs/         design notes and data profiling findings
notebooks/    profiling, rule experiments, baselines, calibration, evaluation
src/          pipeline code, one package per stage
tests/        pytest suite
```
