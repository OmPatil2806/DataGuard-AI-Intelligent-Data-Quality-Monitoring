# DataGuard AI — Project Plan (Version 1)

DataGuard AI finds bad data that pipelines let through, explains what broke and since when,
and recommends what to do about it. Version 1 runs on real NYC TLC Yellow Taxi data (and the
Vehicle Energy Dataset later) with deliberately injected faults, so detection quality is measured
instead of assumed.

Version 1 uses rules, statistics and one ML model (Isolation Forest). The LLM explainer and the
AI agent are Version 2 and plug into interfaces built in Version 1.

## Principles

- **"Pipeline succeeded" is not "data is good".** Every run is shown on the dashboard, clean or not.
- **Bronze is never modified.** Every later layer can be rebuilt from it.
- **No silent fixing.** Rows are standardized (names, types), then accepted or quarantined with a
  reason. Bad source files are fixed by re-ingesting a corrected file.
- **Humans approve actions.** The system suggests; it never runs a fix on its own.
- **Everything is measured.** Injected faults with known ground truth prove what is caught and missed.

## Key decisions

| Topic | Decision |
|---|---|
| Data | Yellow Taxi 2023-01 to 2023-12, plus one older month for a real schema difference; zone lookup; VED in Phase 9 |
| Storage | Parquet files for bulk data, one DuckDB file for metadata, incidents and Gold tables |
| Row rules | Pandera for schema and dtypes; range, NULL, pattern, referential and consistency rules compiled from YAML to DuckDB SQL |
| Rule actions | Each rule has `action: quarantine` or `action: warn` |
| Volume check | Rows per day against a rolling median (handles short months and seasonality) |
| Drift check | PSI per numeric column (KS p-values flag everything at this row count) |
| Freshness | File manifest with `expected_at` and `delivered_at` |
| Uniqueness | Composite trip key (vendor, pickup/dropoff time, locations, fare) |
| Baselines | Trusted after 3 earlier runs; fixed YAML thresholds until then |
| Incident resolve | After 2 consecutive clean runs |
| ML | Isolation Forest on Silver rows; flags only (warn), never quarantines |
| Investigation | Deterministic evidence pack + template explainer behind an `Explainer` interface |
| Alerts | Local outbox (`data/alerts/outbox.jsonl`) and `alert_log` for new Critical and High incidents, shown on the dashboard |
| Runtime | Local Python virtual environment; no Docker, no servers |
| CI | GitHub Actions: lint and tests on every push |

## Default thresholds (calibrated later)

| Item | Default |
|---|---|
| Volume tolerance | ±30% rows per day against the rolling median |
| Drift | PSI > 0.2 High, 0.1–0.2 Medium |
| Baseline minimum | 3 earlier runs |
| Auto-resolve | 2 clean runs |
| ML false-positive target | ≤ 0.5% of rows on clean months |
| Development data | 10% sample first, then full months |

## Phases

### P0 — Setup and data profiling

| Step | Content | Check |
|---|---|---|
| 0.1 | Repository structure, `.gitignore`, README skeleton | Tree matches the plan |
| 0.2 | `requirements.txt`, `requirements-dev.txt`, `pyproject.toml`, smoke tests | `pytest` and `ruff check .` pass |
| 0.3 | GitHub Actions CI (lint, format, tests) | Green check on GitHub |
| 0.4 | `configs/*.yml` and a validated config loader | Tests pass; broken config gives a clear error |
| 0.5 | `download_data.py`: idempotent, retries, Parquet check, zone lookup | One month downloads; a rerun skips it |
| 0.6 | 10% sample mode and committed test fixtures | Sample builds in seconds; fixture loads in a test |
| 0.7 | `01_data_profiling.ipynb` | Runs top to bottom |
| 0.8 | `docs/data_profile.md` findings and README update | Every check category has a finding or a fault to test it |

### P1 — Bronze ingestion and control tables
`ingest_data.py`, `bronze_loader.py`. Write Parquet unchanged plus `source_file`,
`ingestion_timestamp`, `pipeline_run_id`, `bronze_row_id`. Tables `pipeline_runs` and
`file_manifest`. Reloading the same file is detected and skipped.
**Done when:** one month loads into Bronze, the run is recorded, and a reload changes nothing.

### P2 — Row-level rules
`standardize.py`, `row_rules.py`, `quality_rules.yml`. Schema and dtypes, completeness, uniqueness,
range, pattern and timestamp, referential integrity, consistency. Each row gets a `failed_rules`
list. Zone IDs 264/265 raise a warning. Table `rule_run_log`.
**Done when:** unit tests on hand-made bad rows flag exactly the expected rows.

### P3 — Dataset-level checks and baselines
`dataset_checks.py`. Schema changes, volume (rows per day), freshness (manifest), NULL rate,
drift (PSI). Table `baselines` with cold-start fallback. Notebook 03.
**Done when:** the real schema difference is detected and the clean months raise no volume or
drift alarms.

### P4 — Silver, Quarantine, incidents and quality score
`silver_processor.py`, `incident_manager.py`, `severity_engine.py`. Split rows, enrich Silver
(trip duration, dates, zone names), incident lifecycle (open, de-duplicate, investigating, resolve),
severity and quality score. Notebook 04 calibrates the weights.
**Done when:** incidents open, de-duplicate and close correctly across runs (tested).

### P5 — Gold tables and lineage
`monitoring_tables.py`. `gold_dataset_health`, `gold_trends`, `gold_schema_history`,
`gold_column_profile`, `gold_incident_summary`, `gold_lineage`.
**Done when:** Gold rebuilds from Bronze in one step with identical results.

### P6 — Fault injection and detection evaluation
`fault_injector.py`, `evaluate.py`, notebook 05. Ten faults, one per check category: missing
column, NULL spike, duplicate rows, out-of-range values, orphan location IDs, unit change, volume
drop, stale file, pickup outside the month, swapped pickup/dropoff. Strengths 1%, 5%, 20%.
Ground truth in `fault_manifest.json`.
**Done when:** a table shows what was caught and missed, and every miss is explained.

### P7 — ML anomaly detection
`src/ml/features.py`, `train.py`, `score.py`, `configs/ml.yml`, notebook 06.
- Features: trip duration, speed, fare per mile, fare per minute, total vs. sum of components,
  pickup hour and weekday, distance, passenger count.
- Isolation Forest (scikit-learn) trained on clean Silver months; time-based split (train Jan–Mar,
  calibrate threshold on April, test on later months).
- Every Silver row gets `anomaly_score` and `is_anomaly` (warning only). A rise in the anomaly rate
  above its baseline opens an incident (check category 11).
- Each flag lists the 2–3 most deviating features.
- Fault 11, "plausible but wrong": values in range but inconsistent together (e.g. fares ×3 on short
  trips).
- Model saved to `models/` (ignored by Git) with metadata; retraining policy in `configs/ml.yml`.

**Done when:** a table compares rules only with rules + ML on all 11 faults, including false alarms.

### P8 — Investigation engine (no LLM)
`evidence_pack.py`, `explainer.py`. Evidence: what changed, since when, where, impact (lineage),
similar incidents, sample rows. `TemplateExplainer` maps evidence patterns to root causes and
playbook actions (stored as `pending_approval`). Explanation accuracy measured against the fault
manifest.
**Done when:** every incident has a report and accuracy on injected faults is reported.

### P9 — Second dataset (VED)
Unpack with `py7zr`, ingest into Bronze, `ved_signals` rules, lineage entries.
**Done when:** VED is monitored with YAML config only, no code changes.

### P10 — Dashboard and alerts
Streamlit, five pages: Overview, Dataset Health, Incidents (approve/dismiss), Quarantine Explorer
(including ML flags), Lineage. DuckDB opened read-only, except the decisions taken on the
Incidents page. `notifier.py` writes alerts to a local outbox.
**Done when:** a broken file triggers an alert and appears on the dashboard.

### P11 — Orchestration, tests and demo
`run_pipeline.py` (`--month`, `--all`, `--inject-fault`, `--strength`), `Makefile` / run scripts,
README with architecture, results and screenshots.
**Done when:** a newcomer can clone, run one command and see the dashboard with incidents.

## End of Version 1 deliverables

1. **One-command pipeline:** Bronze → rules → Silver/Quarantine → dataset checks → ML scoring →
   incidents → Gold → investigation → alerts.
2. **Data outputs:** `data/bronze/`, `data/silver/`, `data/quarantine/`, `data/dataguard.duckdb`,
   `models/`.
3. **Five-page dashboard** with incident approval.
4. **Alerts** for new Critical and High incidents (local outbox, on the dashboard).
5. **Measured results:** detection rate per fault, smallest detectable fault, false alarms on clean
   months, ML contribution, explanation accuracy, time to root cause — misses included.
6. **Quality:** tests for every stage, CI, six notebooks, README and docs.

## Version 2 outlook

- **V2a — LLM explainer:** `LLMExplainer` behind the same interface; evidence pack only, quoted
  numbers, template fallback; accuracy compared with the template explainer.
- **V2b — AI agent:** investigates by querying DuckDB, lineage and incident history; proposes actions
  that a human approves; every step logged.
- **V2c — Platform:** learned thresholds, suggested rules from profiles, approved re-ingestion,
  FastAPI with access control, checks before data lands in Bronze, scheduler (Prefect or Dagster).
