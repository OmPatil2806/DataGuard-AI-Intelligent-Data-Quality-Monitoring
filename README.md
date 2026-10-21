# DataGuard AI: Intelligent Data Quality Monitoring

[![CI](https://github.com/OmPatil2806/DataGuard-AI-Intelligent-Data-Quality-Monitoring/actions/workflows/ci.yml/badge.svg)](https://github.com/OmPatil2806/DataGuard-AI-Intelligent-Data-Quality-Monitoring/actions/workflows/ci.yml)

DataGuard AI finds bad data that pipelines let through, explains what broke and since when,
and recommends what to do about it. Raw files land in **Bronze**, a rule engine sends valid rows
to **Silver** and invalid rows to **Quarantine** with a reason, dataset-level checks (schema,
freshness, volume, drift) raise incidents, and **Gold** monitoring tables feed a Streamlit dashboard.
Version 1 runs on real NYC TLC Yellow Taxi data with deliberately injected faults, so detection
quality is measured instead of assumed.

**Status:** Phase 5 complete (Gold tables and lineage). Pending a run on the full data: the
total-amount rule (open item O1), the measured rule results, the drift calibration check
(notebook 03) and the score calibration (notebook 04). Next: Phase 6, fault injection and
detection evaluation.

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

## Incidents

```bash
python src/incidents/incidents_cli.py update                   # latest check run and rule run -> incidents
python src/incidents/incidents_cli.py list                     # active incidents (--all for resolved too)
python src/incidents/incidents_cli.py show INC-0001            # details and full history
python src/incidents/incidents_cli.py investigate INC-0001 --note "asked the source"
```

A failed dataset check opens an incident, and so does a row rule whose failure rate clearly rises
above its own history (steady, known problems do not); the same problem in later months updates it instead of opening
a new one; it is resolved after two clean months in a row. A problem that returns later opens a
new incident linked to the old one. Every change is kept in `incident_events`.

## Quality score

```bash
python src/quality/score.py                                    # score 0-100 per month with Silver
```

Each month gets `100 x sum(w x p) / sum(w)` over the categories in `configs/severity_weights.yml`.
For row categories, `p` is the share of rows that fail none of that category's rules. For dataset
checks, `p` comes from `check_scores` (pass 1.0, low 0.75, medium 0.5, high 0.25, critical 0).
Categories without a result are left out. Scores are kept in `quality_scores` and
`quality_score_parts`. `notebooks/04_quality_score_calibration.ipynb` shows the points lost per
category, how much the score depends on the weights, and the incident timeline. Details:
[docs/incidents_and_score.md](docs/incidents_and_score.md).

## Gold tables

```bash
python src/gold/rebuild.py                                    # Silver, checks, rules, incidents, score, Gold
python src/gold/monitoring_tables.py                          # rebuild every Gold table only
python src/gold/monitoring_tables.py --list                   # the Gold tables and what they hold
```

After `ingest_data.py`, `rebuild.py` runs every later stage in order, so one command brings
everything up to date. Running it again on the same Bronze gives the same Gold.

Gold holds the small summary tables the dashboard reads, inside the DuckDB file. It is always
rebuilt in full from the control tables, in one transaction: a failed build keeps the previous
Gold. Each build is logged in `gold_build_log` with a checksum per table that leaves out run ids
and timestamps, so two builds of the same data match. Every column is described in
[docs/gold_tables.md](docs/gold_tables.md).

| Table | One row per | Holds |
|---|---|---|
| `gold_months` | dataset and month | The current Bronze file version |
| `gold_dataset_health` | dataset and month | Rows in Silver and Quarantine, score, failed checks by severity, incidents, health label |
| `gold_trends` | dataset, month, metric and item | Rows per day, Quarantine and warned rate, score, NULL rate per column, PSI per column, failure rate per rule; with the baseline, the change and the previous month |
| `gold_schema_history` | dataset, month and column | Name, position and type as delivered, and the change to the previous month (same rules as the schema check) |
| `gold_column_profile` | dataset, month and column | NULL share, mean, median, p99, rows per drift bin and PSI, with the check results |
| `gold_incident_summary` | incident | Status, severity, months seen, active and until resolved, how it was resolved, recurrence chain, investigation and notes |
| `gold_incident_monthly` | dataset and month | Incidents opened, recurring, seen and resolved, and active at the end of the month by severity |
| `gold_lineage` | dataset and month | Source URL, file and SHA-256, Bronze, Silver and Quarantine files, the runs, and row counts read from the files: `ok`, `not_processed`, `missing_file` or `mismatch` |

The health label is `critical` below a score of 80 or with a critical finding, `warning` below 95
or with a high or medium finding, otherwise `healthy` (limits in `configs/severity_weights.yml`,
`health`). A month without Silver is `not_scored`.

Trace one row from its source file to Silver or Quarantine (`--row` is the row's 1-based
position in the source file, `bronze_row_id`):

```bash
python src/gold/lineage.py --period 2023-01 --row 42
```

## Fault injection

```bash
python src/evaluation/fault_injector.py --list                # the planned experiments
python src/evaluation/run_experiments.py                      # run them all
python src/evaluation/run_experiments.py --only control,null_spike-5pct
```

`run_experiments.py` reads the history and target months and the zone lookup from
`data/incoming/`. It builds the sampled history once (reused while the config and source files
are unchanged), then gives every experiment its own copy of the history database, adds the
broken month and runs `rebuild.py --months <target>`. Results: one folder per experiment under
`data/evaluation/experiments/` with the database, the manifest, `experiment.json` (stage exit
codes and times) and `log.txt`.

Phase 6 measures what the pipeline catches. Each experiment breaks a copy of one clean month
(`target_month` in `configs/faults.yml`) in one known way, at one strength, and records the
ground truth in `fault_manifest.json`: which rows were broken or removed and which detectors
should fire. A control experiment keeps the month unbroken. Experiments run in their own
folders under `data/evaluation/`, never in the real pipeline's database. The same seed always
gives the same sample and the same broken rows.

The ten faults (`src/evaluation/faults.py`), one per check category: missing column, NULL
spike, duplicate rows, out-of-range fares, orphan location IDs, distances in kilometres, volume
drop, late file, pickups moved into the previous month, and swapped pickup and dropoff times.

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
