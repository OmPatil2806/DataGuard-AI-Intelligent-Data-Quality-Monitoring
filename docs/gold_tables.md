# Gold tables and lineage

Phase 5. Gold is the summary layer: small tables in the DuckDB file that answer the dashboard's
questions directly (how healthy is each month, what changed, what is broken, where did a number
come from). The dashboard (Phase 10) reads Gold only.

## Commands

| Command | What it does |
|---|---|
| `python src/gold/rebuild.py` | Everything after Bronze in one step: Silver, checks, rules, anomaly scoring (once a model is trained), incidents, score, Gold |
| `python src/gold/monitoring_tables.py` | Rebuilds the Gold tables only |
| `python src/gold/monitoring_tables.py --list` | The Gold tables and what they hold |
| `python src/gold/lineage.py --period 2023-01 --row 42` | Traces one row from its source file to Silver or Quarantine |

## How Gold is built

- **Always in full**, from the control tables (`file_manifest`, `silver_log`, `rule_run_log`,
  `dataset_metrics`, `baselines`, `dataset_check_results`, `incidents`, `incident_events`,
  `quality_scores`). Gold is never updated piece by piece, so it cannot drift away from them.
- **In one transaction.** A reader sees the previous build or the new one, never half of each.
  A failed build is rolled back and the previous Gold stays.
- **In order.** Later tables read earlier ones: every table is built on `gold_months`, the
  current Bronze version of each month.
- **For the current version only.** A corrected re-export of a month replaces the old version
  everywhere in Gold; the old version stays in Bronze and the control tables.
- **Logged.** Every build is a tracked run (`pipeline_runs`, dataset `gold`) with one row per
  table in `gold_build_log`: row count and checksum.

### Checksums and reproducibility

The checksum of a table is an MD5 over its rows as text, sorted, without its **volatile**
columns: run ids, timestamps and file paths. Those change with every run or with the folder the
data is in; the content (counts, scores, findings, SHA-256 of each file) does not.

`tests/test_rebuild.py` proves Phase 5's "done when": the same source files, loaded into a new
folder and rebuilt with `rebuild.py`, give exactly the same checksums as running every command
by hand; rebuilding again, or with `--force`, changes nothing.

## The tables

Periods are `YYYY-MM`. Shares are fractions (0.055 = 5.5%), except `quarantine_pct`.

### `gold_months`: one row per dataset and month

| Column | Meaning |
|---|---|
| `dataset`, `period`, `sha256` | The month and its current Bronze file version |
| `file_name`, `source_url`, `size_bytes` | The source file |
| `bronze_rows` | Rows in the file (manifest) |
| `loaded_run_id`, `registered_at` | volatile |

### `gold_dataset_health`: one row per dataset and month

| Column | Meaning |
|---|---|
| `bronze_rows`, `silver_rows`, `quarantine_rows`, `warned_rows` | Where the rows went; Silver columns are empty until processed |
| `quarantine_pct` | Quarantine rows in % of Bronze rows |
| `score` | Quality score 0–100 (Phase 4) |
| `ml_rows_flagged`, `ml_flag_rate`, `ml_model_id` | Rows the anomaly model flagged and their share, and the model (Phase 7; empty until a model is trained) |
| `checks_run`, `checks_failed`, `failed_critical` … `failed_low` | Latest dataset check run of this version |
| `incidents_found`, `incidents_opened`, `incidents_active` | Incidents with a finding this month; opened this month; of those, still active now |
| `worst_severity` | Worst severity among the failed checks and incidents of the month |
| `health` | `critical`: score < 80 or a critical finding · `warning`: score < 95 or a high or medium finding · `healthy` · `not_scored`: no Silver yet (limits in `configs/severity_weights.yml`, `health`) |
| `silver_run_id`, `check_run_id` | volatile |

### `gold_trends`: one row per dataset, month, metric and item

| `metric` | `item` | `unit` | Baseline |
|---|---|---|---|
| `rows_per_day` | | rows/day | yes |
| `quarantine_rate`, `warned_rate` | | share | |
| `score` | | score | |
| `ml_flag_rate` | | share | (the incident compares it with the earlier months) |
| `null_rate` | column | share | yes |
| `psi` | binned column | psi | (PSI is itself the comparison) |
| `rule_failure_rate` | rule id | share | |

Other columns: `value`; `baseline` and `baseline_trusted` from the check run; `change` =
value − baseline, only for a trusted baseline; `previous` = the value of the month before in the
table; `check_status` and `severity` of the check behind the value. Rule failure rates have no
baseline column: the incident engine computes their baselines itself.

### `gold_schema_history`: one row per dataset, month and column

Every column delivered that month, plus missing standard columns and removed extra columns.
Uses the schema check's own rules and previous month, so the two never disagree.

| Column | Meaning |
|---|---|
| `column_key` | Lower-case name: the same column across months |
| `column_name`, `position`, `dtype` | As delivered (`position` and `dtype` empty when absent) |
| `in_standard` | Part of the standard schema in `configs/datasets.yml` |
| `previous_period`, `previous_name`, `previous_dtype` | The month it is compared with (at most 12 months back) |
| `change` | `unchanged` or the check's kinds, joined: `missing` (critical), `renamed` (high), `type_changed` (medium), `removed` (medium), `unexpected` (low), `added` (low) |
| `severity`, `detail` | Worst severity and the check's wording |

### `gold_column_profile`: one row per dataset, month and column

| Column | Meaning |
|---|---|
| `null_rate`, `null_rate_baseline`, `null_rate_status`, `null_rate_severity` | NULL share and its check |
| `mean`, `median`, `p99` | Binned columns only (`trip_distance`, `fare_amount`, `total_amount`) |
| `histogram` | Rows per fixed drift bin, in bin order (`defaults.drift.bins` in `configs/quality_rules.yml`) |
| `psi`, `drift_status`, `drift_severity` | Drift against the baseline; empty during the cold start |

### `gold_incident_summary`: one row per incident

| Column | Meaning |
|---|---|
| `incident_id`, `source`, `rule_id`, `column_name`, `title`, `severity`, `status` | The incident |
| `first_seen_period`, `last_seen_period` | First and last month with a finding |
| `resolved_period`, `resolved_by` | `lifecycle` (two clean months) or `hand` (`incidents_cli.py resolve`); a hand resolution counts as resolved in the month of the incident's last lifecycle event |
| `months_seen` | Months with a finding |
| `months_active` | First seen to resolved, or to the latest month while active |
| `months_to_resolve` | Months from first seen to resolved |
| `recurrence_of`, `chain_root`, `chain_position` | A returning problem: the incident it repeats, the first of the chain, and which time this is |
| `investigated`, `notes`, `events` | Manual work and the length of the history |
| `latest_observed`, `latest_expected`, `latest_detail` | The latest finding |

### `gold_incident_monthly`: one row per dataset and month

`opened`, `recurrences` (opened as a recurrence), `seen` (with a finding), `resolved`, and
`active_end` with `active_critical` … `active_low`: incidents active at the end of the month,
by the severity they had reached by then.

### `gold_lineage`: one row per dataset and month

| Column | Meaning |
|---|---|
| `source_url`, `file_name`, `sha256`, `size_bytes`, `versions_registered` | The source file and how many versions of the month were registered |
| `bronze_path`, `silver_path`, `quarantine_path` | Where the files are (volatile) |
| `manifest_rows` | Rows recorded when the file was registered |
| `bronze_rows`, `silver_rows`, `quarantine_rows` | Rows **counted in the files** at build time |
| `logged_silver_rows`, `logged_quarantine_rows` | Rows `silver_log` recorded |
| `status`, `detail` | `ok` (all counts agree) · `not_processed` · `missing_file` · `mismatch` (detail says which count) |
| `loaded_run_id`, `silver_run_id`, `rule_run_id`, `check_run_id` | The runs (volatile) |
| `scored` | The month has a quality score |

## Row lineage

Every Bronze, Silver and Quarantine row keeps `bronze_row_id` (its 1-based position in the
source file), `source_file`, `ingestion_timestamp` and `pipeline_run_id`. `lineage.py` uses them
to follow one row of the current version of a month:

```
row 5 of yellow_taxi 2023-02
  source     https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet
  file       yellow_tripdata_2023-02.parquet (sha256 4df999bc427c)
  Bronze     data/bronze/yellow_taxi/period=2023-02/version=4df999bc427c/part-0.parquet (written by run_0001)
  Quarantine data/quarantine/yellow_taxi/period=2023-02/version=4df999bc427c/part-0.parquet (written by run_0002)
    - pickup 2023-01-18 21:05:34 outside the file's month
  checks     run_0003
  score      69.1 for the month
  values     tpep_pickup_datetime=2023-01-18 21:05:34, ...
```

The trace reads the control tables and the files, not Gold, so it is never stale.

## Known limits

- Gold covers the current version of each month; older versions are only in Bronze and the
  control tables.
- `gold_lineage` counts rows with Parquet metadata, which is fast, but on the full data it
  still opens every file at each build.
- Rule failure rates in `gold_trends` have no baseline column (see above).
