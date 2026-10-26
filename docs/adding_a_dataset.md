# Adding a dataset

Phase 9.4. A new dataset needs configuration only, no code: `tests/test_new_dataset.py`
defines a made-up `air_quality` dataset in a temporary copy of `configs/` and runs it through
every stage (ingest, Silver, checks, rules, incidents, score, investigation, Gold). In its last
month one station stops sending PM2.5; the pipeline raises the NULL-rate incident and the
investigation places it in that station.

## What to configure

| File | Section | What | Needed? |
|---|---|---|---|
| `datasets.yml` | `datasets.<name>` | format, frequency, `file_template`, `url_template` (or `source` / `archives` for files that are not one per month), `periods`, `timestamp_column` | yes |
| | `columns` | the standard schema: `name: TYPE`, `{type, source: "Name in the file"}` (one name or a list), or `{type, sql: "..."}` for a computed column | yes |
| | `lookups` | reference files: `file`, or `files` stacked by column name, with `rename` where the files name a column differently | for referential rules and joins |
| | `silver` | `derived` columns and lookup `joins` added to Silver | optional |
| | `expected_lag_days` | when a month's file is due; without it the freshness check is skipped | optional |
| `quality_rules.yml` | `rules.<name>` | the row rules (completeness, validity, pattern, referential, consistency, uniqueness) | yes |
| | `cold_start.<name>` | rows per day and NULL-rate limits until three months form a baseline | recommended |
| | `defaults.drift.bins.<name>` | bin edges per numeric column for the drift check (and the mean, median and p99 metrics) | for drift |
| `investigation.yml` | `datasets.<name>` | `dimensions` for where/when, `sample_columns` for the evidence pack | optional: without dimensions no where/when; without sample columns the first ten standard columns |
| `ml.yml` | `dataset` | the one dataset the anomaly model is trained for | only that dataset |

The thresholds in `quality_rules.yml` (`defaults`), `severity_weights.yml`, `playbooks.yml`
and the `concentration` section of `investigation.yml` are shared by every dataset.

## Running it

```
python src/ingestion/ingest_data.py --dataset <name>
python src/gold/rebuild.py --dataset <name>
```

Gold covers every dataset in the database. The anomaly model's stage is skipped for every
dataset other than the one in `ml.yml`.

To try a configuration without touching `configs/`, point `DATAGUARD_CONFIG_DIR` at another
folder with the same files.

## Examples

| Dataset | Shows |
|---|---|
| `yellow_taxi` | One Parquet file per month from a URL, zone lookup joined into Silver, drift bins, the anomaly model |
| `ved_signals` | Weekly CSVs in 7z archives regrouped into months (`source`, `archives`), source names with units, a computed row time, no freshness ([ved_profile.md](ved_profile.md)) |
| `air_quality` (test only) | Source names, a computed column, a lookup from two files with a renamed key, its own drift bins and investigation dimensions |
