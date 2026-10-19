# Bronze layer

Phase 1. Bronze holds every source file exactly as delivered, plus four lineage columns.
It is never modified or deleted by the pipeline, so Silver, Quarantine and Gold can always be
rebuilt from it.

## Commands

| Command | What it does |
|---|---|
| `python src/data_download/download_data.py` | Downloads source files into `data/incoming/` |
| `python src/ingestion/ingest_data.py` | Registers, loads into Bronze and records one run |
| `python src/ingestion/verify_bronze.py` | Checks Bronze against the manifest and the source files |
| `python src/common/db.py` | Shows the control tables and the latest runs |
| `python src/ingestion/manifest.py` | Registers files without loading them |

## Layout

```
data/bronze/<dataset>/period=YYYY-MM/version=<sha256[:12]>/part-0.parquet
```

- **One folder per file version.** A corrected re-export of a month gets its own `version=`
  folder next to the old one; nothing is overwritten.
- **Current version per month:** the most recently registered `LOADED` version in
  `file_manifest` (`bronze_loader.current_files()`).
- **Writes are atomic:** the loader writes to `version=<sha>.tmp/`, checks the row count, then
  renames the folder. A crash leaves no partial Bronze file.

## Columns

All source columns keep their name, type, values and order. Bronze does **not** standardize:
`Airport_fee` (2023-02 onward) and `airport_fee` (2023-01) stay as delivered, as do the
BIGINT/INTEGER differences. Standardization is Phase 2.

| Added column | Type | Meaning |
|---|---|---|
| `bronze_row_id` | BIGINT | 1-based position of the row in the source file |
| `source_file` | VARCHAR | Source file name |
| `ingestion_timestamp` | TIMESTAMP | When the row was written to Bronze (UTC) |
| `pipeline_run_id` | VARCHAR | Run that wrote it (`pipeline_runs.run_id`) |

`(source_file, bronze_row_id)` identifies a row; `file_manifest` links the file to the SHA-256
of the exact content loaded.

## Reading Bronze with DuckDB

DuckDB treats `name=value` folders as Hive partitions and adds `period` and `version` columns
automatically when it reads these files.

```sql
-- one file exactly as stored (what the pipeline uses: bronze_loader.read_bronze_sql)
SELECT * FROM read_parquet('data/bronze/yellow_taxi/period=2023-01/version=.../part-0.parquet',
                           hive_partitioning = false);

-- all of Bronze with period and version as columns (schemas differ between months)
SELECT period, version, count(*)
FROM read_parquet('data/bronze/yellow_taxi/*/*/*.parquet', hive_partitioning = true,
                  union_by_name = true)
GROUP BY ALL ORDER BY period;
```

Reading all versions counts a re-exported month twice; use the current version per month for
analysis.

## Control tables (data/dataguard.duckdb)

| Table | One row per | Key columns |
|---|---|---|
| `pipeline_runs` | Run | `run_id`, `status` (RUNNING, SUCCESS, WARNING, FAILED), counters, `error_message` |
| `file_manifest` | File version | `(dataset, period, sha256)`, `row_count`, `expected_at`, `delivered_at`, `status`, `bronze_path`, `loaded_run_id` |
| `schema_version` | Schema change | `version` |

Times are stored in UTC.

## Run outcomes

| Situation | File outcome | Run status |
|---|---|---|
| New file or new version | loaded | SUCCESS |
| Version already in Bronze | skipped | SUCCESS |
| `--force` | rewritten in place | SUCCESS |
| File missing in `incoming/` | missing | WARNING |
| File cannot be read or loaded | failed (manifest `FAILED`), other files continue | FAILED |
| Process killed during a run | — | FAILED ("interrupted"), set by the next run |

## Verification

`verify_bronze.py` checks every loaded version without changing anything:

| Check | Passes when |
|---|---|
| rows | Bronze row count equals the source row count in the manifest |
| row_ids | `bronze_row_id` runs from 1 to the row count with no gaps or repeats |
| lineage | `source_file` matches the manifest, timestamps and run IDs are filled, one run per version |
| schema | Source columns are present with unchanged names and types, followed by the four lineage columns |
| checksum | The file in `incoming/` still has the registered SHA-256 |

`schema` and `checksum` are skipped (`-`) when the source file is no longer in `incoming/` or
has been replaced by a newer download.
