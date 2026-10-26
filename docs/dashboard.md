# Dashboard

Phase 10. A Streamlit app that shows the monitoring results of every dataset: the Gold tables,
plus the alerts and the pipeline runs.

```bash
streamlit run src/dashboard/app.py                                    # http://localhost:8501
DATAGUARD_DATABASE=other.duckdb streamlit run src/dashboard/app.py    # another database
```

PowerShell: `$env:DATAGUARD_DATABASE="other.duckdb"; streamlit run src/dashboard/app.py`.

## How it is built

| File | Holds |
|---|---|
| `src/dashboard/app.py` | The entry point: page setup, navigation, sidebar |
| `src/dashboard/data.py` | Every query, as plain functions (no Streamlit), so they are tested without a browser |
| `src/dashboard/ui.py` | What every page shares: the chosen dataset and month, cached reads, colors and labels |
| `src/dashboard/views/` | One file per page |

- **Read-only.** Each query opens the DuckDB file read-only and closes it straight after. The
  dashboard cannot change the database (`tests/test_dashboard.py` checks the file's SHA-256),
  and it holds DuckDB's file lock only while a query runs, so the pipeline can write between
  two redraws. If a query meets the pipeline's lock, the page says the database is busy.
- **Cached until the file changes.** Reads are cached per database file and its modification
  time: a redraw does not query again, and a rebuild shows on the next redraw. **Refresh** in
  the sidebar empties the cache.
- **Says what to run** when there is nothing to show: no database, Gold not built, no month.

## Pages

| Page | Step | Shows |
|---|---|---|
| Overview | 10.2 | The month at a glance, its trend, open incidents, alerts, runs |
| Dataset Health | 10.3 | coming |
| Incidents | 10.4 | coming |
| Quarantine Explorer | 10.5 | coming |
| Lineage | 10.5 | coming |

### Overview

| Part | What | From |
|---|---|---|
| Headline numbers | Quality score and its change against the month before, health, open incidents, rows checked, share quarantined | `gold_dataset_health`, `gold_incident_summary` |
| Open now | Active incidents by severity, failed checks of the month | `gold_incident_summary`, `gold_dataset_health` |
| Quality score per month | The score, each month's marker colored by its health, the warning and critical limits (`configs/severity_weights.yml`) | `gold_dataset_health` |
| Incidents opened and resolved | Per month, with the incidents active at month end on hover | `gold_incident_monthly` |
| Most severe open incidents | Up to five, by severity then newest, with the likely cause and the actions waiting for approval | `gold_incident_summary`, `gold_investigations` |
| Latest alerts | The dataset's latest alerts and whether they were written | `alert_log` |
| Latest pipeline runs | The last five runs of any dataset | `pipeline_runs` |

Health and severity are always shown as a word (with an icon in tables), never by color
alone. Status colors (healthy, warning, critical) are used only for status, never for a series.

## Tests

`tests/test_dashboard.py` builds the planted pipeline's database (five test months,
investigated, alerted, Gold built) and checks the queries against the Gold tables, the page
itself with Streamlit's `AppTest` (numbers, charts, tables, the month picker), that the file is
unchanged after the page was shown, and the messages when there is no database or no Gold.
