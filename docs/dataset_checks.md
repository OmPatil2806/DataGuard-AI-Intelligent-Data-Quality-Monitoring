# Dataset-level checks

Phase 3. Row rules (Phase 2) judge single rows; these checks judge a **whole month** against
the standard schema and against the months before it. Failed checks are findings about the
data; Phase 4 turns them into incidents with a lifecycle.

## Commands

| Command | What it does |
|---|---|
| `python src/quality/metrics.py` | Computes metrics per month (once per file version) and shows them |
| `python src/quality/run_checks.py` | Runs all five checks on every month in Bronze in one tracked run |
| `notebooks/03_baselines_and_drift.ipynb` | Charts of the latest run and a calibration verdict |

## How a month is judged

1. **Metrics** (`dataset_metrics`): rows, rows per day, raw schema as delivered, NULL share per
   column, row counts per fixed bin, mean, median and p99.
2. **Baseline** (`baselines`, recorded per run): built from the earlier months, ordered by
   month (not by load order), at most 12 months back and at most the six most recent.
   - **Trusted** from three earlier months: rows per day and NULL shares are medians, a
     distribution is the mean share per bin.
   - **Cold start** before that: fixed limits from `configs/quality_rules.yml` (`cold_start`).
     The previous month's schema is used even during a cold start.
3. **Checks** (`dataset_check_results`): one row per month, check and column with status
   `pass`, `fail` (with severity) or `skipped`, the observed and expected values, a detail and
   the baseline used.

For the real data: 2019-01 and 2023-01 have no earlier month within 12 months, 2023-02 and
2023-03 are in cold start, and 2023-04 to 2023-12 have a trusted baseline.

## The checks

| Check | Compares | Fails when | Severity |
|---|---|---|---|
| **schema** | Raw schema with the standard schema and the previous month | Required column missing / renamed by case / type changed or extra column removed / standard column back or new extra column | critical / high / medium / low |
| **volume** | Rows per day with the baseline median | Off by more than ±30% (medium), more than ±60% (high); cold start: outside 50,000–200,000 rows/day (medium) | medium, high |
| **freshness** | Delivery time with `expected_at` (month end + 60 days) | Late beyond 3 days of grace: up to 7 days (low), up to 30 (medium), more (high) | low to high |
| **null_rate** | NULL share per column with the baseline median | Increase above 2 points, or 0.1 points for normally complete columns (baseline below 0.1%): up to 2× the limit low, up to 5× medium, more high; cold start: above 10% (medium) | low to high |
| **drift** | Share of rows per fixed bin with the baseline distribution (PSI) | PSI ≥ 0.10 (medium), ≥ 0.20 (high); skipped during a cold start | medium, high |

All limits are in `configs/quality_rules.yml` (`defaults`, `cold_start`) and are validated when
loaded.

### Freshness replay mode

The historical files were downloaded long after they were published, so in `replay` mode (the
default) a file counts as delivered at `expected_at` unless `file_manifest.simulated_delivered_at`
is set; Phase 6 sets it to inject the stale-file fault. `live` mode uses the real arrival time
(`delivered_at`); with the historical files every month would then be years late.

### Fixed drift bins

`trip_distance`, `fare_amount` and `total_amount` are binned with fixed edges
(`defaults.drift.bins`), plus one open bin below the first and one at or above the last edge, so
extreme values (a 345,729-mile trip) count as "above the last edge" instead of stretching the
scale.

## Calibration

| Item | Result |
|---|---|
| Volume (real 2023 rows per day) | Against each baseline: Apr +5.3%, May +6.1%, Jun +0.6%, Jul −14.5%, Aug −17.0%, Sep −13.5%, Oct +11.1%, Nov +8.5%, Dec +6.2%. Largest 13 points inside the limit |
| 2019-01 volume | 248,278 rows/day, above the 200,000 cold-start limit: medium |
| NULL rate (O2, decided) | Absolute limit. Only September exceeds it (+2.01 points over the median of March–August): low |
| Drift sensitivity (O3, measured) | Distances scaled ×1.15: PSI 0.023 · ×1.3: 0.087 · ×1.4: 0.136 (medium) · ×1.5: 0.20 (high) · ×1.609 miles→km: 0.28 (high). Shifts below about +32% are not reported |
| Schema | January → February 2023: `airport_fee` renamed `Airport_fee` and five type changes, one high finding |

**Still to confirm on the full data** (`notebooks/03_baselines_and_drift.ipynb`, section 6): the
real PSI values per month (expected below 0.10, including the September fare step) and that the
only findings are the 2023-02 schema change, the 2019-01 volume and the September NULL increase.

## Known limits

- Shifts in a distribution smaller than about a third are not reported as drift; Phase 6
  measures the smallest fault each check detects.
- The first months of a dataset have no baseline; volume and NULL rate fall back on fixed limits
  and drift is skipped.
- Metrics are computed per file version; a corrected re-export gets its own metrics and becomes
  the month's current version.
