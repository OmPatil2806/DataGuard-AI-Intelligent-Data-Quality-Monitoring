# Vehicle Energy Dataset in DataGuard

Phase 9. The second dataset: GPS and OBD-II signals of 384 cars in Ann Arbor, November 2017
to November 2018, from [gsoh/VED](https://github.com/gsoh/VED) (Apache License 2.0). It is as
unlike the taxis as a dataset can be: weekly CSVs in 7z archives instead of monthly Parquet,
column names with units, no timestamp column, signals that exist for some engine types only,
and no delivery schedule.

**Done when** VED is monitored with YAML configuration only, no code changes: Step 9.5 changed
no file in `src/` (`tests/test_ved_pipeline.py` also checks that no monitoring code names VED).

| Document | Holds |
|---|---|
| [ved_profile.md](ved_profile.md) | The profile of every row (findings V1 to V20), ingestion, standardization and the monitoring results on the real data |
| [adding_a_dataset.md](adding_a_dataset.md) | What configuration a dataset needs |
| `notebooks/08_ved_profiling.ipynb` | The profile with charts |
| `notebooks/09_ved_monitoring.ipynb` | Health per month, rules, incidents, the fuel-trim finding, the fault experiments |

## The steps

| Step | What it took | Code or configuration |
|---|---|---|
| 9.1 Profile | `fetch_archives.py` (download, check, unpack, Excel to CSV) and the profile of 22.4 million rows. Found that DuckDB's CSV reader silently skips 53.5% of these files' rows (V1) | code: a new source type |
| 9.2 Ingest | `source` blocks: weekly CSVs read with pyarrow, every file's rows checked against its lines, regrouped into one file per month with each row's source line | code: general |
| 9.3 Standardize | column `source` names and computed `sql` columns (VED's row time) | code: general |
| 9.4 Configure | per-dataset drift bins, investigation dimensions and sample columns, multi-file lookups, ML for its own dataset only; proved with a made-up dataset defined only in YAML | code: general |
| 9.5 Monitor | VED's rules, lookup, cold-start limits, drift bins, dimensions | **configuration only** |
| 9.6 Prove | fault experiments on VED (`configs/faults_ved.yml`), notebook 09, this page | configuration, and one explainer fix |

## Commands

```
python src/data_download/fetch_archives.py --dataset ved_signals
python src/ingestion/ingest_data.py --dataset ved_signals
python src/gold/rebuild.py --dataset ved_signals --lookups data/incoming
python src/evaluation/run_experiments.py --config configs/faults_ved.yml
python src/evaluation/evaluate.py --config configs/faults_ved.yml
python src/evaluation/explanation_accuracy.py --config configs/faults_ved.yml
```

## On the real data

Ingest about 2 minutes, the rebuild of 12 months (22,000,556 rows) 84 seconds. Only the 2,702
readings with a negative trip time leave Silver; schema and volume pass every month; 9
incidents, of which the main one is a real change: **from April 2018 between 25 and 52 petrol
cars a month send no fuel trims at all**, against 1 to 4 a month before (notebook 09,
section 4). Details: [ved_profile.md](ved_profile.md#monitoring-step-95-configuration-only).

## Fault experiments

`configs/faults_ved.yml`: the faults of `configs/faults.yml` that do not need the taxis'
columns, on VED's columns, plus one car that stops sending its battery charge. Target month
2018-02 after a history of 2017-11 to 2018-01, a 10% sample (about 150,000 readings a month).
All 25 experiments ran in 2.5 minutes.

**18 caught, 1 flagged, 5 missed; the unbroken control month raises nothing. The right cause
came first in all 18 detected experiments (top-1 100%); no fault spread at random got a
where/when (0 of 8), and the broken car was named (vehicle 560).**

| Experiment | Detector | Outcome | Seen | Best cause | Where/when |
|---|---|---|---|---|---|
| `missing_column` (battery charge) | schema | 🟢 caught, critical | 21 columns, 22 expected | `column_missing` | |
| `null_spike-1pct` (RPM) | NULL rate | 🔴 missed | 2.92% (limit 1.40% + 2 points) | | |
| `null_spike-5pct`, `-20pct` | NULL rate | 🟢 caught, medium / high | 6.85%, 21.56% | `source_nulls` | none, as it should |
| `null_spike_vendor-1pct` (car 560's battery) | rule `battery_signals_present_ev` | 🟡 flagged | 52 rows, below the rule's incident limit | | |
| `null_spike_vendor-5pct`, `-20pct` | rule `battery_signals_present_ev` | 🟢 caught, low | every broken row | `rule_failures_up` | **vehicle = 560** |
| `duplicate_rows-1pct`, `-5pct`, `-20pct` | rule `reading_unique` | 🟢 caught, medium | every copy | `duplicate_delivery` | none |
| `out_of_range-1pct`, `-5pct`, `-20pct` (negative speed) | rule `speed_in_range` | 🟢 caught, high | every broken row | `out_of_range_values` | none |
| `orphan_location-1pct`, `-5pct`, `-20pct` (unknown car) | rule `vehicle_known` | 🟢 caught, high | every broken row | `unknown_location` | vehicle = 9999 (the injected id) |
| `unit_change-1pct`, `-5pct`, `-20pct` (mph) | drift | 🔴 missed | PSI 0.004, 0.006, 0.036 | | |
| `unit_change-100pct` | drift | 🟢 caught, high | PSI 1.181 | `unit_conversion` | |
| `volume_drop-1pct` | volume | 🔴 missed | −28.8% (limit ±30%) | | |
| `volume_drop-5pct`, `-20pct`, `-50pct` | volume | 🟢 caught, medium / high | −31.6%, −42.4%, −64.0% | `rows_missing_evenly` | |

### Why each miss is missed

| Misses | Why |
|---|---|
| `null_spike-1pct` | RPM is 1.4% empty normally (electric cars); 1% more stays inside the 2-point NULL-rate limit, as for the taxis |
| `null_spike_vendor-1pct` | 52 readings of one car: flagged and kept in Silver with a warning, but below the rule's incident limit (0.1 point above its usual rate). Not a NULL-rate check at all: battery charge is 83% empty by nature (petrol and hybrid cars), so one car could never move it; the rule per engine type is what sees it |
| `unit_change` 1%, 5%, 20% | Speeds in mph look like slower driving; a share of rows moves little of the distribution (PSI 0.036 at 20%) |
| `volume_drop-1pct` | February 2018 is already −28.8% against November to January (fewer cars driving, profile V20); 1% stays inside ±30%. **The 5% drop is caught for the same reason**: it adds 2.8 points to a month that was 1.2 points from the limit, so it is a weak result, not a sensitive check |

### What the experiments changed

- **The explainer claimed missing days that were not.** Every volume drop was first explained
  as "rows missing on 2 days": 19 and 20 February 2018 are low in the unbroken month too (19
  February 2018 was Presidents' Day). A low day now counts only when the low days account for
  most of the month's missing rows (`low_days_share_of_drop` in `explainer.py`, more than half
  in `configs/playbooks.yml`). Top-1 rose from 15 to 18 of 18; the taxis' results did not
  change.
- **Playbooks for any dataset**: a general `unit_conversion` cause (the median moved by a
  factor between two common units: 0.62, 1.6, 0.28, 3.6), the uniqueness playbook for VED's
  `reading_unique`, and the referential cause worded without the taxis' zones.

## Taxis and VED side by side

| | Taxis (test months) | VED (real data, 10% sample) |
|---|---|---|
| Experiments | 36 | 24 |
| Caught / flagged / missed | 24 / 5 / 7 | 18 / 1 / 5 |
| Control month raises | nothing | nothing |
| Right cause first | 22 of 24 (92%) | 18 of 18 (100%) |
| False where/when | 0 of 12 | 0 of 8 |
| Concentrated fault placed | vendor 2 (at 20%) | vehicle 560 (at 5% and 20%) |

## Known limits

- **Fleet-driven NULL shares.** VED's NULL shares depend on which cars drive in a month; the
  NULL-rate check compares the whole month with its baseline, so a change in one engine type
  can be diluted by the others, and a change in the fleet can look like one. A NULL-rate check
  per engine type would see both better (a candidate for later).
- **Where/when across many vehicles.** A problem spread over dozens of cars has no single
  segment that holds most of it: the answer stays "no part of the data standing out".
- **No anomaly model for VED.** `configs/ml.yml` trains one model, for the taxis.
- **The last month is partial** (2018-11, 10 days) and left out of the periods.
