# Detection evaluation

Phase 6. Until now the pipeline's checks were calibrated, not tested against problems with a
known answer. Here every check category gets a fault: a clean month is broken in one known
way, at several strengths, the whole pipeline runs on it, and the findings are compared with
what was broken (the ground truth) and with the same month unbroken (the control).

**Done when** a table shows what was caught and missed, and every miss is explained: see
[Results](#results-on-the-test-data) and [Why each miss is missed](#why-each-miss-is-missed).

## Commands

| Command | What it does |
|---|---|
| `python src/evaluation/fault_injector.py --list` | The planned experiments (31) |
| `python src/evaluation/run_experiments.py` | Builds the clean history once, then runs every experiment |
| `python src/evaluation/run_experiments.py --only control,null_spike-5pct` | Some experiments only |
| `python src/evaluation/evaluate.py` | Scores the experiments: caught, flagged or missed |
| `notebooks/05_detection_evaluation.ipynb` | Heatmap, rows flagged, what each check saw, score impact, verdict |

## How an experiment works

1. **Config** (`configs/faults.yml`): target month 2023-06, clean history 2023-01 to 2023-05
   (so June has a trusted baseline), a 10% sample of every month (`sample_fraction`, the same
   rows in every experiment) and seed 42.
2. **History, once.** The sampled history months go through the whole pipeline in
   `data/evaluation/history/`. It is reused while the config and the source files are
   unchanged.
3. **Each experiment** gets its own copy of the history database in
   `data/evaluation/experiments/<id>/`. The June file is broken by one fault
   (`src/evaluation/faults.py`), loaded, and the pipeline runs on it:
   `rebuild.py --months 2023-06`. The control experiment runs June unbroken.
4. **Ground truth.** Every broken file comes with `fault_manifest.json`: the rows changed
   (numbered like `bronze_row_id`), the rows removed, and the detectors that should fire. Tests
   prove that each fault changes exactly those rows and only its own columns.
5. **Evaluation** (`evaluate.py`), on June only:

| Outcome | Meaning |
|---|---|
| 🟢 **caught** | An incident was raised: the fault reaches someone |
| 🟡 **flagged** | The broken rows were flagged (quarantined or warned), but no incident was raised: the bad rows stay out of Silver, nobody is alerted |
| 🔴 **missed** | Nothing |
| ⚪ **error** | The experiment could not run |

A check failing, or an incident seen, in the control month too is never credited.

## The ten faults

| Fault | Category | What it does | Strengths | Expected detector |
|---|---|---|---|---|
| `missing_column` | schema | Drops `passenger_count` | – | schema check |
| `null_spike` | completeness | Empties `fare_amount` | 1%, 5%, 20% | NULL-rate check on `fare_amount` |
| `duplicate_rows` | uniqueness | Delivers rows twice, each copy after its original | 1%, 5%, 20% | rule `trip_unique` |
| `out_of_range` | validity | Makes positive fares negative | 1%, 5%, 20% | rule `fare_not_negative` |
| `orphan_location` | referential | `PULocationID` = 999 | 1%, 5%, 20% | rule `pickup_zone_known` |
| `unit_change` | drift | `trip_distance` × 1.609 (miles → km) | 1%, 5%, 20%, 100% | drift check on `trip_distance` |
| `volume_drop` | volume | Removes rows | 1%, 5%, 20%, 50% | volume check |
| `stale_file` | freshness | File delivered late (data unchanged) | 1, 5, 20 days | freshness check |
| `pickup_outside_month` | pattern | Moves trips into the previous month (duration kept) | 1%, 5%, 20% | rule `pickup_in_file_month` |
| `swapped_times` | consistency | Swaps pickup and dropoff time | 1%, 5%, 20% | rule `dropoff_not_before_pickup` |

A share is a share of the rows the fault can really change (a NULL that is already NULL is not
a fault). The extra strengths (volume 50%, unit change 100%) sit above the check's limit, so
every check is also shown catching its fault.

## Results on the test data

All 31 experiments on the test months: the fixture's 200 trips moved into each month of
2023-01 to 2023-06, every row used (`sample_fraction` 1.0). **20 caught, 4 flagged, 6 missed,
0 errors.** The control month raises nothing: no false alarms. Rule faults: 100% of the
broken rows flagged at every strength, 0 good rows flagged.

| Experiment | Detector | Outcome | Severity | What the detector saw | Score drop |
|---|---|---|---|---|---|
| `missing_column` | check schema | 🟢 caught | critical | 18 columns (limit: 19 columns as in 2023-05) | 24.6 |
| `null_spike-1pct` | check null_rate (fare_amount) | 🟢 caught | high | 1.00% (limit: 0.00% + at most 0.1 points) | 7.6 |
| `null_spike-5pct` | check null_rate (fare_amount) | 🟢 caught | high | 5.00% (limit: 0.00% + at most 0.1 points) | 7.5 |
| `null_spike-20pct` | check null_rate (fare_amount) | 🟢 caught | high | 20.00% (limit: 0.00% + at most 0.1 points) | 7.5 |
| `duplicate_rows-1pct` | rule trip_unique | 🟡 flagged | – | 2/2 rows (100%), 0 extra | 0.1 |
| `duplicate_rows-5pct` | rule trip_unique | 🟢 caught | medium | 10/10 rows (100%), 0 extra | 0.4 |
| `duplicate_rows-20pct` | rule trip_unique | 🟢 caught | medium | 40/40 rows (100%), 0 extra | 1.2 |
| `out_of_range-1pct` | rule fare_not_negative | 🟡 flagged | – | 2/2 rows (100%), 0 extra | 0.1 |
| `out_of_range-5pct` | rule fare_not_negative | 🟢 caught | high | 10/10 rows (100%), 0 extra | 0.5 |
| `out_of_range-20pct` | rule fare_not_negative | 🟢 caught | high | 39/39 rows (100%), 0 extra | 9.7 |
| `orphan_location-1pct` | rule pickup_zone_known | 🟡 flagged | – | 2/2 rows (100%), 0 extra | 0.1 |
| `orphan_location-5pct` | rule pickup_zone_known | 🟢 caught | high | 10/10 rows (100%), 0 extra | 0.3 |
| `orphan_location-20pct` | rule pickup_zone_known | 🟢 caught | high | 40/40 rows (100%), 0 extra | 1.3 |
| `unit_change-1pct` | check drift (trip_distance) | 🔴 missed | – | PSI 0.002 (limit: PSI < 0.1) | 0.0 |
| `unit_change-5pct` | check drift (trip_distance) | 🔴 missed | – | PSI 0.024 (limit: PSI < 0.1) | 0.0 |
| `unit_change-20pct` | check drift (trip_distance) | 🟢 caught | high | PSI 0.287 (limit: PSI < 0.1) | 7.8 |
| `unit_change-100pct` | check drift (trip_distance) | 🟢 caught | high | PSI 1.741 (limit: PSI < 0.1) | 7.8 |
| `volume_drop-1pct` | check volume | 🔴 missed | – | 7 rows/day (limit: 6 rows/day ± 30%) | -0.1 |
| `volume_drop-5pct` | check volume | 🔴 missed | – | 6 rows/day (limit: 6 rows/day ± 30%) | -0.1 |
| `volume_drop-20pct` | check volume | 🔴 missed | – | 5 rows/day (limit: 6 rows/day ± 30%) | -0.2 |
| `volume_drop-50pct` | check volume | 🟢 caught | medium | 3 rows/day (limit: 6 rows/day ± 30%) | 6.5 |
| `stale_file-1d` | check freshness | 🔴 missed | – | delivered 2023-08-31 (limit: by 2023-08-30 + 3 day(s) grace) | 0.0 |
| `stale_file-5d` | check freshness | 🟢 caught | low | delivered 2023-09-04 (limit: by 2023-08-30 + 3 day(s) grace) | 2.6 |
| `stale_file-20d` | check freshness | 🟢 caught | medium | delivered 2023-09-19 (limit: by 2023-08-30 + 3 day(s) grace) | 5.2 |
| `pickup_outside_month-1pct` | rule pickup_in_file_month | 🟡 flagged | – | 2/2 rows (100%), 0 extra | 0.1 |
| `pickup_outside_month-5pct` | rule pickup_in_file_month | 🟢 caught | high | 10/10 rows (100%), 0 extra | 0.3 |
| `pickup_outside_month-20pct` | rule pickup_in_file_month | 🟢 caught | high | 40/40 rows (100%), 0 extra | 1.3 |
| `swapped_times-1pct` | rule dropoff_not_before_pickup | 🟢 caught | high | 2/2 rows (100%), 0 extra | 0.0 |
| `swapped_times-5pct` | rule dropoff_not_before_pickup | 🟢 caught | high | 10/10 rows (100%), 0 extra | 0.3 |
| `swapped_times-20pct` | rule dropoff_not_before_pickup | 🟢 caught | high | 39/39 rows (100%), 0 extra | 1.3 |

Side findings: `missing_column` also fails the NULL-rate check on `passenger_count` (the column
is filled with NULLs when the file is standardized); `out_of_range-20pct` also moves the
`fare_amount` distribution (drift, high). The volume misses rise the score slightly: fewer
rows, the same share of them bad.

## Why each miss is missed

Every miss stays inside a limit set on purpose; the evaluation measures where that limit is.

| Misses | Limit | Why the limit is there | What would catch it |
|---|---|---|---|
| `volume_drop` 1%, 5%, 20% | Rows per day within ±30% of the median of the earlier months (`defaults.volume.tolerance`) | Real 2023 volumes move by up to −17% (July, August) and +11% (October) against their baseline without anything being wrong; a tighter limit would alarm in normal months. Missing rows cannot be caught row by row | A drop of more than 30% (`volume_drop-50pct` is caught). A tighter limit only with a seasonal baseline (same month last year) |
| `stale_file` 1 day | Late beyond 3 days of grace (`defaults.freshness.grace_days`) | A file one day late is normal operations, not an incident | 4 days or more (5 and 20 days are caught, low and medium) |
| `unit_change` 1%, 5% | PSI ≥ 0.10 over fixed distance bins (`defaults.drift`) | Kilometres look like longer trips; when only a few rows change, few rows move to another bin, so the distribution barely moves (PSI 0.002 and 0.024). The rules see nothing either: the distances stay plausible. Measured in Phase 3 (open item O3): shifts of all rows below about +32% stay below 0.10 | A share large enough to move the distribution (20% is caught at PSI 0.29). Row by row: fare per mile drops for a km row, a feature of the Phase 7 anomaly model |

### Why the 1% rule faults are flagged, not caught

All broken rows are flagged and quarantined, so no bad row reaches Silver. Whether an incident
is raised depends on the rule's normal failure rate: the severity engine raises one when the
rate rises by more than `max(0.1 points, 1.0 × the rule's own baseline)`
(`defaults.incidents.row_rules`), so that steady, known problems do not become permanent
incidents.

In the test months each of these rules normally fails on 1% to 1.5% of rows, so a 1% fault
sits at the threshold: `swapped_times` just crosses it, the other four do not. On the real
data the baselines differ (`docs/data_profile.md`): negative fares 1.26%, but dropoff before
pickup 0.027%, pickups outside the month 0.007%, unknown locations and duplicates practically
0%. There the limit is 0.1 points for those four rules, so **a 1% fault is expected to be
caught for all of them except negative fares**, which stays flagged: a 1% rise on a 1.26%
baseline is within the rule's normal range by design. The real run confirms this.

## The score

The score summarises a month; it is not the alert. On the test months a critical schema finding
costs about 25 points, a high check finding 7.5 to 8, a medium one 5 to 6.5, and 1% bad rows
about 0.1.
Incidents, not the score, are what tell someone that a fault happened.

## On the full data (to confirm)

The test months have 200 trips each, so their rows per day and rates are coarse (6 rows/day).
To run on the real data, after `python src/data_download/download_data.py`:

```
python src/evaluation/run_experiments.py      # prints the time of every experiment
python src/evaluation/evaluate.py
```

Then open `notebooks/05_detection_evaluation.ipynb` and paste back the verdict in section 7.
Expected differences: the 1% rule faults caught except negative fares (see above); the
volume, freshness and drift misses the same, at the same limits.

## Known limits

- One target month (2023-06), one fault at a time: faults that mask each other are not tested.
- Faults hit random rows; real problems cluster (one vendor, one zone, one day).
- Detection is per month: a fault that starts mid-month is seen at the month's file.
- The evaluation measures the configured limits; changing a limit in `configs/quality_rules.yml`
  changes the result, which is the point: rerun the experiments after a change.
