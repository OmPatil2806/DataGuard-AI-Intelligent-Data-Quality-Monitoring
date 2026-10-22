# ML anomaly detection

Phase 7. Rules catch values that are wrong on their own: a negative fare, an unknown zone, a
trip that ends before it starts. They cannot see a trip whose values are each plausible but do
not fit together, such as $45 for a one-mile ride. An **Isolation Forest** (scikit-learn) learns
what normal Silver trips look like and scores every row by how unusual the combination of its
values is.

**Done when** a table compares rules only with rules + ML on all 11 faults, including false
alarms: see [Rules only against rules + ML](#rules-only-against-rules--ml).

## Commands

| Command | What it does |
|---|---|
| `python src/ml/features.py --period 2023-01` | The model's features for one month |
| `python src/ml/train.py` | Trains the model on the train months, sets the threshold on the calibration month |
| `python src/ml/score.py` | Scores every month and stores the flags (also the `ml` stage of `rebuild.py`) |
| `notebooks/06_ml_anomaly_detection.ipynb` | What the model learnt, the threshold, flags per month and why, rules against rules + ML |

## Features

SQL over each Silver row, in `configs/ml.yml`; one feature row per Silver row, keyed by
`bronze_row_id`, so every flag traces back to its source row.

| Feature | Expression | Catches |
|---|---|---|
| `trip_duration_min` | from Silver | very short or very long trips |
| `trip_distance` | as delivered | |
| `speed_mph` | distance ÷ hours | a long trip in a few minutes |
| `fare_per_mile` | fare ÷ distance | a high fare for a short trip |
| `fare_per_minute` | fare ÷ minutes | |
| `total_gap` | total − sum of its parts | totals that do not add up (open item O1) |
| `pickup_hour`, `pickup_weekday` | from the pickup time | trips at unusual times |
| `passenger_count` | as delivered | |

A division by zero gives NULL; NULLs are filled with the training medians.

## Training

`train.py` uses a **time-based split**: it learns from a random sample (100,000 rows a month,
seed 42) of the train months 2023-01 to 2023-03, and sets the **threshold** on the next month,
2023-04: the anomaly score above which 1% of its rows lie (`threshold.target_flag_rate`). A
clean month therefore flags about 1% of its trips, and a rise means something.

The model and `metadata.json` are saved to `models/<dataset>/<model_id>/` (ignored by Git) and
registered in `ml_models`. The metadata records the features, the train and calibration months
and their **Silver file versions**, the medians, quartiles and typical spreads (for the
explanations), the threshold and the scikit-learn version. The model id is a hash of the
settings and those versions: training again on the same data gives the same id and the same
scores.

## Scoring and explanations

`score.py` scores every row of each month's current Silver version. Rows above the threshold are
**flagged**; a flag is a warning and the row stays in Silver. Each flag keeps the three features
furthest from their training median, measured in that feature's typical spread (the
interquartile range, or the 5–95% range for a nearly constant feature such as `total_gap`):

```
fare_per_mile 97 (typical 5.9); total_gap -63.7 (typical 0); fare_per_minute 22 (typical 1.49)
```

| Table | Holds |
|---|---|
| `ml_anomalies` | Every flagged row: `bronze_row_id`, score, top features, reasons |
| `ml_month_summary` | Per month, file version and model: rows scored and flagged, flag rate, warnings |

Scoring warns (and the run ends as WARNING) when the model learnt from months more than 12
before the scored month (`retrain.max_age_months`), or when a month it learnt from has a newer
Silver version since training. Retraining is by hand: `python src/ml/train.py`.

## In the pipeline

- **`rebuild.py`** has an `ml` stage after the rules. It skips until a model is trained.
- **Incidents.** A month whose flag rate rises above the same model's flag rate in the earlier
  months, by more than `max(0.5 points, 1 × that rate)`, opens an incident **"Unusual trips
  (ML)"** (`ml_anomaly`). Before three earlier months a fixed 3% limit applies. The detail
  names the most frequent reasons. The incident follows the same lifecycle as the others.
- **Quality score.** `ml_anomaly` is the 11th category (weight 1): the share of rows not
  flagged, so a clean month scores about 0.99 there.
- **Gold.** `gold_dataset_health` has `ml_rows_flagged`, `ml_flag_rate` and `ml_model_id`;
  `gold_trends` has the `ml_flag_rate` series.
- **Tests** use their own models folder (`DATAGUARD_MODELS_DIR`), never a model trained on the
  real data.

## Rules only against rules + ML

The detection experiments of Phase 6 ([detection_evaluation.md](detection_evaluation.md))
train the model on the clean history and score the broken month with it. Fault 11,
`plausible_but_wrong`, triples the fare of short trips (0.5 to 3 miles) and raises the total to
match: every value stays in range and the total still adds up.

**Rules** count as caught when any check or row-rule incident is new against the control month,
as flagged when a rule newly flags a broken row. **ML** counts as caught when an `ml_anomaly`
incident is new, as flagged when the model flags a broken row it does not already flag in the
control. A row flagged anyway is never credited.

On the test months (34 experiments, every row used):

| | Rules only | ML only | Rules + ML |
|---|---|---|---|
| 🟢 caught | 20 | 2 | **21** |
| 🟡 flagged | 4 | 1 | **5** |
| 🔴 missed | 9 | 30 | **7** |

**False alarms:** in the unbroken control month the model flags 1.06% of the rows (2 of 188,
its calibrated normal level) and raises no incident; the rules raise nothing either.

| Experiment | Rules only | ML | Rules + ML | ML recall | ML adds |
|---|---|---|---|---|---|
| `duplicate_rows-1pct` | 🟡 flagged | 🔴 missed | 🟡 flagged | 0% | |
| `duplicate_rows-5pct` | 🟢 caught | 🔴 missed | 🟢 caught | 0% | |
| `duplicate_rows-20pct` | 🟢 caught | 🔴 missed | 🟢 caught | 0% | |
| `missing_column` | 🟢 caught | 🔴 missed | 🟢 caught | | |
| `null_spike-1pct` | 🟢 caught | 🔴 missed | 🟢 caught | 0% | |
| `null_spike-5pct` | 🟢 caught | 🔴 missed | 🟢 caught | 0% | |
| `null_spike-20pct` | 🟢 caught | 🟢 caught | 🟢 caught | 10% | |
| `orphan_location-1pct` | 🟡 flagged | 🔴 missed | 🟡 flagged | 0% | |
| `orphan_location-5pct` | 🟢 caught | 🔴 missed | 🟢 caught | 0% | |
| `orphan_location-20pct` | 🟢 caught | 🔴 missed | 🟢 caught | 0% | |
| `out_of_range-1pct` | 🟡 flagged | 🔴 missed | 🟡 flagged | 0% | |
| `out_of_range-5pct` | 🟢 caught | 🔴 missed | 🟢 caught | 0% | |
| `out_of_range-20pct` | 🟢 caught | 🔴 missed | 🟢 caught | 0% | |
| `pickup_outside_month-1pct` | 🟡 flagged | 🔴 missed | 🟡 flagged | 0% | |
| `pickup_outside_month-5pct` | 🟢 caught | 🔴 missed | 🟢 caught | 0% | |
| `pickup_outside_month-20pct` | 🟢 caught | 🔴 missed | 🟢 caught | 0% | |
| `plausible_but_wrong-1pct` | 🔴 missed | 🟡 flagged | 🟡 flagged | 100% | yes |
| `plausible_but_wrong-5pct` | 🔴 missed | 🔴 missed | 🔴 missed | 0% | |
| `plausible_but_wrong-20pct` | 🔴 missed | 🟢 caught | 🟢 caught | 22% | yes |
| `stale_file-1d` | 🔴 missed | 🔴 missed | 🔴 missed | | |
| `stale_file-5d` | 🟢 caught | 🔴 missed | 🟢 caught | | |
| `stale_file-20d` | 🟢 caught | 🔴 missed | 🟢 caught | | |
| `swapped_times-1pct` | 🟢 caught | 🔴 missed | 🟢 caught | 0% | |
| `swapped_times-5pct` | 🟢 caught | 🔴 missed | 🟢 caught | 0% | |
| `swapped_times-20pct` | 🟢 caught | 🔴 missed | 🟢 caught | 0% | |
| `unit_change-1pct` | 🔴 missed | 🔴 missed | 🔴 missed | 0% | |
| `unit_change-5pct` | 🔴 missed | 🔴 missed | 🔴 missed | 0% | |
| `unit_change-20pct` | 🟢 caught | 🔴 missed | 🟢 caught | 0% | |
| `unit_change-100pct` | 🟢 caught | 🔴 missed | 🟢 caught | 0% | |
| `volume_drop-1pct` | 🔴 missed | 🔴 missed | 🔴 missed | | |
| `volume_drop-5pct` | 🔴 missed | 🔴 missed | 🔴 missed | | |
| `volume_drop-20pct` | 🔴 missed | 🔴 missed | 🔴 missed | | |
| `volume_drop-50pct` | 🟢 caught | 🔴 missed | 🟢 caught | | |

### Reading the table

- **ML adds detection exactly where no rule can look:** the plausible-but-wrong fares, flagged
  at 1% and caught as an incident at 20%.
- **It does not compete with the rules.** Rows that break a rule go to Quarantine and are never
  scored, and month-level faults (schema, volume, late file) leave the rows themselves normal.
  The model is a second line, not a replacement.
- **Distances in kilometres stay hidden.** Fare per mile only falls by a factor of 1.6, well
  inside its normal spread, so neither the rules nor the model see 1% or 5% of them; the drift
  check catches 20%.
- **NULL fares at 20%** also raise an ML incident: empty fares are filled with the median, which
  makes the fare disagree with the trip.
- **Plausible but wrong at 5%** is missed: on the test months 5% is 2 rows; one of them the
  model flags in the control month too, the other scores just below the threshold.

## Known limits

- **An Isolation Forest does not grade extremes.** It isolates values by random splits within
  the training range, so a value far beyond the range scores like one just beyond it (notebook
  06, section 5: a fare per mile 2×, 5×, 20× or 100× the normal range all score the same).
  Unusual trips are ranked well, but a month's natural oddities share the 1% flag budget with
  the broken rows.
- **The flag budget is a choice.** 1% of a clean month is flagged by design; raising
  `target_flag_rate` catches more and alarms more.
- **Totals that do not add up lead the flags** on the test months (`total_gap`): whether that is
  a real problem waits for open item O1.
- **Retraining is by hand.** Scoring warns when the model is old or its training data changed.
- **Results on the full data** are pending (notebook 06, section 7): the test months have 200
  trips each, so 1% is two rows; on the full data it is about a thousand a month.
