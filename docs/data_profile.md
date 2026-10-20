# Data profile: NYC Yellow Taxi (2019-01, 2023-01 to 2023-12)

Phase 0, Step 0.8. Findings from `notebooks/01_data_profiling.ipynb` run on the full files
(13 files, 7,696,617 rows for 2019-01 and 38,310,226 rows for 2023). These are measured values,
not expectations. Thresholds marked *proposed* are decided in Phase 2 (row rules) and Phase 3
(dataset checks).

## Key findings

1. **The schema changed inside 2023.** From February, `airport_fee` is renamed `Airport_fee` and five
   columns change type. A pipeline reading all months as one table breaks or silently loses data.
2. **About 28% of rows in at least one file fail the "total = sum of parts" check.** This is too
   large to be bad data alone; the check's definition must be investigated before it becomes a rule
   (open item O1).
3. **The NULL group always arrives together.** When `passenger_count` is NULL, the other four
   related columns are NULL too, in 100% of cases. Its rate more than doubled during 2023
   (2.34% → 5.33%), with a jump in September.
4. **Distances and fares contain impossible extremes** (345,729-mile trips, a $386,983 fare), while
   medians are stable, so drift must be measured on robust statistics, not means.
5. **Some problems do not occur in the real data** (unknown location IDs: 0; exact duplicates: 2 rows
   in 38M). Their checks can only be tested with fault injection (Phase 6).

## 1. Volume

| File | Rows | Days | Rows per day | vs. 2023 median |
|---|---:|---:|---:|---:|
| 2019-01 | 7,696,617 | 31 | 248,278 | +127% |
| 2023-01 | 3,066,766 | 31 | 98,928 | −9.5% |
| 2023-02 | 2,913,955 | 28 | 104,070 | −4.8% |
| 2023-03 | 3,403,766 | 31 | 109,799 | +0.5% |
| 2023-04 | 3,288,250 | 30 | 109,608 | +0.3% |
| 2023-05 | 3,513,649 | 31 | 113,344 | +3.7% |
| 2023-06 | 3,307,234 | 30 | 110,241 | +0.9% |
| 2023-07 | 2,907,108 | 31 | 93,778 | −14.2% |
| 2023-08 | 2,824,209 | 31 | 91,104 | −16.6% |
| 2023-09 | 2,846,722 | 30 | 94,891 | −13.2% |
| 2023-10 | 3,522,285 | 31 | 113,622 | +4.0% |
| 2023-11 | 3,339,715 | 30 | 111,324 | +1.9% |
| 2023-12 | 3,376,567 | 31 | 108,922 | −0.3% |

2023 median: 109,265 rows per day.

- **Seasonality:** July to September run 13–17% below the median.
- **Largest month-to-month change:** September → October, +19.7%. Against the rolling median of the
  three previous runs (July–September, 93,778/day), October is +21%.
- **Threshold:** a ±30% tolerance on rows per day passes every 2023 month; ±20% would raise a false
  alarm on October. **Keep ±30%** (already the default in `configs/quality_rules.yml`).
- **With the Phase 3 baseline** (median of up to six earlier months, from 2023-04 on), the changes
  are: Apr +5.3%, May +6.1%, Jun +0.6%, Jul −14.5%, Aug −17.0%, Sep −13.5%, Oct +11.1%, Nov +8.5%,
  Dec +6.2%. The largest is 13 points inside the ±30% tolerance.
- **2019-01** is 2.27× the 2023 median. It has no earlier month within 12 months, so it is judged
  by the cold-start limits (50,000–200,000 rows/day) and flagged as above them.

## 2. Schema

| Column | 2019-01 | 2023-01 | 2023-02 to 2023-12 |
|---|---|---|---|
| `airport_fee` | INTEGER | DOUBLE | missing |
| `Airport_fee` | missing | missing | DOUBLE |
| `VendorID` | BIGINT | BIGINT | INTEGER |
| `PULocationID` | BIGINT | BIGINT | INTEGER |
| `DOLocationID` | BIGINT | BIGINT | INTEGER |
| `RatecodeID` | DOUBLE | DOUBLE | BIGINT |
| `passenger_count` | DOUBLE | DOUBLE | BIGINT |

All other columns have the same name and type in all 13 files.

- **Three schema versions:** 2019-01, 2023-01, and 2023-02 onward.
- **Rename by case only:** `airport_fee` → `Airport_fee`. DuckDB matches column names without case,
  pandas and Parquet do not, so the same file behaves differently depending on the tool.
- **Type changes:** IDs move from BIGINT to INTEGER, and `RatecodeID` and `passenger_count` move from
  DOUBLE to BIGINT. The values are whole numbers in both versions.
- **Decision:** keep 2019-01 in the dataset. The main schema change is inside 2023, and 2019-01 adds a
  third schema version and a real volume difference.
- **Phase 2 standardization (proposed canonical schema):** lower-case column names
  (`airport_fee`), `VendorID`/`PULocationID`/`DOLocationID`/`payment_type` as INTEGER,
  `passenger_count`/`RatecodeID` as nullable INTEGER, amounts as DOUBLE, timestamps as TIMESTAMP.
- **Phase 3 schema check:** must report the January → February 2023 change as one incident with the
  rename and the four type changes listed.

## 3. Completeness

`passenger_count`, `RatecodeID`, `store_and_fwd_flag`, `congestion_surcharge` and `airport_fee` are
NULL together in 100% of the rows where `passenger_count` is NULL, in every file.

| File | Rows with the group NULL | Share |
|---|---:|---:|
| 2019-01 | 28,672 | 0.37% |
| 2023-01 | 71,743 | 2.34% |
| 2023-02 | 76,817 | 2.64% |
| 2023-03 | 87,619 | 2.57% |
| 2023-04 | 90,690 | 2.76% |
| 2023-05 | 101,796 | 2.90% |
| 2023-06 | 99,887 | 3.02% |
| 2023-07 | 85,086 | 2.93% |
| 2023-08 | 87,886 | 3.11% |
| 2023-09 | 140,225 | 4.93% |
| 2023-10 | 154,929 | 4.40% |
| 2023-11 | 132,675 | 3.97% |
| 2023-12 | 180,003 | 5.33% |

- **One rule, not five:** "trip metadata missing" (category completeness), action **warn**. The
  fares and locations of these rows are present, so the trips are real.
- **NULL-rate baseline:** the rate rises during 2023 and jumps from 3.11% to 4.93% in September
  (+59% relative). A relative threshold would raise an incident for September; an absolute
  threshold of +2 percentage points would not. *Proposed:* absolute threshold, calibrated in Phase 3.
  Either way the September jump is a real change in the source and is worth reporting at Medium or
  Low severity.

## 4–7. Row-level problems

Totals across all 13 files (45.98M rows). `max %` is the share in the worst file.

| Check | Category | Rows | Max % | Files | Proposed action |
|---|---|---:|---:|---:|---|
| total ≠ sum of parts | consistency | 10,243,395 | 27.94% | 13 | none yet; see O1 |
| `trip_distance` = 0 | validity | 828,546 | 3.46% | 13 | warn |
| zone 264/265 (Unknown / Outside of NYC) | referential | 821,149 | 2.44% | 13 | warn |
| `passenger_count` = 0 | validity | 700,386 | 1.73% | 13 | warn |
| `fare_amount` < 0 | validity | 388,779 | 1.26% | 13 | quarantine |
| `total_amount` < 0 | validity | 384,010 | 1.18% | 13 | quarantine |
| dropoff = pickup | consistency | 19,647 | 0.085% | 13 | warn |
| `fare_amount` = 0 | validity | 15,494 | 0.036% | 13 | warn |
| dropoff before pickup | consistency | 2,479 | 0.027% | 13 | quarantine |
| `trip_distance` > 100 miles | validity | 1,336 | 0.005% | 13 | quarantine |
| pickup outside the file's month | pattern | 1,267 | 0.007% | 13 | quarantine |
| `passenger_count` > 6 | validity | 466 | 0.002% | 13 | quarantine |
| trip longer than 24 hours | consistency | 217 | 0.001% | 13 | quarantine |
| location ID not in the zone lookup | referential | 0 | 0% | 0 | quarantine; test by fault injection |

Every problem except unknown location IDs appears in all 13 files: these are steady properties of the
source, not one-off accidents.

**Expected quarantine share:** the proposed quarantine rules together cover roughly 1.3% of rows in
the worst file (negative amounts dominate; the other quarantine rules add under 0.05%). The warn
rules flag up to about 3.5% (zero distance) without removing rows.

### Notes per finding

- **Negative amounts** are refunds and disputes. They are valid records of the business but not
  trips, so they go to Quarantine with the reason `negative amount (refund or dispute)`. From
  2023-07 the 1st percentile of `fare_amount` is negative (−3.00 to −4.40), so negative fares rise
  above 1% of rows in the second half of 2023.
- **Zero distance** with a paid fare is often a cancelled or flat-rate trip; warn only.
- **Zones 264/265** are valid IDs in the lookup with no real location; warn and count them in Gold.
- **Zero passengers** is a driver-entered field; warn only, as in the example row found during setup.

## 8. Uniqueness

| File | Exact duplicates | Composite-key duplicates |
|---|---:|---:|
| 2019-01 | 0 | 3 |
| 2023-01 | 0 | 1 |
| 2023-04 | 0 | 1 |
| 2023-10 | 0 | 1 |
| 2023-11 | 2 | 2 |
| 2023-12 | 0 | 1 |
| other 2023 months | 0 | 0 |

Composite key: `VendorID`, pickup and dropoff time, `PULocationID`, `DOLocationID`, `fare_amount`,
`total_amount`.

- Duplicates are almost absent (at most 3 per file). The uniqueness rule stays, with action
  quarantine, but its detection quality can only be measured with the duplicate-rows fault (Phase 6).

## 9. Distributions (drift baseline)

| Statistic | 2019-01 | 2023 range (12 files) |
|---|---:|---:|
| `trip_distance` median | 1.53 | 1.70 – 1.83 |
| `trip_distance` p99 | 18.96 | 19.90 – 20.58 |
| `trip_distance` max | 831.80 | 90,771.91 – 345,729.44 |
| `trip_distance` mean | 2.83 | 3.63 – 4.78 |
| `fare_amount` median | 9.00 | 12.80 – 14.20 |
| `fare_amount` p01 | 3.00 | −4.40 – 3.00 |
| `fare_amount` max | 623,259.86 | 1,040.40 – 386,983.63 |
| `total_amount` median | 11.30 | 20.16 – 21.84 |

- **Means are unreliable:** a handful of impossible values (a 345,729-mile trip, a $386,983 fare)
  move the mean, while medians and p99 stay stable. The drift check (PSI) must work on binned values
  with the extremes capped at fixed edges, and the range rules remove the extremes from Silver first.
- **`trip_distance` is the cleanest drift signal:** median 1.70–1.83 all year. A miles-to-kilometres
  change (×1.61) would move the median to about 2.8, far outside this band.
- **Fares shift during the year:** the `fare_amount` median steps from 12.80–13.50 (January–August)
  to 14.20 (September–December). The drift check should see this as at most Medium drift, not a
  fault; it is a calibration case for Phase 3.
- **2019 vs 2023:** fares and totals are much lower in 2019 (median fare 9.00 vs. 12.80+), so 2019-01
  is expected to fail the drift check against a 2023 baseline. This is a real difference, not an error.

## Coverage of the check categories

| # | Category | Real finding | Testable on real data? |
|---|---|---|---|
| 1 | Schema | Rename and type changes, January → February 2023 | Yes |
| 2 | Completeness | NULL group, 0.37% – 5.33% | Yes |
| 3 | Uniqueness | At most 3 duplicates per file | Needs fault injection |
| 4 | Validity (range) | Zero/negative amounts, zero distance, extremes | Yes |
| 5 | Pattern and timestamp | 1,267 pickups outside the month | Yes |
| 6 | Referential integrity | 0 unknown IDs; 821,149 rows in zones 264/265 | Needs fault injection for unknown IDs |
| 7 | Consistency | Dropoff before pickup, zero duration, total mismatch | Yes (total mismatch after O1) |
| 8 | Freshness | Not measurable on historical files | Needs fault injection (manifest) |
| 9 | Volume | Seasonal ±17%; 2019-01 at 2.27× | Yes |
| 10 | Distribution drift | Fare median step in September; 2019 vs 2023 | Yes |

Every category has a real finding or is covered by a fault in Phase 6.

## Open items

- **O1 — total mismatch (PENDING: waiting for the diagnosis run on the full data).** 10.2M rows, up to 27.9% in one file, fail
  `|total_amount − sum(parts)| > 0.01`.
  - Finding so far (commit `868f281`): VendorID 1 books `congestion_surcharge` and `airport_fee`
    inside `extra` as well, so its totals match the sum *without* those two columns. The profiling
    check now accepts either formula, for every vendor.
  - Open: whether the second formula applies to VendorID 1 only, and how many rows still match
    neither. `python src/profiling/diagnose_totals.py` measures both per vendor and lists the most
    common differences. Until then the rule `total_matches_parts` is in
    `configs/quality_rules.yml` with `enabled: false`, so it flags nothing.
- **O2 — NULL-rate threshold (Phase 3).** Absolute (+2 pp) or relative threshold; the September 2023
  jump is the calibration case.
- **O3 — fare drift (Phase 3).** Calibrate PSI so the September fare step is Medium at most and a
  ×1.61 unit change is High.
- **O4 — canonical schema (Phase 2).** Confirm the standard types listed in section 2.
