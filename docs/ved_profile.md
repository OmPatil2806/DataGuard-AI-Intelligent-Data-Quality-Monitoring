# Vehicle Energy Dataset: profile

Phase 9, Step 9.1. The second dataset, profiled on all of it before any rule or check is
written for it. Source: [gsoh/VED](https://github.com/gsoh/VED) (Oh, LeBlanc, Peng, *Vehicle
Energy Dataset (VED), A Large-scale Dataset for Vehicle Energy Consumption Research*, IEEE
T-ITS 2020), Apache License 2.0. GPS and OBD-II signals of cars in Ann Arbor, Michigan, from
November 2017 to November 2018.

**Done when** the facts the configuration of Steps 9.2 to 9.5 depends on are measured on the
real files, with the numbers below.

## Commands

| Command | What it does |
|---|---|
| `python src/data_download/fetch_archives.py --dataset ved_signals` | Downloads the 4 source files (176 MB), checks them, unpacks the weekly CSVs (3.2 GB) and writes the static data as CSV, into `data/incoming/ved_signals/` |
| `python src/profiling/ved_profile.py` | Profiles them (about 1.5 minutes); summary in `data/profile/ved_signals/profile.json` |
| `notebooks/08_ved_profiling.ipynb` | The same, with charts and a verdict |

The source files and their unpacked contents are listed in `configs/datasets.yml`
(`ved_signals.archives`); `fetch_manifest.json` records the SHA-256 of each archive and every
file it gave.

## The files

| Source file | Size | Gives |
|---|---|---|
| `VED_DynamicData_Part1.7z` | 82.7 MB | 22 weekly CSVs, 2017-11-01 to 2018-03-28 (1.46 GB) |
| `VED_DynamicData_Part2.7z` | 93.7 MB | 32 weekly CSVs, 2018-04-04 to 2018-11-07 (1.74 GB) |
| `VED_Static_Data_ICE&HEV.xlsx` | 21 KB | 357 vehicles: petrol (ICE) and hybrid (HEV) |
| `VED_Static_Data_PHEV&EV.xlsx` | 10 KB | 27 vehicles: plug-in hybrid (PHEV) and electric (EV) |

**54 weekly files, 22,436,808 rows, 384 vehicles, 32,552 trips**, from 2017-11-01 00:04 to
2018-11-10 12:10. All 54 files have the same 22 columns.

## Findings

| # | Finding | Measured | What it means for the configuration |
|---|---|---|---|
| V1 | **Mixed line endings: DuckDB loses rows without an error.** Every header ends with CR LF, every data row with LF. DuckDB 1.5's CSV reader cannot sniff the files; told the dialect, it skips each file's first 32 MB | All 54 files. First week: 489,414 rows in the file, 254,852 read (**234,562 lost**); all files: 22,436,808 rows, 10,430,640 read (**53.5% lost**) | Ingestion (9.2) must not read these CSVs with DuckDB as they are: read with pyarrow (every file's row count is checked against its line count here), or normalise the header first (then DuckDB reads every row). A Bronze check that rows read = lines in the file |
| V2 | File names are **yymmdd**, not the "mmddyy" of the README | `VED_171101_week.csv` is the week of 2017-11-01 | Parse the week from the name as yymmdd |
| V3 | Missing values are the text `NaN` | Every signal column | `NaN` is the NULL marker when reading |
| V4 | One header differs from the README: `OAT[DegC]`, not "Outside Air Temperature[DegC]"; units are part of every name (`Vehicle Speed[km/h]`) | 22 columns | Standard names need aliases from the source names (9.3), not only a case-insensitive match |
| V5 | **A row has no timestamp.** `DayNum` is the trip's start in days (1.0 = 2017-11-01 00:00), the same on every row of a trip; `Timestamp(ms)` is the time since the trip started | No trip has more than one DayNum; no trip spans two files | Row time = 2017-11-01 + (DayNum − 1) days + Timestamp(ms): a computed column before the rules (9.3) |
| V6 | Trips that start at a week's end run into the next week but stay in their start's file | 17,669 rows after their file's 7 days; none before | Group rows into months by their own time (9.2); "outside the period" must use the row time |
| V7 | Negative `Timestamp(ms)` | 2,702 rows in 4 trips, down to −1,658,400 ms (−27.6 min) | A validity rule: time since the trip started ≥ 0 (quarantine) |
| V8 | **No duplicates** on (VehId, Trip, Timestamp(ms)) | 0 | Uniqueness rule on that key: it should stay at 0 |
| V9 | **384 vehicles**, not the README's 383: 264 ICE, 93 HEV, 24 PHEV, 3 EV (README: 92 HEV, 27 PHEV/EV) | Same 384 in the dynamic and the static data; no id repeated | The static data is a complete lookup: a referential rule (every VehId known) should find 0 |
| V10 | The two static files name the engine type column differently: `Vehicle Type` (ICE & HEV) and `EngineType` (PHEV & EV) | Both files | One lookup from two files needs a rename of that column (9.2/9.5) |
| V11 | The static data is mostly placeholders | `Vehicle Class` "NO DATA" for 352 of 384, `Drive Wheels` 357, `Transmission` 286 (and 3 empty), `Generalized_Weight` 16 | Only VehId and engine type are usable as dimensions; "NO DATA" is a NULL |
| V12 | **Which signals exist depends on the engine type** (notebook 08, section 3) | EV: no MAF, RPM, load or fuel trims (100% NULL); ICE: no battery signals (100%); fuel rate is NULL for 99.99% of ICE and HEV rows and present for 27.6% of PHEV rows; AC power in kW only for PHEV, in W for EV and PHEV | Completeness rules per engine type (`when:` on the engine type). A NULL fuel rate on a petrol car is normal here: its fuel use can only come from MAF |
| V13 | Outside air temperature is often missing, and sits on the sensor's limits | 41.8% NULL overall (58.5% for ICE); 1,441 rows at −40 °C and 1,460 at 60 °C | Completeness as a warning; values at exactly −40 or 60 as a warning (sensor limits) |
| V14 | Engine load above 100% | 29,219 rows above 100%, 481 above 200%, up to 22,463%, in 127 vehicles | Validity rule 0–100% for load, as a warning first (measure before quarantining) |
| V15 | Battery charge just above 100 | 358 values of 100.00003% (float rounding); none below 0 | A 0–100 rule needs a small tolerance (≤ 100.001) |
| V16 | Speed and RPM: an electric motor turns the wheels without the engine | Moving with RPM 0 or NULL: EV 100%, PHEV 71%, HEV 28%, ICE 0.04% | "Moving means RPM > 0" only for ICE |
| V17 | GPS is inside a small box around Ann Arbor | Latitude 42.220–42.326, longitude −83.804 to −83.674: no row outside this box | A pattern rule: GPS inside 42.2–42.35 and −83.82 to −83.65 should find 0 |
| V18 | Speed range | 0 to 173 km/h, p99 115; 0.006% NULL; 11.5% of rows at 0 | Validity 0–200 km/h |
| V19 | Sampling | Median 0.6 s between rows of a trip (p10 0.1 s, p90 1.4 s, p99 2.7 s); 507 gaps over a minute | Not a monitoring signal; documented |
| V20 | Fleet and volume shrink over the year | Vehicles per month 332 (2017-11) to 204 (2018-09); rows per month 2.41 M (2017-12) to 1.50 M (2018-08); November 2018 has 10 days (436,252 rows); rows per day 5,761 to 158,314 (median 60,003) | The volume check must live with a slow decline: February 2018 has 54,262 rows/day against a median of 75,419 for November to January, −28%, just inside the ±30% limit (to watch in 9.5); the last month is partial (exclude 2018-11 or expect a volume finding) |

## Rows per month

By each row's own time (V5, V6):

| Month | Rows | Vehicles | Trips |
|---|---|---|---|
| 2017-11 | 2,262,581 | 332 | 3,575 |
| 2017-12 | 2,414,949 | 320 | 3,431 |
| 2018-01 | 2,093,093 | 309 | 3,112 |
| 2018-02 | 1,519,330 | 278 | 2,259 |
| 2018-03 | 1,795,078 | 293 | 2,777 |
| 2018-04 | 1,959,780 | 286 | 2,789 |
| 2018-05 | 1,904,031 | 271 | 2,623 |
| 2018-06 | 1,804,636 | 266 | 2,539 |
| 2018-07 | 1,603,437 | 238 | 2,392 |
| 2018-08 | 1,504,330 | 214 | 2,091 |
| 2018-09 | 1,511,844 | 204 | 2,135 |
| 2018-10 | 1,627,467 | 215 | 2,253 |
| 2018-11 | 436,252 | 163 | 584 (10 days) |

## Decisions for Steps 9.2 to 9.5

| Decision | Because |
|---|---|
| VED is monitored **per month** (2017-11 to 2018-10; 2018-11 is partial), rows grouped by their own time | V5, V6, V20: the pipeline's checks, baselines and Gold are monthly |
| CSV sources are read with pyarrow, with every file's rows checked against its lines | V1 |
| Standard names are mapped from source names with aliases | V4 |
| The row time is a computed column, before the rules | V5 |
| Completeness rules depend on the engine type, from the static data | V10, V12 |
| No freshness check for VED: it is a closed, historical dataset | The source has no delivery schedule |
| First rules (9.5): time since trip start ≥ 0; speed 0–200; load 0–100 (warn); SOC 0–100.001; GPS box; key unique; VehId known; RPM > 0 when moving (ICE, warn) | V7, V8, V9, V13 to V18 |

## Ingestion (Step 9.2)

```
python src/ingestion/ingest_data.py --dataset ved_signals
```

`datasets.yml` gives VED a `source` block (`kind: csv_files`): the weekly CSVs are not loaded
as they are. `ingest_data.py` first runs `src/ingestion/prepare_source.py`, which reads every
weekly file with pyarrow (`NaN` read as NULL, V3), stops if a file's rows differ from its lines
(V1), stages it as Parquet with `origin_file` and `origin_line` (the line after the header) and
the row's month from `source.period_sql` (V5), and writes one file per month,
`ved_signals_<YYYY-MM>.parquet`, with the rows in source order. From there VED is registered and
loaded into Bronze like the taxi files.

On the real files:

| | Result |
|---|---|
| Staging and regrouping the 54 files | about 1.5 minutes |
| Months written and loaded | 12 (2017-11 to 2018-10), **22,000,556 rows** |
| Left out | 2018-11: 436,252 rows (10 days, V20), reported as outside the configured periods |
| Every row accounted for | 22,000,556 + 436,252 = 22,436,808, every row of the 54 files |
| Lineage | Bronze row 1 of 2017-11 is line 1 of `VED_171101_week.csv`, with the same values |
| Run again | nothing staged or written, 12 files skipped |
| A month file written again | the same SHA-256: no new version in the manifest |
| `verify_bronze.py --dataset ved_signals` | 12 file versions, 0 failures |

The taxi files have no `source` block and take the same path as before.

## Standardization (Step 9.3)

The standard schema in `datasets.yml` (`ved_signals.columns`) gives every source column a
snake_case name with its unit, mapped from the source name (V4):

```yaml
speed_kmh: {type: DOUBLE, source: "Vehicle Speed[km/h]"}
outside_temp_c: {type: DOUBLE, source: ["OAT[DegC]", "Outside Air Temperature[DegC]"]}
row_time: {type: TIMESTAMP, sql: "TIMESTAMP '2017-11-01' + ... day_num ... trip_ms ..."}
```

- **`source`**: the name (or names, first found wins) of the column in the source file. The
  schema check (Phase 3) compares a month's file with these names, so a file using the README's
  longer name for the temperature is not a missing column.
- **`sql`**: a computed column, made from the other standard columns right after they are
  renamed and cast, so the rules see it like any other column. VED's `row_time` (V5) is the
  dataset's `timestamp_column`.
- `origin_file` and `origin_line` (Step 9.2) pass through to Silver and Quarantine with the
  Bronze lineage columns.
- **Freshness**: VED has no `expected_lag_days`, so no expected arrival: the freshness check
  reports `skipped`, never a late file.

On the real February 2018 file: 22 columns renamed, 5 cast (ids to INTEGER, RPM and powers to
DOUBLE) without a value lost, `row_time` computed for all 1,519,330 rows, every one inside
February, and the schema check finds no change.

## Known limits

- Profiled once, on the files as downloaded in Phase 9.1 (their SHA-256 is in
  `fetch_manifest.json`); a new release of the source would need a new profile.
- The fixture used by the tests (`tests/ved_fixture.py`) copies the files' shape (header, line
  endings, `NaN`, engine types, the static files' two column names), not their size: the
  DuckDB row loss (V1) appears only above 32 MB and is measured here, not in the tests.
