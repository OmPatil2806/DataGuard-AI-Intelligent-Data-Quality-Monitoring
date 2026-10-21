# Silver, incidents and the quality score

Phase 4. The row rules (Phase 2) and the dataset checks (Phase 3) produce findings; this phase
writes the clean rows to **Silver** and the rejected rows to **Quarantine**, turns findings into
**incidents** with a lifecycle, and sums each month up in one **quality score**.

## Commands

| Command | What it does |
|---|---|
| `python src/silver/process_silver.py` | Splits every month into Silver and Quarantine, adds derived and zone columns |
| `python src/incidents/incidents_cli.py update` | Applies the latest check run and rule run to the incidents |
| `python src/incidents/incidents_cli.py list` / `show INC-0001` | Active incidents / details and history |
| `python src/incidents/incidents_cli.py investigate\|reopen\|resolve\|note INC-0001 --note "..."` | Manual status changes and notes |
| `python src/quality/score.py` | Scores every month with Silver and stores the result |
| `notebooks/04_quality_score_calibration.ipynb` | Score per month, points lost, weight sensitivity, incident timeline, verdict |

## Silver and Quarantine

Every Bronze row ends up in exactly one place:

- **Quarantine**: the row fails at least one rule with action `quarantine`. It keeps
  `failed_rules` and `reasons`.
- **Silver**: every other row, including rows that only fail `warn` rules (still listed in
  `failed_rules`). Silver adds `trip_duration_min`, `pickup_date`, `pickup_hour` and the pickup
  and dropoff borough and zone from the zone lookup (`configs/datasets.yml`, `silver`).

Both are Parquet under `period=YYYY-MM/version=<sha256 prefix>/`, written atomically, once per
Bronze file version (`silver_log`).

## Incidents

An incident is one problem in one dataset, identified by **(dataset, source, rule, column)**:
`dataset_check` incidents come from failed checks (schema, volume, freshness, null_rate, drift),
and `row_rule` incidents come from the severity engine.

### When a row rule becomes an incident

Real data has steady, known problems, such as about 1.2% negative fares every month, so a rule
failing is not an incident on its own. A rule becomes an incident when its failure rate rises
clearly above its own history:

```
trusted baseline (3+ earlier months):  rate - baseline > max(0.1 points, 1.0 x baseline)
cold start:                            rate > 5% (quarantine rules) or 10% (warn rules)
```

The baseline is the median rate of the earlier months, chosen like the dataset baselines. The
limits are in `configs/quality_rules.yml` (`defaults.incidents.row_rules`).

### Lifecycle

Months are applied in month order:

| Situation | Result |
|---|---|
| Finding, no active incident | A new incident is opened. If the same problem was resolved before, it links to it (`recurrence_of`) |
| Finding, active incident | De-duplicated: occurrences + 1, last seen updated, clean streak back to 0, severity raised if worse |
| Check ran clean, active incident | Clean streak + 1; **resolved after 2 clean months** in a row |
| Check did not run (month missing, skipped) | Unchanged |

Statuses: `OPEN` → `INVESTIGATING` (by hand) → `OPEN` again or `RESOLVED`. Every change is kept
in `incident_events`. Each incident remembers the last month it handled, so re-running
`update`, or applying an older month, never counts a result twice.

## Quality score

```
score = 100 x sum(w_i x p_i) / sum(w_i)
```

| Part | Pass rate `p` |
|---|---|
| Row categories (completeness, uniqueness, validity, pattern, referential, consistency) | Share of the month's rows (Silver + Quarantine) that fail no enabled rule of the category |
| Dataset checks (schema, volume, freshness, drift) | `check_scores` of the worst result: pass 1.0, low 0.75, medium 0.5, high 0.25, critical 0 |
| Completeness | The lower of the row pass rate and the null_rate check |
| No result (a skipped check, no rule, ml_anomaly before Phase 7) | Left out of both sums |

Weights and check scores are in `configs/severity_weights.yml`: schema 5, volume 4,
completeness, validity, freshness and drift 3, the rest 2, ml_anomaly 1. Scores are stored per
file version in `quality_scores`, with the parts and where each pass rate came from in
`quality_score_parts`.

### Reading the score

A row category moves the score only as much as its share of bad rows: 1.2% negative fares cost
about 0.1 points. A failed dataset check costs much more (a high schema finding about 14 points
out of 100), because it affects the whole month. Notebook 04 shows the **points lost** per
category, `100 x w x (1 - p) / sum(w)`, which add up to `100 - score`.

### Calibration on the test months

The five planted months of the test suite (`tests/test_run_checks.py`):

| Month | Score | Main loss |
|---|---|---|
| 2023-01 | 91.2 | volume (cold start, medium) |
| 2023-02 | 69.1 | schema change (high), pickups outside the month |
| 2023-03 | 83.6 | pickups outside the month |
| 2023-04 | 85.4 | pickups outside the month, volume drop (medium) |
| 2023-05 | 76.8 | NULL spike and drift (both high) |

With all weights equal, the months keep the same order, so the ranking does not hinge on the
exact weights.

**Still to confirm on the full data** (notebook 04, section 6): the clean 2023 months are
expected to score close to 100, 2023-02 lowest because of the schema change, and 2023-09
slightly lower because of the low NULL-rate finding.

## Known limits

- The score summarises a month; it does not replace the incidents, which say what broke.
- A steady problem in every month (a rule failing at the same rate) is not an incident, but it
  still lowers the score of every month by its share of rows.
- Total amount against its parts (open item O1) is not scored yet: the rule is disabled until
  it is confirmed on the full data.
