# Investigation engine

Phase 8. An incident says *what* failed: a check, a rule, a rate above its limit. The
investigation engine adds the next questions an engineer asks: since when, how big, where and
when in the month, has it happened before, what is the likely cause, what to do next. It
answers them from the pipeline's own data, without a language model: every sentence comes from
a template and a fact that can be checked, and when the facts do not point to a cause the answer
is **"cause unknown"**, not a guess.

**Done when** every incident has an evidence pack, a ranked explanation with its evidence,
suggested actions waiting for approval and a report, and the explanations are measured against
faults whose cause is known: see [Accuracy](#accuracy-on-the-fault-experiments).

## Commands

| Command | What it does |
|---|---|
| `python src/investigation/evidence_pack.py INC-0004` | The evidence of one incident, as JSON (`--all`: every incident) |
| `python src/investigation/explainer.py INC-0004` | Its likely causes and next steps (`--all`, `--json`) |
| `python src/investigation/investigate.py` | All of it for every incident, stored, with a report each (also the `investigate` stage of `rebuild.py`) |
| `python src/investigation/investigate.py --decide INC-0004 contact_provider approved --by om` | Record a decision on a suggested action |
| `python src/evaluation/explanation_accuracy.py` | Judge the explanations of the fault experiments |
| `notebooks/07_investigation.ipynb` | The incidents' causes, one incident in depth with its report, where/when, actions, accuracy, verdict |

## How an incident is investigated

```
incident ──> evidence pack ──> facts ──> playbook ──> ranked causes ──> actions ──> report
             (evidence_pack)   (explainer)  (configs/      (score =          (pending     (Markdown,
             + where/when                    playbooks.yml)  prior + signals)  approval)    Gold row)
               (concentration)
```

`investigate.py` runs these steps for every incident in one tracked run (`pipeline_runs`, command
`investigate`). It reads only the control tables and the Silver and Quarantine files, never
Gold, so in `rebuild.py` it runs after the score and before Gold:

| Stage | Writes |
|---|---|
| evidence | `investigation_evidence`: the pack as JSON, one row per incident |
| explain | `investigations`: playbook, best cause and confidence, headline, where/when, the full explanation |
| actions | `investigation_actions`: one row per suggested action, with its decision |
| report | `data/reports/investigations/<dataset>/<incident>.md` (`paths.reports`; `--reports`, `--no-reports`) |
| Gold | `gold_investigations` (built by the Gold stage, [gold_tables.md](gold_tables.md)) |

Everything is rebuilt each time, and nothing holds a build time: the same data gives the same
packs, explanations and reports.

## The evidence pack

`src/investigation/evidence_pack.py`, one JSON document per incident:

| Section | Answers | From |
|---|---|---|
| `incident` | What was found: rule or check, column, severity, status, first and last month, latest observed and expected | `incidents` |
| `definition` | What the rule or check tests | `configs/quality_rules.yml`; the check; the model's features |
| `history` | Since when: the rule's failure rate, the check's result or the model's flag rate in every month | `rule_run_log`, `dataset_check_results`, `ml_month_summary` |
| `events` | The incident's own history: opened, seen again, clean, resolved | `incident_events` |
| `versions` | The file versions of the first and last month seen: SHA-256, rows, delivery | `file_manifest` |
| `impact` | Rows of the last month, how many affected, the quality score and its change | `silver_log`, the files, `quality_scores` |
| `similar` | The same problem before or after (same rule or check and column), and the other incidents of the month | `incidents`, `incident_events` |
| `metrics` | The column's NULL rate, mean, median and p99 per month (rows per day for volume) | `dataset_metrics` |
| `samples` | Up to five affected rows with key values, failed rules and reasons | Silver and Quarantine files |
| `concentration` | Where and when the affected rows sit | see below |

Row-level problems (row rules, NULL rates, ML flags) have affected rows; month-level findings
(schema, volume, freshness, drift) do not, so they have no samples. Row rules and NULL rates
look at Silver and Quarantine; the model scored Silver only, so its incidents are compared with
Silver.

## Where and when

`src/investigation/concentration.py`, settings in `configs/investigation.yml` (thresholds
shared; dimensions and sample columns per dataset, under `datasets:`). The affected rows of a
row-level incident are compared with all rows of the month, segment by segment; for the taxis:

| Kind | Dimensions |
|---|---|
| where | vendor, pickup borough, pickup zone, dropoff zone, payment type, rate code |
| when | pickup date, weekday, hour |

A segment is named only when all four hold:

| Test | Default | Why |
|---|---|---|
| rows in the segment | ≥ 20 | too few rows cannot be judged |
| share of the affected rows in it | ≥ 50% | it must hold most of the problem |
| affected rate inside ÷ rate outside | ≥ 3 | the problem must be clearly worse there |
| two-proportion z statistic | ≥ 4 | far beyond chance, with ~600 segments tested per incident |

The rate inside is compared with the rate *outside* the segment, not with the month's overall
rate, so a fault in a large segment is still found (all broken rows in vendor 2, which has 75%
of the trips). When fewer rows are affected than a segment needs, the answer says so: "only 5
affected row(s), too few to tell". A problem spread evenly gets "unknown". On random faults at
1% to 50%, over 40 seeds, no segment was ever named (`tests/test_concentration.py`).

For volume incidents, days with fewer than half the rows of the month's median day show whether
rows are missing on some days or thinly everywhere (only for months with at least 50 rows on the
median day).

## The explainer

`src/investigation/explainer.py` turns a pack into flat **facts** (`FACTS` in the module): the
detail text, each metric's ratio to its median over the earlier months (`mean_ratio`,
`null_rate_ratio`, `rows_per_day_ratio`, ...), the strongest concentration, the other
incidents of the month, whether one file is registered for two months, the model's leading
reason. It picks the first playbook of `configs/playbooks.yml` whose `match` fits the incident
and scores each of its causes:

```
score = prior + the weights of the signals that hold (at most 1)
```

A cause is a candidate only when all its `requires` hold. Confidence is **high** from 0.7,
**medium** from 0.4, **low** below; when the best cause is low, the headline says "cause
unknown" and names it only as a weak lead. Up to three causes are kept, each with the sentences
of the signals that supported it.

| Playbook | Matches | Causes |
|---|---|---|
| `schema` | schema check | `column_missing`, `column_renamed`, `type_changed`, `column_removed_or_added` |
| `volume` | volume check | `cold_start_limits`, `missing_days`, `rows_missing_evenly`, `rows_added` |
| `freshness` | freshness check | `late_delivery` |
| `null_rate` | NULL-rate check | `column_missing_upstream`, `segment_sends_nulls`, `nulls_in_period`, `source_nulls` |
| `drift` | drift check | `unit_change`, `unit_conversion`, `nulls_shift_distribution`, `scale_error`, `real_change` |
| `duplicates` | rule `trip_unique` | `duplicate_delivery` |
| `outside_month` | rule `pickup_in_file_month` | `wrong_file`, `trips_from_other_months` |
| `swapped_times` | rule `dropoff_not_before_pickup` | `times_swapped` |
| `referential` | referential rules | `unknown_location` |
| `validity` | validity rules | `out_of_range_values` |
| `ml_anomaly` | the anomaly model | `fare_does_not_fit_trip`, `totals_do_not_add_up`, `trips_unusual` |
| `default` | anything else | `rule_failures_up` |

Example, the planted drift in the test months:

```
INC-0004  Distribution drift: trip_distance in 2023-05: most likely trip_distance looks like it
is partly or fully in kilometres instead of miles (mean ×1.61, p99 ×1.61) (high confidence)
  1. [high 0.90] trip_distance looks like it is partly or fully in kilometres instead of miles
       - the median grew by about 1.6, the factor from miles to kilometres
       - the longest trips grew too (p99 ×1.61)
  2. [low 0.30] trip_distance changed by a large factor (median ×1.61): a unit or decimal error
  -> Ask the data provider about Distribution drift: trip_distance in 2023-05 [contact_provider, pending_approval]
```

To add a cause: add it to a playbook with a prior, its `requires`, its signals and actions, and
name it in `explain.causes` of the fault that should produce it (`configs/faults.yml`), so its
accuracy is measured.

## Actions

The best cause's actions (`actions` in `configs/playbooks.yml`: contact the provider, check the
column mapping, download again, reprocess the month, review a rule or baseline, accept the
change, ...) are stored as `pending_approval`. **Nothing runs them.** A person records a
decision: `approved`, `rejected` or `done`, with their name. A decision is kept while the
explanation still suggests the action, so investigating again never resets it; a pending
action the explanation no longer suggests is removed. When the cause is unknown, the only
suggestion is to look at the sample rows (or ask the provider for month-level findings).

## The report

One Markdown file per incident: headline, status table, likely causes with their evidence,
where and when (with the segments or low days), suggested actions with their decisions, since
when, impact, metrics, sample rows, file versions, related incidents and what is tested.
Notebook 07 renders the report of the most severe active incident.

## Accuracy on the fault experiments

Every fault in `configs/faults.yml` names the causes that count as right (`explain.causes`) and
where the problem sits (`explain.where`: `none` for a fault spread at random, or the segment
that must be named). `src/evaluation/explanation_accuracy.py` reads the investigation of the
incident each experiment's detector raised, new against the control month.

On the test months (36 experiments, results in
[detection_evaluation.md](detection_evaluation.md#explanation-accuracy)):

| | Result |
|---|---|
| Experiments with an incident to explain | 24 |
| Right cause first (top-1) | **22 (92%)** |
| Right cause among the three best (top-3) | 22 (92%) |
| Wrong / cause unknown | 2 / 0 |
| Random faults given a where/when | **0 of 12** |
| Concentrated fault placed (`null_spike_vendor-20pct`) | vendor = 2 |

The two wrong ones are the one-vendor fault at 1% and 5%: on the test months that is 1 and 5
broken rows, too few to show a concentration, so the explainer falls back to the general cause
("with no part of the data standing out"). On the full data a 1% share of one vendor is
thousands of rows.

## On the full data (to confirm)

After `python src/gold/rebuild.py` (which investigates every incident) and the experiments:

```
python src/evaluation/run_experiments.py
python src/evaluation/explanation_accuracy.py
```

Open `notebooks/07_investigation.ipynb` and paste back the verdict in section 7. To check: the
causes given for the real incidents (for example the volume cold start of the first months and
the schema change between 2019 and 2023), whether any real incident is placed
in one vendor, zone or day, and the accuracy with 10% samples (about 300,000 trips a month).

## Known limits

- **Template-based.** The explainer knows only the causes in the playbooks; a new kind of
  problem gets its playbook's general cause or "cause unknown". That is by design: no cause is
  invented. Version 2 can add a language model on top of the same evidence.
- **Priors and weights are set by hand**, from the fault experiments on the test months; they
  are not learnt. The accuracy table is the check that they still work after a change.
- **One fault at a time.** The experiments break one thing per month; two faults in the same
  month (for example a schema change and a real drift) may share one explanation.
- **Where/when needs rows.** A concentration needs a segment of at least 20 rows and z ≥ 4;
  small problems stay "unknown" even when they are concentrated.
- **Month-level findings have no where/when** (schema, freshness, drift) except the low days of
  volume incidents, and those need 50 rows on the median day.
- **Actions are records, not automation.** Approving an action does not run it.
