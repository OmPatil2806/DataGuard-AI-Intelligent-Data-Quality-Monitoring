import json
from pathlib import Path

import duckdb
import pandera.errors
import pytest
import yaml

from common.config import load_datasets_config
from quality import row_rules
from quality.row_rules import RuleError, check_schema, count_failures, evaluate_sql
from quality.rules import load_rules
from quality.standardize import standardize

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TRIPS = FIXTURES / "yellow_tripdata_fixture.parquet"
ISSUES = json.loads((FIXTURES / "yellow_tripdata_fixture_issues.json").read_text())
PARTS = [
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "congestion_surcharge",
    "airport_fee",
]

# One rule per planted problem in the fixture (answer key: yellow_tripdata_fixture_issues.json).
RULES = {
    "zero_passengers": {
        "category": "validity",
        "check": "range",
        "column": "passenger_count",
        "min": 0,
        "exclusive_min": True,
        "action": "warn",
    },
    "null_group": {
        "category": "completeness",
        "check": "not_null",
        "mode": "all",
        "columns": [
            "passenger_count",
            "RatecodeID",
            "store_and_fwd_flag",
            "congestion_surcharge",
            "airport_fee",
        ],
        "action": "warn",
    },
    "negative_fare": {"category": "validity", "check": "range", "column": "fare_amount", "min": 0},
    "zero_distance": {
        "category": "validity",
        "check": "range",
        "column": "trip_distance",
        "min": 0,
        "exclusive_min": True,
        "action": "warn",
    },
    "dropoff_before_pickup": {
        "category": "consistency",
        "check": "compare",
        "left": "tpep_dropoff_datetime",
        "op": ">=",
        "right": "tpep_pickup_datetime",
    },
    "pickup_outside_month": {
        "category": "pattern",
        "check": "within_period",
        "column": "tpep_pickup_datetime",
    },
    "unknown_location_id": {
        "category": "referential",
        "check": "foreign_key",
        "column": "PULocationID",
        "lookup": "taxi_zone",
        "lookup_column": "LocationID",
    },
    "unknown_zone_264_265": {
        "category": "referential",
        "check": "range",
        "column": "DOLocationID",
        "max": 263,
        "action": "warn",
    },
    "total_mismatch": {
        "category": "consistency",
        "check": "sum_equals",
        "total": "total_amount",
        "parts": PARTS,
        "tolerance": 0.01,
    },
    "duplicate_row": {
        "category": "uniqueness",
        "check": "unique",
        "columns": [c for c, _ in load_datasets_config().dataset("yellow_taxi").columns],
    },
}


@pytest.fixture
def taxi():
    return load_datasets_config().dataset("yellow_taxi")


@pytest.fixture
def con(taxi, tmp_path):
    with duckdb.connect() as connection:
        lookups = tmp_path / "lookups"
        lookups.mkdir()
        (lookups / "taxi_zone_lookup.csv").write_bytes(
            (FIXTURES / "taxi_zone_lookup_fixture.csv").read_bytes()
        )
        row_rules.load_lookups(connection, taxi, lookups)
        yield connection


@pytest.fixture
def source(con, taxi):
    """The fixture as it comes out of Bronze and standardization (with bronze_row_id)."""
    bronze_like = (
        f"(SELECT * EXCLUDE (file_row_number), file_row_number + 1 AS bronze_row_id "
        f"FROM read_parquet('{TRIPS}', file_row_number = true))"
    )
    sql, _ = standardize(con, bronze_like, taxi)
    return f"({sql})"


def make_rules(tmp_path, specs):
    rules = [
        {
            "id": rule_id,
            "severity": "high",
            "reason": f"{rule_id} failed",
            "action": "quarantine",
            **spec,
        }
        for rule_id, spec in specs.items()
    ]
    path = tmp_path / "quality_rules.yml"
    defaults = {
        "baseline": {"min_runs": 3},
        "volume": {"tolerance": 0.3},
        "drift": {"psi_medium": 0.1, "psi_high": 0.2},
        "incidents": {"resolve_after_clean_runs": 2},
    }
    path.write_text(yaml.safe_dump({"defaults": defaults, "rules": {"yellow_taxi": rules}}))
    return load_rules("yellow_taxi", path)


@pytest.fixture
def evaluated(con, source, tmp_path):
    rules = make_rules(tmp_path, RULES)
    sql = evaluate_sql(con, rules, source, period="2023-01")
    return con.execute(f"SELECT * FROM ({sql}) ORDER BY bronze_row_id").df(), rules


@pytest.mark.parametrize("issue", sorted(ISSUES))
def test_each_rule_flags_exactly_the_planted_rows(evaluated, issue):
    df, _ = evaluated
    flagged = [
        int(r) - 1
        for r, failed in zip(df["bronze_row_id"], df["failed_rules"], strict=True)
        if issue in failed
    ]
    assert flagged == ISSUES[issue]


def test_clean_rows_fail_nothing(evaluated):
    df, _ = evaluated
    planted = {r for rows in ISSUES.values() for r in rows}
    clean = df[~(df["bronze_row_id"] - 1).isin(planted)]
    assert len(clean) == 200 - len(planted)
    assert all(len(f) == 0 for f in clean["failed_rules"])
    assert not clean["quarantine"].any()


def test_output_keeps_every_row_and_column(evaluated, taxi):
    df, _ = evaluated
    assert len(df) == 200
    expected = [c for c, _ in taxi.columns] + [
        "bronze_row_id",
        "failed_rules",
        "reasons",
        "quarantine",
    ]
    assert list(df.columns) == expected


def test_quarantine_only_for_quarantine_rules(evaluated):
    df, rules = evaluated
    actions = {r.id: r.action for r in rules}
    for failed, quarantined in zip(df["failed_rules"], df["quarantine"], strict=True):
        assert quarantined == any(actions[f] == "quarantine" for f in failed)
    warn_only = df[df["failed_rules"].apply(lambda f: list(f) == ["zero_passengers"])]
    assert len(warn_only) == 3 and not warn_only["quarantine"].any()


def test_reasons_line_up_with_failed_rules(evaluated):
    df, _ = evaluated
    for failed, reasons in zip(df["failed_rules"], df["reasons"], strict=True):
        assert list(reasons) == [f"{rule_id} failed" for rule_id in failed]


def test_count_failures_matches_evaluation(con, source, tmp_path):
    rules = make_rules(tmp_path, RULES)
    counts = {c.rule_id: c for c in count_failures(con, rules, source, period="2023-01")}
    for issue, rows in ISSUES.items():
        assert counts[issue].failed == len(rows), issue
        assert counts[issue].rows == 200
    assert counts["negative_fare"].pct == 1.5


def test_reason_placeholders_show_row_values(con, source, tmp_path):
    rules = make_rules(
        tmp_path,
        {
            "negative_fare": {
                **RULES["negative_fare"],
                "reason": "fare_amount {fare_amount} below 0 (payment {payment_type})",
            }
        },
    )
    sql = evaluate_sql(con, rules, source, period="2023-01")
    reasons = con.execute(f"SELECT reasons[1] FROM ({sql}) WHERE len(failed_rules) > 0").fetchall()
    assert len(reasons) == 3
    assert all(r[0].startswith("fare_amount -") and r[0].endswith("(payment 4)") for r in reasons)


def test_when_limits_a_rule(con, source, tmp_path):
    rules = make_rules(
        tmp_path, {"negative_fare_vendor_1": {**RULES["negative_fare"], "when": "VendorID = 1"}}
    )
    sql = evaluate_sql(con, rules, source, period="2023-01")
    vendors = con.execute(f"SELECT VendorID FROM ({sql}) WHERE quarantine").fetchall()
    expected = con.execute(
        f"SELECT count_if(fare_amount < 0 AND VendorID = 1) FROM {source}"
    ).fetchone()[0]
    assert len(vendors) == expected and all(v == (1,) for v in vendors)


def test_null_values_pass_value_rules(con, source, tmp_path):
    rules = make_rules(tmp_path, {"passengers_positive": RULES["zero_passengers"]})
    sql = evaluate_sql(con, rules, source, period="2023-01")
    nulls = con.execute(
        f"SELECT count(*) FROM ({sql}) WHERE passenger_count IS NULL AND len(failed_rules) > 0"
    ).fetchone()[0]
    assert nulls == 0


def test_sum_alternatives_and_other_checks(con, tmp_path):
    rows = """(VALUES
        (1, 10.0, 2.5, 2.5, 15.0, 'Y', TIMESTAMP '2023-01-01 10:00', TIMESTAMP '2023-01-01 10:00'),
        (2, 10.0, 2.5, 2.5, 12.5, 'N', TIMESTAMP '2023-01-01 10:00', TIMESTAMP '2023-01-02 11:00'),
        (2, 10.0, 0.0, 0.0, 99.0, 'X', TIMESTAMP '2023-01-01 10:00', TIMESTAMP '2023-01-01 10:20')
    ) t(VendorID, fare_amount, extra, congestion_surcharge, total_amount, store_and_fwd_flag,
        tpep_pickup_datetime, tpep_dropoff_datetime)"""
    specs = {
        "total_ok": {
            "category": "consistency",
            "check": "sum_equals",
            "total": "total_amount",
            "parts": ["fare_amount", "extra", "congestion_surcharge"],
            "alternatives": [["fare_amount", "extra"]],
        },
        "flag_y_or_n": {
            "category": "validity",
            "check": "allowed_values",
            "column": "store_and_fwd_flag",
            "values": ["Y", "N"],
        },
        "under_24h": {
            "category": "consistency",
            "check": "duration",
            "max_seconds": 86400,
            "start": "tpep_pickup_datetime",
            "end": "tpep_dropoff_datetime",
        },
        "has_duration": {
            "category": "consistency",
            "check": "compare",
            "left": "tpep_dropoff_datetime",
            "op": "!=",
            "right": "tpep_pickup_datetime",
        },
    }
    sql = evaluate_sql(con, make_rules(tmp_path, specs), f"(SELECT * FROM {rows})")
    failed = [list(r[0]) for r in con.execute(f"SELECT failed_rules FROM ({sql})").fetchall()]
    # row 1: total 15.0 = 10 + 2.5 + 2.5 ; zero duration
    # row 2: total 12.5 matches the alternative (fare + extra) ; trip of 25 hours
    # row 3: total 99 matches neither ; flag 'X'
    assert sorted(map(sorted, failed)) == sorted(
        map(sorted, [["has_duration"], ["under_24h"], ["total_ok", "flag_y_or_n"]])
    )


def test_within_period_needs_a_period(con, source, tmp_path):
    rules = make_rules(tmp_path, {"in_month": RULES["pickup_outside_month"]})
    with pytest.raises(RuleError, match="needs the file's period"):
        evaluate_sql(con, rules, source)


def test_no_rules(con, source):
    sql = evaluate_sql(con, [], source)
    row = con.execute(
        f"SELECT count(*), max(len(failed_rules)), bool_or(quarantine) FROM ({sql})"
    ).fetchone()
    assert row == (200, 0, False)
    assert count_failures(con, [], source) == []


def test_lookup_keeps_text_na(con):
    boroughs = con.execute("SELECT Borough FROM lookup_taxi_zone WHERE LocationID = 265").fetchall()
    assert boroughs == [("N/A",)]


def test_missing_lookup_file(taxi, tmp_path):
    with (
        duckdb.connect() as connection,
        pytest.raises(FileNotFoundError, match="lookup file not found"),
    ):
        row_rules.load_lookups(connection, taxi, tmp_path)


def test_check_schema_accepts_standardized_data(con, source, taxi):
    check_schema(con, f"SELECT * FROM {source}", taxi)


def test_check_schema_rejects_raw_data(con, taxi):
    with pytest.raises(pandera.errors.SchemaError, match="VendorID.*int32"):
        check_schema(con, f"SELECT * FROM read_parquet('{TRIPS}')", taxi)
