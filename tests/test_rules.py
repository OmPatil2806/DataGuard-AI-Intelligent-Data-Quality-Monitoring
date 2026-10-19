import pytest
import yaml

from common.config import ConfigError, load_datasets_config
from quality.rules import CHECKS, load_rules

DEFAULTS = {
    "baseline": {"min_runs": 3},
    "volume": {"tolerance": 0.3},
    "drift": {"psi_medium": 0.1, "psi_high": 0.2},
    "incidents": {"resolve_after_clean_runs": 2},
}

# One valid example per check type.
VALID = [
    {
        "id": "trip_metadata_missing",
        "category": "completeness",
        "check": "not_null",
        "columns": ["passenger_count", "RatecodeID"],
        "mode": "all",
    },
    {
        "id": "fare_not_negative",
        "category": "validity",
        "check": "range",
        "column": "fare_amount",
        "min": 0,
    },
    {
        "id": "passengers_1_to_6",
        "category": "validity",
        "check": "range",
        "column": "passenger_count",
        "min": 0,
        "max": 6,
        "exclusive_min": True,
    },
    {
        "id": "known_payment_type",
        "category": "validity",
        "check": "allowed_values",
        "column": "payment_type",
        "values": [1, 2, 3, 4, 5, 6],
    },
    {
        "id": "zone_not_unknown",
        "category": "referential",
        "check": "forbidden_values",
        "column": "DOLocationID",
        "values": [264, 265],
    },
    {
        "id": "dropoff_after_pickup",
        "category": "consistency",
        "check": "compare",
        "left": "tpep_dropoff_datetime",
        "op": ">=",
        "right": "tpep_pickup_datetime",
    },
    {
        "id": "trip_under_24h",
        "category": "consistency",
        "check": "duration",
        "start": "tpep_pickup_datetime",
        "end": "tpep_dropoff_datetime",
        "max_seconds": 86400,
    },
    {
        "id": "pickup_in_month",
        "category": "pattern",
        "check": "within_period",
        "column": "tpep_pickup_datetime",
    },
    {
        "id": "pickup_zone_known",
        "category": "referential",
        "check": "foreign_key",
        "column": "PULocationID",
        "lookup": "taxi_zone",
        "lookup_column": "LocationID",
    },
    {
        "id": "trip_unique",
        "category": "uniqueness",
        "check": "unique",
        "columns": ["VendorID", "tpep_pickup_datetime", "fare_amount"],
    },
    {
        "id": "total_matches_parts",
        "category": "consistency",
        "check": "sum_equals",
        "total": "total_amount",
        "parts": ["fare_amount", "extra", "tip_amount"],
        "tolerance": 0.01,
        "alternatives": [["fare_amount", "tip_amount"]],
        "when": "VendorID = 1",
    },
]


def with_defaults(rule):
    return {"action": "quarantine", "severity": "high", "reason": "bad {fare_amount}", **rule}


def write(tmp_path, rules, dataset="yellow_taxi"):
    path = tmp_path / "quality_rules.yml"
    data = {"version": 1, "defaults": DEFAULTS, "rules": {dataset: rules}}
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def load(tmp_path, *rules):
    return load_rules("yellow_taxi", write(tmp_path, [with_defaults(r) for r in rules]))


def test_repository_rules_load():
    assert isinstance(load_rules("yellow_taxi"), list)


def test_every_check_type_has_a_valid_example():
    assert {r["check"] for r in VALID} == set(CHECKS)


def test_valid_rules_load_in_order(tmp_path):
    rules = load(tmp_path, *VALID)
    assert [r.id for r in rules] == [r["id"] for r in VALID]
    total = rules[-1]
    assert total.when == "VendorID = 1"
    assert total.params["alternatives"] == [["fare_amount", "tip_amount"]]
    assert total.columns == ["total_amount", "fare_amount", "extra", "tip_amount"]
    assert rules[0].action == "quarantine" and rules[0].severity == "high"


def test_disabled_rules_are_skipped_unless_asked(tmp_path):
    rules = [with_defaults(VALID[1]), with_defaults({**VALID[2], "enabled": False})]
    path = write(tmp_path, rules)
    assert [r.id for r in load_rules("yellow_taxi", path)] == ["fare_not_negative"]
    assert len(load_rules("yellow_taxi", path, include_disabled=True)) == 2


def test_dataset_without_rules(tmp_path):
    assert load_rules("ved_signals", write(tmp_path, [])) == []


BROKEN = [
    ({**VALID[1], "id": "Fare-Rule"}, "'id' must be snake_case"),
    ({**VALID[1], "column": "fare"}, "unknown column 'fare'"),
    ({**VALID[1], "check": "between"}, "unknown check 'between'"),
    ({**VALID[1], "category": "freshness"}, "category must be one of"),
    ({**VALID[1], "action": "drop"}, "action must be quarantine or warn"),
    ({**VALID[1], "severity": "urgent"}, "severity must be one of"),
    ({k: v for k, v in VALID[1].items() if k != "column"}, "check 'range' needs column"),
    ({**VALID[1], "maximum": 5}, "unknown key\\(s\\) for check 'range': maximum"),
    ({**VALID[1], "min": "zero"}, "'min' must be a number"),
    ({**VALID[1], "min": 10, "max": 5}, "'min' is greater than 'max'"),
    (
        {"id": "r", "category": "validity", "check": "range", "column": "fare_amount"},
        "range needs 'min' and/or 'max'",
    ),
    ({**VALID[1], "column": "store_and_fwd_flag"}, "range needs a numeric column"),
    ({**VALID[0], "mode": "some"}, "'mode' must be 'any' or 'all'"),
    ({**VALID[0], "columns": []}, "non-empty list"),
    ({**VALID[5], "op": "=>"}, "'op' must be one of"),
    ({**VALID[6], "start": "fare_amount"}, "must be a timestamp column"),
    ({**VALID[7], "column": "trip_distance"}, "must be a timestamp column"),
    ({**VALID[8], "lookup": "zones"}, "unknown lookup 'zones'"),
    ({**VALID[9], "columns": ["VendorID", "VendorID"]}, "lists a column twice"),
    ({**VALID[10], "tolerance": -1}, "'tolerance' must be 0 or more"),
    ({**VALID[10], "alternatives": [[]]}, "'alternatives' must be a list"),
    ({**VALID[4], "values": []}, "'values' must be a non-empty list"),
    ({**VALID[1], "enabled": "yes"}, "'enabled' must be true or false"),
]


@pytest.mark.parametrize(("rule", "message"), BROKEN)
def test_broken_rules_give_clear_errors(tmp_path, rule, message):
    with pytest.raises(ConfigError, match=message):
        load(tmp_path, rule)


def test_error_names_the_rule(tmp_path):
    with pytest.raises(ConfigError, match=r"rules\.yellow_taxi\.fare_not_negative: unknown column"):
        load(tmp_path, {**VALID[1], "column": "fare"})


def test_missing_common_key(tmp_path):
    rule = {k: v for k, v in with_defaults(VALID[1]).items() if k != "severity"}
    with pytest.raises(ConfigError, match="missing required key 'severity'"):
        load_rules("yellow_taxi", write(tmp_path, [rule]))


def test_reason_placeholders_must_be_columns(tmp_path):
    with pytest.raises(ConfigError, match="reason uses '\\{fare\\}'"):
        load(tmp_path, {**VALID[1], "reason": "fare {fare} is negative"})


def test_duplicate_ids(tmp_path):
    with pytest.raises(ConfigError, match="'fare_not_negative' is used twice"):
        load(tmp_path, VALID[1], VALID[1])


def test_rules_must_be_a_list(tmp_path):
    path = tmp_path / "quality_rules.yml"
    path.write_text(yaml.safe_dump({"defaults": DEFAULTS, "rules": {"yellow_taxi": {"a": 1}}}))
    with pytest.raises(ConfigError, match="must be a list of rules"):
        load_rules(load_datasets_config().dataset("yellow_taxi"), path)
