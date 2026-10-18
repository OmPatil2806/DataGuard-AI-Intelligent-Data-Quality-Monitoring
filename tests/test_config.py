import pytest

from common.config import (
    REPO_ROOT,
    ConfigError,
    expand_periods,
    load_datasets_config,
    load_quality_rules,
    load_severity_weights,
)


def write(tmp_path, text, name="config.yml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


VALID_DATASETS = """
paths:
  incoming: data/incoming
  bronze: data/bronze
  silver: data/silver
  quarantine: data/quarantine
  duckdb: data/dataguard.duckdb
datasets:
  yellow_taxi:
    description: taxi trips
    enabled: true
    format: parquet
    frequency: monthly
    url_template: https://example.com/yellow_{period}.parquet
    file_template: yellow_{period}.parquet
    timestamp_column: pickup
    periods: {start: "2023-11", end: "2024-02", extra: ["2019-01"]}
"""


def test_repository_configs_load():
    datasets = load_datasets_config()
    taxi = datasets.dataset("yellow_taxi")
    assert len(taxi.periods) == 13
    assert taxi.periods[0] == "2019-01"
    assert taxi.lookups[0].name == "taxi_zone"
    assert datasets.paths["bronze"] == REPO_ROOT / "data" / "bronze"

    assert load_quality_rules()["defaults"]["baseline"]["min_runs"] == 3
    assert load_severity_weights()["severity_levels"]["critical"] == 4


def test_url_and_file_for_period(tmp_path):
    taxi = load_datasets_config(write(tmp_path, VALID_DATASETS)).dataset("yellow_taxi")
    assert taxi.url_for("2023-01") == "https://example.com/yellow_2023-01.parquet"
    assert taxi.file_for("2023-01") == "yellow_2023-01.parquet"


def test_periods_cross_year_and_include_extra(tmp_path):
    taxi = load_datasets_config(write(tmp_path, VALID_DATASETS)).dataset("yellow_taxi")
    assert taxi.periods == ("2019-01", "2023-11", "2023-12", "2024-01", "2024-02")


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        ("2023-13", "2023-12", "invalid period '2023-13'"),
        ("2023-1", "2023-12", "invalid period '2023-1'"),
        ("2023-06", "2023-01", "is after end"),
    ],
)
def test_expand_periods_rejects_bad_input(start, end, message):
    with pytest.raises(ConfigError, match=message):
        expand_periods(start, end)


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(ConfigError, match="config file not found"):
        load_datasets_config(tmp_path / "nope.yml")


def test_invalid_yaml_gives_clear_error(tmp_path):
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_datasets_config(write(tmp_path, "paths: [unclosed"))


def test_missing_key_names_the_location(tmp_path):
    broken = VALID_DATASETS.replace("    format: parquet\n", "")
    with pytest.raises(ConfigError, match=r"missing required key 'format' in datasets.yellow_taxi"):
        load_datasets_config(write(tmp_path, broken))


def test_enabled_dataset_needs_url_template(tmp_path):
    broken = VALID_DATASETS.replace(
        "    url_template: https://example.com/yellow_{period}.parquet\n", ""
    )
    with pytest.raises(ConfigError, match="'url_template'.*dataset is enabled"):
        load_datasets_config(write(tmp_path, broken))


def test_unknown_dataset(tmp_path):
    config = load_datasets_config(write(tmp_path, VALID_DATASETS))
    with pytest.raises(ConfigError, match="unknown dataset 'green_taxi'"):
        config.dataset("green_taxi")


def test_drift_thresholds_must_be_ordered(tmp_path):
    text = """
defaults:
  baseline: {min_runs: 3}
  volume: {tolerance: 0.3}
  drift: {psi_medium: 0.3, psi_high: 0.2}
  incidents: {resolve_after_clean_runs: 2}
rules: {}
"""
    with pytest.raises(ConfigError, match="psi_medium < psi_high"):
        load_quality_rules(write(tmp_path, text))


def test_severity_weights_reject_unknown_category_and_bad_weight(tmp_path):
    levels = "severity_levels: {critical: 4, high: 3, medium: 2, low: 1}\n"
    with pytest.raises(ConfigError, match="unknown check categories speed"):
        load_severity_weights(write(tmp_path, levels + "category_weights: {speed: 1}\n"))
    with pytest.raises(ConfigError, match="'schema' must be a positive number"):
        load_severity_weights(write(tmp_path, levels + "category_weights: {schema: 0}\n"))
