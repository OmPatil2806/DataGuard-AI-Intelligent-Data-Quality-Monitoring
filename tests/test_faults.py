"""The ten faults (Phase 6.2): each changes exactly the rows it reports, and nothing else.

The faults run on every row of the fixture (sample_fraction 1.0), against the control output.
For the faults caught by a row rule, the rule is evaluated on the broken file: it must flag
every broken row and no row that the control does not already flag.
"""

import copy
import shutil
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
import yaml

from common.config import REPO_ROOT, load_datasets_config
from evaluation import fault_injector as fi
from evaluation.faults import DROPOFF, PICKUP
from quality import row_rules
from quality.rules import load_rules

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FIXTURE = FIXTURES / "yellow_tripdata_fixture.parquet"
FILE = "yellow_tripdata_2023-06.parquet"


@pytest.fixture(scope="module")
def config(tmp_path_factory):
    data = yaml.safe_load((REPO_ROOT / "configs" / "faults.yml").read_text())
    data = copy.deepcopy(data) | {"sample_fraction": 1.0}
    path = tmp_path_factory.mktemp("cfg") / "faults.yml"
    path.write_text(yaml.safe_dump(data))
    return fi.load_faults_config(path)


@pytest.fixture(scope="module")
def run(config, tmp_path_factory):
    """Inject once per experiment and keep (manifest, frame)."""
    root = tmp_path_factory.mktemp("faults")
    cache = {}

    def get(exp_id):
        if exp_id not in cache:
            fault_id, strength = next((f, s) for e, f, s in config.experiments() if e == exp_id)
            manifest = fi.inject(config, fault_id, strength, FIXTURE, root / exp_id)
            frame = duckdb.sql(f"SELECT * FROM read_parquet('{root / exp_id / FILE}')").df()
            cache[exp_id] = (manifest, frame, root / exp_id / FILE)
        return cache[exp_id]

    return get


def changed_rows(a: pd.DataFrame, b: pd.DataFrame, columns=None) -> list[int]:
    columns = columns or list(a.columns)
    same = (a[columns] == b[columns]) | (a[columns].isna() & b[columns].isna())
    return sorted(np.flatnonzero(~same.all(axis=1)).tolist())


def changed_columns(a, b, rows):
    out = set()
    for column in a.columns:
        x, y = a.loc[rows, column], b.loc[rows, column]
        if not ((x == y) | (x.isna() & y.isna())).all():
            out.add(column)
    return out


def test_every_configured_fault_is_implemented(config):
    assert set(config.faults) <= set(fi.FAULTS)
    assert fi.plan_table(config)["implemented"].all()


def test_every_experiment_runs_and_is_reproducible(config, run, tmp_path):
    for exp_id, fault_id, strength in config.experiments():
        manifest, frame, _ = run(exp_id)
        assert manifest.rows_out == len(frame)
        again = fi.inject(config, fault_id, strength, FIXTURE, tmp_path / exp_id)
        assert again.output_sha256 == manifest.output_sha256, exp_id


ROW_FAULTS = {
    # fault: (columns it may change, share of which rows)
    "null_spike": {"fare_amount"},
    "out_of_range": {"fare_amount"},
    "orphan_location": {"PULocationID"},
    "unit_change": {"trip_distance"},
    "pickup_outside_month": {PICKUP, DROPOFF},
    "swapped_times": {PICKUP, DROPOFF},
}


@pytest.mark.parametrize("fault_id", sorted(ROW_FAULTS))
@pytest.mark.parametrize("pct", ["1pct", "5pct", "20pct"])
def test_row_faults_change_only_the_reported_rows_and_columns(run, fault_id, pct):
    _, control, _ = run("control")
    manifest, frame, _ = run(f"{fault_id}-{pct}")
    assert len(frame) == len(control) == 200 and manifest.removed_rows == []
    affected = [r - 1 for r in manifest.affected_rows]
    assert changed_rows(control, frame) == affected
    assert changed_columns(control, frame, affected) == ROW_FAULTS[fault_id]
    assert not manifest.month_level and manifest.delivery_delay_days is None


@pytest.mark.parametrize(("pct", "share"), [("1pct", 0.01), ("5pct", 0.05), ("20pct", 0.2)])
def test_share_is_a_share_of_the_eligible_rows(run, pct, share):
    _, control, _ = run("control")
    manifest, _, _ = run(f"out_of_range-{pct}")
    eligible = int((control["fare_amount"] > 0).sum())
    assert len(manifest.affected_rows) == max(1, round(share * eligible))


def test_null_spike(run):
    _, control, _ = run("control")
    manifest, frame, _ = run("null_spike-20pct")
    rows = [r - 1 for r in manifest.affected_rows]
    assert frame.loc[rows, "fare_amount"].isna().all()
    assert control.loc[rows, "fare_amount"].notna().all()  # only values that existed


def test_out_of_range_orphan_and_unit_change(run):
    _, control, _ = run("control")
    m, frame, _ = run("out_of_range-5pct")
    rows = [r - 1 for r in m.affected_rows]
    assert (frame.loc[rows, "fare_amount"] == -control.loc[rows, "fare_amount"]).all()
    assert (frame.loc[rows, "fare_amount"] < 0).all()
    m, frame, _ = run("orphan_location-5pct")
    assert (frame.loc[[r - 1 for r in m.affected_rows], "PULocationID"] == 999).all()
    m, frame, _ = run("unit_change-100pct")
    rows = [r - 1 for r in m.affected_rows]
    assert len(rows) == int((control["trip_distance"] > 0).sum())
    ratio = frame.loc[rows, "trip_distance"] / control.loc[rows, "trip_distance"]
    assert np.allclose(ratio, 1.609344)


def test_pickup_outside_month_keeps_the_duration(run):
    _, control, _ = run("control")
    m, frame, _ = run("pickup_outside_month-20pct")
    rows = [r - 1 for r in m.affected_rows]
    before, after = control.loc[rows, PICKUP], frame.loc[rows, PICKUP]
    previous = (before.dt.to_period("M") - 1).astype(str)
    assert (after.dt.to_period("M").astype(str) == previous).all()
    assert (after.dt.time == before.dt.time).all()
    duration = lambda f: f.loc[rows, DROPOFF] - f.loc[rows, PICKUP]  # noqa: E731
    assert (duration(frame) == duration(control)).all()


def test_swapped_times(run):
    _, control, _ = run("control")
    m, frame, _ = run("swapped_times-5pct")
    rows = [r - 1 for r in m.affected_rows]
    assert (frame.loc[rows, PICKUP] == control.loc[rows, DROPOFF]).all()
    assert (frame.loc[rows, DROPOFF] == control.loc[rows, PICKUP]).all()
    assert (frame.loc[rows, DROPOFF] < frame.loc[rows, PICKUP]).all()


@pytest.mark.parametrize(("pct", "copies"), [("1pct", 2), ("5pct", 10), ("20pct", 40)])
def test_duplicate_rows_follow_their_originals(run, pct, copies):
    _, control, _ = run("control")
    m, frame, _ = run(f"duplicate_rows-{pct}")
    assert m.rows_out == 200 + copies and len(m.affected_rows) == copies
    for row in m.affected_rows:  # 1-based: the copy is row, its original row - 1
        assert changed_rows(frame.iloc[[row - 2]].reset_index(drop=True),
                            frame.iloc[[row - 1]].reset_index(drop=True)) == []  # fmt: skip
    rest = frame.drop(index=[r - 1 for r in m.affected_rows]).reset_index(drop=True)
    assert changed_rows(control, rest) == []


@pytest.mark.parametrize(("pct", "removed"), [("1pct", 2), ("20pct", 40), ("50pct", 100)])
def test_volume_drop(run, pct, removed):
    _, control, _ = run("control")
    m, frame, _ = run(f"volume_drop-{pct}")
    assert m.month_level and m.affected_rows == [] and len(m.removed_rows) == removed
    kept = control.drop(index=[r - 1 for r in m.removed_rows]).reset_index(drop=True)
    assert changed_rows(kept, frame) == []


def test_missing_column(run):
    _, control, _ = run("control")
    m, frame, path = run("missing_column")
    assert m.month_level and m.affected_rows == []
    assert "passenger_count" not in frame.columns
    assert changed_rows(control.drop(columns=["passenger_count"]), frame) == []
    assert (
        "passenger_count"
        not in duckdb.sql(f"DESCRIBE SELECT * FROM '{path}'").df()["column_name"].tolist()
    )


@pytest.mark.parametrize("days", [1, 5, 20])
def test_stale_file_changes_no_data(run, days):
    control_manifest, _, _ = run("control")
    m, _, _ = run(f"stale_file-{days}d")
    assert m.month_level and m.delivery_delay_days == days
    assert m.output_sha256 == control_manifest.output_sha256


def test_missing_parameter_column_is_an_error(config, tmp_path, monkeypatch):
    spec = config.faults["null_spike"]
    broken = fi.FaultsConfig(
        **{**config.__dict__, "faults": {**config.faults, "null_spike": spec.__class__(
            **{**spec.__dict__, "params": {"column": "no_such_column"}})}}
    )  # fmt: skip
    with pytest.raises(ValueError, match="no_such_column"):
        fi.inject(broken, "null_spike", 0.05, FIXTURE, tmp_path)


# --- the expected row rule fires on exactly the broken rows ---------------------------------


@pytest.fixture(scope="module")
def flagged(tmp_path_factory):
    taxi = load_datasets_config().dataset("yellow_taxi")
    rules = {r.id: r for r in load_rules(taxi)}
    lookups = tmp_path_factory.mktemp("lookups")
    shutil.copy(FIXTURES / "taxi_zone_lookup_fixture.csv", lookups / "taxi_zone_lookup.csv")
    con = duckdb.connect()
    row_rules.load_lookups(con, taxi, lookups)

    def rows_failing(path, rule_id):
        # the fixture's trips are January 2023, so the file month is 2023-01 here
        # numbered as Bronze numbers rows: the unique rule fails repeats after the first
        source = (
            f"(SELECT * EXCLUDE (file_row_number), file_row_number + 1 AS bronze_row_id "
            f"FROM read_parquet('{path}', file_row_number = true))"
        )
        sql = row_rules.evaluate_sql(con, [rules[rule_id]], source, "2023-01")
        frame = con.execute(sql).df()
        return set(frame.loc[frame["failed_rules"].map(len) > 0, "bronze_row_id"].tolist())

    return rows_failing


RULE_FAULTS = ["duplicate_rows", "out_of_range", "orphan_location", "pickup_outside_month",
               "swapped_times"]  # fmt: skip


@pytest.mark.parametrize("fault_id", RULE_FAULTS)
@pytest.mark.parametrize("pct", ["1pct", "5pct", "20pct"])
def test_expected_rule_flags_every_broken_row(config, run, flagged, fault_id, pct):
    (rule,) = [e.id for e in config.faults[fault_id].expect]
    _, _, control_path = run("control")
    manifest, _, path = run(f"{fault_id}-{pct}")
    broken = set(manifest.affected_rows)
    found = flagged(path, rule)
    assert broken <= found, f"{rule} missed rows {sorted(broken - found)}"
    already = flagged(control_path, rule)
    if fault_id == "duplicate_rows":  # copies shift later rows: compare by content instead
        assert len(found) == len(broken) + len(already)
    else:
        assert found - broken <= already  # no new false flags
