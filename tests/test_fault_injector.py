"""The fault injection framework (Phase 6.1): config, sampling, injection and manifest."""

import copy
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml

from common.config import REPO_ROOT, ConfigError
from evaluation import fault_injector as fi
from evaluation.fault_injector import FaultResult, Manifest

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "yellow_tripdata_fixture.parquet"
RAW = yaml.safe_load((REPO_ROOT / "configs" / "faults.yml").read_text())


@pytest.fixture
def config():
    return fi.load_faults_config()


def write_config(tmp_path, change):
    data = copy.deepcopy(RAW)
    change(data)
    path = tmp_path / "faults.yml"
    path.write_text(yaml.safe_dump(data))
    return path


# --- configuration ---------------------------------------------------------------------------


def test_config_loads(config):
    assert config.dataset == "yellow_taxi" and config.target_month == "2023-06"
    assert config.history == ("2023-01", "2023-02", "2023-03", "2023-04", "2023-05")
    assert (config.sample_fraction, config.seed) == (0.1, 42)
    assert config.workspace == REPO_ROOT / "data" / "evaluation"
    assert len(config.faults) == 12
    assert {f.category for f in config.faults.values()} == {
        "schema", "completeness", "uniqueness", "validity", "referential",
        "drift", "volume", "freshness", "pattern", "consistency", "ml_anomaly",
    }  # fmt: skip
    assert config.faults["missing_column"].strengths == (None,)
    assert config.faults["null_spike"].strengths == (0.01, 0.05, 0.2)
    assert config.faults["volume_drop"].strengths == (0.01, 0.05, 0.2, 0.5)
    assert config.faults["unit_change"].strengths == (0.01, 0.05, 0.2, 1.0)
    assert config.faults["stale_file"].strengths == (1, 5, 20)
    assert config.faults["null_spike"].expect == (fi.Expect("check", "null_rate", "fare_amount"),)
    assert config.faults["duplicate_rows"].expect == (fi.Expect("rule", "trip_unique"),)
    assert config.faults["plausible_but_wrong"].expect == (fi.Expect("ml", "anomaly"),)
    assert all(f.causes for f in config.faults.values())  # every fault says what is right
    assert config.faults["null_spike"].where == "none"
    assert config.faults["null_spike_vendor"].where == {"dimension": "vendor", "value": "2"}
    assert config.faults["missing_column"].where is None


def test_experiment_plan(config):
    plan = config.experiments()
    assert plan[0] == ("control", "control", None)
    ids = [p[0] for p in plan]
    assert len(ids) == 37 == len(set(ids))
    for expected in ("missing_column", "null_spike-1pct", "volume_drop-50pct", "stale_file-20d",
                     "unit_change-100pct", "swapped_times-20pct"):  # fmt: skip
        assert expected in ids
    assert fi.experiment_dir(config, "null_spike-5pct") == (
        config.workspace / "experiments" / "null_spike-5pct"
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.update(target_month="2023-6"), "must be YYYY-MM"),
        (lambda d: d.update(history={"start": "2023-01", "end": "2023-06"}), "before 2023-06"),
        (lambda d: d.update(sample_fraction=0), "sample_fraction"),
        (lambda d: d.update(sample_fraction=1.5), "sample_fraction"),
        (lambda d: d.update(seed=-1), "seed"),
        (lambda d: d.update(seed=True), "seed"),
        (lambda d: d["strengths"].update(share=[0.5, 2]), "at most 1"),
        (lambda d: d["strengths"].update(days=[0]), "days must be above 0"),
        (lambda d: d["faults"]["null_spike"].update(category="nulls"), "unknown category"),
        (lambda d: d["faults"]["null_spike"].update(kind="percent"), "kind must be one of"),
        (lambda d: d["faults"]["missing_column"].update(extra_strengths=[0.5]), "kind none"),
        (lambda d: d["faults"]["null_spike"].update(expect=[{"rule": "no_rule"}]), "unknown rule"),
        (lambda d: d["faults"]["null_spike"].update(expect=[{"check": "speed"}]), "unknown check"),
        (
            lambda d: d["faults"]["null_spike"].update(
                expect=[{"rule": "trip_unique", "check": "volume"}]
            ),
            "exactly one of",
        ),
        (lambda d: d["faults"]["null_spike"].update(expect=[]), "at least one detector"),
        (lambda d: d["faults"]["null_spike"].update(expect=[{"ml": "forest"}]), "is 'anomaly'"),
        (lambda d: d["faults"].update(control=d["faults"]["null_spike"]), "invalid fault id"),
        (lambda d: d.update(faults={}), "no faults"),
        (
            lambda d: d["faults"]["null_spike"].update(explain={"causes": ["a_guess"]}),
            "not in configs/playbooks.yml",
        ),
        (lambda d: d["faults"]["null_spike"].update(explain={"causes": "x"}), "list of cause"),
        (
            lambda d: d["faults"]["null_spike"].update(explain={"where": {"dimension": "v"}}),
            "where must be",
        ),
        (lambda d: d["faults"]["null_spike"].update(explain=[1]), "must be a mapping"),
    ],
)
def test_config_is_validated(tmp_path, change, message):
    with pytest.raises(ConfigError, match=message):
        fi.load_faults_config(write_config(tmp_path, change))


def test_disabled_rule_can_be_expected(tmp_path):
    # the total-amount rule is disabled (open item O1) but may still be named as a detector
    path = write_config(
        tmp_path,
        lambda d: d["faults"]["out_of_range"].update(expect=[{"rule": "total_matches_parts"}]),
    )
    assert fi.load_faults_config(path).faults["out_of_range"].expect[0].id == "total_matches_parts"


# --- sampling and choosing -------------------------------------------------------------------


def test_sample_is_deterministic_sorted_and_sized():
    a = fi.sample_positions(1000, 0.1, 42)
    assert len(a) == 100 and (np.diff(a) > 0).all() and a.max() < 1000
    assert (a == fi.sample_positions(1000, 0.1, 42)).all()
    assert not (a == fi.sample_positions(1000, 0.1, 43)).all()
    assert (fi.sample_positions(5, 1.0, 1) == np.arange(5)).all()
    assert len(fi.sample_positions(5, 0.01, 1)) == 1  # never empty
    assert len(fi.sample_positions(0, 0.5, 1)) == 0


def test_choose(config):
    rng = fi.rng_for(config, "x")
    picked = fi.choose(np.arange(10, 210), 0.05, rng)
    assert len(picked) == 10 and set(picked) <= set(range(10, 210)) and (np.diff(picked) > 0).all()
    assert len(fi.choose(np.arange(10), 0.01, rng)) == 1  # at least one row
    assert len(fi.choose(np.arange(10), 1.0, rng)) == 10
    assert len(fi.choose([], 0.5, rng)) == 0
    same = fi.choose(np.arange(100), 0.2, fi.rng_for(config, "a"))
    assert (same == fi.choose(np.arange(100), 0.2, fi.rng_for(config, "a"))).all()
    assert not (same == fi.choose(np.arange(100), 0.2, fi.rng_for(config, "b"))).all()


# --- injection ---------------------------------------------------------------------------


def read(path):
    return duckdb.sql(f"SELECT * FROM read_parquet('{path}')").df()


def test_control_keeps_the_sampled_rows_unchanged(config, tmp_path):
    manifest = fi.inject(config, "control", None, FIXTURE, tmp_path)
    out = tmp_path / "yellow_tripdata_2023-06.parquet"
    assert pq.read_schema(out) == pq.read_schema(FIXTURE)
    positions = fi.sample_positions(200, 0.1, 42)
    expected = read(FIXTURE).iloc[positions].reset_index(drop=True)
    assert read(out).equals(expected)
    assert (manifest.rows_source, manifest.rows_sampled, manifest.rows_out) == (200, 20, 20)
    assert manifest.affected_rows == [] and manifest.removed_rows == []
    assert manifest.output_sha256 == fi.sha256_of(out)
    assert manifest.source_sha256 == fi.sha256_of(FIXTURE)
    assert Manifest.read(tmp_path / fi.MANIFEST) == manifest


def test_injection_is_reproducible(config, tmp_path):
    a = fi.inject(config, "control", None, FIXTURE, tmp_path / "a")
    b = fi.inject(config, "control", None, FIXTURE, tmp_path / "b")
    assert a.output_sha256 == b.output_sha256
    assert (tmp_path / "a" / fi.MANIFEST).read_text() == (tmp_path / "b" / fi.MANIFEST).read_text()


@pytest.fixture
def test_fault(monkeypatch, config):
    """A made-up fault: NULL fares in a share of rows, drop the first row and a column."""

    def broken(frame, strength, params, rng):
        rows = fi.choose(frame.index[1:], strength, rng)
        frame.loc[rows, params["column"]] = None
        out = frame.drop(index=0).drop(columns=["store_and_fwd_flag"]).reset_index(drop=True)
        return FaultResult(out, affected=[r - 1 for r in rows], removed=[0], note="test")

    monkeypatch.setitem(fi.FAULTS, "null_spike", broken)
    return broken


def test_affected_and_removed_rows_are_recorded(config, tmp_path, test_fault):
    manifest = fi.inject(config, "null_spike", 0.2, FIXTURE, tmp_path)
    out = read(tmp_path / "yellow_tripdata_2023-06.parquet")
    assert manifest.experiment_id == "null_spike-20pct"
    assert manifest.rows_out == 19 and len(out) == 19 and manifest.removed_rows == [1]
    assert len(manifest.affected_rows) == 4  # 20% of 19 eligible rows
    nulls = [i + 1 for i in out.index[out["fare_amount"].isna()]]
    assert manifest.affected_rows == nulls  # 1-based positions in the broken file
    schema = pq.read_schema(tmp_path / "yellow_tripdata_2023-06.parquet")
    source = pq.read_schema(FIXTURE)
    assert "store_and_fwd_flag" not in schema.names
    assert all(schema.field(n).type == source.field(n).type for n in schema.names)
    assert manifest.expect == [{"kind": "check", "id": "null_rate", "column": "fare_amount"}]


def test_fault_results_are_checked(config, tmp_path, monkeypatch):
    def outside(frame, strength, params, rng):
        return FaultResult(frame, affected=[len(frame)])

    def new_column(frame, strength, params, rng):
        frame["speed"] = 1.0
        return FaultResult(frame)

    monkeypatch.setitem(fi.FAULTS, "null_spike", outside)
    with pytest.raises(ValueError, match="inside the frame"):
        fi.inject(config, "null_spike", 0.05, FIXTURE, tmp_path)
    monkeypatch.setitem(fi.FAULTS, "null_spike", new_column)
    with pytest.raises(ValueError, match="unknown column"):
        fi.inject(config, "null_spike", 0.05, FIXTURE, tmp_path)


def test_unknown_or_unconfigured(config, tmp_path, monkeypatch):
    with pytest.raises(KeyError, match="not in faults.yml"):
        fi.inject(config, "speeding", 0.05, FIXTURE, tmp_path)
    with pytest.raises(ValueError, match="not configured"):
        fi.inject(config, "null_spike", 0.5, FIXTURE, tmp_path)
    monkeypatch.delitem(fi.FAULTS, "null_spike", raising=False)
    with pytest.raises(KeyError, match="not implemented"):
        fi.inject(config, "null_spike", 0.05, FIXTURE, tmp_path)


def test_manifest_must_be_complete(tmp_path):
    (tmp_path / "m.json").write_text('{"experiment_id": "x"}')
    with pytest.raises(ValueError, match="manifest misses"):
        Manifest.read(tmp_path / "m.json")


def test_main(tmp_path, capsys):
    assert fi.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "37 experiments" in out and "volume_drop-50pct" in out
    assert fi.main(["--fault", "control", "--source", str(FIXTURE), "--out", str(tmp_path)]) == 0
    assert "control: 20 of 200 rows sampled" in capsys.readouterr().out
    assert fi.main(["--fault", "control"]) == 2
    assert fi.main(["--fault", "speeding", "--source", str(FIXTURE), "--out", str(tmp_path)]) == 2
    bad = write_config(tmp_path, lambda d: d.update(seed=-1))
    assert fi.main(["--config", str(bad), "--list"]) == 2
