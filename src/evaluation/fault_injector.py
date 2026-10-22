"""Inject known faults into a copy of a month, and record exactly what was broken.

    python src/evaluation/fault_injector.py --list                     # the planned experiments
    python src/evaluation/fault_injector.py --fault control --source FILE --out DIR

An experiment takes the target month's source file (configs/faults.yml), keeps the same
sample of its rows as every other experiment, breaks it with one fault at one strength, and
writes two files to the experiment folder:

    yellow_tripdata_<month>.parquet   the broken file, with the source file's schema unchanged
    fault_manifest.json               the ground truth: which rows were broken, which removed,
                                      and which detectors should fire

Rows are numbered as the pipeline numbers them: 1-based positions in a file, so an affected
row's position in the broken file is its bronze_row_id once loaded. Everything is
deterministic: the sample depends on the seed only, the broken rows on the seed and the
experiment, so the same config always gives the same files.

A fault is a function registered with @fault("name") in evaluation/faults.py. It gets the
sampled rows as a pandas frame (pyarrow types, so NULLs and timestamps survive) and returns a
FaultResult. `control` (nothing broken) is the baseline every experiment is compared with.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

if __package__ in (None, ""):
    # Allow running this file directly: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import (  # noqa: E402
    CHECK_CATEGORIES,
    CONFIG_DIR,
    PERIOD_PATTERN,
    REPO_ROOT,
    ConfigError,
    _parse_periods,
    _require,
    load_datasets_config,
    load_yaml,
)
from evaluation.faults import FAULTS, FaultResult, choose, fault  # noqa: E402, F401
from quality.rules import load_rules  # noqa: E402

KINDS = ("share", "days", "none")
CHECKS = ("schema", "volume", "freshness", "null_rate", "drift")
MANIFEST = "fault_manifest.json"
CONTROL = "control"


# --- configuration -------------------------------------------------------------------------


@dataclass(frozen=True)
class Expect:
    """A detector that should fire: a row rule, or a dataset check (optionally per column)."""

    kind: str  # rule | check | ml
    id: str
    column: str = ""


@dataclass(frozen=True)
class FaultSpec:
    id: str
    category: str
    kind: str
    description: str
    strengths: tuple[float | None, ...]
    params: dict[str, Any]
    expect: tuple[Expect, ...]


@dataclass(frozen=True)
class FaultsConfig:
    dataset: str
    target_month: str
    history: tuple[str, ...]
    sample_fraction: float
    seed: int
    workspace: Path
    faults: dict[str, FaultSpec]

    def experiments(self) -> list[tuple[str, str, float | None]]:
        """(experiment id, fault id, strength) for the control and every fault and strength."""
        plan = [(CONTROL, CONTROL, None)]
        for spec in self.faults.values():
            plan += [(experiment_id(spec, s), spec.id, s) for s in spec.strengths]
        return plan


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{where} must be a number, got {value!r}")
    return float(value)


def _strengths(raw: Any, kind: str, where: str) -> list[float]:
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{where} must be a non-empty list")
    values = [_number(v, where) for v in raw]
    for v in values:
        if kind == "share" and not 0 < v <= 1:
            raise ConfigError(f"{where}: a share must be above 0 and at most 1, got {v}")
        if kind == "days" and v <= 0:
            raise ConfigError(f"{where}: days must be above 0, got {v}")
    return values


def load_faults_config(path: Path | None = None) -> FaultsConfig:
    path = Path(path) if path else CONFIG_DIR / "faults.yml"
    data = load_yaml(path)
    where = path.name

    dataset = load_datasets_config().dataset(str(_require(data, "dataset", where)))
    target = str(_require(data, "target_month", where))
    if not PERIOD_PATTERN.match(target):
        raise ConfigError(f"{where}: target_month '{target}' must be YYYY-MM")
    history = _parse_periods(_require(data, "history", where), f"{where} history")
    if not history or history[-1] >= target:
        raise ConfigError(f"{where}: history must be a range of months before {target}")
    fraction = _number(_require(data, "sample_fraction", where), f"{where} sample_fraction")
    if not 0 < fraction <= 1:
        raise ConfigError(f"{where}: sample_fraction must be above 0 and at most 1")
    seed = _require(data, "seed", where)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError(f"{where}: seed must be a whole number of at least 0")
    defaults = _require(data, "strengths", where)
    default = {
        kind: _strengths(_require(defaults, kind, f"{where} strengths"), kind, f"strengths.{kind}")
        for kind in ("share", "days")
    }

    rule_ids = {r.id for r in load_rules(dataset, include_disabled=True)}
    faults: dict[str, FaultSpec] = {}
    for fault_id, raw in (_require(data, "faults", where) or {}).items():
        at = f"{where} faults.{fault_id}"
        if fault_id == CONTROL or not str(fault_id).replace("_", "").isalnum():
            raise ConfigError(f"{at}: invalid fault id")
        category = str(_require(raw, "category", at))
        if category not in CHECK_CATEGORIES:
            raise ConfigError(f"{at}: unknown category '{category}'")
        kind = str(_require(raw, "kind", at))
        if kind not in KINDS:
            raise ConfigError(f"{at}: kind must be one of {', '.join(KINDS)}")
        if kind == "none":
            if raw.get("extra_strengths"):
                raise ConfigError(f"{at}: a fault of kind none has no strengths")
            strengths: tuple[float | None, ...] = (None,)
        else:
            extra = (
                _strengths(raw["extra_strengths"], kind, f"{at}.extra_strengths")
                if raw.get("extra_strengths")
                else []
            )
            strengths = tuple(sorted(set(default[kind]) | set(extra)))
        expects = []
        for item in _require(raw, "expect", at) or []:
            if not isinstance(item, dict) or len({"rule", "check", "ml"} & set(item)) != 1:
                raise ConfigError(f"{at}.expect: each entry needs exactly one of rule, check or ml")
            if "ml" in item:
                if item["ml"] != "anomaly":
                    raise ConfigError(f"{at}.expect: the ml detector is 'anomaly'")
                expects.append(Expect("ml", "anomaly"))
            elif "rule" in item:
                if item["rule"] not in rule_ids:
                    raise ConfigError(f"{at}.expect: unknown rule '{item['rule']}'")
                expects.append(Expect("rule", str(item["rule"])))
            else:
                if item["check"] not in CHECKS:
                    raise ConfigError(f"{at}.expect: unknown check '{item['check']}'")
                expects.append(Expect("check", str(item["check"]), str(item.get("column", ""))))
        if not expects:
            raise ConfigError(f"{at}: expect needs at least one detector")
        params = raw.get("params") or {}
        if not isinstance(params, dict):
            raise ConfigError(f"{at}.params must be a mapping")
        faults[fault_id] = FaultSpec(
            fault_id,
            category,
            kind,
            str(raw.get("description", "")),
            strengths,
            params,
            tuple(expects),
        )
    if not faults:
        raise ConfigError(f"{where}: no faults configured")

    workspace = Path(str(_require(data, "workspace", where)))
    return FaultsConfig(
        dataset.name,
        target,
        history,
        fraction,
        seed,
        workspace if workspace.is_absolute() else REPO_ROOT / workspace,
        faults,
    )


# --- experiments and workspace -------------------------------------------------------------


def _strength_label(spec: FaultSpec, strength: float | None) -> str:
    if spec.kind == "none":
        return ""
    if spec.kind == "days":
        return f"{strength:g}d"
    return f"{100 * strength:g}pct"


def experiment_id(spec: FaultSpec, strength: float | None) -> str:
    """null_spike-5pct, stale_file-20d, missing_column."""
    label = _strength_label(spec, strength)
    return f"{spec.id}-{label}" if label else spec.id


def experiment_dir(config: FaultsConfig, exp_id: str) -> Path:
    return config.workspace / "experiments" / exp_id


def history_dir(config: FaultsConfig) -> Path:
    return config.workspace / "history"


# --- sampling and choosing rows ------------------------------------------------------------


def sample_positions(rows: int, fraction: float, seed: int) -> np.ndarray:
    """0-based positions of the rows kept, in file order: the same for every experiment."""
    if fraction >= 1:
        return np.arange(rows)
    keep = max(1, round(rows * fraction)) if rows else 0
    return np.sort(np.random.default_rng(seed).choice(rows, size=keep, replace=False))


def rng_for(config: FaultsConfig, exp_id: str) -> np.random.Generator:
    """A generator per experiment: reproducible, and different between experiments."""
    return np.random.default_rng([config.seed, zlib.crc32(exp_id.encode())])


# --- faults ------------------------------------------------------------------------------
# The fault functions and their registry live in evaluation/faults.py.

# --- injection ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Manifest:
    experiment_id: str
    fault_id: str
    category: str
    kind: str
    strength: float | None
    description: str
    dataset: str
    target_month: str
    sample_fraction: float
    seed: int
    source_file: str
    source_sha256: str
    output_file: str
    output_sha256: str
    rows_source: int
    rows_sampled: int
    rows_out: int
    month_level: bool
    affected_rows: list[int]  # 1-based positions in the output file (= bronze_row_id)
    removed_rows: list[int]  # 1-based positions in the sampled input that were dropped
    delivery_delay_days: float | None
    expect: list[dict[str, str]]
    note: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2) + "\n"

    @classmethod
    def read(cls, path: Path) -> Manifest:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        missing = sorted(set(cls.__dataclass_fields__) - set(data))
        if missing:
            raise ValueError(f"{path}: manifest misses {', '.join(missing)}")
        return cls(**{k: data[k] for k in cls.__dataclass_fields__})


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _to_arrow(frame: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    """Back to Arrow with the source file's types, for the columns still there."""
    kept = pa.schema([f for f in schema if f.name in frame.columns])
    if list(frame.columns) != kept.names:
        extra = sorted(set(frame.columns) - set(schema.names))
        if extra:
            raise ValueError(f"a fault added unknown column(s): {', '.join(extra)}")
        frame = frame[kept.names]
    return pa.Table.from_pandas(frame, schema=kept, preserve_index=False)


def _spec(config: FaultsConfig, fault_id: str) -> FaultSpec:
    if fault_id == CONTROL:
        return FaultSpec(CONTROL, "none", "none", "the month unbroken", (None,), {}, ())
    if fault_id not in config.faults:
        raise KeyError(f"fault '{fault_id}' is not in faults.yml")
    return config.faults[fault_id]


def inject(
    config: FaultsConfig,
    fault_id: str,
    strength: float | None,
    source: Path,
    out_dir: Path,
    period: str | None = None,
) -> Manifest:
    """Sample the source file, break it, write the file and its manifest to out_dir.

    period names the output file (default: the target month); the history months are written
    with fault 'control', so they hold the same sample as every experiment."""
    period = period or config.target_month
    spec = _spec(config, fault_id)
    if strength not in spec.strengths:
        raise ValueError(f"{fault_id}: strength {strength} is not configured {spec.strengths}")
    if fault_id not in FAULTS:
        raise KeyError(f"fault '{fault_id}' is not implemented")
    exp_id = experiment_id(spec, strength) if fault_id != CONTROL else CONTROL

    table = pq.read_table(source)
    positions = sample_positions(table.num_rows, config.sample_fraction, config.seed)
    sampled = table.take(pa.array(positions, type=pa.int64()))
    frame = sampled.to_pandas(types_mapper=pd.ArrowDtype).reset_index(drop=True)

    result = FAULTS[fault_id](frame.copy(), strength, dict(spec.params), rng_for(config, exp_id))
    rows_out = len(result.frame)
    for name, values, limit in (
        ("affected", result.affected, rows_out),
        ("removed", result.removed, len(frame)),
    ):
        if any(not 0 <= v < limit for v in values) or len(set(values)) != len(values):
            raise ValueError(f"{fault_id}: {name} positions must be unique and inside the frame")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / load_datasets_config().dataset(config.dataset).file_for(period)
    pq.write_table(_to_arrow(result.frame.reset_index(drop=True), table.schema), target)

    manifest = Manifest(
        experiment_id=exp_id,
        fault_id=fault_id,
        category=spec.category,
        kind=spec.kind,
        strength=strength,
        description=spec.description,
        dataset=config.dataset,
        target_month=period,
        sample_fraction=config.sample_fraction,
        seed=config.seed,
        source_file=Path(source).name,
        source_sha256=sha256_of(Path(source)),
        output_file=target.name,
        output_sha256=sha256_of(target),
        rows_source=table.num_rows,
        rows_sampled=len(frame),
        rows_out=rows_out,
        month_level=result.month_level,
        affected_rows=sorted(int(v) + 1 for v in result.affected),
        removed_rows=sorted(int(v) + 1 for v in result.removed),
        delivery_delay_days=result.delivery_delay_days,
        expect=[{k: v for k, v in asdict(e).items() if v} for e in spec.expect],
        note=result.note,
    )
    name = MANIFEST if period == config.target_month else f"fault_manifest_{period}.json"
    (out_dir / name).write_text(manifest.to_json(), encoding="utf-8")
    return manifest


def plan_table(config: FaultsConfig) -> pd.DataFrame:
    records = []
    for exp_id, fault_id, strength in config.experiments():
        spec = _spec(config, fault_id)
        records.append(
            {
                "experiment": exp_id,
                "category": spec.category,
                "strength": "" if strength is None else f"{strength:g}",
                "expect": ", ".join(
                    f"{e.kind} {e.id}" + (f" ({e.column})" if e.column else "") for e in spec.expect
                )
                or "-",
                "implemented": fault_id in FAULTS,
            }
        )
    return pd.DataFrame(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inject a known fault into a copy of a month.")
    parser.add_argument("--config", type=Path, help="faults config (default: configs/faults.yml)")
    parser.add_argument("--list", action="store_true", help="show the planned experiments")
    parser.add_argument("--fault", help="fault id, or control")
    parser.add_argument("--strength", type=float, help="share of rows or days (per the fault)")
    parser.add_argument("--source", type=Path, help="clean source file of the target month")
    parser.add_argument("--out", type=Path, help="experiment folder to write to")
    args = parser.parse_args(argv)
    try:
        config = load_faults_config(args.config)
    except ConfigError as exc:
        print(exc)
        return 2
    if args.list or not args.fault:
        plan = plan_table(config)
        print(
            f"target {config.target_month}, history {config.history[0]} to {config.history[-1]}, "
            f"sample {config.sample_fraction:g}, seed {config.seed}: {len(plan)} experiments"
        )
        with pd.option_context("display.width", 200, "display.max_colwidth", 60):
            print(plan.to_string(index=False))
        return 0
    if not args.source or not args.out:
        print("--fault needs --source and --out")
        return 2
    try:
        manifest = inject(config, args.fault, args.strength, args.source, args.out)
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    print(
        f"{manifest.experiment_id}: {manifest.rows_sampled} of {manifest.rows_source} rows "
        f"sampled, {len(manifest.affected_rows)} affected, {len(manifest.removed_rows)} removed "
        f"-> {Path(args.out) / manifest.output_file}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
