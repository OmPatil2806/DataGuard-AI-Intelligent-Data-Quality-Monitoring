"""Run the detection experiments: a clean history once, then every fault on the target month.

    python src/evaluation/run_experiments.py                      # every experiment in faults.yml
    python src/evaluation/run_experiments.py --only control,null_spike-5pct
    python src/evaluation/run_experiments.py --fresh              # rebuild the history too

Source files come from data/incoming (or --source): the history months, the target month and
the zone lookup. Everything is written under the workspace (configs/faults.yml, default
data/evaluation/), never to the real pipeline's database or folders:

    history/      the history months, sampled like every experiment, through the whole pipeline
                  (ingest, then rebuild), then the anomaly model trained on them (configs/ml.yml,
                  saved in history/models/) and the history scored with it: built once and
                  reused while the configs and the source files are unchanged (history.json
                  holds their fingerprint)
    experiments/<id>/
        incoming/              the broken target month and fault_manifest.json
        dataguard.duckdb       a copy of the history database, then the target month added:
                               ingest, delivery delay (stale_file), rebuild --months <target>
                               (its ml stage scores the month with the history's model)
        bronze/, silver/, quarantine/   the target month's files (history files are shared)
        experiment.json        what ran: stage exit codes and seconds
        log.txt                the output of every stage

Every experiment starts from the same copy of the history, so the control and the faults are
compared on equal terms. A stage failing on a broken month (Silver rejecting a file without a
required column, say) is a result, not an error: it is recorded and the later stages still
run. Exit code: 0 when every experiment ran, 1 when one could not run, 2 for invalid input.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    # Allow running this file directly: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import db  # noqa: E402
from common.config import ConfigError, config_dir, load_datasets_config  # noqa: E402
from evaluation import fault_injector as fi  # noqa: E402
from gold import monitoring_tables, rebuild  # noqa: E402
from incidents import incidents_cli  # noqa: E402
from ingestion import ingest_data  # noqa: E402
from ml import score as ml_score  # noqa: E402
from ml import train as ml_train  # noqa: E402
from ml.features import MODELS_DIR_ENV, load_ml_config  # noqa: E402
from quality import score as quality_score  # noqa: E402

DATABASE = "dataguard.duckdb"
RECORD = "experiment.json"


class ExperimentError(RuntimeError):
    """Raised when an experiment cannot be set up (not when a stage finds the fault)."""


@dataclass
class ExperimentRecord:
    experiment_id: str
    fault_id: str
    strength: float | None
    stages: dict[str, int] = field(default_factory=dict)  # stage -> exit code
    seconds: dict[str, float] = field(default_factory=dict)
    delivery_delay_days: float | None = None
    error: str | None = None

    @property
    def ran(self) -> bool:
        return self.error is None and all(code != 2 for code in self.stages.values())

    def write(self, folder: Path) -> None:
        (folder / RECORD).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, folder: Path) -> ExperimentRecord:
        return cls(**json.loads((Path(folder) / RECORD).read_text(encoding="utf-8")))


# --- sources ---------------------------------------------------------------------------------


def source_files(config: fi.FaultsConfig, source: Path) -> dict[str, Path]:
    """Month (or lookup name) -> source file; raises when one is missing."""
    dataset = load_datasets_config().dataset(config.dataset)
    files = {m: source / dataset.file_for(m) for m in (*config.history, config.target_month)}
    # lookup files keep their path below the source folder (e.g. ved_signals/static_ice_hev.csv)
    files |= {
        f"lookup:{lookup.name}:{name}": source / name
        for lookup in dataset.lookups
        for name in lookup.paths
    }
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise ExperimentError("missing source file(s): " + ", ".join(missing))
    return files


def fingerprint(config: fi.FaultsConfig, files: dict[str, Path]) -> str:
    """What the history depends on: the config's history settings and the source files."""
    parts = {
        "dataset": config.dataset,
        "history": list(config.history),
        "target_month": config.target_month,
        "sample_fraction": config.sample_fraction,
        "seed": config.seed,
        "files": {
            key: fi.sha256_of(path)
            for key, path in sorted(files.items())
            if key != config.target_month
        },
        "ml_config": fi.sha256_of(config_dir() / "ml.yml"),
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


# --- running -------------------------------------------------------------------------------


@contextlib.contextmanager
def _models(history: Path):
    """Point the anomaly model's relative models_dir at the history's own models folder."""
    previous = os.environ.get(MODELS_DIR_ENV)
    os.environ[MODELS_DIR_ENV] = str(history / "models")
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(MODELS_DIR_ENV, None)
        else:
            os.environ[MODELS_DIR_ENV] = previous


def _train(folder: Path, record: ExperimentRecord) -> None:
    """Train the anomaly model on the history and score it, so the target month of every
    experiment is compared with the history's flag rates. A model that cannot be trained (the
    history lacks the train or calibration months) leaves the experiments without ML."""
    database = ["--database", str(folder / DATABASE)]
    log = folder / "log.txt"
    for name, main, argv in (
        ("train", ml_train.main, database),
        ("ml", ml_score.main, [*database, "--if-trained"]),
        ("incidents", incidents_cli.main, [*database, "update"]),
        ("score", quality_score.main, database),
        ("gold", monitoring_tables.main, database),
    ):
        start = time.perf_counter()
        record.stages[f"{name}_after_training"] = _quiet(log, main, argv)
        record.seconds[f"{name}_after_training"] = round(time.perf_counter() - start, 2)
        if name == "train" and record.stages["train_after_training"] != 0:
            return


def _quiet(log: Path, function, *args):
    """Run a stage with its printed output going to the experiment's log."""
    with open(log, "a", encoding="utf-8") as handle, contextlib.redirect_stdout(handle):
        return function(*args)


def _rebuild(
    dataset: str, folder: Path, lookups: Path, months: str | None, record: ExperimentRecord
) -> None:
    argv = [
        "--dataset", dataset, "--lookups", str(lookups),
        "--silver", str(folder / "silver"), "--quarantine", str(folder / "quarantine"),
        "--reports", str(folder / "reports"), "--database", str(folder / DATABASE),
        "--no-alerts",  # experiments never notify anyone
    ]  # fmt: skip
    if months:
        argv += ["--months", months]
    results = _quiet(folder / "log.txt", rebuild.run, rebuild.parse_args(argv))
    for r in results:
        record.stages[r.name] = r.exit_code
        record.seconds[r.name] = round(r.seconds, 2)


def _ingest(
    dataset: str, folder: Path, incoming: Path, months: str, record: ExperimentRecord
) -> None:
    start = time.perf_counter()
    argv = [
        "--dataset", dataset, "--months", months, "--source", str(incoming),
        "--bronze", str(folder / "bronze"), "--database", str(folder / DATABASE),
    ]  # fmt: skip
    record.stages["ingest"] = _quiet(folder / "log.txt", ingest_data.main, argv)
    record.seconds["ingest"] = round(time.perf_counter() - start, 2)
    if record.stages["ingest"] != 0:
        raise ExperimentError(f"ingest failed with exit code {record.stages['ingest']}")


def build_history(
    config: fi.FaultsConfig, files: dict[str, Path], fresh: bool = False
) -> tuple[Path, bool]:
    """The sampled history months through the whole pipeline. Returns (folder, built now)."""
    folder = fi.history_dir(config)
    stamp = fingerprint(config, files)
    marker = folder / "history.json"
    if not fresh and marker.is_file() and (folder / DATABASE).is_file():
        if json.loads(marker.read_text(encoding="utf-8")).get("fingerprint") == stamp:
            return folder, False
    shutil.rmtree(folder, ignore_errors=True)
    incoming = folder / "incoming"
    incoming.mkdir(parents=True)
    for key, path in files.items():
        if key.startswith("lookup:"):
            target = incoming / key.split(":", 2)[2]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, target)
        elif key != config.target_month:
            fi.inject(config, fi.CONTROL, None, path, incoming, period=key)
    record = ExperimentRecord("history", fi.CONTROL, None)
    months = f"{config.history[0]}:{config.history[-1]}"
    with _models(folder):
        _ingest(config.dataset, folder, incoming, months, record)
        _rebuild(config.dataset, folder, incoming, None, record)
        if load_ml_config().dataset == config.dataset:  # the model is for one dataset only
            _train(folder, record)
    record.write(folder)
    if not record.ran:
        raise ExperimentError(f"history did not build: {record.stages}")
    marker.write_text(json.dumps({"fingerprint": stamp}, indent=2) + "\n", encoding="utf-8")
    return folder, True


def set_delivery_delay(database: Path, config: fi.FaultsConfig, days: float) -> None:
    """Count the target month's file as delivered `days` after it was expected."""
    with db.connect(database) as con:
        rows = con.execute(
            "UPDATE file_manifest SET simulated_delivered_at = expected_at + ? "
            "WHERE dataset = ? AND period = ? AND status = 'LOADED' AND expected_at IS NOT NULL "
            "RETURNING period",
            [timedelta(days=days), config.dataset, config.target_month],
        ).fetchall()
    if not rows:
        raise ExperimentError(f"{config.target_month} has no expected delivery time to delay")


def run_experiment(
    config: fi.FaultsConfig,
    exp_id: str,
    fault_id: str,
    strength: float | None,
    source: Path,
    history: Path,
) -> ExperimentRecord:
    folder = fi.experiment_dir(config, exp_id)
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    record = ExperimentRecord(exp_id, fault_id, strength)
    try:
        shutil.copy(history / DATABASE, folder / DATABASE)
        manifest = fi.inject(config, fault_id, strength, source, folder / "incoming")
        _ingest(config.dataset, folder, folder / "incoming", config.target_month, record)
        if manifest.delivery_delay_days is not None:
            set_delivery_delay(folder / DATABASE, config, manifest.delivery_delay_days)
            record.delivery_delay_days = manifest.delivery_delay_days
        with _models(history):
            _rebuild(config.dataset, folder, history / "incoming", config.target_month, record)
    except (ExperimentError, ValueError, KeyError, OSError) as exc:
        record.error = f"{type(exc).__name__}: {exc}"
    record.write(folder)
    return record


def run_all(
    config: fi.FaultsConfig,
    source: Path,
    only: list[str] | None = None,
    fresh: bool = False,
    out=print,
) -> list[ExperimentRecord]:
    plan = config.experiments()
    if only:
        unknown = sorted(set(only) - {exp_id for exp_id, _, _ in plan})
        if unknown:
            raise ExperimentError(f"unknown experiment(s): {', '.join(unknown)}")
        plan = [p for p in plan if p[0] in only]
    files = source_files(config, source)
    start = time.perf_counter()
    history, built = build_history(config, files, fresh)
    out(f"history {config.history[0]} to {config.history[-1]}: "
        f"{'built' if built else 'reused'} ({time.perf_counter() - start:.1f}s)")  # fmt: skip
    records = []
    for i, (exp_id, fault_id, strength) in enumerate(plan, start=1):
        start = time.perf_counter()
        record = run_experiment(
            config, exp_id, fault_id, strength, files[config.target_month], history
        )
        failed = [s for s, code in record.stages.items() if code != 0]
        status = (
            f"ERROR {record.error}"
            if record.error
            else ("ok" if not failed else "stage(s) failed: " + ", ".join(failed))
        )
        out(f"[{i:>2}/{len(plan)}] {exp_id:<28} {time.perf_counter() - start:6.1f}s  {status}")
        records.append(record)
    return records


def summary(records: list[ExperimentRecord]) -> dict[str, Any]:
    return {
        "experiments": len(records),
        "ran": sum(r.ran for r in records),
        "errors": {r.experiment_id: r.error for r in records if not r.ran},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the fault injection experiments.")
    parser.add_argument("--config", type=Path, help="faults config (default: configs/faults.yml)")
    parser.add_argument(
        "--source", type=Path, help="folder with the source files (default: incoming)"
    )
    parser.add_argument("--only", help="comma-separated experiment ids (default: all)")
    parser.add_argument("--fresh", action="store_true", help="rebuild the history as well")
    args = parser.parse_args(argv)
    try:
        config = fi.load_faults_config(args.config)
        source = args.source or load_datasets_config().paths["incoming"]
        only = [e.strip() for e in args.only.split(",")] if args.only else None
        records = run_all(config, Path(source), only, args.fresh)
    except (ConfigError, ExperimentError) as exc:
        print(f"error: {exc}")
        return 2
    result = summary(records)
    print(f"\n{result['ran']} of {result['experiments']} experiment(s) ran"
          f" -> {config.workspace / 'experiments'}")  # fmt: skip
    for exp_id, error in result["errors"].items():
        print(f"  {exp_id}: {error}")
    return 0 if not result["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
