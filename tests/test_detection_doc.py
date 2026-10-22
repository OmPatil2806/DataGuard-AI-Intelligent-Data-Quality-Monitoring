"""docs/detection_evaluation.md covers every planned experiment and every fault."""

from common.config import REPO_ROOT
from evaluation import fault_injector as fi

DOC = (REPO_ROOT / "docs" / "detection_evaluation.md").read_text(encoding="utf-8")


def test_every_experiment_has_a_result_row():
    config = fi.load_faults_config()
    missing = [e for e, _, _ in config.experiments() if e != fi.CONTROL and f"| `{e}` |" not in DOC]
    assert not missing, f"add these experiments to the results table: {missing}"


def test_every_fault_is_described():
    for fault_id in fi.load_faults_config().faults:
        assert f"| `{fault_id}` |" in DOC, fault_id


def test_every_miss_is_explained():
    rows = [line for line in DOC.splitlines() if "🔴 missed" in line]
    faults = {line.split("`")[1].rsplit("-", 1)[0] for line in rows}
    explained = DOC.split("## Why each miss is missed", 1)[1]
    for fault_id in faults:
        assert f"`{fault_id}`" in explained, f"explain why {fault_id} is missed"
