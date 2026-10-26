"""fetch_archives.py (Phase 9.1): archives fetched, checked, unpacked and recorded."""

import json

import pandas as pd
import pytest
from test_download import FakeResponse, FakeSession
from ved_fixture import WEEKS, make_archives

from common.config import ConfigError, load_datasets_config
from data_download import fetch_archives as fa


@pytest.fixture
def source(tmp_path):
    return make_archives(tmp_path / "source")


def test_config_lists_the_archives():
    dataset = load_datasets_config().dataset("ved_signals")
    assert [a.kind for a in dataset.archives] == ["7z", "7z", "xlsx", "xlsx"]
    assert all(
        a.url.startswith("https://raw.githubusercontent.com/gsoh/VED/") for a in dataset.archives
    )
    assert dataset.archives[0].members == "VED_*_week.csv"


def test_copy_unpack_and_convert(source, tmp_path):
    incoming = tmp_path / "incoming"
    results = fa.fetch_all("ved_signals", incoming, source=source)
    assert [r.status for r in results] == ["copied"] * 4
    folder = incoming / "ved_signals"
    weeks = sorted(p.name for p in folder.glob("VED_*_week.csv"))
    assert weeks == [f"VED_{s}_week.csv" for s in WEEKS]
    assert not (folder / "readme.txt").exists()  # not a member matching VED_*_week.csv
    ice_hev = pd.read_csv(folder / "static_ice_hev.csv")
    assert "Vehicle Type" in ice_hev.columns and len(ice_hev) == 3
    assert "EngineType" in pd.read_csv(folder / "static_phev_ev.csv").columns
    manifest = json.loads((folder / fa.MANIFEST).read_text())
    first = manifest["archives"][0]
    assert first["sha256"] == fa.sha256(folder / "raw" / "VED_DynamicData_Part1.7z")
    assert first["outputs"] == [{"file": "VED_171101_week.csv",
                                 "size": (folder / "VED_171101_week.csv").stat().st_size,
                                 "status": "unpacked"}]  # fmt: skip


def test_again_skips_what_is_there(source, tmp_path):
    incoming = tmp_path / "incoming"
    fa.fetch_all("ved_signals", incoming, source=source)
    again = fa.fetch_all("ved_signals", incoming, source=source)
    assert [r.status for r in again] == ["skipped"] * 4
    assert again[0].outputs[0]["status"] == "skipped"
    (incoming / "ved_signals" / "VED_171101_week.csv").write_text("cut short")
    third = fa.fetch_all("ved_signals", incoming, source=source)
    assert third[0].outputs[0]["status"] == "unpacked"  # wrong size: unpacked again


def test_download_and_failures(source, tmp_path):
    files = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}

    def respond(url):
        name = url.rsplit("/", 1)[1].replace("%26", "&")
        return FakeResponse(files[name]) if "Part2" not in name else FakeResponse(b"<html>")

    results = fa.fetch_all("ved_signals", tmp_path / "in", session=FakeSession(default=respond))
    assert [r.status for r in results] == ["downloaded", "failed", "downloaded", "downloaded"]
    assert "not valid 7z" in results[1].error
    assert fa.main(["--dataset", "ved_signals", "--dest", str(tmp_path / "in2"),
                    "--source", str(tmp_path / "nowhere")]) == 1  # fmt: skip


def test_unknown_or_archive_less_dataset(tmp_path):
    with pytest.raises(ConfigError, match="no archives"):
        fa.fetch_all("yellow_taxi", tmp_path)
    assert fa.main(["--dataset", "yellow_taxi", "--dest", str(tmp_path)]) == 2


def test_archive_kind_is_validated(tmp_path):
    import yaml

    from common.config import REPO_ROOT

    data = yaml.safe_load((REPO_ROOT / "configs" / "datasets.yml").read_text())
    data["datasets"]["ved_signals"]["archives"]["dynamic_part1"]["kind"] = "rar"
    path = tmp_path / "datasets.yml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match="kind must be one of"):
        load_datasets_config(path)
