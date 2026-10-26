"""Fetch a dataset's archives (configs/datasets.yml, `archives`) and unpack them (Phase 9).

    python src/data_download/fetch_archives.py --dataset ved_signals
    python src/data_download/fetch_archives.py --dataset ved_signals --source ../VED/Data

Some sources are not one file per period: VED ships its weekly CSVs inside two 7z archives and
its vehicle data as Excel workbooks. For each archive of the dataset this:

    1 downloads it to <incoming>/<dataset>/raw/ (or copies it from --source), checks its
      signature and records its SHA-256
    2 7z:   unpacks the members matching `members` into <incoming>/<dataset>/
      xlsx: writes the first sheet as <incoming>/<dataset>/<archive name>.csv
      csv:  copies it there as <archive name>.csv
    3 writes <incoming>/<dataset>/fetch_manifest.json: every archive and the files it gave

Safe to repeat: an archive already in raw/ with a valid signature is not fetched again (unless
--force), and a member already unpacked with the right size is not unpacked again.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import logging
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import requests

if __package__ in (None, ""):
    # Allow running this file directly: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import Archive, ConfigError, load_datasets_config  # noqa: E402
from data_download.download_data import Target, download, is_valid_file  # noqa: E402

log = logging.getLogger("fetch_archives")
MANIFEST = "fetch_manifest.json"


@dataclass
class Fetched:
    archive: str
    file: str
    url: str
    kind: str
    status: str  # downloaded | copied | skipped | failed
    sha256: str | None = None
    size: int | None = None
    outputs: list[dict] = field(default_factory=list)  # {file, size, status}
    error: str | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(
    archive: Archive,
    raw_dir: Path,
    source: Path | None,
    session: requests.Session,
    force: bool = False,
) -> tuple[Path | None, str, str | None]:
    """The archive in raw_dir: (path, status, error)."""
    dest = raw_dir / archive.file
    raw_dir.mkdir(parents=True, exist_ok=True)
    if source is not None:
        original = source / archive.file
        if not original.is_file():
            return None, "failed", f"not in {source}: {archive.file}"
        if (
            force
            or not is_valid_file(dest, archive.kind)
            or dest.stat().st_size != (original.stat().st_size)
        ):
            tmp = dest.with_name(dest.name + ".part")
            shutil.copyfile(original, tmp)
            os.replace(tmp, dest)
            status = "copied"
        else:
            status = "skipped"
        if not is_valid_file(dest, archive.kind):
            return None, "failed", f"{archive.file} is not a valid {archive.kind} file"
        return dest, status, None
    result = download(Target(archive.url, dest, archive.kind), session, force=force)
    if result.status == "failed":
        return None, "failed", result.error
    return dest, result.status, None


def unpack_7z(path: Path, out_dir: Path, members: str | None) -> list[dict]:
    import py7zr

    with py7zr.SevenZipFile(path) as z:
        infos = [i for i in z.list() if not i.is_directory]
    wanted = [
        i for i in infos if members is None or fnmatch.fnmatch(Path(i.filename).name, members)
    ]
    outputs, todo = [], []
    for info in wanted:
        target = out_dir / Path(info.filename).name
        done = target.is_file() and target.stat().st_size == info.uncompressed
        outputs.append({"file": target.name, "size": info.uncompressed,
                        "status": "skipped" if done else "unpacked"})  # fmt: skip
        if not done:
            todo.append(info.filename)
    if todo:
        tmp = out_dir / f".unpack-{path.stem}"
        shutil.rmtree(tmp, ignore_errors=True)
        with py7zr.SevenZipFile(path) as z:
            z.extract(path=tmp, targets=todo)
        for name in todo:
            os.replace(tmp / name, out_dir / Path(name).name)
        shutil.rmtree(tmp, ignore_errors=True)
    return outputs


def to_csv(archive: Archive, path: Path, out_dir: Path) -> list[dict]:
    target = out_dir / f"{archive.name}.csv"
    tmp = target.with_name(target.name + ".part")
    if archive.kind == "xlsx":
        pd.read_excel(path, sheet_name=0).to_csv(tmp, index=False)
    else:
        shutil.copyfile(path, tmp)
    os.replace(tmp, target)
    return [{"file": target.name, "size": target.stat().st_size, "status": "written"}]


def fetch_all(
    dataset_name: str,
    incoming: Path,
    source: Path | None = None,
    session: requests.Session | None = None,
    force: bool = False,
    config_path: Path | None = None,
) -> list[Fetched]:
    dataset = load_datasets_config(config_path).dataset(dataset_name)
    if not dataset.archives:
        raise ConfigError(f"dataset '{dataset.name}' has no archives in datasets.yml")
    out_dir = incoming / dataset.name
    raw_dir = out_dir / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    session = session or requests.Session()
    results = []
    for archive in dataset.archives:
        path, status, error = fetch(archive, raw_dir, source, session, force)
        item = Fetched(archive.name, archive.file, archive.url, archive.kind, status, error=error)
        if path is not None:
            item.sha256, item.size = sha256(path), path.stat().st_size
            try:
                if archive.kind == "7z":
                    item.outputs = unpack_7z(path, out_dir, archive.members)
                else:
                    item.outputs = to_csv(archive, path, out_dir)
            except Exception as exc:  # a corrupt archive or workbook: report, keep going
                item.status, item.error = "failed", f"{type(exc).__name__}: {exc}"
        results.append(item)
        log.info(
            "%-10s %s%s", item.status, archive.file,
            f" -> {len(item.outputs)} file(s)" if item.outputs else f": {item.error}",
        )  # fmt: skip
    manifest = {"dataset": dataset.name, "archives": [asdict(r) for r in results]}
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return results


def main(argv: list[str] | None = None, session: requests.Session | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and unpack a dataset's archives.")
    parser.add_argument("--dataset", required=True, help="dataset name in datasets.yml")
    parser.add_argument("--source", type=Path, help="copy the archives from this folder")
    parser.add_argument("--dest", type=Path, help="incoming folder (default: paths.incoming)")
    parser.add_argument("--config", type=Path, help="path to datasets.yml")
    parser.add_argument("--force", action="store_true", help="fetch even if already there")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        incoming = args.dest or load_datasets_config(args.config).paths["incoming"]
        results = fetch_all(args.dataset, incoming, args.source, session, args.force, args.config)
    except ConfigError as exc:
        log.error("%s", exc)
        return 2
    failed = [r for r in results if r.status == "failed"]
    files = sum(len(r.outputs) for r in results)
    log.info("done: %d archive(s), %d file(s), %d failed -> %s",
             len(results), files, len(failed), incoming / args.dataset)  # fmt: skip
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
