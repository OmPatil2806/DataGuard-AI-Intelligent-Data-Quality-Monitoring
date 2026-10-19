"""Diagnose open item O1: why total_amount differs from the sum of its parts.

    python src/profiling/diagnose_totals.py

Prints, and writes to data/o1_total_diagnosis.txt:
  1. per VendorID over all files: how many rows match the full sum, only the sum without
     congestion_surcharge and airport_fee, or neither
  2. the same per file
  3. the most common differences (total - sum of all parts) for rows that do not match
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Allow running this file directly: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import load_datasets_config  # noqa: E402
from profiling import taxi_profile as tp  # noqa: E402


def build_report(files: list[Path], top: int = 10) -> str:
    with tp.connect() as con:
        per_file, diffs = tp.total_diagnosis(con, files, top=top)

    overall = per_file.groupby("vendor")[["full", "base", "none", "rows"]].sum()
    for name in ("full", "base", "none"):
        overall[f"{name}_pct"] = (100 * overall[name] / overall["rows"]).round(2)
    overall_diffs = (
        diffs.groupby(["vendor", "difference"], as_index=False)["rows"]
        .sum()
        .sort_values(["vendor", "rows"], ascending=[True, False])
        .groupby("vendor")
        .head(top)
    )
    sections = [
        ("1. ALL FILES BY VENDOR", overall.reset_index()),
        ("2. PER FILE AND VENDOR", per_file),
        ("3. MOST COMMON DIFFERENCES (total - sum of all parts), ALL FILES", overall_diffs),
    ]
    return "\n".join(
        f"\n===== {title} =====\n{frame.to_string(index=False)}" for title, frame in sections
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose total_amount vs. its parts (O1).")
    parser.add_argument("--source", type=Path, help="input folder (default: paths.incoming)")
    parser.add_argument(
        "--out", type=Path, help="report file (default: data/o1_total_diagnosis.txt)"
    )
    parser.add_argument("--top", type=int, default=10, help="differences to show per vendor")
    args = parser.parse_args(argv)

    config = load_datasets_config()
    source = args.source or config.paths["incoming"]
    files = sorted(source.glob("yellow_tripdata_*.parquet"))
    if not files:
        print(f"no yellow_tripdata_*.parquet files in {source}: run download_data.py first")
        return 1

    report = build_report(files, top=args.top)
    out = args.out or config.paths["data_dir"] / "o1_total_diagnosis.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.lstrip() + "\n", encoding="utf-8")
    print(report)
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
