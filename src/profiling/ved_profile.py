"""Profile the Vehicle Energy Dataset before it is monitored (Phase 9.1).

    python src/data_download/fetch_archives.py --dataset ved_signals   # unpack the archives
    python src/profiling/ved_profile.py                                # profile them

Reads the weekly CSVs and the vehicles' static data from <incoming>/ved_signals/ and measures
what the configuration of Phase 9.2 to 9.5 needs: the files and how to read them, the columns
and their ranges, NULLs by engine type, vehicles and trips, duplicates, how DayNum and
Timestamp(ms) give a time, rows per month, GPS area and signal consistency.

Every weekly file is read with pyarrow and its row count is checked against the file's own
line count: the VED files end their header with CR LF and their rows with LF, and DuckDB's CSV
reader (1.5) then silently skips the file's first 32 MB. The rows are cached as Parquet in
<data_dir>/profile/ved_signals/ and profiled with DuckDB from there.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

if __package__ in (None, ""):
    # Allow running this file directly: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import load_datasets_config  # noqa: E402

DATASET = "ved_signals"
WEEK_FILE = re.compile(r"^VED_(\d{6})_week\.csv$")
REFERENCE = "2017-11-01"  # DayNum 1 = 2017-11-01 00:00 (README of the dataset)
KEY = ["VehId", "Trip", "Timestamp(ms)"]
STATIC_FILES = ("static_ice_hev.csv", "static_phev_ev.csv")
# The two static files name the engine type column differently.
ENGINE_TYPE_NAMES = ("EngineType", "Vehicle Type")


@dataclass
class WeekFile:
    file: str
    week_start: str  # from the file name, read as yymmdd
    size_bytes: int
    data_lines: int  # lines after the header, counted in the file
    rows_read: int  # rows pyarrow read: must equal data_lines
    header_crlf: bool
    rows_crlf: bool


def folders() -> tuple[Path, Path]:
    """The unpacked files and the profile cache (env overrides for tests and notebooks)."""
    paths = load_datasets_config().paths
    source = Path(os.environ.get("DATAGUARD_VED_DIR", paths["incoming"] / DATASET))
    cache = Path(os.environ.get("DATAGUARD_VED_CACHE", paths["data_dir"] / "profile" / DATASET))
    return source, cache


def week_files(source: Path) -> list[Path]:
    return sorted(p for p in source.glob("VED_*_week.csv") if WEEK_FILE.match(p.name))


def _line_endings(path: Path) -> tuple[bool, bool, int]:
    """Header ends with CR LF, a row ends with CR LF, and the number of data lines."""
    with path.open("rb") as f:
        header = f.readline()
        rows_crlf, lines = False, 0
        for line in f:
            lines += 1
            if not rows_crlf and line.endswith(b"\r\n"):
                rows_crlf = True
    return header.endswith(b"\r\n"), rows_crlf, lines


def read_week(path: Path, cache: Path) -> WeekFile:
    """Read one weekly CSV into the Parquet cache (once) and check that no row was lost."""
    header_crlf, rows_crlf, lines = _line_endings(path)
    target = cache / path.with_suffix(".parquet").name
    if not target.is_file() or target.stat().st_mtime < path.stat().st_mtime:
        table = pacsv.read_csv(
            path,
            convert_options=pacsv.ConvertOptions(null_values=["NaN", ""]),
        )
        cache.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".parquet.part")
        pq.write_table(table.append_column("file", [[path.name] * table.num_rows]), tmp)
        os.replace(tmp, target)
    rows = pq.ParquetFile(target).metadata.num_rows
    stamp = WEEK_FILE.match(path.name).group(1)
    week = f"20{stamp[:2]}-{stamp[2:4]}-{stamp[4:]}"
    return WeekFile(path.name, week, path.stat().st_size, lines, rows, header_crlf, rows_crlf)


def static_frame(source: Path) -> pd.DataFrame:
    """Both static files with one engine type column, and where each vehicle came from."""
    frames = []
    for name in STATIC_FILES:
        path = source / name
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        engine = next((c for c in ENGINE_TYPE_NAMES if c in frame.columns), None)
        frame = frame.rename(columns={engine: "EngineType"}) if engine else frame
        frames.append(frame.assign(static_file=name))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def connect(cache: Path, static: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(
        f"CREATE VIEW ved AS SELECT * FROM read_parquet('{(cache / '*.parquet').as_posix()}')"
    )
    con.register("static_raw", static if not static.empty else pd.DataFrame({"VehId": []}))
    con.execute("CREATE VIEW static AS SELECT * FROM static_raw")
    engine = "s.EngineType" if "EngineType" in static.columns else "NULL"
    # A row's time: DayNum is the trip's start (days since REFERENCE - 1), Timestamp(ms) the
    # time since the trip started.
    con.execute(
        f"""
        CREATE VIEW rows AS
        SELECT v.*, coalesce({engine}, 'unknown') AS engine_type,
               TIMESTAMP '{REFERENCE}' + to_microseconds(CAST(round((v.DayNum - 1) * 86400e6
                   + v."Timestamp(ms)" * 1000) AS BIGINT)) AS row_time
        FROM ved v LEFT JOIN static s USING (VehId)
        """
    )
    return con


# --- the measurements ---------------------------------------------------------------------------


def files_table(con, files: list[WeekFile]) -> pd.DataFrame:
    frame = pd.DataFrame([asdict(f) for f in files])
    days = con.execute(
        "SELECT file, min(DayNum) AS first_daynum, max(DayNum) AS last_daynum, "
        "min(row_time) AS first_time, max(row_time) AS last_time, count(DISTINCT VehId) AS "
        "vehicles, count(DISTINCT (VehId, Trip)) AS trips FROM rows GROUP BY 1"
    ).df()
    return frame.merge(days, on="file", how="left")


def columns_table(con) -> pd.DataFrame:
    names = [r[0] for r in con.execute("DESCRIBE ved").fetchall() if r[0] != "file"]
    types = dict(con.execute("SELECT column_name, column_type FROM (DESCRIBE ved)").fetchall())
    parts = []
    for name in names:
        q = f'"{name}"'
        row = con.execute(
            f"SELECT count(*), count({q}), min({q}), quantile_cont({q}, 0.01), median({q}), "
            f"quantile_cont({q}, 0.99), max({q}), count(DISTINCT {q}) FROM ved"
        ).fetchone()
        n, present, lo, p01, p50, p99, hi, distinct = row
        parts.append({
            "column": name, "type": types[name], "null_share": round(1 - present / n, 6),
            "min": lo, "p01": p01, "median": p50, "p99": p99, "max": hi, "distinct": distinct,
        })  # fmt: skip
    return pd.DataFrame(parts)


def nulls_by_engine(con) -> pd.DataFrame:
    names = [r[0] for r in con.execute("DESCRIBE ved").fetchall() if r[0] != "file"]
    shares = ", ".join(f'round(1 - count("{n}") / count(*), 4) AS "{n}"' for n in names)
    return con.execute(
        f"SELECT engine_type, count(*) AS rows, {shares} FROM rows GROUP BY 1 ORDER BY 1"
    ).df()


def vehicles_table(con) -> dict[str, object]:
    dynamic = {r[0] for r in con.execute("SELECT DISTINCT VehId FROM ved").fetchall()}
    static = {r[0] for r in con.execute("SELECT VehId FROM static").fetchall()}
    by_type = dict(
        con.execute(
            "SELECT engine_type, count(DISTINCT VehId) FROM rows GROUP BY 1 ORDER BY 1"
        ).fetchall()
    )
    static_types = dict(
        con.execute("SELECT EngineType, count(*) FROM static GROUP BY 1 ORDER BY 1").fetchall()
        if static
        else []
    )
    return {
        "vehicles_with_rows": len(dynamic),
        "vehicles_in_static_data": len(static),
        "static_ids_repeated": int(
            con.execute("SELECT count(*) - count(DISTINCT VehId) FROM static").fetchone()[0]
        ),
        "with_rows_not_in_static": sorted(dynamic - static),
        "in_static_without_rows": len(static - dynamic),
        "vehicles_with_rows_by_engine_type": by_type,
        "static_vehicles_by_engine_type": static_types,
    }


def static_quality(static: pd.DataFrame) -> pd.DataFrame:
    """Placeholders ('NO DATA') and NULLs per column of the static data."""
    rows = []
    for column in static.columns:
        if column == "static_file":
            continue
        values = static[column]
        rows.append({
            "column": column,
            "no_data": int((values.astype(str).str.upper() == "NO DATA").sum()),
            "null": int(values.isna().sum()),
            "distinct": int(values.nunique()),
        })  # fmt: skip
    return pd.DataFrame(rows)


def trips_table(con) -> dict[str, object]:
    row = con.execute(
        """
        WITH t AS (
            SELECT VehId, Trip, count(*) AS rows, count(DISTINCT DayNum) AS daynums,
                   count(DISTINCT file) AS files,
                   max("Timestamp(ms)") / 60000.0 AS minutes
            FROM ved GROUP BY 1, 2
        )
        SELECT count(*), quantile_cont(rows, 0.5), quantile_cont(rows, 0.99), max(rows),
               count_if(daynums > 1), count_if(files > 1), quantile_cont(minutes, 0.5),
               quantile_cont(minutes, 0.99), max(minutes)
        FROM t
        """
    ).fetchone()
    keys = (
        "trips", "rows_median", "rows_p99", "rows_max", "trips_with_more_than_one_daynum",
        "trips_in_more_than_one_file", "minutes_median", "minutes_p99", "minutes_max",
    )  # fmt: skip
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in zip(keys, row, strict=True)}


def duplicates(con) -> dict[str, int]:
    key = ", ".join(f'"{k}"' for k in KEY)
    groups, extra = con.execute(
        f"SELECT count(*), coalesce(sum(n - 1), 0) FROM (SELECT {key}, count(*) AS n FROM ved "
        "GROUP BY ALL HAVING count(*) > 1)"
    ).fetchone()
    exact = con.execute(
        "SELECT coalesce(sum(n - 1), 0) FROM (SELECT * EXCLUDE (file), count(*) AS n FROM ved "
        "GROUP BY ALL HAVING count(*) > 1)"
    ).fetchone()[0]
    return {"key_groups_repeated": int(groups), "extra_rows_on_key": int(extra),
            "extra_rows_identical_in_every_column": int(exact)}  # fmt: skip


def months_table(con) -> pd.DataFrame:
    return con.execute(
        """
        SELECT strftime(row_time, '%Y-%m') AS month, count(*) AS rows,
               count(DISTINCT VehId) AS vehicles, count(DISTINCT (VehId, Trip)) AS trips,
               count(DISTINCT CAST(row_time AS DATE)) AS days
        FROM rows GROUP BY 1 ORDER BY 1
        """
    ).df()


def outside_week(con, files: list[WeekFile]) -> dict[str, int]:
    """Rows whose time falls outside the 7 days that start at their file's date."""
    starts = pd.DataFrame(
        {
            "file": [f.file for f in files],
            "week_start": pd.to_datetime([f.week_start for f in files]),
        }
    )
    con.register("weeks", starts)
    before, after = con.execute(
        """
        SELECT count_if(r.row_time < w.week_start),
               count_if(r.row_time >= w.week_start + INTERVAL 7 DAY)
        FROM rows r JOIN weeks w USING (file)
        """
    ).fetchone()
    return {"rows_before_week_start": int(before), "rows_after_week_end": int(after)}


def gps_table(con) -> dict[str, object]:
    row = con.execute(
        """
        SELECT quantile_cont("Latitude[deg]", [0.001, 0.5, 0.999]),
               quantile_cont("Longitude[deg]", [0.001, 0.5, 0.999]),
               min("Latitude[deg]"), max("Latitude[deg]"),
               min("Longitude[deg]"), max("Longitude[deg]")
        FROM ved
        """
    ).fetchone()
    return {
        "latitude_p0.1_p50_p99.9": [round(v, 4) for v in row[0]],
        "longitude_p0.1_p50_p99.9": [round(v, 4) for v in row[1]],
        "latitude_min_max": [row[2], row[3]],
        "longitude_min_max": [row[4], row[5]],
    }


def consistency(con) -> pd.DataFrame:
    """Signals that should agree, by engine type."""
    return con.execute(
        """
        SELECT engine_type, count(*) AS rows,
               round(count_if("Vehicle Speed[km/h]" > 0) / count(*), 4) AS moving,
               round(count_if("Vehicle Speed[km/h]" > 0 AND coalesce("Engine RPM[RPM]", 0) = 0)
                     / greatest(count_if("Vehicle Speed[km/h]" > 0), 1), 4)
                   AS moving_with_rpm_0_or_null,
               round(count_if("Fuel Rate[L/hr]" IS NOT NULL) / count(*), 4) AS fuel_rate_present,
               round(count_if("HV Battery SOC[%]" IS NOT NULL) / count(*), 4) AS soc_present,
               round(count_if("Air Conditioning Power[kW]" IS NOT NULL) / count(*), 4)
                   AS ac_kw_present,
               round(count_if("Air Conditioning Power[Watts]" IS NOT NULL) / count(*), 4)
                   AS ac_watts_present,
               round(count_if("Vehicle Speed[km/h]" > 200) / count(*), 6) AS speed_over_200,
               round(count_if("HV Battery SOC[%]" NOT BETWEEN 0 AND 100) / count(*), 6)
                   AS soc_outside_0_100
        FROM rows GROUP BY 1 ORDER BY 1
        """
    ).df()


def profile(source: Path, cache: Path) -> dict[str, object]:
    """Every measurement, as DataFrames and dicts (see the notebook and docs/ved_profile.md)."""
    weeks = week_files(source)
    if not weeks:
        raise FileNotFoundError(
            f"no VED_*_week.csv in {source} (python src/data_download/fetch_archives.py "
            "--dataset ved_signals)"
        )
    files = [read_week(p, cache) for p in weeks]
    static = static_frame(source)
    con = connect(cache, static)
    try:
        return {
            "files": files_table(con, files),
            "columns": columns_table(con),
            "nulls_by_engine": nulls_by_engine(con),
            "vehicles": vehicles_table(con),
            "static_quality": static_quality(static),
            "static_columns": {
                name: list(pd.read_csv(source / name, nrows=0).columns)
                for name in STATIC_FILES
                if (source / name).is_file()
            },
            "trips": trips_table(con),
            "duplicates": duplicates(con),
            "months": months_table(con),
            "outside_week": outside_week(con, files),
            "gps": gps_table(con),
            "consistency": consistency(con),
        }
    finally:
        con.close()


def summary(result: dict[str, object]) -> dict[str, object]:
    """The headline numbers, JSON-friendly (the notebook's verdict)."""
    files: pd.DataFrame = result["files"]
    months: pd.DataFrame = result["months"]
    return {
        "files": len(files),
        "rows": int(files["rows_read"].sum()),
        "rows_lost_by_reader": int((files["data_lines"] - files["rows_read"]).sum()),
        "files_header_crlf": int(files["header_crlf"].sum()),
        "files_rows_crlf": int(files["rows_crlf"].sum()),
        "first_time": str(files["first_time"].min()),
        "last_time": str(files["last_time"].max()),
        "months": dict(zip(months["month"], months["rows"].astype(int), strict=True)),
        "vehicles": result["vehicles"],
        "trips": result["trips"],
        "duplicates": result["duplicates"],
        "outside_week": result["outside_week"],
        "gps": result["gps"],
        "static_columns": result["static_columns"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile the Vehicle Energy Dataset.")
    parser.add_argument("--source", type=Path, help="unpacked files (default: incoming)")
    parser.add_argument("--cache", type=Path, help="Parquet cache (default: data/profile)")
    args = parser.parse_args(argv)
    source, cache = folders()
    source, cache = args.source or source, args.cache or cache
    try:
        result = profile(source, cache)
    except FileNotFoundError as exc:
        print(exc)
        return 2
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        for name in ("files", "columns", "nulls_by_engine", "months", "consistency",
                     "static_quality"):  # fmt: skip
            print(f"\n== {name} ==")
            print(result[name].to_string(index=False))
    out = summary(result)
    print("\n== summary ==")
    print(json.dumps(out, indent=2, default=str))
    (cache / "profile.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"-> {cache / 'profile.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
