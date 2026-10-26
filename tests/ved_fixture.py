"""A small Vehicle Energy Dataset in the shape of the real one (Phase 9).

Two weekly files with the real header (ending CR LF, rows ending LF, missing values as NaN),
vehicles of the four engine types, a trip whose last rows fall after its week, a trip with
negative milliseconds, and the two static workbooks with their two names for the engine type
column. make_archives packs them as the source repository ships them.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HEADER = [
    "DayNum", "VehId", "Trip", "Timestamp(ms)", "Latitude[deg]", "Longitude[deg]",
    "Vehicle Speed[km/h]", "MAF[g/sec]", "Engine RPM[RPM]", "Absolute Load[%]", "OAT[DegC]",
    "Fuel Rate[L/hr]", "Air Conditioning Power[kW]", "Air Conditioning Power[Watts]",
    "Heater Power[Watts]", "HV Battery Current[A]", "HV Battery SOC[%]",
    "HV Battery Voltage[V]", "Short Term Fuel Trim Bank 1[%]", "Short Term Fuel Trim Bank 2[%]",
    "Long Term Fuel Trim Bank 1[%]", "Long Term Fuel Trim Bank 2[%]",
]  # fmt: skip
VEHICLES = {8: "ICE", 10: "EV", 11: "HEV", 12: "PHEV", 13: "ICE"}
WEEKS = {"171101": 1.0, "171108": 8.0, "171129": 29.0}  # stamp -> DayNum of its first day
# the week of 2017-11-29 runs into December: its third day is 2017-12-01
ROWS_PER_TRIP = 40


def _trip(vehicle: int, trip: int, daynum: float, rng, start_ms: int = 0) -> list[list]:
    kind = VEHICLES[vehicle]
    rows = []
    for i in range(ROWS_PER_TRIP):
        speed = float(rng.integers(0, 90))
        engine = kind in ("ICE", "HEV", "PHEV")
        battery = kind in ("EV", "HEV", "PHEV")
        rows.append([
            daynum, vehicle, trip, start_ms + 1000 * i, 42.27 + rng.normal(0, 0.01),
            -83.73 + rng.normal(0, 0.01), speed,
            round(rng.uniform(1, 20), 2) if engine else None,
            int(rng.integers(700, 3000)) if engine else None,
            round(rng.uniform(10, 80), 2) if engine else None,
            float(rng.integers(-5, 30)) if i % 3 else None,
            round(rng.uniform(0, 5), 2) if kind == "PHEV" else None,
            None, 0 if kind == "EV" else None, None,
            round(rng.normal(-5, 20), 2) if battery else None,
            round(rng.uniform(20, 90), 1) if battery else None,
            round(rng.uniform(250, 380), 1) if battery else None,
            round(rng.normal(0, 3), 2) if engine else None, None,
            round(rng.normal(1, 3), 2) if engine else None, None,
        ])  # fmt: skip
    return rows


def week_rows(stamp: str, seed: int = 0) -> list[list]:
    rng = np.random.default_rng(seed)
    first = WEEKS[stamp]
    rows, trip = [], 100 * int(first)
    for vehicle in VEHICLES:
        for day in range(3):
            trip += 1
            rows += _trip(vehicle, trip, first + day + 0.3 + 0.01 * vehicle, rng)
    if stamp == "171101":
        rows += _trip(8, 900, first + 6.9999, rng)  # ends after the week: next file's week
        rows += _trip(13, 901, first + 2.5, rng, start_ms=-5000)  # negative milliseconds
    return rows


def _cell(value) -> str:
    if value is None:
        return "NaN"
    return str(value)


def write_week(folder: Path, stamp: str) -> Path:
    path = folder / f"VED_{stamp}_week.csv"
    lines = [",".join(_cell(v) for v in row) for row in week_rows(stamp)]
    path.write_bytes((",".join(HEADER) + "\r\n" + "\n".join(lines) + "\n").encode())
    return path


def static_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    def row(vehicle, kind):
        return [vehicle, kind, "NO DATA", "4-GAS 2.0L", "NO DATA", "FWD", 3500]

    columns = ["VehId", "Engine Configuration & Displacement", "Transmission", "Drive Wheels",
               "Generalized_Weight"]  # fmt: skip
    ice_hev = pd.DataFrame(
        [row(v, k) for v, k in VEHICLES.items() if k in ("ICE", "HEV")],
        columns=["VehId", "Vehicle Type", "Vehicle Class", *columns[1:]],
    )
    phev_ev = pd.DataFrame(
        [row(v, k) for v, k in VEHICLES.items() if k in ("PHEV", "EV")],
        columns=["VehId", "EngineType", "Vehicle Class", *columns[1:]],
    )
    return ice_hev, phev_ev


def make_unpacked(folder: Path) -> Path:
    """The files as fetch_archives.py leaves them in <incoming>/ved_signals/."""
    folder.mkdir(parents=True, exist_ok=True)
    for stamp in WEEKS:
        write_week(folder, stamp)
    ice_hev, phev_ev = static_frames()
    ice_hev.to_csv(folder / "static_ice_hev.csv", index=False)
    phev_ev.to_csv(folder / "static_phev_ev.csv", index=False)
    return folder


def make_archives(folder: Path) -> Path:
    """The four source files as the repository ships them (names from configs/datasets.yml)."""
    import py7zr

    folder.mkdir(parents=True, exist_ok=True)
    weeks = folder / "weeks"
    weeks.mkdir(exist_ok=True)
    paths = [write_week(weeks, stamp) for stamp in WEEKS]
    (weeks / "readme.txt").write_text("not a week file")
    for name, members in (("Part1", paths[:1] + [weeks / "readme.txt"]), ("Part2", paths[1:])):
        with py7zr.SevenZipFile(folder / f"VED_DynamicData_{name}.7z", "w") as z:
            for member in members:
                z.write(member, member.name)
    ice_hev, phev_ev = static_frames()
    ice_hev.to_excel(folder / "VED_Static_Data_ICE&HEV.xlsx", index=False)
    phev_ev.to_excel(folder / "VED_Static_Data_PHEV&EV.xlsx", index=False)
    return folder
