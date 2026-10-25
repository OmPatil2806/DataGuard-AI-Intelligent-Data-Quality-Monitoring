"""The faults of the detection evaluation: one per check category, and a concentrated variant.

Every fault gets the sampled rows of the target month (a pandas frame with pyarrow types),
its strength and params from configs/faults.yml, and a random generator seeded for the
experiment. It changes only the rows it reports, and only the columns it is about:

    missing_column        drop a required column                         month level
    null_spike            empty a column in a share of rows (rows that had a value)
    null_spike_vendor     the same, only in the rows of one segment (one vendor): a
                          concentrated fault, for the where/when of the investigation (Phase 8)
    duplicate_rows        deliver a share of rows twice, each copy right after its original
    out_of_range          make a share of positive fares negative
    orphan_location       point a share of pickups to a zone that does not exist
    unit_change           multiply a share of distances (miles -> kilometres)
    volume_drop           remove a share of rows                         month level
    stale_file            deliver the file late (strength = days)        month level
    pickup_outside_month  move a share of trips into the previous month (duration kept)
    swapped_times         swap pickup and dropoff time in a share of trips
    plausible_but_wrong   triple the fare of a share of short trips and raise the total to match:
                          every value stays in range and the total still adds up, only the fare
                          does not fit the trip (for the anomaly model, Phase 7)

Rows are chosen only among rows the fault really changes (a NULL that is already NULL is not
a fault), so the share is a share of those rows.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

PICKUP = "tpep_pickup_datetime"
DROPOFF = "tpep_dropoff_datetime"


@dataclass
class FaultResult:
    """What a fault produced.

    frame         the broken rows, in output order (pyarrow-backed pandas frame)
    affected      0-based positions *in frame* of the rows the fault changed or added
    removed       0-based positions *in the sampled input* of rows the fault dropped
    month_level   the fault concerns the whole month (schema, volume, delivery), not rows
    delivery_delay_days  for a late file: days after expected_at it counts as delivered
    """

    frame: pd.DataFrame
    affected: list[int] = field(default_factory=list)
    removed: list[int] = field(default_factory=list)
    month_level: bool = False
    delivery_delay_days: float | None = None
    note: str = ""


FaultFunction = Callable[
    [pd.DataFrame, float | None, dict[str, Any], np.random.Generator], FaultResult
]
FAULTS: dict[str, FaultFunction] = {}


def fault(name: str) -> Callable[[FaultFunction], FaultFunction]:
    def register(function: FaultFunction) -> FaultFunction:
        if name in FAULTS:
            raise ValueError(f"fault '{name}' is registered twice")
        FAULTS[name] = function
        return function

    return register


def choose(
    eligible: np.ndarray | pd.Index | list[int], share: float, rng: np.random.Generator
) -> np.ndarray:
    """round(share x eligible) of the eligible positions (at least one), sorted."""
    eligible = np.asarray(eligible)
    if len(eligible) == 0 or share <= 0:
        return np.array([], dtype=int)
    count = min(len(eligible), max(1, round(share * len(eligible))))
    return np.sort(rng.choice(eligible, size=count, replace=False))


def _eligible(mask: pd.Series) -> np.ndarray:
    return np.flatnonzero(mask.fillna(False).to_numpy(dtype=bool))


def _column(frame: pd.DataFrame, params: dict[str, Any]) -> str:
    column = params.get("column")
    if column not in frame.columns:
        raise ValueError(f"column '{column}' is not in the file")
    return column


@fault("control")
def control(frame, strength, params, rng):
    return FaultResult(frame, note="nothing broken")


@fault("missing_column")
def missing_column(frame, strength, params, rng):
    column = _column(frame, params)
    return FaultResult(frame.drop(columns=[column]), month_level=True, note=f"dropped {column}")


@fault("null_spike")
def null_spike(frame, strength, params, rng):
    column = _column(frame, params)
    rows = choose(_eligible(frame[column].notna()), strength, rng)
    frame.loc[rows, column] = None
    return FaultResult(frame, affected=rows.tolist(), note=f"{column} set to NULL")


@fault("null_spike_vendor")
def null_spike_vendor(frame, strength, params, rng):
    column = _column(frame, params)
    segment = _column(frame, {"column": params.get("segment_column")})
    inside = frame[segment].astype("string") == str(params["segment_value"])
    rows = choose(_eligible(frame[column].notna() & inside), strength, rng)
    frame.loc[rows, column] = None
    note = f"{column} set to NULL where {segment} = {params['segment_value']}"
    return FaultResult(frame, affected=rows.tolist(), note=note)


@fault("duplicate_rows")
def duplicate_rows(frame, strength, params, rng):
    rows = set(choose(np.arange(len(frame)), strength, rng).tolist())
    order, affected = [], []
    for position in range(len(frame)):
        order.append(position)
        if position in rows:  # the copy follows its original, so the copy is the repeat
            affected.append(len(order))
            order.append(position)
    out = frame.iloc[order].reset_index(drop=True)
    return FaultResult(out, affected=affected, note="copies inserted after their originals")


@fault("out_of_range")
def out_of_range(frame, strength, params, rng):
    column = _column(frame, params)
    rows = choose(_eligible(frame[column] > 0), strength, rng)
    frame.loc[rows, column] = -frame.loc[rows, column]
    return FaultResult(frame, affected=rows.tolist(), note=f"{column} made negative")


@fault("orphan_location")
def orphan_location(frame, strength, params, rng):
    column, value = _column(frame, params), params["value"]
    rows = choose(_eligible(frame[column] != value), strength, rng)
    frame.loc[rows, column] = value
    return FaultResult(frame, affected=rows.tolist(), note=f"{column} set to {value}")


@fault("unit_change")
def unit_change(frame, strength, params, rng):
    column, factor = _column(frame, params), float(params["factor"])
    rows = choose(_eligible(frame[column] > 0), strength, rng)
    frame.loc[rows, column] = frame.loc[rows, column] * factor
    return FaultResult(frame, affected=rows.tolist(), note=f"{column} x {factor:g}")


@fault("volume_drop")
def volume_drop(frame, strength, params, rng):
    rows = choose(np.arange(len(frame)), strength, rng)
    out = frame.drop(index=rows).reset_index(drop=True)
    return FaultResult(out, removed=rows.tolist(), month_level=True, note="rows removed")


@fault("stale_file")
def stale_file(frame, strength, params, rng):
    return FaultResult(
        frame,
        month_level=True,
        delivery_delay_days=float(strength),
        note=f"delivered {strength:g} day(s) after expected",
    )


@fault("pickup_outside_month")
def pickup_outside_month(frame, strength, params, rng):
    rows = choose(_eligible(frame[PICKUP].notna() & frame[DROPOFF].notna()), strength, rng)
    # back by the day of the month: the same time on the last day of the previous month
    shift = pd.to_timedelta(frame.loc[rows, PICKUP].dt.day.astype("int64"), unit="D")
    for column in (PICKUP, DROPOFF):
        frame.loc[rows, column] = frame.loc[rows, column] - shift.to_numpy()
    return FaultResult(frame, affected=rows.tolist(), note="moved into the previous month")


@fault("swapped_times")
def swapped_times(frame, strength, params, rng):
    rows = choose(_eligible(frame[DROPOFF] > frame[PICKUP]), strength, rng)
    pickup = frame.loc[rows, PICKUP].copy()
    frame.loc[rows, PICKUP] = frame.loc[rows, DROPOFF].to_numpy()
    frame.loc[rows, DROPOFF] = pickup.to_numpy()
    return FaultResult(frame, affected=rows.tolist(), note="pickup and dropoff swapped")


@fault("plausible_but_wrong")
def plausible_but_wrong(frame, strength, params, rng):
    factor, longest = float(params["factor"]), float(params["max_distance"])
    short = frame["trip_distance"].between(0.5, longest) & (frame["fare_amount"] > 0)
    rows = choose(_eligible(short & frame["total_amount"].notna()), strength, rng)
    extra = frame.loc[rows, "fare_amount"] * (factor - 1)
    frame.loc[rows, "fare_amount"] = frame.loc[rows, "fare_amount"] + extra
    frame.loc[rows, "total_amount"] = frame.loc[rows, "total_amount"] + extra
    return FaultResult(
        frame, affected=rows.tolist(), note=f"fare x {factor:g} on trips up to {longest:g} miles"
    )
