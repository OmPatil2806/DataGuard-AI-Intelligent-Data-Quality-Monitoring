"""Profiling queries for Yellow Taxi files, used by notebooks/01_data_profiling.ipynb.

Every function runs one DuckDB query per file, so files with different schemas
(for example an older month) can be profiled side by side.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

PERIOD_IN_NAME = re.compile(r"(\d{4})-(\d{2})")

PICKUP = "tpep_pickup_datetime"
DROPOFF = "tpep_dropoff_datetime"
AMOUNT_PARTS = [
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "congestion_surcharge",
    "airport_fee",
]
# VendorID 1 also books the congestion surcharge and airport fee inside `extra`,
# so its totals match the sum without those two columns (found in Step 0.7).
SURCHARGES_ALSO_IN_EXTRA = ["congestion_surcharge", "airport_fee"]
NULL_GROUP = [
    "passenger_count",
    "RatecodeID",
    "store_and_fwd_flag",
    "congestion_surcharge",
    "airport_fee",
]
COMPOSITE_KEY = [
    "VendorID",
    PICKUP,
    DROPOFF,
    "PULocationID",
    "DOLocationID",
    "fare_amount",
    "total_amount",
]
UNKNOWN_ZONES = (264, 265)


@dataclass(frozen=True)
class Check:
    name: str
    category: str
    condition: str
    description: str


CHECKS = [
    Check("zero_passengers", "validity", "passenger_count = 0", "passenger_count is 0"),
    Check("many_passengers", "validity", "passenger_count > 6", "passenger_count above 6"),
    Check("negative_fare", "validity", "fare_amount < 0", "fare_amount below 0"),
    Check("zero_fare", "validity", "fare_amount = 0", "fare_amount is 0"),
    Check("negative_total", "validity", "total_amount < 0", "total_amount below 0"),
    Check("zero_distance", "validity", "trip_distance = 0", "trip_distance is 0"),
    Check("long_distance", "validity", "trip_distance > 100", "trip_distance above 100 miles"),
    Check(
        "dropoff_before_pickup",
        "consistency",
        f"{DROPOFF} < {PICKUP}",
        "dropoff earlier than pickup",
    ),
    Check("zero_duration", "consistency", f"{DROPOFF} = {PICKUP}", "dropoff equals pickup"),
    Check(
        "long_duration",
        "consistency",
        f"date_diff('second', {PICKUP}, {DROPOFF}) > 24 * 3600",
        "trip longer than 24 hours",
    ),
    Check(
        "pickup_outside_month",
        "pattern",
        f"({PICKUP} < {{start}} OR {PICKUP} >= {{end}})",
        "pickup outside the file's month",
    ),
    Check(
        "unknown_location_id",
        "referential",
        "(PULocationID NOT IN (SELECT id FROM zones)"
        " OR DOLocationID NOT IN (SELECT id FROM zones))",
        "location ID not in the zone lookup",
    ),
    Check(
        "unknown_zone_264_265",
        "referential",
        f"(PULocationID IN {UNKNOWN_ZONES} OR DOLocationID IN {UNKNOWN_ZONES})",
        "zone 264/265 (Unknown / Outside of NYC)",
    ),
    Check(
        "total_mismatch",
        "consistency",
        "abs(total_amount - ({parts})) > 0.01 AND abs(total_amount - ({base})) > 0.01",
        "total_amount matches neither the sum of its parts nor the sum without surcharges",
    ),
]


def connect() -> duckdb.DuckDBPyConnection:
    return duckdb.connect()


def _sql_path(path: Path) -> str:
    return "'" + str(path).replace("\\", "/").replace("'", "''") + "'"


def _source(path: Path) -> str:
    return f"read_parquet({_sql_path(path)})"


def period_of(path: Path) -> str | None:
    """'yellow_tripdata_2023-01.parquet' -> '2023-01'."""
    match = PERIOD_IN_NAME.search(Path(path).name)
    return f"{match.group(1)}-{match.group(2)}" if match else None


def columns_of(con: duckdb.DuckDBPyConnection, path: Path) -> dict[str, str]:
    rows = con.execute(f"DESCRIBE SELECT * FROM {_source(path)}").fetchall()
    return {name: dtype for name, dtype, *_ in rows}


def _resolve(columns: dict[str, str], name: str) -> str | None:
    """Find a column regardless of case (e.g. 'airport_fee' vs 'Airport_fee')."""
    lowered = {c.lower(): c for c in columns}
    return lowered.get(name.lower())


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _amount_sum(columns: dict[str, str], names: list[str]) -> str:
    """SQL sum of the amount columns present in a file, NULLs counted as 0."""
    present = [c for c in (_resolve(columns, n) for n in names) if c]
    return " + ".join(f"coalesce({_quote(c)}, 0)" for c in present)


def rows_per_file(con: duckdb.DuckDBPyConnection, files: list[Path]) -> pd.DataFrame:
    records = []
    for path in files:
        rows = con.execute(f"SELECT count(*) FROM {_source(path)}").fetchone()[0]
        period = period_of(path)
        days = pd.Period(period, freq="M").days_in_month if period else None
        records.append(
            {
                "file": path.name,
                "period": period,
                "rows": rows,
                "days": days,
                "rows_per_day": round(rows / days) if days else None,
            }
        )
    return pd.DataFrame(records)


def schema_by_file(con: duckdb.DuckDBPyConnection, files: list[Path]) -> pd.DataFrame:
    records = [
        {"file": path.name, "column": name, "type": dtype}
        for path in files
        for name, dtype in columns_of(con, path).items()
    ]
    return pd.DataFrame(records)


def schema_differences(schema: pd.DataFrame) -> pd.DataFrame:
    """Columns whose exact name or type is not the same in every file."""
    table = schema.pivot(index="column", columns="file", values="type").fillna("missing")
    differs = table.nunique(axis=1) > 1
    return table[differs].reset_index()


def null_rates(con: duckdb.DuckDBPyConnection, files: list[Path]) -> pd.DataFrame:
    """Percentage of NULLs per column (rows: columns, one column per file)."""
    frames = []
    for path in files:
        cols = list(columns_of(con, path))
        select = ", ".join(
            f"round(100.0 * count_if({_quote(c)} IS NULL) / count(*), 3) AS {_quote(c)}"
            for c in cols
        )
        row = con.execute(f"SELECT {select} FROM {_source(path)}").df()
        frames.append(row.T.rename(columns={0: path.name}))
    return pd.concat(frames, axis=1).rename_axis("column").reset_index()


def null_group(con: duckdb.DuckDBPyConnection, files: list[Path]) -> pd.DataFrame:
    """How often passenger_count is NULL, and whether the related columns are NULL with it."""
    records = []
    for path in files:
        cols = columns_of(con, path)
        present = [c for c in (_resolve(cols, name) for name in NULL_GROUP) if c]
        all_null = " AND ".join(f"{_quote(c)} IS NULL" for c in present)
        rows, null_pc, grouped = con.execute(
            f"""
            SELECT count(*),
                   count_if(passenger_count IS NULL),
                   count_if({all_null})
            FROM {_source(path)}
            """
        ).fetchone()
        records.append(
            {
                "file": path.name,
                "rows": rows,
                "passenger_count_null": null_pc,
                "whole_group_null": grouped,
                "group_columns": ", ".join(present),
            }
        )
    return pd.DataFrame(records)


def issue_counts(
    con: duckdb.DuckDBPyConnection, files: list[Path], zones_csv: Path
) -> pd.DataFrame:
    """One row per (file, check) with the number and share of rows that fail the check."""
    con.execute(
        f"CREATE OR REPLACE TEMP TABLE zones AS "
        f"SELECT LocationID AS id FROM read_csv({_sql_path(zones_csv)}, header = true)"
    )
    records = []
    for path in files:
        cols = columns_of(con, path)

        parts = _amount_sum(cols, AMOUNT_PARTS)
        base = _amount_sum(cols, [p for p in AMOUNT_PARTS if p not in SURCHARGES_ALSO_IN_EXTRA])
        period = period_of(path)
        selects, active = [], []
        for check in CHECKS:
            if check.name == "pickup_outside_month":
                if not period:
                    continue
                start = f"TIMESTAMP '{period}-01'"
                condition = check.condition.format(start=start, end=f"({start} + INTERVAL 1 MONTH)")
            elif check.name == "total_mismatch":
                condition = check.condition.format(parts=parts, base=base)
            else:
                condition = check.condition
            selects.append(f"count_if({condition})")
            active.append(check)
        values = con.execute(
            f"SELECT count(*), {', '.join(selects)} FROM {_source(path)}"
        ).fetchone()
        total = values[0]
        for check, failed in zip(active, values[1:], strict=True):
            records.append(
                {
                    "file": path.name,
                    "check": check.name,
                    "category": check.category,
                    "description": check.description,
                    "rows": failed,
                    "pct": round(100.0 * failed / total, 4) if total else 0.0,
                }
            )
    return pd.DataFrame(records)


def duplicate_counts(con: duckdb.DuckDBPyConnection, files: list[Path]) -> pd.DataFrame:
    records = []
    for path in files:
        cols = columns_of(con, path)
        key = [c for c in (_resolve(cols, k) for k in COMPOSITE_KEY) if c]
        key_sql = ", ".join(_quote(c) for c in key)
        rows, distinct_rows, distinct_keys = con.execute(
            f"""
            SELECT (SELECT count(*) FROM {_source(path)}),
                   (SELECT count(*) FROM (SELECT DISTINCT * FROM {_source(path)})),
                   (SELECT count(*) FROM (SELECT DISTINCT {key_sql} FROM {_source(path)}))
            """
        ).fetchone()
        records.append(
            {
                "file": path.name,
                "rows": rows,
                "exact_duplicates": rows - distinct_rows,
                "key_duplicates": rows - distinct_keys,
            }
        )
    return pd.DataFrame(records)


def distribution(
    con: duckdb.DuckDBPyConnection,
    files: list[Path],
    columns: tuple[str, ...] = ("trip_distance", "fare_amount", "total_amount"),
) -> pd.DataFrame:
    """Quantiles per file and column; the baseline the drift check (Phase 3) builds on."""
    quantiles = [0.01, 0.25, 0.5, 0.75, 0.99]
    records = []
    for path in files:
        cols = columns_of(con, path)
        for name in columns:
            col = _resolve(cols, name)
            if not col:
                continue
            mean, maximum, qs = con.execute(
                f"SELECT avg({_quote(col)}), max({_quote(col)}), "
                f"quantile_cont({_quote(col)}, {quantiles}) FROM {_source(path)}"
            ).fetchone()
            record = {"file": path.name, "column": name, "mean": mean, "max": maximum}
            record.update({f"p{round(q * 100):02d}": v for q, v in zip(quantiles, qs, strict=True)})
            records.append(record)
    return pd.DataFrame(records).round(3)


def summary(issues: pd.DataFrame) -> pd.DataFrame:
    """Worst file per check, ordered by how often the problem occurs."""
    grouped = issues.groupby(["check", "category", "description"], as_index=False).agg(
        total_rows=("rows", "sum"),
        max_pct=("pct", "max"),
        files_affected=("rows", lambda s: int((s > 0).sum())),
    )
    return grouped.sort_values("total_rows", ascending=False).reset_index(drop=True)
