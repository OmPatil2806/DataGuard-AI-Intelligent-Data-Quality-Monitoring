"""Regenerate the synthetic test fixtures in this folder.

    python tests/fixtures/make_fixtures.py

The trip fixture follows the 2023 Yellow Taxi schema. Most rows are clean and
internally consistent; a known set of rows carries one deliberate problem each,
listed in yellow_tripdata_fixture_issues.json so tests can check that the rules
catch exactly those rows. The data is synthetic, not taken from the TLC files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TRIPS_FILE = HERE / "yellow_tripdata_fixture.parquet"
ISSUES_FILE = HERE / "yellow_tripdata_fixture_issues.json"
ZONES_FILE = HERE / "taxi_zone_lookup_fixture.csv"

N_ROWS = 200
SEED = 7
MONTH_START = pd.Timestamp("2023-01-01")

ZONES = [
    (1, "EWR", "Newark Airport", "EWR"),
    (43, "Manhattan", "Central Park", "Yellow Zone"),
    (48, "Manhattan", "Clinton East", "Yellow Zone"),
    (68, "Manhattan", "East Chelsea", "Boro Zone"),
    (132, "Queens", "JFK Airport", "Airports"),
    (138, "Queens", "LaGuardia Airport", "Airports"),
    (161, "Manhattan", "Midtown Center", "Yellow Zone"),
    (236, "Manhattan", "Upper East Side North", "Yellow Zone"),
    (237, "Manhattan", "Upper East Side South", "Yellow Zone"),
    (264, "Unknown", "N/A", "N/A"),
    (265, "N/A", "Outside of NYC", "N/A"),
]
REAL_ZONES = [z[0] for z in ZONES if z[0] < 264]
AIRPORT_ZONES = {132, 138}

COLUMNS = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "airport_fee",
]
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

# issue -> number of rows that get it; every issue row is distinct and starts clean.
ISSUES = {
    "zero_passengers": 3,
    "null_group": 4,
    "negative_fare": 3,
    "zero_distance": 3,
    "dropoff_before_pickup": 2,
    "pickup_outside_month": 2,
    "unknown_location_id": 2,
    "unknown_zone_264_265": 2,
    "total_mismatch": 2,
    "duplicate_row": 2,
}


def clean_trips(rng: np.random.Generator, n: int) -> pd.DataFrame:
    pickup = MONTH_START + pd.to_timedelta(rng.integers(0, 31 * 24 * 3600 - 7200, n), unit="s")
    minutes = rng.uniform(3, 45, n).round(2)
    distance = (minutes * rng.uniform(0.15, 0.35, n)).round(2)
    pu = rng.choice(REAL_ZONES, n)
    do = rng.choice(REAL_ZONES, n)
    fare = (3.0 + 2.5 * distance + 0.7 * minutes).round(2)
    payment = rng.choice([1, 2], n, p=[0.8, 0.2])
    tip = np.where(payment == 1, (fare * rng.uniform(0.1, 0.25, n)).round(2), 0.0)
    df = pd.DataFrame(
        {
            "VendorID": rng.choice([1, 2], n).astype("int64"),
            "tpep_pickup_datetime": pickup,
            "tpep_dropoff_datetime": pickup + pd.to_timedelta(np.round(minutes * 60), unit="s"),
            "passenger_count": rng.choice([1.0, 1.0, 1.0, 2.0, 3.0], n),
            "trip_distance": distance,
            "RatecodeID": 1.0,
            "store_and_fwd_flag": "N",
            "PULocationID": pu.astype("int64"),
            "DOLocationID": do.astype("int64"),
            "payment_type": payment.astype("int64"),
            "fare_amount": fare,
            "extra": rng.choice([0.0, 1.0, 2.5], n),
            "mta_tax": 0.5,
            "tip_amount": tip,
            "tolls_amount": 0.0,
            "improvement_surcharge": 1.0,
            "congestion_surcharge": np.where(np.isin(pu, AIRPORT_ZONES), 0.0, 2.5),
            "airport_fee": np.where(np.isin(pu, AIRPORT_ZONES), 1.25, 0.0),
        }
    )
    df["total_amount"] = df[AMOUNT_PARTS].sum(axis=1).round(2)
    return df[COLUMNS]


def inject_issues(df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    total = sum(ISSUES.values())
    rows = rng.choice(len(df), total, replace=False)
    issues: dict[str, list[int]] = {}
    start = 0
    for name, count in ISSUES.items():
        issues[name] = sorted(int(r) for r in rows[start : start + count])
        start += count

    def at(name):
        return issues[name]

    df.loc[at("zero_passengers"), "passenger_count"] = 0.0
    for col in [
        "passenger_count",
        "RatecodeID",
        "store_and_fwd_flag",
        "congestion_surcharge",
        "airport_fee",
    ]:
        df.loc[at("null_group"), col] = None
    df.loc[at("null_group"), "total_amount"] = (
        df.loc[at("null_group"), AMOUNT_PARTS].fillna(0).sum(axis=1).round(2)
    )
    for col in AMOUNT_PARTS + ["total_amount"]:
        df.loc[at("negative_fare"), col] = -df.loc[at("negative_fare"), col]
    df.loc[at("negative_fare"), "payment_type"] = 4
    df.loc[at("zero_distance"), "trip_distance"] = 0.0
    rows = at("dropoff_before_pickup")
    df.loc[rows, "tpep_dropoff_datetime"] = df.loc[
        rows, "tpep_pickup_datetime"
    ].to_numpy() - np.timedelta64(5, "m")
    # Move the whole trip to the last hour of the previous month.
    rows = at("pickup_outside_month")
    duration = df.loc[rows, "tpep_dropoff_datetime"] - df.loc[rows, "tpep_pickup_datetime"]
    df.loc[rows, "tpep_pickup_datetime"] = np.datetime64("2022-12-31T23:10:00", "ns")
    df.loc[rows, "tpep_dropoff_datetime"] = (
        df.loc[rows, "tpep_pickup_datetime"].to_numpy() + duration.to_numpy()
    )
    df.loc[at("unknown_location_id"), "PULocationID"] = 999
    df.loc[at("unknown_zone_264_265"), "DOLocationID"] = [264, 265]
    df.loc[at("total_mismatch"), "total_amount"] += 4.5
    # Duplicates: overwrite the chosen rows with an exact copy of the nearest clean row before them.
    issue_rows = {r for rows in issues.values() for r in rows}
    for r in at("duplicate_row"):
        source = r - 1
        while source in issue_rows:
            source -= 1
        df.loc[r] = df.loc[source % len(df)]
    return df, issues


def build() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    trips, issues = inject_issues(clean_trips(rng, N_ROWS), rng)
    trips["store_and_fwd_flag"] = trips["store_and_fwd_flag"].astype("string")
    for col in ["tpep_pickup_datetime", "tpep_dropoff_datetime"]:
        trips[col] = trips[col].astype("datetime64[us]")
    zones = pd.DataFrame(ZONES, columns=["LocationID", "Borough", "Zone", "service_zone"])
    return trips, issues, zones


def main() -> None:
    trips, issues, zones = build()
    trips.to_parquet(TRIPS_FILE, index=False)
    ISSUES_FILE.write_text(json.dumps(issues, indent=2) + "\n", encoding="utf-8")
    zones.to_csv(ZONES_FILE, index=False)
    print(f"wrote {TRIPS_FILE.name} ({len(trips)} rows), {ISSUES_FILE.name}, {ZONES_FILE.name}")


if __name__ == "__main__":
    main()
