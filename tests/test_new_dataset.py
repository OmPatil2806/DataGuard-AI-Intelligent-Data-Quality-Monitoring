"""Phase 9.4 'done when': a dataset defined only in YAML runs through the whole pipeline.

`air_quality` is made up for this test: ten stations, readings every 2.4 hours, six months.
It exists only in a temporary copy of the config folder (DATAGUARD_CONFIG_DIR); nothing in
src/ knows it. Its source names differ from its standard names, one column is computed, its
station lookup comes in two files that name the key differently, and it has no delivery
schedule and no anomaly model. In June station 7 stops sending PM2.5: the pipeline must raise
the NULL-rate incident and the investigation must place it in station 7.
"""

import shutil
from datetime import datetime, timedelta

import duckdb
import numpy as np
import pandas as pd
import pytest
import yaml

from common.config import CONFIG_DIR, CONFIG_DIR_ENV
from gold import rebuild
from ingestion import ingest_data

MONTHS = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06"]
STATIONS = range(1, 11)
BROKEN_STATION = 7

DATASET = {
    "description": "Air quality readings of ten stations (made up for the Phase 9.4 test)",
    "enabled": True,
    "format": "parquet",
    "frequency": "monthly",
    "url_template": "https://example.org/air_quality_{period}.parquet",
    "file_template": "air_quality_{period}.parquet",
    "timestamp_column": "measured_at",
    "periods": {"start": MONTHS[0], "end": MONTHS[-1]},
    "columns": {
        "station_id": {"type": "INTEGER", "source": "Station"},
        "measured_at": {"type": "TIMESTAMP", "source": "Time (UTC)"},
        "pm25": {"type": "DOUBLE", "source": "PM2.5 [ug/m3]"},
        "temperature_c": {"type": "DOUBLE", "source": "Temp [C]"},
        "humidity_pct": {"type": "DOUBLE", "source": ["RH [%]", "Humidity [%]"]},
        "measured_hour": {"type": "INTEGER", "sql": "hour(measured_at)"},
    },
    "lookups": {
        "stations": {
            "files": ["air_quality/stations_north.csv", "air_quality/stations_south.csv"],
            "rename": {"Station_ID": "station_id"},
        }
    },
    "silver": {
        "derived": {"measured_date": "CAST(measured_at AS DATE)"},
        "joins": [
            {
                "lookup": "stations",
                "key": "station_id",
                "column": "station_id",
                "columns": {"city": "city"},
            },
        ],  # fmt: skip
    },
}
RULES = [
    {"id": "reading_identified", "category": "completeness", "check": "not_null",
     "columns": ["station_id", "measured_at"], "action": "quarantine", "severity": "high",
     "reason": "station or time missing"},
    {"id": "pm25_in_range", "category": "validity", "check": "range", "column": "pm25",
     "min": 0, "max": 1000, "action": "quarantine", "severity": "high",
     "reason": "pm25 {pm25} outside 0-1000"},
    {"id": "humidity_in_range", "category": "validity", "check": "range",
     "column": "humidity_pct", "min": 0, "max": 100, "action": "warn", "severity": "low",
     "reason": "humidity {humidity_pct} outside 0-100"},
    {"id": "measured_in_file_month", "category": "pattern", "check": "within_period",
     "column": "measured_at", "action": "quarantine", "severity": "high",
     "reason": "reading at {measured_at} outside the file's month"},
    {"id": "station_known", "category": "referential", "check": "foreign_key",
     "column": "station_id", "lookup": "stations", "lookup_column": "station_id",
     "action": "quarantine", "severity": "high", "reason": "station {station_id} unknown"},
    {"id": "reading_unique", "category": "uniqueness", "check": "unique",
     "columns": ["station_id", "measured_at"], "action": "quarantine", "severity": "medium",
     "reason": "a second reading of the station at the same time"},
]  # fmt: skip
INVESTIGATION = {
    "sample_columns": ["station_id", "measured_at", "pm25", "temperature_c", "city"],
    "dimensions": {
        "station": {"sql": "station_id", "kind": "where"},
        "city": {"sql": "city", "kind": "where"},
        "hour": {"sql": "measured_hour", "kind": "when"},
    },
}


def write_configs(folder, root):
    """configs/ copied, with air_quality added to every file that has a part per dataset."""
    shutil.copytree(CONFIG_DIR, folder)

    def edit(name, change):
        path = folder / name
        data = yaml.safe_load(path.read_text())
        change(data)
        path.write_text(yaml.safe_dump(data, sort_keys=False))

    def datasets(d):
        d["paths"] |= {"incoming": str(root / "incoming"), "bronze": str(root / "bronze"),
                       "silver": str(root / "silver"), "quarantine": str(root / "quarantine"),
                       "duckdb": str(root / "dataguard.duckdb")}  # fmt: skip
        d["datasets"]["air_quality"] = DATASET

    def rules(d):
        d["rules"]["air_quality"] = RULES
        d["cold_start"]["air_quality"] = {"rows_per_day": {"min": 50, "max": 200},
                                          "null_rate_max": 0.05}  # fmt: skip
        d["defaults"]["drift"]["bins"]["air_quality"] = {
            "pm25": [0, 10, 20, 30, 40, 50, 60, 80], "temperature_c": [-10, 0, 10, 20, 30],
        }  # fmt: skip

    edit("datasets.yml", datasets)
    edit("quality_rules.yml", rules)
    edit("investigation.yml", lambda d: d["datasets"].update(air_quality=INVESTIGATION))


def readings(period, rng):
    start = datetime.strptime(period, "%Y-%m")
    days = pd.Period(period, freq="M").days_in_month
    rows = []
    for day in range(days):
        for step in range(10):  # every 2.4 hours: 100 readings a day over ten stations
            at = start + timedelta(days=day, hours=2.4 * step)
            for station in STATIONS:
                pm25 = round(float(rng.gamma(4, 6 + station)), 1)
                if period == MONTHS[-1] and station == BROKEN_STATION:
                    pm25 = None  # the fault: station 7 stops sending PM2.5
                rows.append((station, at, pm25, round(float(rng.normal(8, 9)), 1),
                             round(float(rng.uniform(30, 95)), 1)))  # fmt: skip
    return pd.DataFrame(rows, columns=["Station", "Time (UTC)", "PM2.5 [ug/m3]", "Temp [C]",
                                       "RH [%]"])  # fmt: skip


def write_data(incoming):
    rng = np.random.default_rng(7)
    for period in MONTHS:
        readings(period, rng).to_parquet(incoming / f"air_quality_{period}.parquet", index=False)
    lookups = incoming / "air_quality"
    lookups.mkdir()
    north = pd.DataFrame({"station_id": [1, 2, 3, 4, 5], "city": ["Northport"] * 5})
    south = pd.DataFrame({"Station_ID": [6, 7, 8, 9, 10], "city": ["Southbay"] * 5})
    north.to_csv(lookups / "stations_north.csv", index=False)
    south.to_csv(lookups / "stations_south.csv", index=False)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("new_dataset")
    write_configs(root / "configs", root)
    (root / "incoming").mkdir()
    write_data(root / "incoming")
    database = str(root / "dataguard.duckdb")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(CONFIG_DIR_ENV, str(root / "configs"))
        common = ["--dataset", "air_quality", "--database", database]
        ingested = ingest_data.main([*common, "--source", str(root / "incoming"),
                                     "--bronze", str(root / "bronze")])  # fmt: skip
        folders = {"--lookups": "incoming", "--silver": "silver", "--quarantine": "quarantine",
                   "--reports": "reports"}  # fmt: skip
        argv = [*common, *(x for flag, name in folders.items() for x in (flag, str(root / name)))]
        results = rebuild.run(rebuild.parse_args(argv))
    return {"db": database, "ingested": ingested, "stages": {r.name: r.exit_code for r in results},
            "root": root}  # fmt: skip


def query(env, sql):
    with duckdb.connect(env["db"], read_only=True) as con:
        return con.execute(sql).fetchall()


def test_every_stage_runs(env):
    assert env["ingested"] == 0
    assert env["stages"] == {s: 0 for s in ("silver", "checks", "rules", "ml", "incidents",
                                             "score", "investigate", "gold")}  # fmt: skip


def test_standard_names_computed_columns_and_lookup(env):
    path = query(env, "SELECT silver_path FROM silver_log WHERE period = '2024-01' LIMIT 1")[0][0]
    columns = [r[0] for r in query(env, f"DESCRIBE SELECT * FROM read_parquet('{path}')")]
    assert columns[:9] == ["station_id", "measured_at", "pm25", "temperature_c", "humidity_pct",
                           "measured_hour", "measured_date", "city", "bronze_row_id"]  # fmt: skip
    cities = query(env, f"SELECT city, count(DISTINCT station_id) FROM read_parquet('{path}') "
                        "GROUP BY 1 ORDER BY 1")  # fmt: skip
    assert cities == [("Northport", 5), ("Southbay", 5)]  # both lookup files, renamed key
    assert query(env, "SELECT sum(silver_rows), sum(quarantine_rows) FROM silver_log") == [
        (sum(pd.Period(m, freq="M").days_in_month for m in MONTHS) * 100, 0)
    ]


def test_checks_see_the_dataset(env):
    results = dict(query(env, """
        SELECT "check" || ':' || column_name, status FROM dataset_check_results
        WHERE period = '2024-06' AND "check" IN ('schema', 'volume', 'freshness', 'null_rate')
        AND column_name IN ('', 'pm25')
    """))  # fmt: skip
    assert results == {"schema:": "pass", "volume:": "pass", "freshness:": "skipped",
                       "null_rate:pm25": "fail"}  # fmt: skip
    drift = query(env, "SELECT DISTINCT column_name FROM dataset_check_results "
                       "WHERE \"check\" = 'drift' ORDER BY 1")  # fmt: skip
    assert drift == [("pm25",), ("temperature_c",)]  # its own bins
    assert query(env, "SELECT count(*) FROM ml_month_summary") == [(0,)]  # no model for it


def test_the_fault_is_found_and_placed(env):
    (incident,) = query(env, "SELECT incident_id, rule_id, column_name, first_seen_period, "
                             "severity FROM incidents WHERE status = 'OPEN'")  # fmt: skip
    assert incident[1:] == ("null_rate", "pm25", "2024-06", "high")
    sql = "SELECT top_cause, explained, where_verdict FROM investigations WHERE incident_id = "
    top_cause, explained, where = query(env, f"{sql}'{incident[0]}'")[0]
    assert (top_cause, explained) == ("segment_sends_nulls", True)
    assert where.startswith(f"where: station = {BROKEN_STATION} holds 100%")
    report = env["root"] / "reports" / "investigations" / "air_quality" / f"{incident[0]}.md"
    text = report.read_text(encoding="utf-8")
    assert "station = 7" in text and "| Row | station_id | measured_at | pm25 |" in text


def test_gold_covers_it(env):
    health = query(env, "SELECT period, health FROM gold_dataset_health "
                        "WHERE dataset = 'air_quality' ORDER BY 1")  # fmt: skip
    assert [p for p, _ in health] == MONTHS
    assert dict(health)["2024-06"] != "healthy" and dict(health)["2024-05"] == "healthy"
    lineage = query(env, "SELECT DISTINCT status FROM gold_lineage WHERE dataset = 'air_quality'")
    assert lineage == [("ok",)]
    assert query(env, "SELECT count(*) FROM gold_investigations")[0][0] >= 1
