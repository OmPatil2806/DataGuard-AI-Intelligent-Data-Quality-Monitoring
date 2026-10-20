"""Split a Bronze month into Silver (trusted rows) and Quarantine (rejected rows).

For one Bronze file version: standardize, check the schema (Pandera), evaluate the enabled
row rules once, then write

    data/silver/<dataset>/period=YYYY-MM/version=<sha[:12]>/part-0.parquet
        rows without any quarantine failure; rows that only failed warn rules stay here
        with their failed_rules and reasons
    data/quarantine/<dataset>/period=YYYY-MM/version=<sha[:12]>/part-0.parquet
        rows with at least one quarantine failure, with every column, failed_rules,
        reasons and quarantine_rules (the failed rules whose action is quarantine)

Both files add the dataset's enrichment columns (configs/datasets.yml: silver.derived and
silver.joins, for the taxi data trip duration, pickup date and hour, and zone and borough
names), keep the Bronze lineage columns (bronze_row_id, source_file, ingestion_timestamp,
pipeline_run_id) and add silver_run_id. Both are written to temporary folders first, the
row counts are verified (Silver + Quarantine = Bronze, no row twice), and only then are both
moved into place, so a failure never leaves one half written.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import duckdb

from common.config import Dataset
from quality import row_rules
from quality.rules import Rule
from quality.standardize import standardize_file

PART_FILE = "part-0.parquet"
EVALUATED = "__dg_evaluated"


class SilverError(RuntimeError):
    """Raised when a month cannot be split into Silver and Quarantine correctly."""


@dataclass(frozen=True)
class SilverResult:
    period: str
    silver_path: Path
    quarantine_path: Path
    rows: int
    silver_rows: int
    quarantine_rows: int
    warned_rows: int  # Silver rows that failed at least one warn rule
    status: str  # "processed" or "skipped" (this version was already processed)


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def enrich_sql(con: duckdb.DuckDBPyConnection, dataset: Dataset, source: str) -> str:
    """SQL adding the configured derived and lookup columns to `source` (a FROM-able).

    Column order: standard columns, derived columns, lookup columns, then everything else
    (lineage, failed_rules, reasons, ...). Lookup tables must be loaded (lookup_<name>) and
    their join key must be unique, so a join can never duplicate rows.
    """
    names = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {source}").fetchall()]
    standard = [c for c, _ in dataset.columns if c in names]
    rest = [c for c in names if c not in standard]

    derived = ", ".join(f"({expr}) AS {_q(name)}" for name, expr in dataset.silver_derived)
    inner = f"(SELECT *{', ' + derived if derived else ''} FROM {source}) s"
    joins, joined = [], []
    for i, join in enumerate(dataset.silver_joins):
        table, alias = _q(row_rules.LOOKUP_PREFIX + join.lookup), f"__j{i}"
        rows, keys = con.execute(
            f"SELECT count(*), count(DISTINCT {_q(join.key)}) FROM {table}"
        ).fetchone()
        if rows != keys:
            raise SilverError(f"lookup '{join.lookup}' has duplicate values in {join.key}")
        joins.append(f"LEFT JOIN {table} {alias} ON s.{_q(join.column)} = {alias}.{_q(join.key)}")
        joined += [f"{alias}.{_q(src)} AS {_q(dst)}" for src, dst in join.columns]

    select = [f"s.{_q(c)}" for c in standard]
    select += [f"s.{_q(name)}" for name, _ in dataset.silver_derived]
    select += joined
    select += [f"s.{_q(c)}" for c in rest]
    return f"SELECT {', '.join(select)} FROM {inner} {' '.join(joins)}"


def _sql_path(path: Path) -> str:
    return "'" + str(path).replace("\\", "/").replace("'", "''") + "'"


def version_dir(root: Path, dataset: str, period: str, sha256: str) -> Path:
    return Path(root) / dataset / f"period={period}" / f"version={sha256[:12]}"


def _count(con: duckdb.DuckDBPyConnection, path: Path, where: str = "TRUE") -> int:
    return con.execute(
        f"SELECT count(*) FROM read_parquet({_sql_path(path)}, hive_partitioning = false) "
        f"WHERE {where}"
    ).fetchone()[0]


def process_version(
    con: duckdb.DuckDBPyConnection,
    dataset: Dataset,
    rules: list[Rule],
    period: str,
    bronze_path: Path,
    sha256: str,
    silver_root: Path,
    quarantine_root: Path,
    run_id: str,
    force: bool = False,
) -> SilverResult:
    """Write Silver and Quarantine for one Bronze version. Lookups must already be loaded."""
    silver_dir = version_dir(silver_root, dataset.name, period, sha256)
    quarantine_dir = version_dir(quarantine_root, dataset.name, period, sha256)
    silver_path, quarantine_path = silver_dir / PART_FILE, quarantine_dir / PART_FILE

    if silver_path.exists() and quarantine_path.exists() and not force:
        silver_rows, quarantine_rows = _count(con, silver_path), _count(con, quarantine_path)
        warned = _count(con, silver_path, "len(failed_rules) > 0")
        return SilverResult(
            period,
            silver_path,
            quarantine_path,
            silver_rows + quarantine_rows,
            silver_rows,
            quarantine_rows,
            warned,
            "skipped",
        )

    sql, _ = standardize_file(con, bronze_path, dataset)
    row_rules.check_schema(con, sql, dataset)
    evaluated = row_rules.evaluate_sql(con, rules, f"({sql})", period=period)
    quarantine_ids = [r.id for r in rules if r.action == "quarantine"]
    ids = "[" + ", ".join("'" + i.replace("'", "''") + "'" for i in quarantine_ids) + "]::VARCHAR[]"

    enriched = enrich_sql(con, dataset, f"({evaluated})")
    con.execute(f"CREATE OR REPLACE TEMP TABLE {EVALUATED} AS {enriched}")
    try:
        rows, quarantined, warned, distinct = con.execute(
            f"SELECT count(*), count_if(quarantine), "
            f"count_if(NOT quarantine AND len(failed_rules) > 0), count(DISTINCT bronze_row_id) "
            f"FROM {EVALUATED}"
        ).fetchone()
        if distinct != rows:
            raise SilverError(f"{period}: bronze_row_id is not unique ({distinct} of {rows})")

        temps = [d.with_name(d.name + ".tmp") for d in (silver_dir, quarantine_dir)]
        for tmp in temps:
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
        try:
            silver_tmp, quarantine_tmp = (t / PART_FILE for t in temps)
            con.execute(
                f"COPY (SELECT * EXCLUDE (quarantine), ?::VARCHAR AS silver_run_id "
                f"FROM {EVALUATED} WHERE NOT quarantine ORDER BY bronze_row_id) "
                f"TO {_sql_path(silver_tmp)} (FORMAT parquet, COMPRESSION zstd)",
                [run_id],
            )
            con.execute(
                f"COPY (SELECT * EXCLUDE (quarantine), "
                f"list_filter(failed_rules, x -> list_contains({ids}, x)) AS quarantine_rules, "
                f"?::VARCHAR AS silver_run_id "
                f"FROM {EVALUATED} WHERE quarantine ORDER BY bronze_row_id) "
                f"TO {_sql_path(quarantine_tmp)} (FORMAT parquet, COMPRESSION zstd)",
                [run_id],
            )
            silver_rows, quarantine_rows = _count(con, silver_tmp), _count(con, quarantine_tmp)
            if silver_rows + quarantine_rows != rows or quarantine_rows != quarantined:
                raise SilverError(
                    f"{period}: wrote {silver_rows} Silver + {quarantine_rows} Quarantine rows "
                    f"for {rows} Bronze rows ({quarantined} to quarantine)"
                )
            for final, tmp in ((silver_dir, temps[0]), (quarantine_dir, temps[1])):
                if final.exists():
                    shutil.rmtree(final)
                os.replace(tmp, final)
        finally:
            for tmp in temps:
                shutil.rmtree(tmp, ignore_errors=True)
    finally:
        con.execute(f"DROP TABLE IF EXISTS {EVALUATED}")

    return SilverResult(
        period,
        silver_path,
        quarantine_path,
        rows,
        silver_rows,
        quarantine_rows,
        warned,
        "processed",
    )
