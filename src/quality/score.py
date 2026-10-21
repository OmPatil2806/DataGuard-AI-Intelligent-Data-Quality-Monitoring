"""Quality score per month: one number from 0 to 100, built from the category pass rates.

    score = 100 x sum(w_i x p_i) / sum(w_i)        weights: configs/severity_weights.yml

Row categories (completeness, uniqueness, validity, pattern, referential, consistency):
    p = share of the month's rows that fail no enabled rule of the category, counted on the
    written Silver and Quarantine files (every Bronze row is in exactly one of them).
Dataset checks (schema, volume, freshness, drift; null_rate counts as completeness):
    p = check_scores of the worst result of that check in the month (pass 1.0 ... critical 0.0),
    from the latest dataset check run of the month's file version.
Completeness combines both: the lower of the row pass rate and the NULL-rate check.
Categories without a result (a skipped check, no rule in the category, ml_anomaly before
Phase 7) are left out of both sums, so they neither help nor hurt.

    python src/quality/score.py              # score every month with Silver and show them
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

if __package__ in (None, ""):
    # Allow running this file directly: make src/ importable.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import db  # noqa: E402
from common.config import (  # noqa: E402
    ConfigError,
    Dataset,
    load_datasets_config,
    load_severity_weights,
)
from common.runs import utc_now  # noqa: E402
from ingestion.bronze_loader import current_versions  # noqa: E402
from quality.rules import ROW_CATEGORIES, Rule, load_rules  # noqa: E402
from silver.process_silver import current_silver  # noqa: E402

CHECK_CATEGORY = {
    "schema": "schema",
    "volume": "volume",
    "freshness": "freshness",
    "drift": "drift",
    "null_rate": "completeness",
}
DEFAULT_CHECK_SCORES = {"pass": 1.0, "low": 0.75, "medium": 0.5, "high": 0.25, "critical": 0.0}


@dataclass(frozen=True)
class Part:
    category: str
    weight: float
    pass_rate: float
    source: str


@dataclass(frozen=True)
class MonthScore:
    period: str
    sha256: str
    score: float
    rows: int
    parts: tuple[Part, ...]
    check_run: str | None


def combine(parts: list[Part]) -> float:
    total = sum(p.weight for p in parts)
    if total <= 0:
        raise ValueError("no category with a weight to score")
    return 100 * sum(p.weight * p.pass_rate for p in parts) / total


def _sql_path(path: Path) -> str:
    return "'" + str(path).replace("\\", "/").replace("'", "''") + "'"


def row_pass_rates(
    con: duckdb.DuckDBPyConnection, silver: Path, quarantine: Path, rules: list[Rule]
) -> tuple[int, dict[str, float]]:
    """Rows of the month and, per row category with rules, the share failing none of them."""
    by_category: dict[str, list[str]] = {}
    for rule in rules:
        by_category.setdefault(rule.category, []).append(rule.id)
    source = (
        f"(SELECT failed_rules FROM read_parquet({_sql_path(silver)}, hive_partitioning = false) "
        f"UNION ALL SELECT failed_rules FROM read_parquet({_sql_path(quarantine)}, "
        "hive_partitioning = false))"
    )
    categories = [c for c in ROW_CATEGORIES if c in by_category]
    if not categories:
        rows = con.execute(f"SELECT count(*) FROM {source}").fetchone()[0]
        return rows, {}
    counts = ", ".join(
        "count_if(list_has_any(failed_rules, ["
        + ", ".join("'" + i.replace("'", "''") + "'" for i in by_category[c])
        + "]::VARCHAR[]))"
        for c in categories
    )
    values = con.execute(f"SELECT count(*), {counts} FROM {source}").fetchone()
    rows = values[0]
    return rows, {
        c: (1 - failed / rows if rows else 1.0)
        for c, failed in zip(categories, values[1:], strict=True)
    }


def check_results(
    con: duckdb.DuckDBPyConnection, dataset: str, period: str, sha256: str
) -> tuple[str | None, dict[str, str]]:
    """Latest check run of this file version and, per check, its worst result."""
    row = con.execute(
        "SELECT max(run_id) FROM dataset_check_results WHERE dataset = ? AND period = ? "
        "AND sha256 = ?",
        [dataset, period, sha256],
    ).fetchone()
    run_id = row[0] if row else None
    if run_id is None:
        return None, {}
    rows = con.execute(
        'SELECT "check", status, severity FROM dataset_check_results '
        "WHERE run_id = ? AND dataset = ? AND period = ?",
        [run_id, dataset, period],
    ).fetchall()
    order = ["critical", "high", "medium", "low", "pass"]
    worst: dict[str, str] = {}
    for check, status, severity in rows:
        if status == "skipped":
            continue
        result = severity if status == "fail" else "pass"
        if check not in worst or order.index(result) < order.index(worst[check]):
            worst[check] = result
    return run_id, worst


def score_month(
    con: duckdb.DuckDBPyConnection,
    dataset: Dataset,
    period: str,
    sha256: str,
    silver: Path,
    quarantine: Path,
    rules: list[Rule],
    weights: dict[str, float],
    check_scores: dict[str, float],
) -> MonthScore:
    rows, row_rates = row_pass_rates(con, silver, quarantine, rules)
    check_run, worst = check_results(con, dataset.name, period, sha256)

    rates: dict[str, tuple[float, str]] = {c: (rate, "row rules") for c, rate in row_rates.items()}
    for check, result in worst.items():
        category = CHECK_CATEGORY.get(check)
        if category is None:
            continue
        value = check_scores[result]
        source = f"{check} check: {result}"
        if category in rates:
            if value < rates[category][0]:  # the weaker signal decides
                rates[category] = (value, source)
            else:
                rates[category] = (rates[category][0], f"{rates[category][1]}; {source}")
        else:
            rates[category] = (value, source)

    parts = [
        Part(category, float(weights[category]), rate, source)
        for category, (rate, source) in rates.items()
        if weights.get(category, 0) > 0
    ]
    return MonthScore(period, sha256, combine(parts), rows, tuple(parts), check_run)


def store(
    con: duckdb.DuckDBPyConnection, dataset: str, s: MonthScore, now: datetime | None = None
) -> None:
    key = [dataset, s.period, s.sha256]
    con.execute("DELETE FROM quality_scores WHERE dataset = ? AND period = ? AND sha256 = ?", key)
    con.execute(
        "DELETE FROM quality_score_parts WHERE dataset = ? AND period = ? AND sha256 = ?", key
    )
    con.execute(
        "INSERT INTO quality_scores VALUES (?, ?, ?, ?, ?, ?, ?)",
        [*key, s.score, s.rows, s.check_run, now or utc_now()],
    )
    con.executemany(
        "INSERT INTO quality_score_parts VALUES (?, ?, ?, ?, ?, ?, ?)",
        [[*key, p.category, p.weight, p.pass_rate, p.source] for p in s.parts],
    )


def score_all(
    con: duckdb.DuckDBPyConnection,
    dataset: Dataset,
    months: list[str] | None = None,
    rules: list[Rule] | None = None,
    weights_config: dict | None = None,
) -> list[MonthScore]:
    """Score every month with Silver (current Bronze version) and store the results."""
    rules = load_rules(dataset) if rules is None else rules
    config = weights_config or load_severity_weights()
    weights = config["category_weights"]
    check_scores = config.get("check_scores") or DEFAULT_CHECK_SCORES
    versions = current_versions(con, dataset)
    silver = current_silver(con, dataset)
    scores = []
    for period in months or sorted(silver):
        if period not in silver:
            continue
        s = score_month(
            con, dataset, period, versions[period][1], *silver[period], rules, weights, check_scores
        )
        store(con, dataset.name, s)
        scores.append(s)
    return scores


def overview(scores: list[MonthScore]) -> pd.DataFrame:
    records = []
    for s in scores:
        record = {"period": s.period, "score": round(s.score, 1), "rows": s.rows}
        record.update({p.category: round(100 * p.pass_rate, 2) for p in s.parts})
        records.append(record)
    return pd.DataFrame(records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute the quality score per month.")
    parser.add_argument("--dataset", default="yellow_taxi", help="dataset name in datasets.yml")
    parser.add_argument("--database", type=Path, help="DuckDB file (default: paths.duckdb)")
    args = parser.parse_args(argv)
    try:
        dataset = load_datasets_config().dataset(args.dataset)
    except ConfigError as exc:
        print(exc)
        return 2
    with db.connect(args.database) as con:
        db.init_schema(con)
        scores = score_all(con, dataset)
    if not scores:
        print(f"no Silver for '{dataset.name}' (run process_silver.py)")
        return 1
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(overview(scores).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
