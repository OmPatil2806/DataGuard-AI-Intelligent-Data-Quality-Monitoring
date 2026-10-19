"""Row-level rule engine: compile the YAML rules (quality/rules.py) into DuckDB SQL.

    sql = evaluate_sql(rules, source_sql, period="2023-01")

returns every source column plus

    failed_rules   VARCHAR[]  ids of the rules the row fails, in rule order
    reasons        VARCHAR[]  one reason per failed rule, with the row's values filled in
    quarantine     BOOLEAN    true when at least one failed rule has action 'quarantine'

Every rule becomes a boolean "row fails" expression that is FALSE when the values it reads
are NULL (missing values are reported once, by not_null rules) or when its `when`
condition does not hold. The rules run in one pass over the data, so millions of rows are
evaluated in seconds.

check_schema() uses Pandera to confirm that standardized data has the standard column
names and types before the rules run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd
import pandera.pandas as pa
import pyarrow as pyarrow_types

from common.config import Dataset
from quality.rules import PLACEHOLDER, Rule

ORDER_COLUMN = "bronze_row_id"  # rows that repeat a unique key are failed after the first
FALLBACK_ORDER = "__dg_row"
LOOKUP_PREFIX = "lookup_"
SCHEMA_SAMPLE_ROWS = 10_000

ARROW_TYPES = {
    "BOOLEAN": pyarrow_types.bool_(),
    "INTEGER": pyarrow_types.int32(),
    "BIGINT": pyarrow_types.int64(),
    "DOUBLE": pyarrow_types.float64(),
    "VARCHAR": pyarrow_types.string(),
    "DATE": pyarrow_types.date32(),
    "TIMESTAMP": pyarrow_types.timestamp("us"),
}


class RuleError(ValueError):
    """Raised when a rule cannot be compiled for the given data."""


@dataclass(frozen=True)
class RuleCount:
    rule_id: str
    action: str
    failed: int
    rows: int

    @property
    def pct(self) -> float:
        return round(100.0 * self.failed / self.rows, 4) if self.rows else 0.0


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _literal(value: object) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int | float):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def _sql_path(path: Path) -> str:
    return "'" + str(path).replace("\\", "/").replace("'", "''") + "'"


def _amount_sum(columns: list[str]) -> str:
    return " + ".join(f"coalesce({_q(c)}, 0)" for c in columns)


def failure_sql(rule: Rule, period: str | None = None) -> str:
    """SQL that is TRUE exactly when a row fails the rule."""
    p = rule.params
    check = rule.check
    if check == "not_null":
        joiner = " OR " if p.get("mode", "any") == "any" else " AND "
        expr = joiner.join(f"{_q(c)} IS NULL" for c in p["columns"])
    elif check == "range":
        col = _q(p["column"])
        parts = []
        if "min" in p:
            parts.append(f"{col} {'<=' if p.get('exclusive_min') else '<'} {_literal(p['min'])}")
        if "max" in p:
            parts.append(f"{col} {'>=' if p.get('exclusive_max') else '>'} {_literal(p['max'])}")
        expr = " OR ".join(parts)
    elif check == "allowed_values":
        values = ", ".join(_literal(v) for v in p["values"])
        expr = f"{_q(p['column'])} NOT IN ({values})"
    elif check == "forbidden_values":
        values = ", ".join(_literal(v) for v in p["values"])
        expr = f"{_q(p['column'])} IN ({values})"
    elif check == "compare":
        op = "<>" if p["op"] == "!=" else p["op"]
        expr = f"NOT ({_q(p['left'])} {op} {_q(p['right'])})"
    elif check == "duration":
        seconds = f"(epoch({_q(p['end'])}) - epoch({_q(p['start'])}))"
        parts = []
        if "min_seconds" in p:
            parts.append(f"{seconds} < {_literal(p['min_seconds'])}")
        if "max_seconds" in p:
            parts.append(f"{seconds} > {_literal(p['max_seconds'])}")
        expr = " OR ".join(parts)
    elif check == "within_period":
        if not period:
            raise RuleError(f"rule '{rule.id}' needs the file's period (YYYY-MM)")
        start = f"TIMESTAMP '{period}-01'"
        col = _q(p["column"])
        expr = f"{col} < {start} OR {col} >= {start} + INTERVAL 1 MONTH"
    elif check == "foreign_key":
        table = _q(LOOKUP_PREFIX + p["lookup"])
        key = _q(p["lookup_column"])
        expr = f"{_q(p['column'])} NOT IN (SELECT {key} FROM {table} WHERE {key} IS NOT NULL)"
    elif check == "unique":
        key = ", ".join(_q(c) for c in p["columns"])
        expr = f"row_number() OVER (PARTITION BY {key} ORDER BY {{order}}) > 1"
    elif check == "sum_equals":
        tolerance = _literal(p.get("tolerance", 0.01))
        total = _q(p["total"])
        options = [p["parts"], *p.get("alternatives", [])]
        expr = " AND ".join(
            f"abs({total} - ({_amount_sum(parts)})) > {tolerance}" for parts in options
        )
    else:  # pragma: no cover - rules.py rejects unknown checks
        raise RuleError(f"unknown check '{check}'")

    expr = f"coalesce(({expr}), FALSE)"
    if rule.when:
        expr = f"(coalesce(({rule.when}), FALSE) AND {expr})"
    return expr


def reason_sql(rule: Rule) -> str:
    """SQL text for the rule's reason with {column} placeholders replaced by row values."""
    pieces: list[str] = []
    position = 0
    for match in PLACEHOLDER.finditer(rule.reason):
        if match.start() > position:
            pieces.append(_literal(rule.reason[position : match.start()]))
        pieces.append(f"coalesce(CAST({_q(match.group(1))} AS VARCHAR), 'NULL')")
        position = match.end()
    if position < len(rule.reason):
        pieces.append(_literal(rule.reason[position:]))
    return "concat(" + ", ".join(pieces) + ")" if len(pieces) > 1 else pieces[0]


def _ordered_source(con: duckdb.DuckDBPyConnection, source_sql: str) -> tuple[str, str]:
    """Make sure the rows have a stable order for unique rules."""
    names = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {source_sql}").fetchall()]
    if ORDER_COLUMN in names:
        return f"(SELECT * FROM {source_sql})", _q(ORDER_COLUMN)
    return (
        f"(SELECT *, row_number() OVER () AS {FALLBACK_ORDER} FROM {source_sql})",
        FALLBACK_ORDER,
    )


def _flag_columns(rules: list[Rule], period: str | None, order: str) -> list[str]:
    return [
        f"{failure_sql(rule, period).replace('{order}', order)} AS __f{i}"
        for i, rule in enumerate(rules)
    ]


def evaluate_sql(
    con: duckdb.DuckDBPyConnection,
    rules: list[Rule],
    source_sql: str,
    period: str | None = None,
) -> str:
    """SQL returning the source rows with failed_rules, reasons and quarantine."""
    source, order = _ordered_source(con, source_sql)
    if not rules:
        drop = f" EXCLUDE ({FALLBACK_ORDER})" if order == FALLBACK_ORDER else ""
        return (
            f"SELECT *{drop}, []::VARCHAR[] AS failed_rules, []::VARCHAR[] AS reasons, "
            f"FALSE AS quarantine FROM {source}"
        )
    flags = _flag_columns(rules, period, order)
    ids = ", ".join(f"CASE WHEN __f{i} THEN {_literal(r.id)} END" for i, r in enumerate(rules))
    reasons = ", ".join(f"CASE WHEN __f{i} THEN {reason_sql(r)} END" for i, r in enumerate(rules))
    quarantine = (
        " OR ".join(f"__f{i}" for i, r in enumerate(rules) if r.action == "quarantine") or "FALSE"
    )
    hidden = [f"__f{i}" for i in range(len(rules))]
    if order == FALLBACK_ORDER:
        hidden.append(FALLBACK_ORDER)
    return f"""
        SELECT * EXCLUDE ({", ".join(hidden)}),
               list_filter([{ids}], x -> x IS NOT NULL) AS failed_rules,
               list_filter([{reasons}], x -> x IS NOT NULL) AS reasons,
               ({quarantine}) AS quarantine
        FROM (SELECT *, {", ".join(flags)} FROM {source})
    """


def count_failures(
    con: duckdb.DuckDBPyConnection,
    rules: list[Rule],
    source_sql: str,
    period: str | None = None,
) -> list[RuleCount]:
    """How many rows fail each rule, computed in one pass."""
    source, order = _ordered_source(con, source_sql)
    if not rules:
        return []
    flags = _flag_columns(rules, period, order)
    counts = ", ".join(f"count_if(__f{i})" for i in range(len(rules)))
    values = con.execute(
        f"SELECT count(*), {counts} FROM (SELECT {', '.join(flags)} FROM {source})"
    ).fetchone()
    rows = values[0]
    return [
        RuleCount(rule.id, rule.action, failed, rows)
        for rule, failed in zip(rules, values[1:], strict=True)
    ]


def load_lookups(con: duckdb.DuckDBPyConnection, dataset: Dataset, folder: Path) -> list[str]:
    """Create one lookup_<name> table per dataset lookup, read from `folder`."""
    created = []
    for lookup in dataset.lookups:
        path = Path(folder) / lookup.file
        if not path.is_file():
            raise FileNotFoundError(f"lookup file not found: {path} (run download_data.py)")
        table = _q(LOOKUP_PREFIX + lookup.name)
        # Default CSV settings keep text such as "N/A" as text, not as a missing value.
        con.execute(
            f"CREATE OR REPLACE TEMP TABLE {table} AS SELECT * FROM read_csv({_sql_path(path)})"
        )
        created.append(LOOKUP_PREFIX + lookup.name)
    return created


def pandera_schema(dataset: Dataset) -> pa.DataFrameSchema:
    """The standard schema as a Pandera schema (Arrow-backed pandas types)."""
    return pa.DataFrameSchema(
        {
            name: pa.Column(pd.ArrowDtype(ARROW_TYPES[dtype]), nullable=True)
            for name, dtype in dataset.columns
        },
        strict=False,
        ordered=True,
        name=f"{dataset.name}_standard",
    )


def check_schema(
    con: duckdb.DuckDBPyConnection,
    sql: str,
    dataset: Dataset,
    sample_rows: int = SCHEMA_SAMPLE_ROWS,
) -> None:
    """Validate column names, order and types of `sql` against the standard schema.

    Types are fixed for a whole query, so a sample of rows is enough. Raises
    pandera.errors.SchemaError when the data does not match.
    """
    table = con.execute(f"SELECT * FROM ({sql}) LIMIT {int(sample_rows)}").arrow()
    if hasattr(table, "read_all"):  # newer DuckDB returns a record batch reader
        table = table.read_all()
    pandera_schema(dataset).validate(table.to_pandas(types_mapper=pd.ArrowDtype))
