from dataclasses import replace
from pathlib import Path

import duckdb
import pandas as pd
import pytest
import yaml

from common import db
from common.config import ConfigError, load_datasets_config, load_severity_weights
from ingestion import ingest_data
from quality import baselines, run_checks, score
from quality.rules import load_rules
from quality.score import Part, combine
from silver import process_silver

FIXTURES = Path(__file__).resolve().parent / "fixtures"
WEIGHTS = load_severity_weights()["category_weights"]


def test_combine_is_a_weighted_average():
    parts = [Part("schema", 5, 1.0, ""), Part("volume", 4, 0.5, ""), Part("validity", 3, 0.9, "")]
    assert combine(parts) == pytest.approx(100 * (5 * 1.0 + 4 * 0.5 + 3 * 0.9) / 12)
    assert combine([Part("schema", 5, 1.0, "")]) == 100
    with pytest.raises(ValueError, match="no category"):
        combine([])


def test_check_scores_from_the_config():
    scores = load_severity_weights()["check_scores"]
    assert scores == {"pass": 1.0, "low": 0.75, "medium": 0.5, "high": 0.25, "critical": 0.0}


@pytest.mark.parametrize(
    ("scores", "message"),
    [
        ({"pass": 1.0}, "needs exactly"),
        ({"pass": 1.0, "low": 2, "medium": 0.5, "high": 0.25, "critical": 0}, "between 0 and 1"),
    ],
)
def test_check_scores_are_validated(tmp_path, scores, message):
    data = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "severity_weights.yml").read_text()
    )
    data["check_scores"] = scores
    path = tmp_path / "w.yml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError, match=message):
        load_severity_weights(path)


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    """Five fixture months with planted problems, written to Silver and checked."""
    from test_run_checks import FIXTURE, MONTHS

    tmp = tmp_path_factory.mktemp("score")
    incoming = tmp / "incoming"
    incoming.mkdir()
    for period, query in MONTHS.items():
        target = incoming / f"yellow_tripdata_{period}.parquet"
        duckdb.sql(f"COPY ({query.format(src=FIXTURE)}) TO '{target}' (FORMAT parquet)")
    (incoming / "taxi_zone_lookup.csv").write_bytes(
        (FIXTURES / "taxi_zone_lookup_fixture.csv").read_bytes()
    )
    database = tmp / "d.duckdb"
    common = ["--database", str(database)]
    assert (
        ingest_data.main(
            [
                "--months",
                "2023-01:2023-05",
                "--source",
                str(incoming),
                "--bronze",
                str(tmp / "bronze"),
                *common,
            ]
        )
        == 0
    )
    assert (
        process_silver.main(
            [
                "--lookups",
                str(incoming),
                "--silver",
                str(tmp / "silver"),
                "--quarantine",
                str(tmp / "quarantine"),
                *common,
            ]
        )
        == 0
    )
    taxi = load_datasets_config().dataset("yellow_taxi")
    limits = replace(
        baselines.settings_for("yellow_taxi"), rows_per_day_min=1, rows_per_day_max=100
    )
    with db.connect(database) as con:
        run_checks.run_checks(con, taxi, limits=limits)
        scores = {s.period: s for s in score.score_all(con, taxi)}
        silver = process_silver.current_silver(con, taxi)
    return {"db": database, "scores": scores, "silver": silver}


def parts(s):
    return {p.category: p for p in s.parts}


def test_row_pass_rates_match_the_written_rows(processed):
    s = processed["scores"]["2023-01"]
    silver, quarantine = processed["silver"]["2023-01"]
    rows = pd.concat([pd.read_parquet(silver), pd.read_parquet(quarantine)])
    rules = load_rules("yellow_taxi")
    for category in ("validity", "referential", "uniqueness", "pattern"):
        ids = {r.id for r in rules if r.category == category}
        failing = rows["failed_rules"].apply(lambda f, ids=ids: bool(set(f) & ids)).sum()
        assert parts(s)[category].pass_rate == pytest.approx(1 - failing / len(rows)), category
    assert s.rows == 200


def test_clean_month_score_by_hand(processed):
    s = processed["scores"]["2023-01"]
    p = parts(s)
    assert p["schema"].pass_rate == 1.0 and p["volume"].pass_rate == 1.0
    assert "drift" not in p  # skipped during the cold start: left out
    assert "ml_anomaly" not in p  # no data before Phase 7
    expected = (
        100 * sum(WEIGHTS[c] * q.pass_rate for c, q in p.items()) / sum(WEIGHTS[c] for c in p)
    )
    assert s.score == pytest.approx(expected)
    assert 95 < s.score < 100


def test_schema_change_lowers_february(processed):
    feb = processed["scores"]["2023-02"]
    assert parts(feb)["schema"].pass_rate == 0.25  # high finding
    assert parts(feb)["schema"].source == "schema check: high"
    assert feb.score < processed["scores"]["2023-03"].score


def test_null_spike_decides_completeness_in_may(processed):
    may = processed["scores"]["2023-05"]
    completeness = parts(may)["completeness"]
    assert completeness.pass_rate == 0.25 and completeness.source == "null_rate check: high"
    assert parts(may)["drift"].pass_rate == 0.25
    others = processed["scores"]
    assert may.score < min(others[p].score for p in ("2023-01", "2023-03", "2023-04"))


def test_scores_are_stored(processed):
    with db.connect(processed["db"], read_only=True) as con:
        stored = dict(con.execute("SELECT period, round(score, 6) FROM quality_scores").fetchall())
        n_parts = con.execute(
            "SELECT count(*) FROM quality_score_parts WHERE period = '2023-01'"
        ).fetchone()[0]
    assert stored == {p: round(s.score, 6) for p, s in processed["scores"].items()}
    assert n_parts == len(processed["scores"]["2023-01"].parts)


def test_main(processed, capsys):
    assert score.main(["--database", str(processed["db"])]) == 0
    out = capsys.readouterr().out
    assert "2023-05" in out and "score" in out


def test_main_without_silver(tmp_path):
    assert score.main(["--database", str(tmp_path / "e.duckdb")]) == 1
