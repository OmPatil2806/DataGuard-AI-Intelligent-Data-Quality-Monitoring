import duckdb
import pytest

from data_download.make_sample import main, sample_parquet


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "incoming" / "yellow_tripdata_2023-01.parquet"
    path.parent.mkdir()
    query = "SELECT range AS id, range * 1.5 AS fare FROM range(20000)"
    duckdb.sql(f"COPY ({query}) TO '{path}' (FORMAT parquet)")
    return path


def ids(path):
    return [r[0] for r in duckdb.sql(f"SELECT id FROM read_parquet('{path}')").fetchall()]


def test_sample_size_order_and_columns(source, tmp_path):
    dest = tmp_path / "sample" / source.name
    result = sample_parquet(source, dest, fraction=0.1, seed=42)
    assert result.status == "created"
    assert result.source_rows == 20000
    assert 1800 <= result.sample_rows <= 2200
    sampled = ids(dest)
    assert sampled == sorted(sampled)
    assert set(sampled) <= set(range(20000))
    assert duckdb.sql(f"SELECT * FROM read_parquet('{dest}')").columns == ["id", "fare"]
    assert not dest.with_name(dest.name + ".part").exists()


def test_same_seed_same_rows_other_seed_other_rows(source, tmp_path):
    a, b, c = (tmp_path / name / source.name for name in ("a", "b", "c"))
    sample_parquet(source, a, seed=42)
    sample_parquet(source, b, seed=42)
    sample_parquet(source, c, seed=7)
    assert ids(a) == ids(b)
    assert ids(a) != ids(c)


def test_skips_existing_unless_forced(source, tmp_path):
    dest = tmp_path / "sample" / source.name
    sample_parquet(source, dest)
    assert sample_parquet(source, dest).status == "skipped"
    assert sample_parquet(source, dest, fraction=0.5, force=True).sample_rows > 9000


def test_missing_source(tmp_path):
    result = sample_parquet(tmp_path / "nope.parquet", tmp_path / "out.parquet")
    assert result.status == "missing"


def test_rejects_bad_fraction(source, tmp_path):
    with pytest.raises(ValueError, match="fraction"):
        sample_parquet(source, tmp_path / "out.parquet", fraction=0)


def test_main_samples_and_copies_lookup(source, tmp_path):
    (source.parent / "taxi_zone_lookup.csv").write_text("LocationID,Borough\n1,EWR\n")
    dest = tmp_path / "sample"
    argv = ["--months", "2023-01:2023-02", "--source", str(source.parent), "--dest", str(dest)]
    assert main(argv) == 0  # 2023-02 is missing, but 2023-01 was sampled
    assert sorted(p.name for p in dest.iterdir()) == [
        "taxi_zone_lookup.csv",
        "yellow_tripdata_2023-01.parquet",
    ]


def test_main_fails_when_nothing_to_sample(tmp_path):
    argv = ["--months", "2023-01", "--source", str(tmp_path), "--dest", str(tmp_path / "s")]
    assert main(argv) == 1
    assert main(["--fraction", "1.5", "--source", str(tmp_path)]) == 2
