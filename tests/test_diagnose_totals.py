import duckdb
import pytest

from profiling import diagnose_totals
from profiling import taxi_profile as tp

COLUMNS = (
    "VendorID, fare_amount, extra, mta_tax, tip_amount, tolls_amount, "
    "improvement_surcharge, congestion_surcharge, Airport_fee, total_amount"
)
# fare 10, extra 1, mta 0.5, tip 2, tolls 0, improvement 1, congestion 2.5, airport 0 -> 17.0
ROWS = [
    (2, 10.0, 1.0, 0.5, 2.0, 0.0, 1.0, 2.5, 0.0, 17.0),  # full
    (2, 10.0, 1.0, 0.5, 2.0, 0.0, 1.0, 2.5, 0.0, 17.0),  # full
    (1, 10.0, 3.5, 0.5, 2.0, 0.0, 1.0, 2.5, 0.0, 17.0),  # base: extra already holds 2.50
    (1, 10.0, 1.0, 0.5, 2.0, 0.0, 1.0, 2.5, 0.0, 17.0),  # full
    (2, 10.0, 1.0, 0.5, 2.0, 0.0, 1.0, 2.5, 0.0, 30.0),  # none: +13.00
    (2, 10.0, 1.0, 0.5, 2.0, 0.0, 1.0, 2.5, 0.0, 30.0),  # none: +13.00
    (2, 10.0, 1.0, 0.5, 2.0, 0.0, 1.0, 2.5, 0.0, 16.0),  # none: -1.00
]


@pytest.fixture
def files(tmp_path):
    path = tmp_path / "yellow_tripdata_2023-07.parquet"
    values = ", ".join(str(r) for r in ROWS)
    duckdb.sql(f"COPY (SELECT * FROM (VALUES {values}) t({COLUMNS})) TO '{path}' (FORMAT parquet)")
    return [path]


def test_classes_per_vendor(files):
    with tp.connect() as con:
        per_file, _ = tp.total_diagnosis(con, files)
    by_vendor = per_file.set_index("vendor")
    assert by_vendor.loc[1, ["full", "base", "none"]].tolist() == [1, 1, 0]
    assert by_vendor.loc[2, ["full", "base", "none"]].tolist() == [2, 0, 3]
    assert by_vendor.loc[2, "none_pct"] == 60.0


def test_most_common_differences(files):
    with tp.connect() as con:
        _, diffs = tp.total_diagnosis(con, files)
    vendor2 = diffs[diffs["vendor"] == 2]
    assert vendor2[["difference", "rows"]].values.tolist() == [[13.0, 2], [-1.0, 1]]
    vendor1 = diffs[diffs["vendor"] == 1]
    # extra already holds the 2.50, so the full sum counts it twice: total - sum = -2.50
    assert vendor1[["difference", "rows"]].values.tolist() == [[-2.5, 1]]


def test_main_writes_report(files, tmp_path, capsys):
    out = tmp_path / "report.txt"
    assert diagnose_totals.main(["--source", str(files[0].parent), "--out", str(out)]) == 0
    text = out.read_text()
    assert "1. ALL FILES BY VENDOR" in text and "3. MOST COMMON DIFFERENCES" in text
    assert "written to" in capsys.readouterr().out


def test_main_without_files(tmp_path):
    assert diagnose_totals.main(["--source", str(tmp_path)]) == 1
