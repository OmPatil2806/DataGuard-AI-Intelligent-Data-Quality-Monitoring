"""ved_profile.py (Phase 9.1): the profile of a small VED in the real files' shape."""

import duckdb
import pytest
from ved_fixture import ROWS_PER_TRIP, VEHICLES, make_unpacked, write_week

from profiling import ved_profile as vp


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    root = tmp_path_factory.mktemp("ved")
    source = make_unpacked(root / "incoming")
    return vp.profile(source, root / "cache")


def test_every_row_is_read(result):
    files = result["files"]
    assert files["file"].tolist() == [
        "VED_171101_week.csv", "VED_171108_week.csv", "VED_171129_week.csv",
    ]  # fmt: skip
    assert (files["data_lines"] == files["rows_read"]).all()
    assert files["header_crlf"].all() and not files["rows_crlf"].any()
    assert files["week_start"].tolist()[:2] == ["2017-11-01", "2017-11-08"]  # yymmdd
    summary = vp.summary(result)
    assert summary["rows_lost_by_reader"] == 0
    assert summary["rows"] == (5 * 3 * 3 + 2) * ROWS_PER_TRIP


def test_engine_types_from_both_static_files(result):
    vehicles = result["vehicles"]
    assert vehicles["vehicles_with_rows"] == len(VEHICLES) == vehicles["vehicles_in_static_data"]
    assert vehicles["vehicles_with_rows_by_engine_type"] == {"EV": 1, "HEV": 1, "ICE": 2, "PHEV": 1}
    assert vehicles["with_rows_not_in_static"] == []
    nulls = result["nulls_by_engine"].set_index("engine_type")
    assert (
        nulls.loc["EV", "Engine RPM[RPM]"] == 1.0 and nulls.loc["ICE", "HV Battery SOC[%]"] == 1.0
    )
    assert nulls.loc["PHEV", "Fuel Rate[L/hr]"] == 0.0
    quality = result["static_quality"].set_index("column")
    assert quality.loc["Vehicle Class", "no_data"] == len(VEHICLES)


def test_time_from_daynum_and_milliseconds(result):
    files = result["files"].set_index("file")
    assert str(files.loc["VED_171101_week.csv", "first_time"]).startswith(
        "2017-11-01 09:07:12"
    )  # DayNum 1.38
    assert result["outside_week"] == {
        "rows_before_week_start": 0,
        "rows_after_week_end": 31,
    }  # 9 of 40 before midnight
    assert result["trips"]["trips_with_more_than_one_daynum"] == 0
    assert result["months"]["month"].tolist() == ["2017-11", "2017-12"]
    columns = result["columns"].set_index("column")
    assert columns.loc["Timestamp(ms)", "min"] == -5000  # a trip that starts negative


def test_no_duplicates_and_gps(result):
    assert result["duplicates"] == {"key_groups_repeated": 0, "extra_rows_on_key": 0,
                                    "extra_rows_identical_in_every_column": 0}  # fmt: skip
    lat = result["gps"]["latitude_min_max"]
    assert 42.1 < lat[0] < lat[1] < 42.4


def test_consistency(result):
    c = result["consistency"].set_index("engine_type")
    assert c.loc["EV", "moving_with_rpm_0_or_null"] == 1.0  # no engine
    assert c.loc["ICE", "fuel_rate_present"] == 0.0 and c.loc["PHEV", "fuel_rate_present"] == 1.0


def test_duplicates_are_found(tmp_path):
    source = make_unpacked(tmp_path / "incoming")
    week = source / "VED_171108_week.csv"
    lines = week.read_bytes().split(b"\n")
    week.write_bytes(b"\n".join([*lines[:-1], lines[1], lines[2], b""]))  # two rows again
    dup = vp.profile(source, tmp_path / "cache")["duplicates"]
    assert dup == {"key_groups_repeated": 2, "extra_rows_on_key": 2,
                   "extra_rows_identical_in_every_column": 2}  # fmt: skip


def test_the_duckdb_csv_pitfall_is_detected(tmp_path):
    """The reason the profile reads with pyarrow: the line endings are mixed."""
    path = write_week(tmp_path, "171101")
    header_crlf, rows_crlf, lines = vp._line_endings(path)
    assert (header_crlf, rows_crlf) == (True, False)
    assert lines == len(path.read_bytes().split(b"\n")) - 2  # header and the final newline
    assert duckdb.__version__  # the reader in use; see docs/ved_profile.md for the measured loss


def test_main(tmp_path, capsys, monkeypatch):
    source = make_unpacked(tmp_path / "incoming")
    monkeypatch.setenv("DATAGUARD_VED_DIR", str(source))
    monkeypatch.setenv("DATAGUARD_VED_CACHE", str(tmp_path / "cache"))
    assert vp.main([]) == 0
    out = capsys.readouterr().out
    assert "== summary ==" in out and (tmp_path / "cache" / "profile.json").is_file()
    assert vp.main(["--source", str(tmp_path / "empty")]) == 2
