import pytest
import requests

from common.config import ConfigError, load_datasets_config
from data_download.download_data import (
    Target,
    build_targets,
    download,
    is_valid_file,
    main,
    parse_months,
)

PARQUET = b"PAR1" + b"\x00" * 32 + b"PAR1"
CSV = b'"LocationID","Borough","Zone","service_zone"\n1,"EWR","Newark Airport","EWR"\n'
HTML = b"<html><body>Access denied</body></html>"


class FakeResponse:
    def __init__(self, body=b"", status=200):
        self.body = body
        self.status_code = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)

    def iter_content(self, chunk_size):
        for i in range(0, len(self.body), chunk_size):
            yield self.body[i : i + chunk_size]


class FakeSession:
    """Returns queued responses per URL; an Exception in the queue is raised instead."""

    def __init__(self, responses=None, default=None):
        self.responses = {url: list(items) for url, items in (responses or {}).items()}
        self.default = default
        self.calls = []

    def get(self, url, stream, timeout):
        self.calls.append(url)
        queue = self.responses.get(url)
        item = queue.pop(0) if queue else self.default(url)
        if isinstance(item, Exception):
            raise item
        return item


def by_extension(url):
    return FakeResponse(CSV if url.endswith(".csv") else PARQUET)


@pytest.fixture
def target(tmp_path):
    return Target("https://example.com/a.parquet", tmp_path / "a.parquet", "parquet")


def test_downloads_and_validates(target):
    result = download(target, FakeSession(default=by_extension))
    assert result.status == "downloaded"
    assert target.dest.read_bytes() == PARQUET
    assert not target.dest.with_name("a.parquet.part").exists()


def test_skips_existing_valid_file(target):
    target.dest.write_bytes(PARQUET)
    session = FakeSession(default=by_extension)
    assert download(target, session).status == "skipped"
    assert session.calls == []


def test_force_downloads_again(target):
    target.dest.write_bytes(PARQUET)
    session = FakeSession(default=by_extension)
    assert download(target, session, force=True).status == "downloaded"
    assert len(session.calls) == 1


def test_replaces_existing_corrupt_file(target):
    target.dest.write_bytes(b"truncated")
    assert download(target, FakeSession(default=by_extension)).status == "downloaded"
    assert target.dest.read_bytes() == PARQUET


def test_retries_with_backoff_then_succeeds(target):
    session = FakeSession(
        {target.url: [requests.ConnectionError("reset"), FakeResponse(status=503)]},
        default=by_extension,
    )
    waits = []
    result = download(target, session, sleep=waits.append)
    assert result.status == "downloaded"
    assert waits == [2, 4]


def test_gives_up_after_retries_and_leaves_no_file(target):
    session = FakeSession(default=lambda url: requests.ConnectionError("down"))
    waits = []
    result = download(target, session, retries=3, sleep=waits.append)
    assert result.status == "failed"
    assert "down" in result.error
    assert waits == [2, 4]
    assert not target.dest.exists()
    assert not target.dest.with_name("a.parquet.part").exists()


def test_does_not_retry_not_found(target):
    session = FakeSession(default=lambda url: FakeResponse(status=404))
    waits = []
    assert download(target, session, sleep=waits.append).status == "failed"
    assert waits == []
    assert len(session.calls) == 1


def test_rejects_html_saved_as_parquet(target):
    result = download(target, FakeSession(default=lambda url: FakeResponse(HTML)))
    assert result.status == "failed"
    assert "not valid parquet" in result.error
    assert not target.dest.exists()


@pytest.mark.parametrize(
    ("content", "kind", "valid"),
    [
        (PARQUET, "parquet", True),
        (b"PAR1 no footer", "parquet", False),
        (b"PAR1", "parquet", False),
        (CSV, "csv", True),
        (HTML, "csv", False),
        (b"", "csv", False),
    ],
)
def test_is_valid_file(tmp_path, content, kind, valid):
    path = tmp_path / f"f.{kind}"
    path.write_bytes(content)
    assert is_valid_file(path, kind) is valid


def test_parse_months():
    configured = ("2023-01", "2023-02")
    assert parse_months(None, configured) == ["2023-01", "2023-02"]
    assert parse_months("2023-03", configured) == ["2023-03"]
    assert parse_months("2023-11:2024-01, 2019-01", configured) == [
        "2019-01",
        "2023-11",
        "2023-12",
        "2024-01",
    ]
    with pytest.raises(ConfigError, match="invalid month 'March'"):
        parse_months("March", configured)


def test_build_targets_uses_config(tmp_path):
    taxi = load_datasets_config().dataset("yellow_taxi")
    targets = build_targets(taxi, ["2023-01"], tmp_path)
    assert [t.dest.name for t in targets] == [
        "yellow_tripdata_2023-01.parquet",
        "taxi_zone_lookup.csv",
    ]
    assert targets[0].url.endswith("/trip-data/yellow_tripdata_2023-01.parquet")
    assert [t.kind for t in targets] == ["parquet", "csv"]
    assert len(build_targets(taxi, ["2023-01"], tmp_path, lookups=False)) == 1


def test_main_end_to_end(tmp_path):
    session = FakeSession(default=by_extension)
    argv = ["--months", "2023-01:2023-02", "--dest", str(tmp_path)]
    assert main(argv, session=session) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "taxi_zone_lookup.csv",
        "yellow_tripdata_2023-01.parquet",
        "yellow_tripdata_2023-02.parquet",
    ]
    # A second run downloads nothing.
    assert main(argv, session=session) == 0
    assert len(session.calls) == 3


def test_main_reports_failures(tmp_path):
    session = FakeSession(default=lambda url: FakeResponse(status=404))
    assert main(["--months", "2023-01", "--dest", str(tmp_path)], session=session) == 1


def test_main_rejects_bad_arguments(tmp_path):
    assert main(["--months", "2023-13", "--dest", str(tmp_path)]) == 2
    assert main(["--dataset", "green_taxi", "--dest", str(tmp_path)]) == 2
