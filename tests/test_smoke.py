import importlib

import pytest

PACKAGES = [
    "alerts",
    "common",
    "dashboard",
    "data_download",
    "evaluation",
    "gold",
    "incidents",
    "ingestion",
    "investigation",
    "quality",
    "silver",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name):
    importlib.import_module(name)
