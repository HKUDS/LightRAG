"""Operator selection is a name, with CLI taking precedence over .env."""

import sys

import pytest

from lightrag.api.config import parse_args

pytestmark = pytest.mark.offline


@pytest.mark.parametrize("value", ["", "company-chunker"])
def test_custom_chunker_env(monkeypatch, value):
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("CUSTOM_CHUNKER", value)
    assert parse_args().custom_chunker == value


def test_custom_chunker_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("CUSTOM_CHUNKER", "env-chunker")
    monkeypatch.setattr(
        sys, "argv", ["lightrag-server", "--custom-chunker", "cli-chunker"]
    )
    assert parse_args().custom_chunker == "cli-chunker"
