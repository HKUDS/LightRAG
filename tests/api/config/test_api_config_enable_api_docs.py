"""Offline tests for the ENABLE_API_DOCS switch parsing (issue #3666, RFC #3671)."""

from __future__ import annotations

import sys

import pytest

from lightrag.api.config import parse_args


pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    monkeypatch.delenv("ENABLE_API_DOCS", raising=False)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("LLM_BINDING", "openai")


def test_enable_api_docs_defaults_to_true():
    args = parse_args()

    assert args.enable_api_docs is True


@pytest.mark.parametrize(
    "value,expected",
    [
        ("true", True),
        ("false", False),
        ("True", True),
        ("False", False),
    ],
)
def test_enable_api_docs_parses_as_a_bool(monkeypatch, value, expected):
    monkeypatch.setenv("ENABLE_API_DOCS", value)

    args = parse_args()

    assert args.enable_api_docs is expected
