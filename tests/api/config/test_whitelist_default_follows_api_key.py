"""WHITELIST_PATHS defaults to following LIGHTRAG_API_KEY.

Unset, it exempts the Ollama-compatible ``/api/*`` routes only while no API key
is configured: the key is the one credential an Ollama client can carry, so once
it exists the routes require it. ``AUTH_ACCOUNTS`` alone does not change the
default, because requiring a login there would lock every Ollama client out. An
explicit value always wins, the old default included.
"""

from __future__ import annotations

import sys

import pytest

from lightrag.api.config import default_whitelist_paths, parse_args

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    # The module is imported above, so its one-time load_dotenv has already
    # run: a deleted variable stays deleted for parse_args.
    monkeypatch.delenv("WHITELIST_PATHS", raising=False)
    monkeypatch.delenv("LIGHTRAG_API_KEY", raising=False)
    monkeypatch.delenv("AUTH_ACCOUNTS", raising=False)
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    monkeypatch.setenv("LLM_BINDING", "openai")


def test_without_a_key_the_ollama_routes_stay_open():
    assert parse_args().whitelist_paths == "/health,/api/*"


def test_a_key_from_the_environment_protects_the_ollama_routes(monkeypatch):
    monkeypatch.setenv("LIGHTRAG_API_KEY", "an-api-key")

    assert parse_args().whitelist_paths == "/health"


def test_a_key_from_the_command_line_counts_too(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["lightrag-server", "--key", "an-api-key"])

    assert parse_args().whitelist_paths == "/health"


def test_accounts_alone_leave_the_default_open(monkeypatch):
    monkeypatch.setenv("AUTH_ACCOUNTS", "reader:a-strong-secret")
    monkeypatch.setenv("TOKEN_SECRET", "a-non-default-token-secret")

    assert parse_args().whitelist_paths == "/health,/api/*"


@pytest.mark.parametrize("explicit", ["/health,/api/*", "/health", ""])
def test_an_explicit_value_always_wins(monkeypatch, explicit):
    monkeypatch.setenv("LIGHTRAG_API_KEY", "an-api-key")
    monkeypatch.setenv("WHITELIST_PATHS", explicit)

    assert parse_args().whitelist_paths == explicit


@pytest.mark.parametrize(
    "api_key,expected",
    [(None, "/health,/api/*"), ("", "/health,/api/*"), ("k", "/health")],
)
def test_default_whitelist_paths(api_key, expected):
    assert default_whitelist_paths(api_key) == expected
