"""The startup banner's "Ollama routes are open" warning, and when it fires.

The warning (GHSA-mmg5-8x8q-v934) tells an operator who enabled authentication
that the default WHITELIST_PATHS still leaves /api/chat and /api/generate
unauthenticated. It used to be gated on a non-loopback bind alone, which silently
skipped the canonical multi-site deployment: with LIGHTRAG_API_PREFIX set, the
reverse proxy holds the public interface and the backend binds loopback, so the
routes are reachable from the network while the bind address says otherwise.

The prefix is normalized before it is believed. LIGHTRAG_API_PREFIX=/ (or
whitespace) means "no prefix", and reading the raw value would warn about a
reverse proxy that does not exist.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.offline

_WARNING_MARKER = "remain publicly accessible"


def _splash(monkeypatch, *, host: str, api_prefix: str | None) -> str:
    """Render the real banner over real parsed args and return its output.

    ASCIIColors writes through its own stream rather than the stdout capsys
    replaces, so the collector below captures the arguments it is handed."""
    # WHITELIST_PATHS is pinned to the shipped default: a developer-local .env
    # value would otherwise decide whether the warning applies at all. setenv,
    # not delenv -- load_dotenv(override=False) would re-populate a deleted one.
    monkeypatch.setenv("WHITELIST_PATHS", "/health,/api/*")
    monkeypatch.setenv("LIGHTRAG_API_KEY", "an-api-key")

    argv = ["lightrag-server", "--host", host]
    if api_prefix is not None:
        argv += ["--api-prefix", api_prefix]

    original_argv = sys.argv.copy()
    try:
        sys.argv = argv
        config = importlib.import_module("lightrag.api.config")
        utils_api = importlib.import_module("lightrag.api.utils_api")
        args = config.parse_args()
        # The matcher reads module state captured at import time; the banner reads
        # args. Keep them consistent.
        monkeypatch.setattr(
            utils_api, "whitelist_patterns", [("/health", False), ("/api", True)]
        )
        colors = MagicMock()
        monkeypatch.setattr(utils_api, "ASCIIColors", colors)
        utils_api.display_splash_screen(args)
    finally:
        sys.argv = original_argv

    return "\n".join(str(arg) for call in colors.mock_calls for arg in call.args)


def test_warns_on_a_network_bind(monkeypatch):
    """The original trigger, unchanged."""
    output = _splash(monkeypatch, host="0.0.0.0", api_prefix=None)

    assert _WARNING_MARKER in output


def test_warns_behind_a_reverse_proxy_prefix_on_loopback(monkeypatch):
    """The canonical multi-site deployment: nginx public, backend on loopback.

    Silent before this change -- the loopback bind suppressed the warning even
    though the routes were served to the internet through the proxy.
    """
    output = _splash(monkeypatch, host="127.0.0.1", api_prefix="/site01")

    assert _WARNING_MARKER in output
    assert "reverse-proxy prefix" in output


def test_silent_on_loopback_without_a_prefix(monkeypatch):
    """Nothing is reachable, so nothing to warn about."""
    output = _splash(monkeypatch, host="127.0.0.1", api_prefix=None)

    assert _WARNING_MARKER not in output


@pytest.mark.parametrize("api_prefix", ["", "/", "   "])
def test_an_empty_prefix_is_not_a_reverse_proxy(monkeypatch, api_prefix):
    """All three values normalize to "no prefix", so the loopback bind still
    means unreachable. Believing the raw value would warn about a proxy that is
    not there."""
    output = _splash(monkeypatch, host="127.0.0.1", api_prefix=api_prefix)

    assert _WARNING_MARKER not in output
