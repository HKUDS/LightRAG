"""The shared pre-auth helpers (LR2 Phase 5-b, §9.3).

The pure-ASGI admission middleware has to answer "may this request proceed?"
*before* the first ``receive()`` — a body it is about to refuse must never be
read. It cannot run the route's dependency to find out, so the acceptance rules
are extracted into two functions used by both layers:

* ``path_is_whitelisted`` — the WHITELIST_PATHS matcher, shared with the
  enforcing route dependency, so the middleware never 401s a path the route
  itself lets through;
* ``credentials_accepted`` — the credential predicate, shared with the
  non-enforcing /health reporter.

The split matters: whitelisted is *reachable without credentials*, which is not
the same as *authenticated*. Folding the whitelist into the predicate would make
/health (itself whitelisted) report every caller as authenticated and reveal
configuration.
"""

from __future__ import annotations

import importlib
import sys

import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_utils_api = importlib.import_module("lightrag.api.utils_api")
_auth = importlib.import_module("lightrag.api.auth")
sys.argv = _original_argv

pytestmark = pytest.mark.offline


@pytest.fixture
def patterns(monkeypatch):
    def _set(entries):
        compiled = []
        for entry in entries:
            if entry.endswith("/*"):
                compiled.append((entry[:-2], True))
            else:
                compiled.append((entry, False))
        monkeypatch.setattr(_utils_api, "whitelist_patterns", compiled)

    return _set


def test_exact_and_prefix_whitelist_entries(patterns):
    patterns(["/health", "/api/*"])

    assert _utils_api.path_is_whitelisted("/health") is True
    assert _utils_api.path_is_whitelisted("/api/chat") is True
    assert _utils_api.path_is_whitelisted("/healthz") is False
    assert _utils_api.path_is_whitelisted("/documents/upload") is False


def test_catch_all_whitelist_covers_ingestion(patterns):
    """An operator who whitelists everything must not get 401s from the
    middleware on paths the route dependency waves through."""
    patterns(["/*"])

    assert _utils_api.path_is_whitelisted("/documents/upload") is True


def test_fully_open_mode_accepts_anything(monkeypatch):
    monkeypatch.setattr(_utils_api, "auth_configured", False)

    assert (
        _utils_api.credentials_accepted(
            token=None, api_key_header_value=None, api_key=None
        )
        is True
    )


def test_api_key_mode_requires_the_key(monkeypatch):
    monkeypatch.setattr(_utils_api, "auth_configured", False)

    assert (
        _utils_api.credentials_accepted(
            token=None, api_key_header_value=None, api_key="secret"
        )
        is False
    )
    assert (
        _utils_api.credentials_accepted(
            token=None, api_key_header_value="wrong", api_key="secret"
        )
        is False
    )
    assert (
        _utils_api.credentials_accepted(
            token=None, api_key_header_value="secret", api_key="secret"
        )
        is True
    )


def test_guest_token_never_substitutes_for_the_api_key(monkeypatch):
    """GHSA-f4vv-55c2-5789: a guest token is obtainable (or forgeable with the
    public default secret) by anyone, so it must not authenticate in
    API-key-only mode."""
    monkeypatch.setattr(_utils_api, "auth_configured", False)
    guest = _auth.auth_handler.create_token(username="guest", role="guest")

    assert (
        _utils_api.credentials_accepted(
            token=guest, api_key_header_value=None, api_key="secret"
        )
        is False
    )


def test_password_mode_accepts_a_non_guest_token(monkeypatch):
    monkeypatch.setattr(_utils_api, "auth_configured", True)
    user = _auth.auth_handler.create_token(username="alice", role="user")
    guest = _auth.auth_handler.create_token(username="guest", role="guest")

    assert (
        _utils_api.credentials_accepted(
            token=user, api_key_header_value=None, api_key=None
        )
        is True
    )
    # A guest token is not a login.
    assert (
        _utils_api.credentials_accepted(
            token=guest, api_key_header_value=None, api_key=None
        )
        is False
    )


def test_garbage_token_is_rejected_not_raised(monkeypatch):
    monkeypatch.setattr(_utils_api, "auth_configured", True)

    assert (
        _utils_api.credentials_accepted(
            token="not-a-jwt", api_key_header_value=None, api_key=None
        )
        is False
    )
