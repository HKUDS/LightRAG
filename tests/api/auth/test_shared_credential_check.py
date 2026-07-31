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


def _scope(path: str, root_path: str = "") -> dict:
    """A minimal ASGI scope. ``path`` is in canonical form: it includes
    ``root_path``, the way FastAPI hands it to the route dependency."""
    return {"type": "http", "path": path, "root_path": root_path}


def test_exact_and_prefix_whitelist_entries(patterns):
    patterns(["/health", "/api/*"])

    assert _utils_api.path_is_whitelisted(_scope("/health")) is True
    assert _utils_api.path_is_whitelisted(_scope("/api/chat")) is True
    assert _utils_api.path_is_whitelisted(_scope("/healthz")) is False
    assert _utils_api.path_is_whitelisted(_scope("/documents/upload")) is False


def test_catch_all_whitelist_covers_ingestion(patterns):
    """An operator who whitelists everything must not get 401s from the
    middleware on paths the route dependency waves through."""
    patterns(["/*"])

    assert _utils_api.path_is_whitelisted(_scope("/documents/upload")) is True


def test_entries_are_matched_against_the_route_path(patterns):
    """Entries are unprefixed route paths, so the mount prefix must come off
    before matching — in either forwarding form.

    Getting this wrong collapsed the shipped default ``/health,/api/*`` into a
    blanket exemption for every route whenever the mount prefix itself started
    with ``/api``, and simultaneously stopped ``/health`` from matching under any
    other prefix.
    """
    patterns(["/health", "/api/*"])

    # Verbatim forwarding: the proxy passes the prefixed path through.
    assert _utils_api.path_is_whitelisted(_scope("/api/v1/health", "/api/v1")) is True
    assert (
        _utils_api.path_is_whitelisted(_scope("/api/v1/documents", "/api/v1")) is False
    )
    assert _utils_api.path_is_whitelisted(_scope("/site01/health", "/site01")) is True
    assert _utils_api.path_is_whitelisted(_scope("/site01/api/tags", "/site01")) is True
    assert (
        _utils_api.path_is_whitelisted(_scope("/site01/documents", "/site01")) is False
    )

    # Proxy-strip forwarding: the admission middleware runs outside the
    # normalizing middleware, so it sees a bare path with root_path already set.
    # Nothing may be removed here.
    assert _utils_api.path_is_whitelisted(_scope("/health", "/api/v1")) is True
    assert _utils_api.path_is_whitelisted(_scope("/documents", "/api/v1")) is False


def test_prefix_entries_match_on_segment_boundaries(patterns):
    """A ``/*`` entry exempts the prefix and what is under it, nothing else.

    Bare ``startswith`` widened every prefix entry past the segment the operator
    wrote: ``/graph/*`` also exempted ``GET /graphs``, a real route in this
    codebase, and ``/api/*`` would exempt any future ``/apikeys``.
    """
    patterns(["/graph/*"])

    assert _utils_api.path_is_whitelisted(_scope("/graph")) is True
    assert _utils_api.path_is_whitelisted(_scope("/graph/label/list")) is True
    assert _utils_api.path_is_whitelisted(_scope("/graphs")) is False

    patterns(["/api/*"])
    assert _utils_api.path_is_whitelisted(_scope("/api/tags")) is True
    assert _utils_api.path_is_whitelisted(_scope("/apikeys")) is False


def test_mount_prefix_falls_back_to_the_explicit_argument(patterns):
    """Callers whose scope was not built by FastAPI (hand-rolled ASGI scopes)
    pass the prefix explicitly."""
    patterns(["/health"])

    assert (
        _utils_api.path_is_whitelisted(
            {"type": "http", "path": "/site01/health"}, mount_prefix="/site01"
        )
        is True
    )


def test_route_path_never_cuts_mid_segment():
    """``/api`` is not a segment prefix of ``/apikeys``, so nothing comes off —
    an unguarded slice would produce ``keys``."""
    assert _utils_api.get_route_path(_scope("/apikeys", "/api")) == "/apikeys"
    assert _utils_api.get_route_path(_scope("/api/tags", "/api")) == "/tags"
    # An exact-prefix hit normalizes to "/" (Starlette returns "" here).
    assert _utils_api.get_route_path(_scope("/site01", "/site01")) == "/"
    assert _utils_api.get_route_path(_scope("/documents")) == "/documents"


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
