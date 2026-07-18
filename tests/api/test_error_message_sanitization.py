"""Tests that API error responses never leak raw exception text (CWE-209).

Security advisory GHSA-964x-f4fr-585m: raw Python exception strings were
propagated into HTTP 500 response bodies across ~31 API-layer handlers via
``raise HTTPException(status_code=500, detail=str(e))`` (and f-string
equivalents). A backend/storage failure could therefore disclose internal
infrastructure — database hosts, credentials, filesystem paths, SQL fragments —
to any API client.

The fix routes every one of those handlers through the shared
``internal_server_error`` chokepoint, and registers a last-resort
``@app.exception_handler(Exception)`` in ``create_app`` for anything that
escapes a route entirely. These tests pin both chokepoints:

- ``internal_server_error`` never echoes the exception text and always returns a
  generic 500 body carrying a fresh correlation id (unit level).
- A route that raises via the helper, and a route that raises a *bare* exception
  caught by the global handler, both return a sanitized body when driven through
  the real ``create_app`` request cycle (integration level).
- A control route that still uses the old ``detail=str(e)`` pattern *does* leak,
  proving the assertions above would fail without the fix.

Importing ``lightrag.api.*`` eagerly instantiates ``AuthHandler`` (auth.py),
which reads ``global_args`` and would call ``parse_args(sys.argv)`` against
pytest's argv. The autouse fixture initializes config with a clean argv first,
and the modules under test are imported lazily inside the tests — mirroring the
pattern used by the other ``tests/api`` suites.
"""

import sys

import pytest
from fastapi import APIRouter, HTTPException
from fastapi.testclient import TestClient

# A stand-in exception message stuffed with the kind of internal detail that
# must never reach a client (mirrors a real asyncpg connection failure).
_SECRET = (
    "connect failed host=db.corp port=5432 user=lightrag key=/etc/lightrag/tls.key"
)
_SECRET_NEEDLES = ("db.corp", "5432", "lightrag", "/etc/lightrag/tls.key", _SECRET)

_ENV_VARS_TO_ISOLATE = (
    "LLM_BINDING",
    "EMBEDDING_BINDING",
    "AUTH_ACCOUNTS",
    "TOKEN_SECRET",
    "LIGHTRAG_API_KEY",
    "WHITELIST_PATHS",
    "LIGHTRAG_API_PREFIX",
)


@pytest.fixture(autouse=True)
def _init_config(monkeypatch):
    """Initialize config with a clean argv so importing lightrag.api.* is safe.

    Without this, the first import of auth.py under pytest calls
    ``parse_args(sys.argv)`` against pytest's argv and aborts the session.
    """
    for var in _ENV_VARS_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AUTH_ACCOUNTS", "")
    monkeypatch.setenv("LIGHTRAG_API_KEY", "")
    monkeypatch.setenv("TOKEN_SECRET", "")
    monkeypatch.setenv("WHITELIST_PATHS", "/_sanitization_probe/*")
    monkeypatch.setenv("LLM_BINDING", "ollama")
    monkeypatch.setenv("EMBEDDING_BINDING", "ollama")

    import lightrag.api.config as config

    config._global_args = None
    config._initialized = False
    original_argv = sys.argv.copy()
    sys.argv = ["lightrag-server"]
    try:
        from lightrag.api.config import parse_args, initialize_config

        initialize_config(parse_args(), force=True)
    finally:
        sys.argv = original_argv
    yield
    config._global_args = None
    config._initialized = False


# --------------------------------------------------------------------------- #
# Unit level: the chokepoint every 500 handler now funnels through.
# --------------------------------------------------------------------------- #


def test_internal_server_error_strips_exception_text():
    from lightrag.api.utils_api import internal_server_error

    http_exc = internal_server_error(RuntimeError(_SECRET))
    assert isinstance(http_exc, HTTPException)
    assert http_exc.status_code == 500
    for needle in _SECRET_NEEDLES:
        assert needle not in http_exc.detail
    # Generic message plus a correlation id an operator can grep the logs for.
    assert "Internal server error" in http_exc.detail
    assert "error_id" in http_exc.detail


def test_internal_server_error_uses_fresh_correlation_id_per_call():
    from lightrag.api.utils_api import internal_server_error

    first = internal_server_error(ValueError("x")).detail
    second = internal_server_error(ValueError("x")).detail
    assert first != second


# --------------------------------------------------------------------------- #
# Integration level: drive the real create_app request cycle.
# --------------------------------------------------------------------------- #


class _FakeLightRAG:
    """Minimal stand-in for LightRAG so create_app touches no backend I/O."""

    def __init__(self, *_args, **_kwargs):
        pass

    def register_role_llm_builder(self, _builder):
        return None

    def set_role_llm_metadata(self, _role, **_metadata):
        return None

    def get_llm_role_config(self):
        return {}


class _FakeOllamaAPI:
    def __init__(self, *_args, **_kwargs):
        self.router = APIRouter()


def _build_client(monkeypatch):
    """Build a TestClient over the real create_app with all backends mocked out.

    Extra probe routes are mounted onto the app *after* construction so they sit
    behind the same exception handlers create_app registered.
    """
    from lightrag.api.config import global_args
    from lightrag.api.utils_api import internal_server_error
    import lightrag.api.lightrag_server as lightrag_server

    monkeypatch.setattr(lightrag_server, "LightRAG", _FakeLightRAG)
    monkeypatch.setattr(lightrag_server, "check_frontend_build", lambda: (True, False))
    monkeypatch.setattr(
        lightrag_server, "create_document_routes", lambda *_a, **_k: APIRouter()
    )
    monkeypatch.setattr(
        lightrag_server, "create_query_routes", lambda *_a, **_k: APIRouter()
    )
    monkeypatch.setattr(
        lightrag_server, "create_graph_routes", lambda *_a, **_k: APIRouter()
    )
    monkeypatch.setattr(lightrag_server, "OllamaAPI", _FakeOllamaAPI)

    app = lightrag_server.create_app(global_args)

    @app.get("/_sanitization_probe/via_helper")
    async def _via_helper():
        try:
            raise RuntimeError(_SECRET)
        except Exception as e:
            raise internal_server_error(e)

    @app.get("/_sanitization_probe/bare")
    async def _bare():
        # Escapes the route entirely -> must be caught by the global handler.
        raise RuntimeError(_SECRET)

    @app.get("/_sanitization_probe/legacy_leak")
    async def _legacy_leak():
        # The pre-fix pattern, kept only as a control for the assertions below.
        try:
            raise RuntimeError(_SECRET)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # raise_server_exceptions=False so the global handler produces the response
    # instead of TestClient re-raising into the test.
    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(
    "path",
    ["/_sanitization_probe/via_helper", "/_sanitization_probe/bare"],
)
def test_http_500_body_is_sanitized(monkeypatch, path):
    resp = _build_client(monkeypatch).get(path)
    assert resp.status_code == 500
    body = resp.text
    for needle in _SECRET_NEEDLES:
        assert needle not in body
    assert resp.json()["detail"].startswith("Internal server error")


def test_legacy_pattern_leaks_control(monkeypatch):
    """Control: the old ``detail=str(e)`` pattern DOES leak, so the sanitized
    assertions above are meaningful and would fail on the pre-fix code."""
    resp = _build_client(monkeypatch).get("/_sanitization_probe/legacy_leak")
    assert resp.status_code == 500
    assert _SECRET in resp.text
