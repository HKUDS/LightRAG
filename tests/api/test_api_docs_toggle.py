"""Tests for the ENABLE_API_DOCS switch (issue #3666, RFC #3671).

The switch is the single authority over every interactive documentation
surface: /docs, /docs/oauth2-redirect, /redoc, /openapi.json and the
/static/swagger-ui mount must all flip together. /health reports the state as
``api_docs_available`` in its unauthenticated liveness payload, and when the
WebUI bundle is absent the `/` and /webui fallbacks must degrade to a JSON
service-info response (root_path-aware ``health_url``) instead of redirecting
into a 404.
"""

import sys
from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

# Every route the flag governs. The static asset is a Starlette Mount, not an
# APIRoute — it is listed explicitly so a regression that conditions only the
# documented routes still fails here.
_DOC_SURFACES = (
    "/docs",
    "/docs/oauth2-redirect",
    "/redoc",
    "/openapi.json",
    "/static/swagger-ui/swagger-ui.css",
)

_ENV_VARS_TO_ISOLATE = (
    "LLM_BINDING",
    "EMBEDDING_BINDING",
    "AUTH_ACCOUNTS",
    "TOKEN_SECRET",
    "LIGHTRAG_API_KEY",
    "WHITELIST_PATHS",
    "LIGHTRAG_API_PREFIX",
    "ENABLE_API_DOCS",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Keep tests hermetic from developer-local .env and global config state."""
    for var in _ENV_VARS_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AUTH_ACCOUNTS", "")
    monkeypatch.setenv("LIGHTRAG_API_KEY", "")
    monkeypatch.setenv("TOKEN_SECRET", "")
    monkeypatch.setenv("LLM_BINDING", "ollama")
    monkeypatch.setenv("EMBEDDING_BINDING", "ollama")

    import lightrag.api.config as config

    config._global_args = None
    config._initialized = False
    yield
    config._global_args = None
    config._initialized = False


class _FakeLightRAG:
    """Minimal stand-in implementing the async surface /health touches."""

    def __init__(self, *_args, **_kwargs):
        pass

    def register_role_llm_builder(self, _builder):
        return None

    def set_role_llm_metadata(self, _role, **_metadata):
        return None

    def get_llm_role_config(self):
        return {}

    async def get_llm_queue_status(self, include_base=True):
        return {}

    async def get_embedding_queue_status(self):
        return {}

    async def get_rerank_queue_status(self):
        return {}


class _FakeOllamaAPI:
    def __init__(self, *_args, **_kwargs):
        self.router = APIRouter()


def _build_client(monkeypatch, *, enable_api_docs=None, webui_available=True, argv=()):
    """Build a TestClient with all backend I/O mocked out.

    ``enable_api_docs`` is applied as the ENABLE_API_DOCS environment variable
    (None keeps it unset to exercise the default), so the test goes through the
    real parse_args() path rather than poking the parsed namespace.
    """
    if enable_api_docs is not None:
        monkeypatch.setenv("ENABLE_API_DOCS", enable_api_docs)

    from lightrag.api.config import parse_args, initialize_config

    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server", *argv]
        args = parse_args()
    finally:
        sys.argv = original_argv
    initialize_config(args, force=True)

    import lightrag.api.lightrag_server as lightrag_server

    monkeypatch.setattr(lightrag_server, "LightRAG", _FakeLightRAG)
    monkeypatch.setattr(
        lightrag_server, "check_frontend_build", lambda: (webui_available, False)
    )
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
    monkeypatch.setattr(
        lightrag_server, "get_namespace_data", AsyncMock(return_value={"busy": False})
    )
    monkeypatch.setattr(lightrag_server, "get_default_workspace", lambda: "default")
    monkeypatch.setattr(
        lightrag_server,
        "cleanup_keyed_lock",
        lambda: {"cleanup_performed": {}, "current_status": {}},
    )

    app = lightrag_server.create_app(args)
    return TestClient(app)


# --------------------------------------------------------------------------- #
# The five documentation surfaces flip together.
# --------------------------------------------------------------------------- #
def test_docs_surfaces_enabled_by_default(monkeypatch):
    client = _build_client(monkeypatch)

    for path in _DOC_SURFACES:
        resp = client.get(path)
        assert resp.status_code == 200, f"{path} should be served by default"


def test_docs_surfaces_all_return_404_when_disabled(monkeypatch):
    client = _build_client(monkeypatch, enable_api_docs="false")

    for path in _DOC_SURFACES:
        resp = client.get(path)
        assert resp.status_code == 404, f"{path} must 404 with ENABLE_API_DOCS=false"


def test_disabling_docs_leaves_other_routes_alive(monkeypatch):
    client = _build_client(monkeypatch, enable_api_docs="false")

    assert client.get("/health").status_code == 200
    assert client.get("/auth-status").status_code == 200


# --------------------------------------------------------------------------- #
# /health capability field (unauthenticated liveness tier).
# --------------------------------------------------------------------------- #
def test_health_reports_api_docs_available_true(monkeypatch):
    client = _build_client(monkeypatch)

    body = client.get("/health").json()
    assert body["api_docs_available"] is True


def test_health_reports_api_docs_available_false(monkeypatch):
    client = _build_client(monkeypatch, enable_api_docs="false")

    body = client.get("/health").json()
    assert body["api_docs_available"] is False


def test_health_capability_field_visible_without_authentication(monkeypatch):
    """The WebUI decides icon visibility from this field before login, so it
    must sit in the liveness tier even when authentication is configured."""
    import lightrag.api.utils_api as utils_api

    client = _build_client(monkeypatch, enable_api_docs="false")
    monkeypatch.setattr(utils_api, "auth_configured", True)
    monkeypatch.setattr(
        utils_api, "whitelist_patterns", [("/health", False), ("/api", True)]
    )

    resp = client.get("/health")
    body = resp.json()
    assert resp.status_code == 200
    assert body["api_docs_available"] is False
    # Still liveness-only for anonymous callers (issue #3294).
    assert "configuration" not in body


# --------------------------------------------------------------------------- #
# Fallbacks when the WebUI bundle is absent.
# --------------------------------------------------------------------------- #
def test_missing_webui_still_redirects_to_docs_when_enabled(monkeypatch):
    client = _build_client(monkeypatch, webui_available=False)

    for path in ("/", "/webui", "/webui/"):
        resp = client.get(path, follow_redirects=False)
        assert resp.status_code in (302, 307), path
        assert resp.headers["location"] == "/docs", path


def _assert_service_info(resp, expected_health_url):
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["service"] == "LightRAG Server"
    assert body["api_version"]
    assert body["health_url"] == expected_health_url
    assert "message" in body


def test_missing_webui_and_disabled_docs_return_service_info(monkeypatch):
    client = _build_client(monkeypatch, enable_api_docs="false", webui_available=False)

    for path in ("/", "/webui", "/webui/"):
        _assert_service_info(client.get(path), "/health")


def test_service_info_health_url_keeps_api_prefix(monkeypatch):
    """Behind LIGHTRAG_API_PREFIX the health_url must include the prefix.

    With root_path set, FastAPI accepts both the proxy-strip request shape
    (natural path) and the verbatim-prefix shape (prefixed path); the
    health_url must carry the prefix in both.
    """
    client = _build_client(
        monkeypatch,
        enable_api_docs="false",
        webui_available=False,
        argv=("--api-prefix", "/site01"),
    )

    for path in (
        "/",
        "/webui",
        "/webui/",
        "/site01/",
        "/site01/webui",
        "/site01/webui/",
    ):
        _assert_service_info(client.get(path), "/site01/health")


def test_webui_available_ignores_docs_switch_for_root_redirect(monkeypatch):
    """With the WebUI bundled, `/` keeps redirecting there even without docs."""
    client = _build_client(monkeypatch, enable_api_docs="false")

    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (302, 307)
    assert resp.headers["location"] == "/webui/"
