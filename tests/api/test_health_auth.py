"""Tests for credential-gated configuration disclosure on ``GET /health``.

Issue #3294: ``/health`` is whitelisted by default so it answers unauthenticated
liveness probes (HTTP 200). It must therefore reveal sensitive runtime
configuration (filesystem paths, LLM/embedding provider + model + host, storage
backends, queue status, ...) ONLY to authenticated callers. These tests pin
that behavior across the three authentication modes:

- fully open (no AUTH_ACCOUNTS, no API key): everything is open anyway, so the
  full configuration is returned to every caller.
- password auth (AUTH_ACCOUNTS): only a valid non-guest JWT (or a configured
  API key) unlocks the configuration; anonymous probes get liveness only.
- API-key-only: only a valid X-API-Key unlocks the configuration.

In every case ``/health`` returns HTTP 200 so external liveness probes keep
working.
"""

import sys
from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

# Fields that must NEVER appear in an unauthenticated response.
_SENSITIVE_TOP_LEVEL = ("working_directory", "input_directory", "configuration")
# Liveness fields that must always be present (safe; already exposed by the
# unauthenticated /auth-status endpoint or pure liveness signals).
_LIVENESS_FIELDS = ("status", "auth_mode", "core_version", "api_version")


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
def _isolate_env(monkeypatch):
    """Keep tests hermetic from developer-local .env and global config state."""
    for var in _ENV_VARS_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AUTH_ACCOUNTS", "")
    monkeypatch.setenv("LIGHTRAG_API_KEY", "")
    monkeypatch.setenv("TOKEN_SECRET", "")
    monkeypatch.setenv("WHITELIST_PATHS", "/health,/api/*")
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


def _build_client(monkeypatch, *, api_key=None):
    """Build a /health-capable TestClient with all backend I/O mocked out."""
    from lightrag.api.config import parse_args, initialize_config

    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server"]
        args = parse_args()
    finally:
        sys.argv = original_argv
    if api_key is not None:
        args.key = api_key
    initialize_config(args, force=True)

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


def _set_auth_mode(monkeypatch, *, auth_configured):
    """Override the module-level auth flags the /health gate reads at runtime.

    Also pin a whitelist that exempts /health so combined_auth keeps returning
    200 for anonymous callers (the gate, not combined_auth, hides the config).
    """
    import lightrag.api.utils_api as utils_api

    monkeypatch.setattr(utils_api, "auth_configured", auth_configured)
    monkeypatch.setattr(
        utils_api, "whitelist_patterns", [("/health", False), ("/api", True)]
    )


def _assert_liveness_only(body):
    for field in _LIVENESS_FIELDS:
        assert field in body, f"liveness field {field!r} missing"
    for field in _SENSITIVE_TOP_LEVEL:
        assert field not in body, f"sensitive field {field!r} leaked"


def _assert_full_config(body):
    assert "configuration" in body
    assert "working_directory" in body
    assert "llm_binding" in body["configuration"]


# --------------------------------------------------------------------------- #
# Fully open mode: everything is open, so config is returned to everyone.
# --------------------------------------------------------------------------- #
def test_open_mode_returns_full_config_to_anonymous(monkeypatch):
    client = _build_client(monkeypatch)
    _set_auth_mode(monkeypatch, auth_configured=False)

    resp = client.get("/health")

    assert resp.status_code == 200
    _assert_full_config(resp.json())


# --------------------------------------------------------------------------- #
# Password auth: anonymous gets liveness only; a valid token unlocks config.
# --------------------------------------------------------------------------- #
def test_password_mode_anonymous_gets_liveness_only(monkeypatch):
    client = _build_client(monkeypatch)
    _set_auth_mode(monkeypatch, auth_configured=True)

    resp = client.get("/health")

    assert resp.status_code == 200  # liveness probe must stay green
    _assert_liveness_only(resp.json())


def test_password_mode_valid_token_unlocks_config(monkeypatch):
    import lightrag.api.utils_api as utils_api

    client = _build_client(monkeypatch)
    _set_auth_mode(monkeypatch, auth_configured=True)
    monkeypatch.setattr(
        utils_api.auth_handler,
        "validate_token",
        lambda token: (
            {"username": "admin", "role": "user"}
            if token == "valid-user-token"
            else (_ for _ in ()).throw(ValueError("bad token"))
        ),
    )

    resp = client.get("/health", headers={"Authorization": "Bearer valid-user-token"})

    assert resp.status_code == 200
    _assert_full_config(resp.json())


def test_password_mode_guest_token_stays_liveness_only(monkeypatch):
    import lightrag.api.utils_api as utils_api

    client = _build_client(monkeypatch)
    _set_auth_mode(monkeypatch, auth_configured=True)
    monkeypatch.setattr(
        utils_api.auth_handler,
        "validate_token",
        lambda token: {"username": "guest", "role": "guest"},
    )

    resp = client.get("/health", headers={"Authorization": "Bearer guest-token"})

    assert resp.status_code == 200
    _assert_liveness_only(resp.json())


# --------------------------------------------------------------------------- #
# API-key-only mode: only a valid X-API-Key unlocks the configuration.
# --------------------------------------------------------------------------- #
def test_api_key_mode_anonymous_gets_liveness_only(monkeypatch):
    client = _build_client(monkeypatch, api_key="secret-key")
    _set_auth_mode(monkeypatch, auth_configured=False)

    resp = client.get("/health")

    assert resp.status_code == 200
    _assert_liveness_only(resp.json())


def test_api_key_mode_valid_key_unlocks_config(monkeypatch):
    client = _build_client(monkeypatch, api_key="secret-key")
    _set_auth_mode(monkeypatch, auth_configured=False)

    resp = client.get("/health", headers={"X-API-Key": "secret-key"})

    assert resp.status_code == 200
    _assert_full_config(resp.json())


# --------------------------------------------------------------------------- #
# Combined mode (AUTH_ACCOUNTS + LIGHTRAG_API_KEY): either a valid JWT or a
# valid X-API-Key unlocks the configuration; an anonymous probe gets liveness.
# --------------------------------------------------------------------------- #
def test_combined_mode_anonymous_gets_liveness_only(monkeypatch):
    client = _build_client(monkeypatch, api_key="secret-key")
    _set_auth_mode(monkeypatch, auth_configured=True)

    resp = client.get("/health")

    assert resp.status_code == 200
    _assert_liveness_only(resp.json())


def test_combined_mode_valid_api_key_unlocks_config(monkeypatch):
    client = _build_client(monkeypatch, api_key="secret-key")
    _set_auth_mode(monkeypatch, auth_configured=True)

    resp = client.get("/health", headers={"X-API-Key": "secret-key"})

    assert resp.status_code == 200
    _assert_full_config(resp.json())
