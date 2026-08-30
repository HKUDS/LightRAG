"""Tests for the ENABLE_AI_CONTENT_NOTICE switch.

The switch decides whether the two query UIs (the admin retrieval panel at
/webui and the query entry at /workspace) label every answer as AI-generated.
It is deployment-level display configuration like WEBUI_TITLE, so it travels
to the frontend on the three responses that already carry that kind of
configuration:

- ``/auth-status`` and ``/login`` — the boot request each entry makes, and the
  only channel the workspace entry has (it never polls /health);
- ``/health`` — the admin shell's poll, which keeps the flag fresh.

On /health the field must sit in the UNAUTHENTICATED liveness tier: the two
entries read it before any credential is available, and it reveals nothing
beyond how the answers are rendered.
"""

import sys
from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

_ENV_VARS_TO_ISOLATE = (
    "LLM_BINDING",
    "EMBEDDING_BINDING",
    "OPENAI_API_KEY",
    "AUTH_ACCOUNTS",
    "TOKEN_SECRET",
    "LIGHTRAG_API_KEY",
    "WHITELIST_PATHS",
    "LIGHTRAG_API_PREFIX",
    "ENABLE_AI_CONTENT_NOTICE",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Keep tests hermetic from developer-local .env and global config state."""
    for var in _ENV_VARS_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AUTH_ACCOUNTS", "")
    monkeypatch.setenv("LIGHTRAG_API_KEY", "")
    monkeypatch.setenv("TOKEN_SECRET", "")
    # openai bindings keep the fixture importable in a minimal test
    # environment (the optional `ollama` package is not always installed);
    # nothing here reaches a provider — every LLM call site is mocked out.
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("EMBEDDING_BINDING", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

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


def _build_client(monkeypatch, *, notice=None):
    """Build a TestClient with all backend I/O mocked out.

    ``notice`` is applied as the ENABLE_AI_CONTENT_NOTICE environment variable
    (None keeps it unset to exercise the default), so the test goes through the
    real parse_args() path rather than poking the parsed namespace.
    """
    if notice is not None:
        monkeypatch.setenv("ENABLE_AI_CONTENT_NOTICE", notice)

    from lightrag.api.config import initialize_config, parse_args

    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server"]
        args = parse_args()
    finally:
        sys.argv = original_argv
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


# --------------------------------------------------------------------------- #
# Default: off, and reported as an explicit boolean everywhere.
# --------------------------------------------------------------------------- #
def test_disabled_by_default(monkeypatch):
    client = _build_client(monkeypatch)

    assert client.get("/auth-status").json()["ai_content_notice_enabled"] is False
    assert client.get("/health").json()["ai_content_notice_enabled"] is False
    assert (
        client.post("/login", data={"username": "u", "password": "p"}).json()[
            "ai_content_notice_enabled"
        ]
        is False
    )


def test_config_default_is_false():
    """The parsed namespace carries the flag, so create_app never has to guess."""
    from lightrag.api.config import parse_args

    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server"]
        args = parse_args()
    finally:
        sys.argv = original_argv

    assert args.enable_ai_content_notice is False


# --------------------------------------------------------------------------- #
# Enabled: every channel the two entries can read reports it.
# --------------------------------------------------------------------------- #
def test_auth_status_reports_the_notice_when_enabled(monkeypatch):
    client = _build_client(monkeypatch, notice="true")

    assert client.get("/auth-status").json()["ai_content_notice_enabled"] is True


def test_login_reports_the_notice_when_enabled(monkeypatch):
    client = _build_client(monkeypatch, notice="true")

    body = client.post("/login", data={"username": "u", "password": "p"}).json()
    assert body["ai_content_notice_enabled"] is True


def test_health_reports_the_notice_when_enabled(monkeypatch):
    client = _build_client(monkeypatch, notice="true")

    assert client.get("/health").json()["ai_content_notice_enabled"] is True


def test_health_reports_the_notice_without_authentication(monkeypatch):
    """The workspace entry has no credentials when it first renders, and the
    admin shell reads /health before login, so the field must stay in the
    liveness tier even when authentication is configured (issue #3294 keeps
    everything else out of it)."""
    import lightrag.api.utils_api as utils_api

    client = _build_client(monkeypatch, notice="true")
    monkeypatch.setattr(utils_api, "auth_configured", True)
    monkeypatch.setattr(
        utils_api, "whitelist_patterns", [("/health", False), ("/api", True)]
    )

    body = client.get("/health").json()
    assert body["ai_content_notice_enabled"] is True
    # Still liveness-only for anonymous callers.
    assert "configuration" not in body
