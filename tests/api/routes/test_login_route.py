"""Tests for the POST /login endpoint.

Focus: the endpoint runs the CPU-bound bcrypt verification off the event loop.

verify_password now performs one bcrypt for *every* attempt (including unknown
usernames, to equalize timing — GHSA-c759-cx9p-mrwq). Because /login is
``async def``, running that bcrypt inline would block the event loop for
~100 ms per request, so a flood of unauthenticated login attempts with random
usernames could starve the whole API (DoS). The route must therefore offload
verification with ``asyncio.to_thread`` and still return the correct result.
"""

import asyncio
import sys
from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

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


def _build_server(monkeypatch):
    """Build the full app via create_app with all backend I/O mocked out.

    Returns the lightrag_server module so callers can reach auth_handler.
    """
    from lightrag.api.config import parse_args, initialize_config

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
    for factory in (
        "create_document_routes",
        "create_query_routes",
        "create_graph_routes",
    ):
        monkeypatch.setattr(lightrag_server, factory, lambda *_a, **_k: APIRouter())
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
    return lightrag_server, app


def _client_with_account(monkeypatch, username="admin", password="s3cret-plain"):
    """Build a client whose auth_handler has a single configured account."""
    lightrag_server, app = _build_server(monkeypatch)
    monkeypatch.setattr(lightrag_server.auth_handler, "accounts", {username: password})
    return lightrag_server, TestClient(app)


def test_login_correct_password_returns_token(monkeypatch):
    _server, client = _client_with_account(monkeypatch)

    resp = client.post("/login", data={"username": "admin", "password": "s3cret-plain"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["access_token"]
    assert body["auth_mode"] == "enabled"


def test_login_wrong_password_is_401(monkeypatch):
    """A wrong password must be rejected. This also guards the ``await``: if the
    to_thread call were not awaited, ``password_ok`` would be a truthy coroutine
    and the wrong password would be accepted.
    """
    _server, client = _client_with_account(monkeypatch)

    resp = client.post("/login", data={"username": "admin", "password": "wrong"})

    assert resp.status_code == 401


def test_login_unknown_username_is_401(monkeypatch):
    _server, client = _client_with_account(monkeypatch)

    resp = client.post("/login", data={"username": "ghost", "password": "whatever"})

    assert resp.status_code == 401


def test_login_offloads_verification_to_thread(monkeypatch):
    """The bcrypt verification must run via asyncio.to_thread so it does not
    block the event loop (unauthenticated DoS surface, since every attempt now
    costs one bcrypt).
    """
    lightrag_server, _app = _build_server(monkeypatch)
    monkeypatch.setattr(
        lightrag_server.auth_handler, "accounts", {"admin": "s3cret-plain"}
    )

    offloaded = []
    real_to_thread = asyncio.to_thread

    async def spy(func, *args, **kwargs):
        offloaded.append(getattr(func, "__name__", func))
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(lightrag_server.asyncio, "to_thread", spy)

    client = TestClient(_app)
    resp = client.post("/login", data={"username": "admin", "password": "s3cret-plain"})

    assert resp.status_code == 200
    assert "verify_password" in offloaded


def test_login_locks_out_after_max_failed_attempts(monkeypatch):
    monkeypatch.setenv("LOGIN_MAX_FAILED_ATTEMPTS", "3")
    monkeypatch.setenv("LOGIN_LOCKOUT_WINDOW_SECONDS", "300")
    _server, client = _client_with_account(monkeypatch)

    for _ in range(3):
        resp = client.post("/login", data={"username": "admin", "password": "bad"})
        assert resp.status_code == 401

    locked = client.post("/login", data={"username": "admin", "password": "bad"})
    assert locked.status_code == 429
    assert int(locked.headers["Retry-After"]) >= 1


def test_login_lockout_returns_429_without_bcrypt(monkeypatch):
    """A locked-out attempt must be rejected before the bcrypt verification, so
    lockout also caps the CPU/DoS cost of a login flood.
    """
    monkeypatch.setenv("LOGIN_MAX_FAILED_ATTEMPTS", "2")
    lightrag_server, app = _build_server(monkeypatch)
    monkeypatch.setattr(
        lightrag_server.auth_handler, "accounts", {"admin": "s3cret-plain"}
    )

    verify_calls = []
    real_to_thread = asyncio.to_thread

    async def spy(func, *args, **kwargs):
        verify_calls.append(getattr(func, "__name__", func))
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(lightrag_server.asyncio, "to_thread", spy)

    client = TestClient(app)
    for _ in range(2):
        client.post("/login", data={"username": "admin", "password": "bad"})
    calls_before_lockout = len(verify_calls)

    resp = client.post("/login", data={"username": "admin", "password": "bad"})
    assert resp.status_code == 429
    assert len(verify_calls) == calls_before_lockout  # no bcrypt on the locked call


def test_login_success_resets_lockout(monkeypatch):
    """A successful login clears the failure counter. With max=2, the sequence
    fail, success, fail, fail must stay 401 throughout; without the reset the
    last attempt would be 429.
    """
    monkeypatch.setenv("LOGIN_MAX_FAILED_ATTEMPTS", "2")
    _server, client = _client_with_account(monkeypatch)

    def status(password):
        return client.post(
            "/login", data={"username": "admin", "password": password}
        ).status_code

    assert status("bad") == 401
    assert status("s3cret-plain") == 200  # resets the counter
    assert status("bad") == 401
    assert status("bad") == 401  # would be 429 if the success had not reset


def test_login_lockout_is_per_username(monkeypatch):
    """Locking one username must not lock a different one from the same IP."""
    monkeypatch.setenv("LOGIN_MAX_FAILED_ATTEMPTS", "2")
    _server, client = _client_with_account(monkeypatch)  # only "admin" configured

    for _ in range(2):
        client.post("/login", data={"username": "admin", "password": "bad"})
    assert (
        client.post("/login", data={"username": "admin", "password": "bad"}).status_code
        == 429
    )

    # A different username has its own bucket and is still only rejected as 401.
    assert (
        client.post("/login", data={"username": "bob", "password": "bad"}).status_code
        == 401
    )
