"""Integration tests for multi-workspace HTTP routing.

Verifies that workspace selection via the ``LIGHTRAG-WORKSPACE`` header
behaves correctly end-to-end: header routing, fallback, path-traversal
rejection, empty-header rejection, and cross-workspace data isolation.

Tests I1–I5 exercise the HTTP layer (header → workspace extraction →
validation) without requiring a real storage backend.  I6 (cross-workspace
isolation) is a true integration test requiring ``--run-integration``.

Uses the same ``_build_client`` / monkeypatch pattern as
``test_health_auth.py``.
"""

import sys
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from lightrag.kg.shared_storage import set_default_workspace


# ────────────────────────────────────────────────────────────────
# Helpers — mirror test_health_auth.py patterns
# ────────────────────────────────────────────────────────────────


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
def _isolate_env_and_config(monkeypatch):
    """Keep tests hermetic from developer-local .env and global config state."""
    for var in _ENV_VARS_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_BINDING", "ollama")
    monkeypatch.setenv("EMBEDDING_BINDING", "ollama")

    import lightrag.api.config as config

    config._global_args = None
    config._initialized = False
    set_default_workspace(None)
    yield
    config._global_args = None
    config._initialized = False
    set_default_workspace(None)


class _FakeLightRAG:
    """Minimal stand-in implementing the async surface /health touches."""

    def __init__(self, *_args, **_kwargs):
        pass

    def register_role_llm_builder(self, _builder):
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


class _FakeDocManagerPoolGet:
    """A DocManagerPool stub whose .get() returns a MagicMock."""

    def __init__(self, base_input_dir: str):
        pass

    async def get(self, workspace: str):
        mgr = MagicMock()
        mgr.workspace = workspace
        return mgr


class _FakeRagPool:
    """A RagPool stub that returns a _FakeLightRAG without real init."""

    def __init__(self, config_factory=None, *, role_llm_builder=None, on_create=None):
        pass

    async def get(self, workspace: str):
        rag = _FakeLightRAG()
        rag.workspace = workspace
        return rag

    async def shutdown_all(self):
        pass


def _build_client(monkeypatch, *, api_key=None, workspace="default"):
    """Build a /health-capable TestClient with all backend I/O mocked out."""
    from lightrag.api.config import parse_args, initialize_config

    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server"]
        if workspace:
            sys.argv.extend(["--workspace", workspace])
        args = parse_args()
    finally:
        sys.argv = original_argv
    if api_key is not None:
        args.key = api_key
    initialize_config(args, force=True)

    # ── Prevent pipmaster from auto-installing packages in tests ──
    import pipmaster as pm

    monkeypatch.setattr(pm, "is_installed", lambda pkg: True)
    monkeypatch.setattr(pm, "install", lambda pkg, *a, **kw: None)

    import lightrag.api.lightrag_server as server

    monkeypatch.setattr(server, "RagPool", _FakeRagPool)
    import lightrag.api.workspace_pool as wp
    monkeypatch.setattr(wp, "LightRAG", _FakeLightRAG)
    monkeypatch.setattr(server, "check_frontend_build", lambda: (True, False))
    monkeypatch.setattr(server, "create_document_routes", lambda *_a, **_k: APIRouter())
    monkeypatch.setattr(server, "create_query_routes", lambda *_a, **_k: APIRouter())
    monkeypatch.setattr(server, "create_graph_routes", lambda *_a, **_k: APIRouter())
    monkeypatch.setattr(server, "OllamaAPI", _FakeOllamaAPI)
    monkeypatch.setattr(
        server, "get_namespace_data", AsyncMock(return_value={"busy": False})
    )
    monkeypatch.setattr(
        server,
        "cleanup_keyed_lock",
        lambda: {"cleanup_performed": {}, "current_status": {}},
    )
    # Replace the real DocManagerPool with our fake that returns MagicMocks
    monkeypatch.setattr(server, "DocManagerPool", _FakeDocManagerPoolGet)

    app = server.create_app(args)
    return TestClient(app)


# ────────────────────────────────────────────────────────────────
# I1 — Workspace header selects correct workspace
# ────────────────────────────────────────────────────────────────


def test_workspace_header_selects_workspace(monkeypatch):
    """I1 — a ``LIGHTRAG-WORKSPACE`` header routes to the named workspace."""
    client = _build_client(monkeypatch, api_key="test-key")

    resp = client.get(
        "/health",
        headers={"LIGHTRAG-WORKSPACE": "projectA", "X-API-Key": "test-key"},
    )
    assert resp.status_code == 200
    assert resp.json()["configuration"]["workspace"] == "projectA"


def test_different_workspace_headers_different_results(monkeypatch):
    """I1-ext — different headers route to different workspaces."""
    client = _build_client(monkeypatch, api_key="test-key")

    r1 = client.get(
        "/health",
        headers={"LIGHTRAG-WORKSPACE": "projectA", "X-API-Key": "test-key"},
    )
    r2 = client.get(
        "/health",
        headers={"LIGHTRAG-WORKSPACE": "projectB", "X-API-Key": "test-key"},
    )
    assert r1.json()["configuration"]["workspace"] == "projectA"
    assert r2.json()["configuration"]["workspace"] == "projectB"


# ────────────────────────────────────────────────────────────────
# I2 — No header falls back to default workspace
# ────────────────────────────────────────────────────────────────


def test_no_header_falls_back_to_default(monkeypatch):
    """I2 — without header, the server's ``--workspace`` is used."""
    client = _build_client(monkeypatch, workspace="default", api_key="test-key")

    resp = client.get(
        "/health",
        headers={"X-API-Key": "test-key"},
    )
    assert resp.status_code == 200
    assert resp.json()["configuration"]["workspace"] == "default"


# ────────────────────────────────────────────────────────────────
# I3 — No header and no default → HTTP 400
# ────────────────────────────────────────────────────────────────


def test_no_header_no_default_returns_400(monkeypatch):
    """I3 — without header or default workspace, the request is rejected."""
    client = _build_client(monkeypatch, workspace="", api_key="test-key")

    resp = client.get(
        "/health",
        headers={"X-API-Key": "test-key"},
    )
    assert resp.status_code == 400
    assert "required" in resp.json()["detail"].lower()


# ────────────────────────────────────────────────────────────────
# I4 — Path traversal rejected
# ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "malicious_workspace",
    [
        "../../etc",  # Unix traversal
        "..\\..\\Windows",  # Windows traversal
        "../secret",  # Single-level traversal
        "proj/../../etc",  # Mid-string traversal
    ],
)
def test_path_traversal_workspace_rejected(monkeypatch, malicious_workspace):
    """I4 — workspace names containing path separators or ``..`` are rejected."""
    client = _build_client(monkeypatch, api_key="test-key")

    resp = client.get(
        "/health",
        headers={
            "LIGHTRAG-WORKSPACE": malicious_workspace,
            "X-API-Key": "test-key",
        },
    )
    assert resp.status_code == 400
    assert (
        "invalid" in resp.json()["detail"].lower()
        or "workspace" in resp.json()["detail"].lower()
    )


# ────────────────────────────────────────────────────────────────
# I5 — Empty workspace header rejected
# ────────────────────────────────────────────────────────────────


def test_empty_workspace_header_rejected(monkeypatch):
    """I5 — an empty (or whitespace-only) header is rejected."""
    client = _build_client(monkeypatch, api_key="test-key")

    resp = client.get(
        "/health",
        headers={"LIGHTRAG-WORKSPACE": "", "X-API-Key": "test-key"},
    )
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


def test_whitespace_only_workspace_header_rejected(monkeypatch):
    """I5-ext — whitespace-only header is treated as empty and rejected."""
    client = _build_client(monkeypatch, api_key="test-key")

    resp = client.get(
        "/health",
        headers={"LIGHTRAG-WORKSPACE": "   ", "X-API-Key": "test-key"},
    )
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"].lower()


# ────────────────────────────────────────────────────────────────
# I7 — /health returns 200 without workspace header (liveness probe)
# ────────────────────────────────────────────────────────────────


def test_health_no_workspace_header_returns_200(monkeypatch):
    """I7 — /health is a liveness probe; returns 200 even without workspace."""
    client = _build_client(monkeypatch, workspace="default")

    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


# ────────────────────────────────────────────────────────────────
# I6 — Cross-workspace data isolation
# ────────────────────────────────────────────────────────────────
# Cross-workspace isolation is tested at the RagPool level in
# ``tests/api/test_workspace_pool.py::TestRagPoolCrossWorkspaceIsolation``.
# That test creates real LightRAG instances (with mock LLM + embedding
# and file-based storage) via RagPool, inserts different data into two
# workspaces, and verifies that full_docs and storage files are fully
# isolated between workspaces.
#
# A true HTTP-level integration test (upload via POST, query via POST
# with different workspace headers) would require a running server with
# a real LLM configured.  The pool-level test covers the same isolation
# contract at the layer closest to the data, without the complexity of
# injecting mock LLMs into create_app().
