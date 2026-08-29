"""Tests for the dual-entry WebUI mounts (/webui + /workspace).

Covers the workspace-entry PRD's server-side contract:

- one static build directory mounted twice, each mount serving only its own
  entry HTML as the directory index;
- cross-entry HTML rejected with 404 while same-entry explicit filenames stay
  reachable;
- independent availability checks (old build directory with only index.html
  degrades /workspace alone);
- the /workspace degradation never redirects to or mentions the API docs,
  regardless of ENABLE_API_DOCS;
- LIGHTRAG_DEFAULT_UI: env + CLI channels, startup fail-fast on illegal
  values, root_path-aware redirects, and '/' following the chosen entry's own
  degradation branch;
- /health exposing webui_available and workspace_available independently.
"""

import sys
from unittest.mock import AsyncMock

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient


PLACEHOLDER = "<!-- __LIGHTRAG_RUNTIME_CONFIG__ -->"

_ENV_VARS_TO_ISOLATE = (
    "LLM_BINDING",
    "EMBEDDING_BINDING",
    "AUTH_ACCOUNTS",
    "TOKEN_SECRET",
    "LIGHTRAG_API_KEY",
    "WHITELIST_PATHS",
    "LIGHTRAG_API_PREFIX",
    "LIGHTRAG_DEFAULT_UI",
    "ENABLE_API_DOCS",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    for var in _ENV_VARS_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AUTH_ACCOUNTS", "")
    monkeypatch.setenv("LIGHTRAG_API_KEY", "")
    monkeypatch.setenv("TOKEN_SECRET", "")
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("EMBEDDING_BINDING", "openai")

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


def _entry_html(name: str) -> str:
    return (
        "<!doctype html><html><head>"
        f"{PLACEHOLDER}"
        f'<script type="module" crossorigin src="./assets/{name}-X.js"></script>'
        f"</head><body><div id=root>{name}</div></body></html>"
    )


def _stage_build(tmp_path, *, webui=True, workspace=True):
    webui_dir = tmp_path / "webui"
    webui_dir.mkdir(exist_ok=True)
    if webui:
        (webui_dir / "index.html").write_text(_entry_html("index"), encoding="utf-8")
    if workspace:
        (webui_dir / "workspace.html").write_text(
            _entry_html("workspace"), encoding="utf-8"
        )
    assets = webui_dir / "assets"
    assets.mkdir(exist_ok=True)
    (assets / "shared-X.js").write_text("//js", encoding="utf-8")


def _build_app(tmp_path, monkeypatch, *cli_args):
    from lightrag.api.config import parse_args, initialize_config

    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server", *cli_args]
        args = parse_args()
    finally:
        sys.argv = original_argv
    initialize_config(args, force=True)

    # Import AFTER initialize_config: module-level code in the server's
    # dependency graph (auth handler) reads global_args on first import, and
    # an uninitialized config would re-run parse_args() against pytest argv.
    import lightrag.api.lightrag_server as lightrag_server

    # Redirect the build checks and the StaticFiles mounts to the staged dir.
    monkeypatch.setattr(
        lightrag_server, "__file__", str(tmp_path / "lightrag_server.py")
    )

    monkeypatch.setattr(lightrag_server, "LightRAG", _FakeLightRAG)
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

    return lightrag_server.create_app(args)


def _client(tmp_path, monkeypatch, *cli_args, **stage_kwargs):
    _stage_build(tmp_path, **stage_kwargs)
    app = _build_app(tmp_path, monkeypatch, *cli_args)
    return TestClient(app)


class TestDualMount:
    def test_each_mount_serves_its_own_entry(self, tmp_path, monkeypatch):
        client = _client(tmp_path, monkeypatch)

        webui = client.get("/webui/")
        assert webui.status_code == 200
        assert ">index<" in webui.text

        workspace = client.get("/workspace/")
        assert workspace.status_code == 200
        assert ">workspace<" in workspace.text

    def test_trailing_slash_redirects_preserved(self, tmp_path, monkeypatch):
        client = _client(tmp_path, monkeypatch)
        for path, target in (("/webui", "/webui/"), ("/workspace", "/workspace/")):
            resp = client.get(path, follow_redirects=False)
            assert resp.status_code in (301, 307, 308), path
            assert resp.headers["location"].endswith(target)

    def test_cross_entry_html_rejected(self, tmp_path, monkeypatch):
        client = _client(tmp_path, monkeypatch)
        assert client.get("/workspace/index.html").status_code == 404
        assert client.get("/webui/workspace.html").status_code == 404

    def test_cross_entry_html_rejected_case_insensitively(self, tmp_path, monkeypatch):
        """The block must not depend on the request's letter case.

        On a case-insensitive filesystem (macOS, Windows) StaticFiles would
        resolve /webui/WORKSPACE.HTML to the real workspace.html, so an
        exact-case check would leave the cross-entry alias open exactly on
        the platforms whose filesystem opens it.
        """
        client = _client(tmp_path, monkeypatch)
        for path in (
            "/webui/WORKSPACE.HTML",
            "/webui/Workspace.html",
            "/workspace/INDEX.HTML",
            "/workspace/Index.Html",
        ):
            assert client.get(path).status_code == 404, path

    def test_same_entry_explicit_filename_still_served(self, tmp_path, monkeypatch):
        client = _client(tmp_path, monkeypatch)

        webui = client.get("/webui/index.html")
        assert webui.status_code == 200
        assert ">index<" in webui.text
        assert "window.__LIGHTRAG_CONFIG__" in webui.text

        workspace = client.get("/workspace/workspace.html")
        assert workspace.status_code == 200
        assert ">workspace<" in workspace.text
        assert "window.__LIGHTRAG_CONFIG__" in workspace.text

    def test_runtime_config_identical_on_both_mounts(self, tmp_path, monkeypatch):
        client = _client(tmp_path, monkeypatch, "--api-prefix", "/site01")

        import re

        def config_script(text):
            match = re.search(
                r"<script>window\.__LIGHTRAG_CONFIG__ = .*?;</script>", text
            )
            return match.group(0) if match else None

        webui = client.get("/site01/webui/").text
        workspace = client.get("/site01/workspace/").text
        assert config_script(webui) is not None
        # Byte-identical injection on both mounts (published shape).
        assert config_script(webui) == config_script(workspace)
        # Published runtime-config shape: { apiPrefix, webuiPrefix } — no
        # entry-mode field.
        assert '"apiPrefix": "/site01"' in webui.replace('":"', '": "')
        assert "uiMode" not in webui and "entryMode" not in workspace

    def test_html_no_cache_and_assets_immutable_on_workspace(
        self, tmp_path, monkeypatch
    ):
        client = _client(tmp_path, monkeypatch)
        html = client.get("/workspace/")
        assert "no-store" in html.headers.get("cache-control", "")
        asset = client.get("/workspace/assets/shared-X.js")
        assert asset.status_code == 200
        assert "immutable" in asset.headers.get("cache-control", "")


class TestWorkspaceDegradation:
    """Old build directory: only index.html exists."""

    @pytest.mark.parametrize("docs_enabled", ["true", "false"])
    def test_workspace_missing_returns_fixed_json(
        self, tmp_path, monkeypatch, docs_enabled
    ):
        monkeypatch.setenv("ENABLE_API_DOCS", docs_enabled)
        client = _client(tmp_path, monkeypatch, workspace=False)

        for path in ("/workspace", "/workspace/"):
            resp = client.get(path, follow_redirects=False)
            assert resp.status_code == 200, path
            data = resp.json()
            assert data["service"] == "LightRAG Server"
            assert data["health_url"] == "/health"
            body = resp.text.lower()
            # Never a redirect, never an API-docs link or guidance.
            assert "docs" not in body, path
            assert "redoc" not in body, path

        # The admin UI must stay fully available.
        assert client.get("/webui/").status_code == 200

    def test_health_reports_entries_independently(self, tmp_path, monkeypatch):
        client = _client(tmp_path, monkeypatch, workspace=False)
        health = client.get("/health").json()
        assert health["webui_available"] is True
        assert health["workspace_available"] is False

    def test_health_reports_both_available(self, tmp_path, monkeypatch):
        client = _client(tmp_path, monkeypatch)
        health = client.get("/health").json()
        assert health["webui_available"] is True
        assert health["workspace_available"] is True


class TestDefaultUi:
    def test_default_is_webui(self, tmp_path, monkeypatch):
        client = _client(tmp_path, monkeypatch)
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code in (302, 307)
        assert resp.headers["location"] == "/webui/"

    def test_workspace_via_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LIGHTRAG_DEFAULT_UI", "workspace")
        client = _client(tmp_path, monkeypatch)
        resp = client.get("/", follow_redirects=False)
        assert resp.headers["location"] == "/workspace/"

    def test_workspace_via_cli(self, tmp_path, monkeypatch):
        client = _client(tmp_path, monkeypatch, "--default-ui", "workspace")
        resp = client.get("/", follow_redirects=False)
        assert resp.headers["location"] == "/workspace/"

    def test_root_path_preserved(self, tmp_path, monkeypatch):
        client = _client(
            tmp_path,
            monkeypatch,
            "--default-ui",
            "workspace",
            "--api-prefix",
            "/site01",
        )
        resp = client.get("/site01/", follow_redirects=False)
        assert resp.headers["location"] == "/site01/workspace/"

    def test_illegal_env_value_fails_startup(self, monkeypatch):
        monkeypatch.setenv("LIGHTRAG_DEFAULT_UI", "workspaces")
        monkeypatch.setattr(sys, "argv", ["lightrag-server"])
        from lightrag.api.config import parse_args

        with pytest.raises(SystemExit):
            parse_args()

    def test_illegal_cli_value_fails_startup(self, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["lightrag-server", "--default-ui", "workspaces"]
        )
        from lightrag.api.config import parse_args

        with pytest.raises(SystemExit):
            parse_args()

    @pytest.mark.parametrize("docs_enabled", ["true", "false"])
    def test_root_follows_workspace_degradation(
        self, tmp_path, monkeypatch, docs_enabled
    ):
        """default-ui=workspace with the workspace entry missing: '/' answers
        with the workspace degradation JSON — it must NOT redirect to the API
        docs, and it must NOT reroute to the still-available /webui/."""
        monkeypatch.setenv("ENABLE_API_DOCS", docs_enabled)
        monkeypatch.setenv("LIGHTRAG_DEFAULT_UI", "workspace")
        client = _client(tmp_path, monkeypatch, workspace=False)

        resp = client.get("/", follow_redirects=False)
        assert resp.status_code == 200
        body = resp.text.lower()
        assert "docs" not in body
        assert "webui" not in body
        # /webui/ itself stays reachable.
        assert client.get("/webui/").status_code == 200
