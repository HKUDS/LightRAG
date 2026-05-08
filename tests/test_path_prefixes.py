"""
Integration tests for API and WebUI path prefix support via root_path.

With the root_path approach, routes always stay at their natural paths
(/docs, /health, /query, /documents/...). The api_prefix is passed to
FastAPI's root_path parameter, which controls the servers URL in the
OpenAPI spec for correct reverse proxy operation.
"""

import os
import sys
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

import pytest


# Env vars that the project's `.env` may have populated (via load_dotenv at
# import time of lightrag.api.config). Tests must be hermetic and not depend
# on developer-local .env values, so we clear/override anything that affects
# parse_args() / create_app().
_ENV_VARS_TO_ISOLATE = (
    "LLM_BINDING",
    "EMBEDDING_BINDING",
    "LLM_BINDING_HOST",
    "LLM_BINDING_API_KEY",
    "LLM_MODEL",
    "EMBEDDING_BINDING_HOST",
    "EMBEDDING_BINDING_API_KEY",
    "EMBEDDING_MODEL",
    "LIGHTRAG_API_PREFIX",
    "LIGHTRAG_KV_STORAGE",
    "LIGHTRAG_VECTOR_STORAGE",
    "LIGHTRAG_GRAPH_STORAGE",
    "LIGHTRAG_DOC_STATUS_STORAGE",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Isolate tests from developer-local .env pollution.

    The lightrag.api.config module loads .env at import time, which can leave
    bindings/hosts/keys in os.environ that mismatch what these tests assume.
    Clear them, then set the minimal viable defaults (ollama bindings) so
    create_app's binding validation passes without touching real services.
    """
    for var in _ENV_VARS_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_BINDING", "ollama")
    monkeypatch.setenv("EMBEDDING_BINDING", "ollama")


@pytest.fixture
def mock_args_api_prefix():
    """Create mock args with API prefix."""
    from lightrag.api.config import parse_args

    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server", "--api-prefix", "/test-api"]
        args = parse_args()
        yield args
    finally:
        sys.argv = original_argv


@pytest.fixture
def mock_args_no_prefix():
    """Create mock args without API prefix."""
    from lightrag.api.config import parse_args

    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server"]
        args = parse_args()
        yield args
    finally:
        sys.argv = original_argv


class TestRootPathConfiguration:
    """Test that root_path is set correctly on the FastAPI app."""

    def test_root_path_set_when_prefix_provided(self, mock_args_api_prefix):
        """Test app.root_path reflects api_prefix."""
        with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
            mock_rag.return_value = MagicMock()
            from lightrag.api.lightrag_server import create_app

            app = create_app(mock_args_api_prefix)
            assert app.root_path == "/test-api"

    def test_root_path_none_when_no_prefix(self, mock_args_no_prefix):
        """Test app.root_path is not set when no prefix is configured."""
        with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
            mock_rag.return_value = MagicMock()
            from lightrag.api.lightrag_server import create_app

            app = create_app(mock_args_no_prefix)
            # When no prefix, root_path is None (not passed to FastAPI)
            # FastAPI stores None as-is, which means no root_path injection
            assert not app.root_path


class TestRoutesAtNaturalPaths:
    """Test that routes stay at their natural paths regardless of root_path."""

    def test_routes_accessible_at_both_paths_with_prefix(self, mock_args_api_prefix):
        """With root_path, routes work at both prefixed and natural paths.

        FastAPI injects root_path into the ASGI scope, and Starlette strips
        it from the path before matching. So /test-api/docs and /docs both work.
        """
        with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
            mock_rag.return_value = MagicMock()
            from lightrag.api.lightrag_server import create_app

            app = create_app(mock_args_api_prefix)
            client = TestClient(app)

            # Natural path works
            response = client.get("/docs")
            assert response.status_code == 200

            response = client.get("/openapi.json")
            assert response.status_code == 200

            # Prefixed path also works (FastAPI strips root_path from scope)
            response = client.get("/test-api/docs")
            assert response.status_code == 200

            response = client.get("/test-api/openapi.json")
            assert response.status_code == 200

    def test_document_routes_at_natural_path(self, mock_args_api_prefix):
        """Test document routes are at /documents/ (their router-level prefix)."""
        with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
            mock_rag.return_value = MagicMock()
            from lightrag.api.lightrag_server import create_app

            app = create_app(mock_args_api_prefix)
            client = TestClient(app)

            response = client.post(
                "/documents/paginated",
                json={},
                headers={"Authorization": "Bearer test"},
            )
            # The route is mounted; the mocked LightRAG may cause 401/422/500,
            # but a missing route (404) or wrong method (405) means routing
            # itself broke and is what we want to catch here.
            assert response.status_code not in (404, 405)

    def test_routes_accessible_at_root_no_prefix(self, mock_args_no_prefix):
        """Test routes are at root when no prefix is set (default)."""
        with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
            mock_rag.return_value = MagicMock()
            from lightrag.api.lightrag_server import create_app

            app = create_app(mock_args_no_prefix)
            client = TestClient(app)

            # API docs accessible at root
            response = client.get("/docs")
            assert response.status_code == 200

            # openapi.json at root
            response = client.get("/openapi.json")
            assert response.status_code == 200

            # Prefixed paths return 404 when no root_path is configured
            response = client.get("/test-api/docs")
            assert response.status_code == 404


class TestOpenAPISpecIntegration:
    """Test that OpenAPI spec uses root_path for servers URL."""

    def test_openapi_spec_has_servers_url_with_prefix(self, mock_args_api_prefix):
        """Test OpenAPI spec servers URL includes the prefix via root_path."""
        with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
            mock_rag.return_value = MagicMock()
            from lightrag.api.lightrag_server import create_app

            app = create_app(mock_args_api_prefix)
            client = TestClient(app)

            # OpenAPI JSON is served at the natural path
            response = client.get("/openapi.json")
            assert response.status_code == 200
            spec = response.json()

            # Servers URL should include the prefix
            servers = spec.get("servers", [])
            assert (
                len(servers) > 0
            ), "OpenAPI spec should have servers entry when root_path is set"
            assert (
                servers[0].get("url") == "/test-api"
            ), f"Expected servers URL to be exactly /test-api, got: {servers[0].get('url')}"

    def test_openapi_spec_no_servers_without_prefix(self, mock_args_no_prefix):
        """Test OpenAPI spec has no servers entry when no root_path."""
        with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
            mock_rag.return_value = MagicMock()
            from lightrag.api.lightrag_server import create_app

            app = create_app(mock_args_no_prefix)
            client = TestClient(app)

            response = client.get("/openapi.json")
            assert response.status_code == 200
            spec = response.json()

            # No servers when root_path is None/empty
            assert "servers" not in spec or spec["servers"] is None

    def test_openapi_spec_paths_at_natural_paths(self, mock_args_api_prefix):
        """Test OpenAPI spec paths are at natural paths (not prefixed)."""
        with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
            mock_rag.return_value = MagicMock()
            from lightrag.api.lightrag_server import create_app

            app = create_app(mock_args_api_prefix)
            client = TestClient(app)

            response = client.get("/openapi.json")
            assert response.status_code == 200
            spec = response.json()
            paths = spec.get("paths", {})

            # Paths should be at natural paths
            for path in paths:
                if path == "/":
                    continue
                assert not path.startswith(
                    "/test-api/"
                ), f"Path {path} should not be prefixed with /test-api/ in root_path mode"


class TestWebUIPrefixIntegration:
    """Test that the WebUI is served at the expected (fixed) /webui path,
    composed with `root_path` when an API prefix is set."""

    def test_webui_at_prefixed_path(self, mock_args_api_prefix):
        """With root_path="/test-api" the WebUI lives at /test-api/webui/
        because FastAPI injects root_path into the ASGI scope."""
        with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
            mock_rag.return_value = MagicMock()
            from lightrag.api.lightrag_server import create_app

            app = create_app(mock_args_api_prefix)
            client = TestClient(app)

            response = client.get("/test-api/webui/")
            assert response.status_code in [200, 307]

    def test_webui_without_api_prefix(self, mock_args_no_prefix):
        """Without an API prefix the WebUI is served at /webui/."""
        with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
            mock_rag.return_value = MagicMock()
            from lightrag.api.lightrag_server import create_app

            app = create_app(mock_args_no_prefix)
            client = TestClient(app)

            response = client.get("/webui/")
            assert response.status_code in [200, 307]


class TestEnvironmentVariables:
    """Test that environment variables are read correctly."""

    def test_env_api_prefix(self):
        """Test LIGHTRAG_API_PREFIX environment variable."""
        from lightrag.api.config import get_env_value

        os.environ["LIGHTRAG_API_PREFIX"] = "unit-test-back/api"
        try:
            value = get_env_value("LIGHTRAG_API_PREFIX", "")
            assert value == "unit-test-back/api"
        finally:
            del os.environ["LIGHTRAG_API_PREFIX"]


class TestPathNormalization:
    """User input for `--api-prefix` may contain trailing slashes, a missing
    leading slash, or be just '/'. create_app must canonicalize these before
    passing to FastAPI's `root_path`, which doesn't accept arbitrary strings."""

    def _build(self, *cli_args):
        # sys.argv must be the lightrag-server form *before* lightrag_server is
        # imported, because importing lightrag.api.utils_api evaluates
        # `global_args.whitelist_paths` at module top level, which triggers
        # parse_args() against whatever sys.argv currently holds.
        original_argv = sys.argv.copy()
        try:
            sys.argv = ["lightrag-server", *cli_args]
            from lightrag.api.config import parse_args
            from lightrag.api.lightrag_server import create_app

            args = parse_args()
            with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
                mock_rag.return_value = MagicMock()
                return create_app(args)
        finally:
            sys.argv = original_argv

    def test_api_prefix_slash_only_treated_as_empty(self):
        """`--api-prefix /` is degenerate; must collapse to no prefix."""
        app = self._build("--api-prefix", "/")
        assert not app.root_path

    def test_api_prefix_trailing_slash_stripped(self):
        """Trailing slash on api_prefix is stripped to keep OpenAPI servers
        URL clean and avoid double-slash artifacts."""
        app = self._build("--api-prefix", "/api/v1/")
        assert app.root_path == "/api/v1"

    def test_api_prefix_missing_leading_slash_added(self):
        app = self._build("--api-prefix", "api/v1")
        assert app.root_path == "/api/v1"


class TestRuntimeConfigInjection:
    """End-to-end tests for the WebUI runtime-config injection.

    The browser-visible URL prefixes are no longer baked into the bundle.
    Instead, the server replaces a placeholder comment in index.html with
    a `<script>window.__LIGHTRAG_CONFIG__ = {...}</script>` snippet on
    every HTML response, so one build can serve any reverse-proxy mount.

    These tests stage a minimal index.html in a tmp dir, patch
    `lightrag_server.__file__` so both `check_frontend_build()` and the
    static-files mount resolve to it, then drive the app via TestClient
    and assert that the body contains the expected injected JSON.
    """

    PLACEHOLDER = "<!-- __LIGHTRAG_RUNTIME_CONFIG__ -->"

    def _stage_index_html(self, tmp_path, *, with_placeholder=True):
        """Mirror what Vite emits: a tiny index.html with the runtime-config
        placeholder in <head> plus a hashed asset reference.

        with_placeholder=False simulates a stale build that pre-dates this
        feature — the server should serve it untouched, not crash.
        """
        webui_dir = tmp_path / "webui"
        webui_dir.mkdir()
        placeholder = self.PLACEHOLDER if with_placeholder else ""
        (webui_dir / "index.html").write_text(
            "<!doctype html><html><head>"
            f"{placeholder}"
            '<script type="module" crossorigin src="./assets/index-X.js"></script>'
            '<link rel="stylesheet" href="./assets/index-X.css">'
            "</head><body><div id=root></div></body></html>",
            encoding="utf-8",
        )

    def _build_app(self, tmp_path, monkeypatch, *cli_args):
        # Force benign argv before the (potentially fresh) module import —
        # see TestPathNormalization._build for the rationale.
        monkeypatch.setattr(sys, "argv", ["lightrag-server", *cli_args])
        from lightrag.api.config import parse_args
        from lightrag.api import lightrag_server
        from lightrag.api.lightrag_server import create_app

        # Redirect both check_frontend_build() and the StaticFiles mount to
        # our staged tmp directory.
        monkeypatch.setattr(
            lightrag_server, "__file__", str(tmp_path / "lightrag_server.py")
        )

        args = parse_args()
        with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
            mock_rag.return_value = MagicMock()
            return create_app(args)

    def test_injection_populates_window_config_with_prefix(self, tmp_path, monkeypatch):
        """With api_prefix=/site01, the injected script must carry both the
        api prefix and the composed webui prefix the browser will see."""
        self._stage_index_html(tmp_path)
        app = self._build_app(tmp_path, monkeypatch, "--api-prefix", "/site01")
        client = TestClient(app)

        response = client.get("/site01/webui/")
        assert response.status_code == 200
        body = response.text

        # Placeholder must be gone and replaced with the runtime config.
        assert self.PLACEHOLDER not in body
        assert "window.__LIGHTRAG_CONFIG__" in body
        assert '"apiPrefix": "/site01"' in body or '"apiPrefix":"/site01"' in body
        assert (
            '"webuiPrefix": "/site01/webui/"' in body
            or '"webuiPrefix":"/site01/webui/"' in body
        )

    def test_injection_default_prefixes_when_unconfigured(self, tmp_path, monkeypatch):
        """No CLI flags → empty api prefix and the default webui mount.
        The injected JSON must reflect this so the SPA falls through to
        same-origin requests."""
        self._stage_index_html(tmp_path)
        app = self._build_app(tmp_path, monkeypatch)
        client = TestClient(app)

        response = client.get("/webui/")
        assert response.status_code == 200
        body = response.text

        assert '"apiPrefix": ""' in body or '"apiPrefix":""' in body
        assert '"webuiPrefix": "/webui/"' in body or '"webuiPrefix":"/webui/"' in body

    def test_missing_placeholder_serves_original_html(self, tmp_path, monkeypatch):
        """Older builds without the placeholder must still serve cleanly —
        no 500, no partial replacement, just the original HTML. Avoids
        breaking anyone whose pre-built bundle is in use during an upgrade."""
        self._stage_index_html(tmp_path, with_placeholder=False)
        app = self._build_app(tmp_path, monkeypatch)
        client = TestClient(app)

        response = client.get("/webui/")
        assert response.status_code == 200
        # No placeholder was present, so no injected script either.
        assert "window.__LIGHTRAG_CONFIG__" not in response.text

    def test_injection_idempotent_across_requests(self, tmp_path, monkeypatch):
        """Each request reads the file fresh; the placeholder must be
        present in the *file* even after replies (we don't mutate it)."""
        self._stage_index_html(tmp_path)
        app = self._build_app(tmp_path, monkeypatch, "--api-prefix", "/abc")
        client = TestClient(app)

        first = client.get("/abc/webui/").text
        second = client.get("/abc/webui/").text
        assert first == second
        # Source file untouched.
        on_disk = (tmp_path / "webui" / "index.html").read_text(encoding="utf-8")
        assert self.PLACEHOLDER in on_disk

    def test_html_response_keeps_no_cache_headers(self, tmp_path, monkeypatch):
        """Injection must not regress the existing no-cache behaviour for
        HTML — otherwise an updated runtime config could be cached client-
        side and never picked up."""
        self._stage_index_html(tmp_path)
        app = self._build_app(tmp_path, monkeypatch, "--api-prefix", "/x")
        client = TestClient(app)

        response = client.get("/x/webui/")
        assert response.status_code == 200
        cache_control = response.headers.get("cache-control", "")
        assert "no-cache" in cache_control
        assert "no-store" in cache_control
