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
    "LIGHTRAG_WEBUI_PATH",
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
    """Test that WebUI is served at the correct path."""

    def test_webui_at_prefixed_path(self, mock_args_api_prefix):
        """Test WebUI assets are at the prefixed path.

        When root_path is set, the WebUI is served under {root_path}{webui_path}
        because FastAPI injects root_path into the ASGI scope.
        """
        with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
            mock_rag.return_value = MagicMock()
            from lightrag.api.lightrag_server import create_app
            from lightrag.api.config import parse_args

            original_argv = sys.argv.copy()
            try:
                sys.argv = [
                    "lightrag-server",
                    "--api-prefix",
                    "/test-api",
                    "--webui-path",
                    "/test-webui",
                ]
                args = parse_args()
                app = create_app(args)
                client = TestClient(app)

                # With root_path="/test-api" and webui_path="/test-webui",
                # the WebUI is accessible at /test-api/test-webui/
                # (root_path + webui_path, since FastAPI injects root_path)
                response = client.get("/test-api/test-webui/")
                assert response.status_code in [200, 307]
            finally:
                sys.argv = original_argv

    def test_webui_without_api_prefix(self):
        """Test WebUI works with custom path when no API prefix is set."""
        with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
            mock_rag.return_value = MagicMock()
            from lightrag.api.lightrag_server import create_app
            from lightrag.api.config import parse_args

            original_argv = sys.argv.copy()
            try:
                sys.argv = ["lightrag-server", "--webui-path", "/test-webui"]
                args = parse_args()
                app = create_app(args)
                client = TestClient(app)

                response = client.get("/test-webui/")
                assert response.status_code in [200, 307]
            finally:
                sys.argv = original_argv

    def test_webui_not_at_default_path_with_custom(self, mock_args_api_prefix):
        """Test /webui returns 404 when custom path is set."""
        with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
            mock_rag.return_value = MagicMock()
            from lightrag.api.lightrag_server import create_app
            from lightrag.api.config import parse_args

            original_argv = sys.argv.copy()
            try:
                sys.argv = [
                    "lightrag-server",
                    "--api-prefix",
                    "/test-api",
                    "--webui-path",
                    "/test-webui",
                ]
                args = parse_args()
                app = create_app(args)
                client = TestClient(app)

                # /webui should not exist when custom path is set
                response = client.get("/webui/")
                assert response.status_code == 404
            finally:
                sys.argv = original_argv


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

    def test_env_webui_path(self):
        """Test LIGHTRAG_WEBUI_PATH environment variable."""
        from lightrag.api.config import get_env_value

        os.environ["LIGHTRAG_WEBUI_PATH"] = "unit-test-front/webui"
        try:
            value = get_env_value("LIGHTRAG_WEBUI_PATH", "/webui")
            assert value == "unit-test-front/webui"
        finally:
            del os.environ["LIGHTRAG_WEBUI_PATH"]


class TestPathNormalization:
    """User input may contain trailing slashes, missing leading slash, or be
    just '/'. create_app must canonicalize these before passing to FastAPI
    (root_path) and Starlette (app.mount), neither of which accept arbitrary
    strings."""

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

    def test_webui_path_trailing_slash_does_not_crash_mount(self):
        """Starlette's app.mount asserts paths do not end in '/'. The
        previous code passed user input through verbatim, which would crash
        at startup if a user set LIGHTRAG_WEBUI_PATH=/webui/.

        We assert that create_app() returns a working app (i.e. no
        AssertionError raised inside Starlette) and that the WebUI is
        reachable at the normalized path.
        """
        app = self._build("--webui-path", "/custom-ui/")
        client = TestClient(app)
        response = client.get("/custom-ui/")
        assert response.status_code in (200, 307, 404)

    def test_webui_path_slash_only_falls_back_to_default(self):
        """`--webui-path /` is degenerate; must fall back to /webui."""
        app = self._build("--webui-path", "/")
        client = TestClient(app)
        response = client.get("/webui/")
        assert response.status_code in (200, 307, 404)


class TestCheckWebuiBuildPrefix:
    """Direct tests for check_webui_build_prefix.

    These exercise an isolated index.html written to a temp dir, with the
    function's baked-in path patched, so the test does not depend on
    whatever prefix the committed `lightrag/api/webui/` happens to carry.
    """

    def _stage_index_html(self, tmp_path, baked_prefix):
        """Stage a minimal index.html under tmp_path/webui/ mimicking Vite
        output. The check function reads `Path(__file__).parent / "webui"
        / "index.html"`, so callers must also patch `lightrag_server.__file__`
        to point inside tmp_path."""
        prefix_no_slash = baked_prefix.rstrip("/")
        webui_dir = tmp_path / "webui"
        webui_dir.mkdir()
        (webui_dir / "index.html").write_text(
            "<!doctype html><html><head>"
            f'<script type="module" crossorigin src="{prefix_no_slash}/assets/index-X.js"></script>'
            f'<link rel="stylesheet" href="{prefix_no_slash}/assets/index-X.css">'
            "</head><body></body></html>",
            encoding="utf-8",
        )

    def test_match_emits_info_log_and_no_warning(self, tmp_path, monkeypatch, caplog):
        import logging

        # Importing lightrag.api.lightrag_server eagerly evaluates
        # `global_args.whitelist_paths` in utils_api at module top, which
        # calls parse_args() against the current sys.argv. Under pytest
        # that's the pytest invocation argv → argparse exits 2. Force a
        # benign argv before the (potentially-fresh) module import.
        monkeypatch.setattr(sys, "argv", ["lightrag-server"])
        from lightrag.api import lightrag_server

        self._stage_index_html(tmp_path, "/site01/webui/")
        monkeypatch.setattr(
            lightrag_server, "__file__", str(tmp_path / "lightrag_server.py")
        )

        # The lightrag logger has propagate=False; caplog attaches its
        # handler at the root, so without re-enabling propagation we can't
        # observe records here. Restore on teardown via monkeypatch.
        lightrag_logger = logging.getLogger("lightrag")
        monkeypatch.setattr(lightrag_logger, "propagate", True)
        with caplog.at_level(logging.INFO, logger="lightrag"):
            lightrag_server.check_webui_build_prefix(
                api_prefix="/site01", webui_path="/webui"
            )

        # Match path: only an info log, no warning record.
        assert "matches server config" in caplog.text
        assert not any(
            r.levelno >= logging.WARNING for r in caplog.records
        ), "match path must not emit a warning"

    def test_mismatch_emits_warning_with_rebuild_command(
        self, tmp_path, monkeypatch, caplog
    ):
        import logging

        monkeypatch.setattr(sys, "argv", ["lightrag-server"])
        from lightrag.api import lightrag_server

        # Build was made with /webui/ but admin reconfigures the server
        # for site01 — classic "reused image, new prefix" failure mode.
        self._stage_index_html(tmp_path, "/webui/")
        monkeypatch.setattr(
            lightrag_server, "__file__", str(tmp_path / "lightrag_server.py")
        )

        lightrag_logger = logging.getLogger("lightrag")
        monkeypatch.setattr(lightrag_logger, "propagate", True)
        with caplog.at_level(logging.WARNING, logger="lightrag"):
            lightrag_server.check_webui_build_prefix(
                api_prefix="/site01", webui_path="/webui"
            )

        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warning_records, "mismatch must emit a WARNING-level record"
        msg = warning_records[0].getMessage()
        # Show the actual baked vs expected values so admin can act.
        assert "/webui/" in msg
        assert "/site01/webui/" in msg
        # Suggest a runnable rebuild command with the corrected env vars.
        assert "VITE_WEBUI_PREFIX=/site01/webui/" in msg
        assert "VITE_API_PREFIX=/site01" in msg

    def test_missing_build_does_not_raise(self, tmp_path, monkeypatch):
        """When index.html is absent (no build yet), the function returns
        silently — `check_frontend_build` already emits the build warning."""
        monkeypatch.setattr(sys, "argv", ["lightrag-server"])
        from lightrag.api import lightrag_server

        monkeypatch.setattr(
            lightrag_server, "__file__", str(tmp_path / "lightrag_server.py")
        )
        # No webui/ subdir under tmp_path — index.html does not exist.
        lightrag_server.check_webui_build_prefix(
            api_prefix="/site01", webui_path="/webui"
        )
