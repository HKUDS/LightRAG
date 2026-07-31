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
            assert len(servers) > 0, (
                "OpenAPI spec should have servers entry when root_path is set"
            )
            assert servers[0].get("url") == "/test-api", (
                f"Expected servers URL to be exactly /test-api, got: {servers[0].get('url')}"
            )

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
                assert not path.startswith("/test-api/"), (
                    f"Path {path} should not be prefixed with /test-api/ in root_path mode"
                )


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


class TestUvicornRootPathSemantics:
    """Lock in the deployment contract that both proxy-strip and verbatim
    forwarding work through FastAPI's app-level ``root_path`` plus a
    ``_RootPathNormalizationMiddleware`` — without ever passing
    ``root_path`` through to uvicorn/gunicorn.

    Background:

      - uvicorn builds ``scope["path"] = uvicorn.root_path + <incoming http
        path>`` and ``scope["root_path"] = uvicorn.root_path``
        (uvicorn/protocols/http/h11_impl.py).
      - FastAPI's ``__call__`` overrides ``scope["root_path"]`` from
        ``app.root_path`` but does NOT touch ``scope["path"]``
        (fastapi/applications.py:1131-1134).
      - Starlette's ``Mount.matches`` mutates the child scope's root_path
        to ``original + matched_path`` and again leaves ``scope["path"]``
        untouched (starlette/routing.py:401-432). The inner sub-app
        (e.g. StaticFiles) then sees ``scope["path"]`` and
        ``scope["root_path"]`` that do not overlap, and 404s.

    Concrete failure mode without the middleware (proxy strips /site01,
    backend sees ``/webui/``):

        outer get_route_path:  /webui/ does not start with /site01 → /webui/
        Mount.matches:         path_regex "^/webui/(?P<path>.*)$" matches
                               child scope.root_path = /site01/webui
                               child scope.path      = /webui/  (unchanged)
        StaticFiles.get_path:  /webui/ does not start with /site01/webui →
                               returns /webui/ → looked up as filename
                               "webui" inside the webui static dir → 404

    ``_RootPathNormalizationMiddleware`` runs before routing and prepends
    ``root_path`` to ``scope["path"]`` whenever the latter does not already
    start with it. This makes both modes converge to the canonical ASGI
    form (path always contains root_path) without requiring uvicorn's
    --root-path — which would break verbatim mode by double-prefixing.
    """

    @staticmethod
    async def _call_with_scope(app, http_path, *, uvicorn_root_path=""):
        """Drive the ASGI app the way uvicorn would.

        Simulates uvicorn's scope construction so we can catch the
        path-doubling bug that TestClient hides.
        """
        full_path = uvicorn_root_path + http_path
        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
            "root_path": uvicorn_root_path,
            "path": full_path,
            "raw_path": full_path.encode("ascii"),
            "query_string": b"",
            "headers": [],
            "state": {},
        }
        status = {"code": None}

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]

        await app(scope, receive, send)
        return status["code"]

    def _build_app_with_prefix(self, prefix):
        original_argv = sys.argv.copy()
        try:
            sys.argv = ["lightrag-server", "--api-prefix", prefix]
            from lightrag.api.config import parse_args
            from lightrag.api.lightrag_server import create_app

            args = parse_args()
            with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
                mock_rag.return_value = MagicMock()
                return create_app(args)
        finally:
            sys.argv = original_argv

    @pytest.mark.asyncio
    async def test_route_strip_mode_matches(self):
        """Plain Route, proxy-strip mode: backend receives /openapi.json."""
        app = self._build_app_with_prefix("/site01")
        status = await self._call_with_scope(app, "/openapi.json")
        assert status == 200

    @pytest.mark.asyncio
    async def test_route_verbatim_mode_matches(self):
        """Plain Route, verbatim mode: backend receives /site01/openapi.json.

        Locks in the contract that PR #3128's original fix broke: setting
        uvicorn root_path would have made this case 404 by doubling the
        prefix in scope["path"].
        """
        app = self._build_app_with_prefix("/site01")
        status = await self._call_with_scope(app, "/site01/openapi.json")
        assert status == 200

    @pytest.mark.asyncio
    async def test_mount_strip_mode_matches(self):
        """WebUI Mount, proxy-strip mode: backend receives /webui/.

        This is the bug the middleware fixes. Without normalization,
        StaticFiles.get_path sees path=/webui/ and root_path=/site01/webui
        (mutated by Mount.matches) and serves the literal "webui" filename
        lookup → 404. With normalization, scope.path becomes
        /site01/webui/ before routing and the lookup resolves to
        index.html.
        """
        app = self._build_app_with_prefix("/site01")
        status = await self._call_with_scope(app, "/webui/")
        assert status == 200

    @pytest.mark.asyncio
    async def test_mount_verbatim_mode_matches(self):
        """WebUI Mount, verbatim mode: backend receives /site01/webui/.

        Already canonical; the middleware is a no-op for this case. The
        test guards against accidentally regressing verbatim while fixing
        strip — symmetric to test_route_verbatim_mode_matches but exercises
        the Mount path with its nested ``get_route_path`` resolution.
        """
        app = self._build_app_with_prefix("/site01")
        status = await self._call_with_scope(app, "/site01/webui/")
        assert status == 200

    @pytest.mark.asyncio
    async def test_simulated_uvicorn_root_path_breaks_verbatim(self):
        """Document the failure mode that the revert prevents.

        If uvicorn's root_path were also "/site01", uvicorn would build
        scope.path = "/site01" + "/site01/openapi.json". Starlette strips
        one prefix; the remaining "/site01/openapi.json" has no route.
        """
        app = self._build_app_with_prefix("/site01")
        status = await self._call_with_scope(
            app, "/site01/openapi.json", uvicorn_root_path="/site01"
        )
        assert status == 404

    def test_launcher_does_not_pass_root_path_to_uvicorn(self, monkeypatch, tmp_path):
        """Guard against re-adding root_path to the uvicorn launcher kwargs.

        Mocks uvicorn.run and exercises lightrag_server.main() far enough
        to capture the config dict, then asserts root_path is absent.
        """
        monkeypatch.setenv("LIGHTRAG_API_PREFIX", "/site01")
        monkeypatch.setattr(sys, "argv", ["lightrag-server"])

        from lightrag.api import lightrag_server

        captured = {}

        def fake_run(**kwargs):
            captured.update(kwargs)

        monkeypatch.setattr(lightrag_server, "uvicorn", MagicMock(run=fake_run))
        monkeypatch.setattr(lightrag_server, "check_env_file", lambda: True)
        monkeypatch.setattr(
            lightrag_server, "check_and_install_dependencies", lambda: None
        )
        monkeypatch.setattr(lightrag_server, "configure_logging", lambda: None)
        monkeypatch.setattr(lightrag_server, "update_uvicorn_mode_config", lambda: None)
        monkeypatch.setattr(lightrag_server, "display_splash_screen", lambda *_: None)
        with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
            mock_rag.return_value = MagicMock()
            # Re-parse args under the patched env so global_args picks up the prefix.
            from lightrag.api.config import parse_args, global_args as _ga

            new_args = parse_args()
            for attr in vars(new_args):
                setattr(_ga, attr, getattr(new_args, attr))

            lightrag_server.main()

        assert "root_path" not in captured, (
            "uvicorn_config must not include root_path; rely on FastAPI's "
            "app-level root_path only — see TestUvicornRootPathSemantics docstring."
        )

    def test_gunicorn_worker_does_not_inject_root_path(self):
        """Symmetric guard for the multi-worker launcher.

        The regression this protects against is a worker class that injects
        ``root_path`` through ``CONFIG_KWARGS``, re-creating the path-doubling
        in worker processes. Asserted as that invariant rather than as an exact
        class name: the configured class IS a subclass now
        (``LightRAGUvicornWorker`` adds the orphan check gunicorn's own workers
        have and uvicorn's drops), and pinning the name would have blocked it
        while proving nothing extra — a subclass could add root_path under any
        name.
        """
        from uvicorn_worker import UvicornWorker

        from lightrag.api import gunicorn_config
        from gunicorn.util import load_class

        worker_class = load_class(gunicorn_config.worker_class)
        assert issubclass(worker_class, UvicornWorker), worker_class
        # Covers the subclass's own kwargs and everything it inherits.
        assert "root_path" not in worker_class.CONFIG_KWARGS, (
            "the gunicorn worker must not inject root_path; rely on FastAPI's "
            "app-level root_path only — see TestUvicornRootPathSemantics docstring."
        )


class TestWhitelistUnderApiPrefix:
    """WHITELIST_PATHS is matched against route paths, on the real application.

    The unit-level matrix lives in tests/api/auth/test_whitelist_path_prefix.py;
    these two cases pin the same contract on the app ``create_app`` actually
    builds, because that is where the mount prefix, the normalizing middleware and
    the routers' own auth dependency all come together.
    """

    @pytest.fixture
    def _default_whitelist(self, monkeypatch):
        """Pin the shipped default so a developer-local WHITELIST_PATHS in .env
        (already baked into the module-level patterns at import time) cannot
        change what these tests assert."""
        original_argv = sys.argv.copy()
        try:
            # config resolves its args on first attribute access; importing under
            # pytest's own argv would argparse it and exit.
            sys.argv = [sys.argv[0]]
            from lightrag.api import utils_api
        finally:
            sys.argv = original_argv

        monkeypatch.setattr(
            utils_api, "whitelist_patterns", [("/health", False), ("/api", True)]
        )

    @staticmethod
    def _args_with_prefix(prefix: str):
        from lightrag.api.config import parse_args

        original_argv = sys.argv.copy()
        try:
            sys.argv = [
                "lightrag-server",
                "--api-prefix",
                prefix,
                "--key",
                "test-api-key",
            ]
            return parse_args()
        finally:
            sys.argv = original_argv

    @pytest.fixture
    def _colliding_prefix_args(self):
        """An api_prefix that collides with the default whitelist's /api entry —
        the value the project's own --help suggests."""
        return self._args_with_prefix("/api/v1")

    @pytest.fixture
    def _noncolliding_prefix_args(self):
        """The prefix style the multi-site docs use. Nothing here collides with
        the whitelist, so this is where the *under*-exemption showed: no entry
        matched and the exempt routes started demanding credentials."""
        return self._args_with_prefix("/site01")

    @pytest.mark.parametrize("mode", ["verbatim", "strip"])
    def test_destructive_route_requires_auth_under_a_colliding_prefix(
        self, _colliding_prefix_args, _default_whitelist, mode
    ):
        """DELETE /documents answered 200 unauthenticated before this fix, in
        both forwarding modes."""
        with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
            mock_rag.return_value = MagicMock()
            from lightrag.api.lightrag_server import create_app

            client = TestClient(create_app(_colliding_prefix_args))
            prefix = "" if mode == "strip" else "/api/v1"

            assert client.delete(f"{prefix}/documents").status_code == 401
            assert client.get(f"{prefix}/documents").status_code == 401

    @pytest.mark.parametrize("mode", ["verbatim", "strip"])
    def test_whitelisted_routes_stay_open_under_a_prefix(
        self, _noncolliding_prefix_args, _default_whitelist, mode
    ):
        """/health must keep answering liveness probes (#3294) and the
        Ollama-compatible routes must keep their documented exemption.

        Both answered 401 under this prefix before the fix. The assertion is
        "not 401" rather than 200 because these handlers reach further into a
        LightRAG that is only a MagicMock here: /health reads shared storage that
        no test initializes, and /api/tags feeds mock attributes to a Pydantic
        response model. Both therefore fail *after* the auth gate, which is why
        server exceptions are surfaced as responses rather than raised. Getting
        past the dependency at all is the property under test; the unit-level
        matrix pins the exact 200.
        """
        with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
            mock_rag.return_value = MagicMock()
            from lightrag.api.lightrag_server import create_app

            client = TestClient(
                create_app(_noncolliding_prefix_args), raise_server_exceptions=False
            )
            prefix = "" if mode == "strip" else "/site01"

            assert client.get(f"{prefix}/health").status_code != 401
            # GET, matching the real registration; GET /api/chat would be a 405
            # from the router without the dependency ever running.
            assert client.get(f"{prefix}/api/tags").status_code != 401
