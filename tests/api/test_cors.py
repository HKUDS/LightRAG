"""
Tests for CORS middleware configuration on the LightRAG API server.

LightRAG authenticates exclusively via request headers (Authorization Bearer
token and X-API-Key) and never via cookies or other ambient credentials. The
Fetch spec forbids pairing the wildcard origin "*" with credentialed responses
("Access-Control-Allow-Origin: *" together with
"Access-Control-Allow-Credentials: true"). These tests pin the resulting
behavior:

- When CORS_ORIGINS is the wildcard "*" (the default), credentials are disabled
  and the server responds with a clean "Access-Control-Allow-Origin: *" and no
  "Access-Control-Allow-Credentials" header.
- When CORS_ORIGINS is an explicit allowlist, the matching origin is reflected
  and credentials are enabled.
"""

import sys
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient


# Env vars that the project's `.env` may have populated (via load_dotenv at
# import time of lightrag.api.config). Tests must be hermetic and not depend on
# developer-local .env values, so we clear/override anything that affects
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
    "CORS_ORIGINS",
    "LIGHTRAG_API_PREFIX",
    "LIGHTRAG_KV_STORAGE",
    "LIGHTRAG_VECTOR_STORAGE",
    "LIGHTRAG_GRAPH_STORAGE",
    "LIGHTRAG_DOC_STATUS_STORAGE",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Isolate tests from developer-local .env pollution and global config state.

    Clear env vars that affect parse_args()/create_app(), then set the minimal
    viable defaults (ollama bindings) so create_app's binding validation passes
    without touching real services. The global config singleton in
    lightrag.api.config is reset before and after each test so neither developer
    .env values nor other tests' parsed args leak into our CORS assertions.
    """
    for var in _ENV_VARS_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_BINDING", "ollama")
    monkeypatch.setenv("EMBEDDING_BINDING", "ollama")

    import lightrag.api.config as config

    config._global_args = None
    config._initialized = False
    yield
    config._global_args = None
    config._initialized = False


def _build_client(cors_origins, monkeypatch):
    """Create a TestClient for an app configured with the given CORS_ORIGINS."""
    from lightrag.api.config import parse_args, initialize_config

    monkeypatch.setenv("CORS_ORIGINS", cors_origins)

    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server"]
        args = parse_args()
    finally:
        sys.argv = original_argv

    # create_app / get_cors_origins read the module-level global_args proxy
    # (not the passed args), which otherwise lazily re-parses sys.argv — by then
    # polluted with pytest's argv. Pin the parsed args explicitly.
    initialize_config(args, force=True)

    with patch("lightrag.api.lightrag_server.LightRAG") as mock_rag:
        mock_rag.return_value = MagicMock()
        from lightrag.api.lightrag_server import create_app

        app = create_app(args)
    return TestClient(app)


def _preflight(client, origin):
    """Issue a CORS preflight request and return the response."""
    return client.options(
        "/query",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type",
        },
    )


class TestWildcardCorsDisablesCredentials:
    """With CORS_ORIGINS="*" the config must stay spec-compliant."""

    def test_preflight_wildcard_no_credentials(self, monkeypatch):
        client = _build_client("*", monkeypatch)
        resp = _preflight(client, "https://evil.example.com")

        assert resp.headers["access-control-allow-origin"] == "*"
        # Wildcard origin must NOT be paired with credentials.
        assert "access-control-allow-credentials" not in resp.headers

    def test_simple_request_wildcard_no_credentials(self, monkeypatch):
        client = _build_client("*", monkeypatch)
        # /health is whitelisted (no auth) so this returns a real response with
        # CORS headers attached by the middleware.
        resp = client.get("/health", headers={"Origin": "https://evil.example.com"})

        assert resp.headers.get("access-control-allow-origin") == "*"
        assert "access-control-allow-credentials" not in resp.headers

    def test_wildcard_mixed_with_explicit_origin_no_credentials(self, monkeypatch):
        # Starlette treats any list containing "*" as allow-all, so a mixed config
        # is still allow-all and must NOT enable credentials.
        client = _build_client("*,https://app.example.com", monkeypatch)
        resp = _preflight(client, "https://evil.example.com")

        assert "access-control-allow-credentials" not in resp.headers

    def test_wildcard_trailing_comma_no_credentials(self, monkeypatch):
        # A trailing comma yields ["*", ""]; the empty entry is dropped and the
        # wildcard still disables credentials.
        client = _build_client("*,", monkeypatch)
        resp = _preflight(client, "https://evil.example.com")

        assert "access-control-allow-credentials" not in resp.headers


class TestEmptyCorsFailsClosed:
    """An explicitly empty CORS_ORIGINS disables cross-origin access (fail closed)."""

    def test_empty_string_allows_no_origin(self, monkeypatch):
        # CORS_ORIGINS= is an explicit "disable cross-origin" config and must not
        # silently widen to "*".
        client = _build_client("", monkeypatch)
        resp = _preflight(client, "https://evil.example.com")

        assert "access-control-allow-origin" not in resp.headers
        assert "access-control-allow-credentials" not in resp.headers

    def test_only_commas_allows_no_origin(self, monkeypatch):
        # A value with no real origins (e.g. ",,") also fails closed.
        client = _build_client(",,", monkeypatch)
        resp = _preflight(client, "https://evil.example.com")

        assert "access-control-allow-origin" not in resp.headers
        assert "access-control-allow-credentials" not in resp.headers


class TestExplicitAllowlistEnablesCredentials:
    """An explicit origin allowlist reflects the origin and allows credentials."""

    def test_preflight_allowed_origin_reflected_with_credentials(self, monkeypatch):
        client = _build_client(
            "https://app.example.com,https://dash.example.com", monkeypatch
        )
        resp = _preflight(client, "https://app.example.com")

        assert resp.headers["access-control-allow-origin"] == "https://app.example.com"
        assert resp.headers["access-control-allow-credentials"] == "true"

    def test_preflight_disallowed_origin_not_reflected(self, monkeypatch):
        client = _build_client("https://app.example.com", monkeypatch)
        resp = _preflight(client, "https://evil.example.com")

        # A non-allowlisted origin is never echoed back.
        assert (
            resp.headers.get("access-control-allow-origin")
            != "https://evil.example.com"
        )
