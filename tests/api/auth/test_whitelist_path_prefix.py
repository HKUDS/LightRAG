"""WHITELIST_PATHS under LIGHTRAG_API_PREFIX, over real HTTP.

``WHITELIST_PATHS`` entries are unprefixed route paths. The enforcing dependency
used to match them against ``request.url.path``, which in ASGI still carries the
mount prefix. Two failures followed from that single mismatch:

* **over-exemption** — the shipped default ``/health,/api/*`` compiles a bare
  ``/api`` prefix entry, so once the mount prefix itself began with ``/api``
  (``/api/v1`` is the value in the project's own ``--help``) every request path
  began with ``/api``, the whitelist degenerated to always-true, and the check
  returned before any credential was read. ``DELETE /documents`` answered 200
  with no credentials, in both forwarding modes;
* **under-exemption** — under any other prefix (``/site01``) no entry matched at
  all, so ``/health`` stopped answering liveness probes and the Ollama-compatible
  routes stopped being exempt.

Every assertion here goes through the real dependency over a real client, not
through ``path_is_whitelisted`` directly: the fix changed that function's
signature, so a direct call would fail on the old code with a ``TypeError``,
which proves only that a new symbol exists. A status-code assertion fails
behaviorally.

Route methods deliberately match the production registrations — ``GET
/api/tags``, never ``GET /api/chat`` (registered as POST). A method mismatch
returns 405 from the router *without running the dependency at all*, which is
identical whether or not the path is exempt.
"""

from __future__ import annotations

import importlib
import sys

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_utils_api = importlib.import_module("lightrag.api.utils_api")
_server = importlib.import_module("lightrag.api.lightrag_server")
sys.argv = _original_argv

pytestmark = pytest.mark.offline

API_KEY = "the-operators-secret-api-key"

# The shipped default: WHITELIST_PATHS=/health,/api/*
_DEFAULT_PATTERNS = [("/health", False), ("/api", True)]


def _client(monkeypatch, api_prefix: str) -> TestClient:
    """The real dependency and the real path-normalizing middleware, mounted the
    way ``create_app`` mounts them for a prefixed deployment."""
    monkeypatch.setattr(_utils_api, "whitelist_patterns", list(_DEFAULT_PATTERNS))
    monkeypatch.setattr(_utils_api, "auth_configured", True)

    dependency = _utils_api.get_combined_auth_dependency(API_KEY)
    app = FastAPI(root_path=api_prefix or None)
    if api_prefix:
        app.add_middleware(_server._RootPathNormalizationMiddleware)

    @app.get("/documents", dependencies=[Depends(dependency)])
    async def list_documents():
        return {"statuses": {}}

    @app.delete("/documents", dependencies=[Depends(dependency)])
    async def clear_documents():
        return {"status": "success"}

    @app.get("/health", dependencies=[Depends(dependency)])
    async def health():
        return {"status": "healthy"}

    # Mirrors the production registration: GET, mounted under the /api router.
    @app.get("/api/tags", dependencies=[Depends(dependency)])
    async def ollama_tags():
        return {"models": []}

    return TestClient(app)


def _request_path(api_prefix: str, route_path: str, mode: str) -> str:
    """The path the backend receives under each forwarding mode.

    verbatim — the proxy (or the Vite dev proxy) forwards unchanged.
    strip — nginx removes the prefix, per the example in MultiSiteDeployment.md.
    """
    return route_path if mode == "strip" else f"{api_prefix}{route_path}"


@pytest.mark.parametrize("mode", ["verbatim", "strip"])
@pytest.mark.parametrize("api_prefix", ["/api/v1", "/api", "/site01"])
def test_protected_routes_stay_authenticated_under_any_prefix(
    monkeypatch, api_prefix, mode
):
    """No prefix may exempt a protected route.

    On the old code the three ``/api...`` prefixes answered 200 here — a full
    unauthenticated read *and* an unauthenticated clear of the document store,
    in both forwarding modes.
    """
    client = _client(monkeypatch, api_prefix)

    path = _request_path(api_prefix, "/documents", mode)
    assert client.get(path).status_code == 401
    assert client.delete(path).status_code == 401


@pytest.mark.parametrize("mode", ["verbatim", "strip"])
@pytest.mark.parametrize("api_prefix", ["/api/v1", "/site01"])
def test_health_stays_exempt_under_any_prefix(monkeypatch, api_prefix, mode):
    """``/health`` promises to answer liveness probes unauthenticated (#3294).

    On the old code a non-colliding prefix broke that promise: no whitelist entry
    matched ``/site01/health``, so container and Kubernetes probes got 401.
    """
    client = _client(monkeypatch, api_prefix)

    response = client.get(_request_path(api_prefix, "/health", mode))
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@pytest.mark.parametrize("mode", ["verbatim", "strip"])
@pytest.mark.parametrize("api_prefix", ["/api/v1", "/site01"])
def test_ollama_routes_stay_exempt_under_any_prefix(monkeypatch, api_prefix, mode):
    """The default whitelist exists for Ollama-client compatibility, which the
    old matcher lost under a non-colliding prefix."""
    client = _client(monkeypatch, api_prefix)

    response = client.get(_request_path(api_prefix, "/api/tags", mode))
    assert response.status_code == 200


def test_no_prefix_behaviour_is_unchanged(monkeypatch):
    """The stability half: with no prefix the normalization is a no-op and every
    verdict must be exactly what it was before this change."""
    client = _client(monkeypatch, "")

    assert client.get("/documents").status_code == 401
    assert client.delete("/documents").status_code == 401
    assert client.get("/health").status_code == 200
    assert client.get("/api/tags").status_code == 200


@pytest.mark.parametrize("mode", ["verbatim", "strip"])
def test_valid_credentials_still_reach_protected_routes(monkeypatch, mode):
    """The prefix must not make an authenticated caller worse off either."""
    client = _client(monkeypatch, "/api/v1")

    response = client.get(
        _request_path("/api/v1", "/documents", mode), headers={"X-API-Key": API_KEY}
    )
    assert response.status_code == 200
