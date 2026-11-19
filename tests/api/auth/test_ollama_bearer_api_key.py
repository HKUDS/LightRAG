"""The Ollama-compatible routes accept LIGHTRAG_API_KEY as a bearer token.

Ollama clients (Open WebUI and the like) can carry one static key and send it as
``Authorization: Bearer <key>``; they never send ``X-API-Key`` and cannot run the
``/login`` flow a JWT needs. Once ``WHITELIST_PATHS`` stops exempting ``/api/*``
whenever a key is configured, these routes must recognize the key in that form,
or every Ollama client would be locked out.

Everything goes through the real ``OllamaAPI`` router over a real client, so the
assertions pin the wiring (``accept_bearer_api_key=True`` on that router) and
not just the dependency in isolation. A plain route built from the default
dependency pins the other half: the bearer form stays an Ollama-only allowance.
"""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_ollama_api = importlib.import_module("lightrag.api.routers.ollama_api")
_utils_api = importlib.import_module("lightrag.api.utils_api")
_auth = importlib.import_module("lightrag.api.auth")
sys.argv = _original_argv

pytestmark = pytest.mark.offline

API_KEY = "the-operators-secret-api-key"


class _StubRAG:
    def __init__(self):
        self.ollama_server_infos = SimpleNamespace(
            LIGHTRAG_NAME="lightrag",
            LIGHTRAG_TAG="latest",
            LIGHTRAG_MODEL="lightrag:latest",
            LIGHTRAG_SIZE=1,
            LIGHTRAG_CREATED_AT="now",
            LIGHTRAG_DIGEST="sha",
        )


def _client(monkeypatch, *, accounts: bool, api_key: str | None = API_KEY):
    """``/api/*`` from the real Ollama router, ``/documents`` from the default
    dependency, under the whitelist the server defaults to once a key is set."""
    monkeypatch.setattr(_utils_api, "whitelist_patterns", [("/health", False)])
    monkeypatch.setattr(_utils_api, "auth_configured", accounts)

    app = FastAPI()
    app.include_router(
        _ollama_api.OllamaAPI(_StubRAG(), api_key=api_key).router, prefix="/api"
    )
    default_dependency = _utils_api.get_combined_auth_dependency(api_key)

    @app.get("/documents", dependencies=[Depends(default_dependency)])
    async def documents():
        return {"statuses": {}}

    return TestClient(app)


def _bearer(value: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {value}"}


@pytest.mark.parametrize("accounts", [False, True])
def test_bearer_api_key_authenticates_the_ollama_routes(monkeypatch, accounts):
    client = _client(monkeypatch, accounts=accounts)

    assert client.get("/api/version", headers=_bearer(API_KEY)).status_code == 200
    assert client.get("/api/tags", headers=_bearer(API_KEY)).status_code == 200


@pytest.mark.parametrize("accounts", [False, True])
def test_x_api_key_still_authenticates_the_ollama_routes(monkeypatch, accounts):
    client = _client(monkeypatch, accounts=accounts)

    response = client.get("/api/version", headers={"X-API-Key": API_KEY})
    assert response.status_code == 200


def test_ollama_routes_require_a_credential_once_a_key_is_set(monkeypatch):
    """No credential: refused, and the message names the form a client can send."""
    client = _client(monkeypatch, accounts=False)

    response = client.get("/api/version")
    assert response.status_code == 403
    assert "Authorization: Bearer <key>" in response.json()["detail"]


@pytest.mark.parametrize("accounts", [False, True])
def test_a_wrong_bearer_key_is_refused(monkeypatch, accounts):
    client = _client(monkeypatch, accounts=accounts)

    response = client.get("/api/version", headers=_bearer("not-the-key"))
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key or token."


def test_a_logged_in_user_token_still_works_on_the_ollama_routes(monkeypatch):
    client = _client(monkeypatch, accounts=True)
    token = _auth.auth_handler.create_token("alice", role="user")

    assert client.get("/api/version", headers=_bearer(token)).status_code == 200


def test_bearer_api_key_is_an_ollama_only_allowance(monkeypatch):
    """The rest of the API keeps ``X-API-Key`` as the one way to send the key."""
    client = _client(monkeypatch, accounts=False)

    assert client.get("/documents", headers=_bearer(API_KEY)).status_code == 401
    assert client.get("/documents", headers={"X-API-Key": API_KEY}).status_code == 200


def test_an_empty_bearer_never_matches_an_unset_key(monkeypatch):
    """Without a configured key there is nothing for a bearer value to match:
    under AUTH_ACCOUNTS it is validated as a JWT and refused."""
    client = _client(monkeypatch, accounts=True, api_key=None)

    assert client.get("/api/version", headers=_bearer("anything")).status_code == 401


@pytest.mark.parametrize(
    "presented,expected",
    [
        (API_KEY, True),
        (API_KEY + "x", False),
        ("", False),
        (None, False),
        ("ключ", False),  # non-ASCII must compare, not raise
    ],
)
def test_api_key_matches(presented, expected):
    assert _utils_api.api_key_matches(presented, API_KEY) is expected
    assert _utils_api.api_key_matches(presented, None) is False
    assert _utils_api.api_key_matches(presented, "") is False
