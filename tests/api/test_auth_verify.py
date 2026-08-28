"""Tests for GET /auth/verify — the explicit credential-validity probe.

/health deliberately stays on the default whitelist as an unauthenticated
liveness probe, so it can never distinguish valid, invalid and missing
credentials. The WebUI's API-key dialogs need an endpoint that ALWAYS runs
the combined auth dependency; /auth/verify is that endpoint (and its path is
outside the default whitelist "/health,/api/*"). Pins, for the API-key-only
profile (LIGHTRAG_API_KEY set, no AUTH_ACCOUNTS):

- no credentials, or only a guest token → 403 "API Key required";
- a wrong X-API-Key → 403 "Invalid API Key";
- the correct X-API-Key → 200 {"status": "ok"};

and for the fully open profile: 200 with no credentials — while /health keeps
answering 200 "healthy" to unauthenticated callers in BOTH profiles (the very
property that makes it unusable as a credential probe).
"""

import pytest
from fastapi.testclient import TestClient

from tests.api.test_workspace_entry_mount import (
    _ENV_VARS_TO_ISOLATE,
    _build_app,
    _stage_build,
)

API_KEY = "test-secret-key"


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


def _client(tmp_path, monkeypatch, *cli_args):
    _stage_build(tmp_path)
    app = _build_app(tmp_path, monkeypatch, *cli_args)
    return TestClient(app)


def _guest_token(client):
    status = client.get("/auth-status").json()
    assert status["auth_configured"] is False
    return status["access_token"]


class TestApiKeyOnlyProfile:
    def test_no_credentials_is_403_api_key_required(self, tmp_path, monkeypatch):
        client = _client(tmp_path, monkeypatch, "--key", API_KEY)
        response = client.get("/auth/verify")
        assert response.status_code == 403
        assert response.json()["detail"] == "API Key required"

    def test_guest_token_alone_is_403_api_key_required(self, tmp_path, monkeypatch):
        # The workspace entry always carries a guest token; in API-key-only
        # mode it authenticates nothing (GHSA-f4vv-55c2-5789) and the probe
        # must surface exactly the message the API-key dialog keys on.
        client = _client(tmp_path, monkeypatch, "--key", API_KEY)
        token = _guest_token(client)
        response = client.get(
            "/auth/verify", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 403
        assert response.json()["detail"] == "API Key required"

    def test_wrong_key_is_403_invalid_api_key(self, tmp_path, monkeypatch):
        client = _client(tmp_path, monkeypatch, "--key", API_KEY)
        response = client.get("/auth/verify", headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 403
        assert response.json()["detail"] == "Invalid API Key"

    def test_correct_key_is_200(self, tmp_path, monkeypatch):
        client = _client(tmp_path, monkeypatch, "--key", API_KEY)
        response = client.get("/auth/verify", headers={"X-API-Key": API_KEY})
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_whitelist_covering_auth_paths_does_not_neuter_the_verifier(
        self, tmp_path, monkeypatch
    ):
        # A deployment may whitelist "/auth/*" (e.g. for /auth-status). The
        # verifier's whole purpose is credential validity, so it must ignore
        # WHITELIST_PATHS — otherwise it answers 200 while protected routes
        # still reject, and the workspace never opens its API-key dialog.
        # whitelist_patterns is compiled once at utils_api import, so the
        # test injects the compiled form directly.
        import lightrag.api.utils_api as utils_api

        client = _client(tmp_path, monkeypatch, "--key", API_KEY)
        monkeypatch.setattr(
            utils_api,
            "whitelist_patterns",
            [("/health", False), ("/api", True), ("/auth", True)],
        )
        response = client.get("/auth/verify")
        assert response.status_code == 403
        assert response.json()["detail"] == "API Key required"
        # The correct key still passes, proving the dependency runs normally.
        ok = client.get("/auth/verify", headers={"X-API-Key": API_KEY})
        assert ok.status_code == 200

    def test_health_stays_an_unauthenticated_liveness_probe(
        self, tmp_path, monkeypatch
    ):
        # The contrast that motivated /auth/verify: /health answers healthy
        # without credentials, so it cannot drive the API-key dialog.
        client = _client(tmp_path, monkeypatch, "--key", API_KEY)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestFullyOpenProfile:
    def test_no_credentials_is_200(self, tmp_path, monkeypatch):
        client = _client(tmp_path, monkeypatch)
        response = client.get("/auth/verify")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
