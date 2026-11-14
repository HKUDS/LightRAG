"""Public WebUI capability metadata is additive; API authentication is unchanged."""

import pytest

from tests.api.test_auth_verify import _client
from tests.api.test_workspace_entry_mount import _ENV_VARS_TO_ISOLATE

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _isolate_discovery_env(monkeypatch):
    import lightrag.api.config as config

    for var in _ENV_VARS_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)
    for var in ("AUTH_ACCOUNTS", "LIGHTRAG_API_KEY", "TOKEN_SECRET"):
        monkeypatch.setenv(var, "")
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("EMBEDDING_BINDING", "openai")
    # Restore the previously initialized singleton after each test, rather
    # than invalidating the configuration used by other auth suites.
    monkeypatch.setattr(config, "_global_args", None)
    monkeypatch.setattr(config, "_initialized", False)


@pytest.mark.parametrize("accounts", [False, True])
@pytest.mark.parametrize("key_source", ["none", "cli", "env", "both"])
def test_discovery_preserves_fields_and_uses_effective_key(
    tmp_path, monkeypatch, accounts, key_source
):
    cli = ("--key", "cli-secret") if key_source in ("cli", "both") else ()
    if key_source in ("env", "both"):
        monkeypatch.setenv("LIGHTRAG_API_KEY", "env-secret")
    client = _client(tmp_path, monkeypatch, *cli)
    import lightrag.api.utils_api as utils_api

    monkeypatch.setattr(
        utils_api.auth_handler, "accounts", {"alice": "password"} if accounts else {}
    )
    monkeypatch.setattr(utils_api, "auth_configured", accounts)

    response = client.get("/auth-status")
    assert response.status_code == 200
    data = response.json()
    assert data["auth_configured"] is accounts
    assert data["api_key_configured"] is (key_source != "none")
    assert data["auth_mode"] == ("enabled" if accounts else "disabled")
    common = {
        "auth_configured",
        "auth_mode",
        "core_version",
        "api_version",
        "webui_title",
        "webui_description",
        "ai_content_notice_enabled",
    }
    expected = common | {"api_key_configured"}
    if not accounts:
        expected |= {"access_token", "token_type", "message"}
        assert data["token_type"] == "bearer"
        assert (
            utils_api.auth_handler.validate_token(data["access_token"])["role"]
            == "guest"
        )
    assert set(data) == expected
    assert "cli-secret" not in response.text
    assert "env-secret" not in response.text

    if key_source != "none":
        effective_key = "env-secret" if key_source in ("env", "both") else "cli-secret"
        assert (
            client.get("/auth/verify", headers={"X-API-Key": effective_key}).status_code
            == 200
        )


def test_key_only_login_still_issues_guest_token(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, "--key", "cli-secret")
    response = client.post("/login", data={"username": "guest", "password": "unused"})
    assert response.status_code == 200
    data = response.json()
    assert data["auth_mode"] == "disabled"
    assert data["access_token"]
    assert "api_key_configured" not in data
    # A guest token authenticates nothing in API-key-only mode. The status code
    # is whatever the route dependency already returned before this transition
    # (403 "API Key required"), not a new token-first 401 -- see #4125,
    # "do not change ... status-code policy".
    assert (
        client.get(
            "/auth/verify", headers={"Authorization": f"Bearer {data['access_token']}"}
        ).status_code
        == 403
    )


def test_empty_environment_key_uses_cli_key(tmp_path, monkeypatch):
    monkeypatch.setenv("LIGHTRAG_API_KEY", "")
    client = _client(tmp_path, monkeypatch, "--key", "cli-secret")
    assert client.get("/auth-status").json()["api_key_configured"] is True
    assert (
        client.get("/auth/verify", headers={"X-API-Key": "cli-secret"}).status_code
        == 200
    )
