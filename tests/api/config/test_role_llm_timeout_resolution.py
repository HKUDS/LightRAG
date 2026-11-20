"""How the API server resolves each role's LLM timeout from the environment.

``{ROLE}_LLM_TIMEOUT`` overrides ``LLM_TIMEOUT``; an explicit 0 means "no
timeout" and must not fall through to the base value.
"""

import sys

import pytest

from lightrag.api import config

pytestmark = pytest.mark.offline


class _Constructed(Exception):
    """Stops startup once the stubbed LightRAG constructor captured its kwargs."""


@pytest.fixture
def role_configs_for(monkeypatch, tmp_path):
    def build(env: dict[str, str]):
        monkeypatch.setattr(config, "_global_args", None)
        monkeypatch.setattr(config, "_initialized", False)
        for key, value in {
            "LLM_BINDING": "openai",
            "EMBEDDING_BINDING": "openai",
            "RERANK_BINDING": "null",
            "UI_TEMPLATES_DIR": "",
            "AUTH_ACCOUNTS": "",
            "LLM_TIMEOUT": "240",
            **env,
        }.items():
            monkeypatch.setenv(key, value)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "lightrag-server",
                "--working-dir",
                str(tmp_path / "rag"),
                "--input-dir",
                str(tmp_path / "inputs"),
            ],
        )
        args = config.initialize_config(force=True)
        from lightrag.api import lightrag_server as server

        captured: list[dict] = []

        def construct(**kwargs):
            captured.append(kwargs)
            raise _Constructed

        monkeypatch.setattr(server, "LightRAG", construct)
        monkeypatch.setattr(server, "check_frontend_build", lambda: (False, False))
        monkeypatch.setattr(server, "check_workspace_frontend_build", lambda: False)
        with pytest.raises(_Constructed):
            server.create_app(args)
        assert len(captured) == 1
        return captured[0]["role_llm_configs"]

    return build


@pytest.mark.parametrize(
    ("query_timeout_env", "expected"),
    [
        pytest.param(None, 240, id="unset-inherits-base"),
        pytest.param("120", 120, id="explicit-override"),
        pytest.param("0", 0, id="zero-means-no-timeout"),
    ],
)
def test_role_timeout_resolution(
    monkeypatch, role_configs_for, query_timeout_env, expected
):
    monkeypatch.delenv("QUERY_LLM_TIMEOUT", raising=False)
    monkeypatch.delenv("EXTRACT_LLM_TIMEOUT", raising=False)
    env = {} if query_timeout_env is None else {"QUERY_LLM_TIMEOUT": query_timeout_env}

    role_configs = role_configs_for(env)

    assert role_configs["query"].timeout == expected
    # Roles without an override keep the base value.
    assert role_configs["extract"].timeout == 240
