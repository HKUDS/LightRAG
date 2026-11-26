"""``get_default_host``: each binding's default endpoint and its override.

A binding missing from the table falls back to the Ollama default
(http://localhost:11434), so every request it makes fails at connect time.
The ``DEFAULT_*_ENDPOINT`` sentinels let the provider SDK pick its own regional
or versioned endpoint unless ``LLM_BINDING_HOST`` overrides it.
"""

import pytest

from lightrag.api.config import get_default_host

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    ("binding", "default"),
    [
        ("gemini", "DEFAULT_GEMINI_ENDPOINT"),
        ("bedrock", "DEFAULT_BEDROCK_ENDPOINT"),
        ("openai_responses", "https://api.openai.com/v1"),
    ],
)
def test_default_host_and_its_override(monkeypatch, binding, default):
    monkeypatch.delenv("LLM_BINDING_HOST", raising=False)
    assert get_default_host(binding) == default

    monkeypatch.setenv("LLM_BINDING_HOST", "https://gateway.example/v1")
    assert get_default_host(binding) == "https://gateway.example/v1"


@pytest.mark.parametrize(
    ("binding", "expected"),
    [
        ("azure_openai", "https://azure.example"),
        # Embedding-only: an LLM gateway URL must never become its endpoint.
        ("voyageai", "DEFAULT_VOYAGEAI_ENDPOINT"),
    ],
)
def test_default_ignores_llm_host(monkeypatch, binding, expected):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://azure.example")
    monkeypatch.setenv("LLM_BINDING_HOST", "https://gateway.example/v1")

    assert get_default_host(binding) == expected


@pytest.mark.parametrize(
    ("binding", "default", "api_key", "role_host"),
    [
        ("gemini", "DEFAULT_GEMINI_ENDPOINT", "gemini-test-key", None),
        ("bedrock", "DEFAULT_BEDROCK_ENDPOINT", None, None),
        (
            "gemini",
            "https://role-gateway.example",
            "gemini-test-key",
            "https://role-gateway.example",
        ),
        ("openai", None, None, None),
    ],
)
def test_role_host_respects_provider_and_explicit_override(
    monkeypatch, lightrag_log_records, binding, default, api_key, role_host
):
    import sys

    from lightrag.api.config import parse_args
    from lightrag.llm_roles import ROLES

    monkeypatch.setattr(sys, "argv", ["lightrag-server"])
    for spec in ROLES:
        for suffix in (
            "LLM_BINDING",
            "LLM_MODEL",
            "LLM_BINDING_HOST",
            "LLM_BINDING_API_KEY",
        ):
            monkeypatch.delenv(f"{spec.env_prefix}_{suffix}", raising=False)
    monkeypatch.setenv("VLM_PROCESS_ENABLE", "false")
    monkeypatch.setenv("LLM_BINDING", "openai")
    monkeypatch.setenv("LLM_BINDING_HOST", "https://base-gateway.example/v1")
    monkeypatch.setenv("QUERY_LLM_BINDING", binding)
    monkeypatch.setenv("QUERY_LLM_MODEL", "role-specific-model")
    if api_key is not None:
        monkeypatch.setenv("QUERY_LLM_BINDING_API_KEY", api_key)

    if role_host is not None:
        monkeypatch.setenv("QUERY_LLM_BINDING_HOST", role_host)

    args = parse_args()

    assert args.llm_binding_host == "https://base-gateway.example/v1"
    assert args.query_llm_binding_host == default
    # Only a defaulted cross-provider host warns that LLM_BINDING_HOST is ignored.
    warned = [
        r
        for r in lightrag_log_records
        if r.levelname == "WARNING" and "QUERY_LLM_BINDING_HOST" in r.getMessage()
    ]
    assert len(warned) == (1 if role_host is None and binding != "openai" else 0)
