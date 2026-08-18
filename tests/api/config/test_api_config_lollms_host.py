"""_create_llm_model_kwargs(): lollms needs base_url, not host.

lollms_model_if_cache()'s parameter is named base_url (unlike ollama's
AsyncClient(host=...)); create_llm_model_kwargs() previously returned the
same {"host": ...} shape for both bindings, so a configured LLM_BINDING_HOST
for lollms silently landed in **kwargs and was never read -- the binding
always talked to its own hardcoded http://localhost:9600 default regardless
of configuration.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from lightrag.api.config import parse_args
from lightrag.llm_roles import ROLES

pytestmark = pytest.mark.offline


# Cleared so a developer-local .env cannot decide the outcome. Two parse_args()
# guards SystemExit on ambient values: VLM_PROCESS_ENABLE=true rejects the lollms
# binding these tests select, and any {ROLE}_LLM_BINDING differing from the base
# one demands a role model and api key. The role names come from the registry so
# a newly added role cannot silently reintroduce the leak.
_ENV_VARS_TO_ISOLATE = (
    "LLM_BINDING",
    "LLM_BINDING_HOST",
    "LLM_BINDING_API_KEY",
    "LLM_MODEL",
    "VLM_PROCESS_ENABLE",
    *(
        f"{spec.env_prefix}_{suffix}"
        for spec in ROLES
        for suffix in (
            "LLM_BINDING",
            "LLM_BINDING_HOST",
            "LLM_BINDING_API_KEY",
            "LLM_MODEL",
        )
    ),
)


@pytest.fixture(autouse=True)
def _isolated_config_env(monkeypatch):
    """Pin sys.argv, then import the server module, then clear the environment.

    The order is load-bearing. Importing lightrag.api.lightrag_server triggers
    auth.py's module-level parse_args() against ambient sys.argv, so argv has to
    be pinned first. That import also runs several module-level
    load_dotenv(override=False) calls, which would re-populate anything deleted
    beforehand straight back out of a developer-local .env -- so the deletions
    only stick once every such module is in sys.modules.
    """
    monkeypatch.setattr(sys, "argv", ["lightrag-server"])

    import lightrag.api.lightrag_server  # noqa: F401

    for var in _ENV_VARS_TO_ISOLATE:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def create_llm_model_kwargs(_isolated_config_env):
    from lightrag.api.lightrag_server import _create_llm_model_kwargs

    return _create_llm_model_kwargs


@pytest.mark.parametrize(
    "binding, expected_key, other_key, host",
    [
        ("lollms", "base_url", "host", "http://myserver:9999"),
        ("ollama", "host", "base_url", "http://myserver:11434"),
    ],
)
def test_custom_host_uses_the_binding_specific_key(
    monkeypatch, create_llm_model_kwargs, binding, expected_key, other_key, host
):
    monkeypatch.setenv("LLM_BINDING", binding)
    monkeypatch.setenv("LLM_BINDING_HOST", host)

    kwargs = create_llm_model_kwargs(binding, parse_args(), llm_timeout=30)

    assert kwargs[expected_key] == host
    assert other_key not in kwargs


def test_default_lollms_host_is_unchanged(monkeypatch, create_llm_model_kwargs):
    monkeypatch.setenv("LLM_BINDING", "lollms")
    monkeypatch.delenv("LLM_BINDING_HOST", raising=False)

    kwargs = create_llm_model_kwargs("lollms", parse_args(), llm_timeout=30)

    assert kwargs["base_url"] == "http://localhost:9600"
    assert "host" not in kwargs


def test_timeout_and_api_key_unchanged_for_both_bindings(
    monkeypatch, create_llm_model_kwargs
):
    monkeypatch.setenv("LLM_BINDING", "lollms")
    monkeypatch.setenv("LLM_BINDING_API_KEY", "secret")

    kwargs = create_llm_model_kwargs("lollms", parse_args(), llm_timeout=42)

    assert kwargs["timeout"] == 42
    assert kwargs["api_key"] == "secret"
    # Ollama-only payload, pinned explicitly rather than borrowed from
    # OllamaLLMOptions.options_dict(), which is empty for lollms only as a
    # side effect of where its arguments are registered.
    assert kwargs["options"] == {}


def test_unsupported_binding_returns_empty_dict(monkeypatch, create_llm_model_kwargs):
    monkeypatch.setenv("LLM_BINDING", "openai")

    assert create_llm_model_kwargs("openai", parse_args(), llm_timeout=30) == {}


def test_configured_lollms_host_reaches_the_actual_request(
    monkeypatch, create_llm_model_kwargs
):
    """End-to-end through the real _create_llm_model_kwargs() output, spread
    into lollms_model_complete() the same way llm_roles.py's
    _wrap_llm_role_func does: partial(raw_func, hashing_kv=..., **model_kwargs).
    """
    monkeypatch.setenv("LLM_BINDING", "lollms")
    monkeypatch.setenv("LLM_BINDING_HOST", "http://myserver:9999")

    model_kwargs = create_llm_model_kwargs("lollms", parse_args(), llm_timeout=30)

    from lightrag.llm.lollms import lollms_model_complete

    captured = {}

    class FakeResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def text(self):
            return "ok"

    class FakeSession:
        def __init__(self, timeout=None, headers=None):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def post(self, url, json=None):
            captured["url"] = url
            return FakeResponse()

    class FakeHashingKV:
        global_config = {"llm_model_name": "fake-model"}

    with patch("lightrag.llm.lollms.aiohttp.ClientSession", FakeSession):
        import asyncio

        asyncio.run(
            lollms_model_complete("hello", hashing_kv=FakeHashingKV(), **model_kwargs)
        )

    assert captured["url"] == "http://myserver:9999/lollms_generate"
