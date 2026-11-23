"""The bundled LLM wrappers size their provider client from LightRAG's timeout.

Inside a role call ``apply_llm_timeout`` hands a bundled ``*_complete``
wrapper that role's effective timeout; outside one it falls back to
``default_llm_timeout``. An explicit ``timeout``, or one in a client
configuration the wrapper forwards, always wins.
"""

import importlib
from types import SimpleNamespace

import pytest

from lightrag.llm_roles import _ROLE_LLM_TIMEOUT, apply_llm_timeout

pytestmark = pytest.mark.offline

_OPENAI_CLIENT_CONFIGS = "openai_client_configs"

# (module, wrapper, inner function the wrapper forwards **kwargs to,
#  client-configuration kwarg whose "timeout" is an explicit override)
_WRAPPERS = [
    (
        "lightrag.llm.openai",
        "openai_complete",
        "openai_complete_if_cache",
        _OPENAI_CLIENT_CONFIGS,
    ),
    (
        "lightrag.llm.openai",
        "gpt_4o_complete",
        "openai_complete_if_cache",
        _OPENAI_CLIENT_CONFIGS,
    ),
    (
        "lightrag.llm.openai",
        "gpt_4o_mini_complete",
        "openai_complete_if_cache",
        _OPENAI_CLIENT_CONFIGS,
    ),
    (
        "lightrag.llm.openai",
        "nvidia_openai_complete",
        "openai_complete_if_cache",
        _OPENAI_CLIENT_CONFIGS,
    ),
    (
        "lightrag.llm.openai",
        "azure_openai_complete",
        "azure_openai_complete_if_cache",
        _OPENAI_CLIENT_CONFIGS,
    ),
    (
        "lightrag.llm.openai_responses",
        "openai_responses_complete",
        "openai_responses_complete_if_cache",
        _OPENAI_CLIENT_CONFIGS,
    ),
    (
        "lightrag.llm.anthropic",
        "anthropic_complete",
        "anthropic_complete_if_cache",
        None,
    ),
    (
        "lightrag.llm.anthropic",
        "claude_3_opus_complete",
        "anthropic_complete_if_cache",
        None,
    ),
    (
        "lightrag.llm.anthropic",
        "claude_3_sonnet_complete",
        "anthropic_complete_if_cache",
        None,
    ),
    (
        "lightrag.llm.anthropic",
        "claude_3_haiku_complete",
        "anthropic_complete_if_cache",
        None,
    ),
    ("lightrag.llm.gemini", "gemini_model_complete", "gemini_complete_if_cache", None),
    ("lightrag.llm.ollama", "ollama_model_complete", "_ollama_model_if_cache", None),
    ("lightrag.llm.lollms", "lollms_model_complete", "lollms_model_if_cache", None),
    ("lightrag.llm.zhipu", "zhipu_complete", "zhipu_complete_if_cache", None),
    ("lightrag.llm.bedrock", "bedrock_complete", "bedrock_complete_if_cache", None),
]

_CLIENT_CONFIG_WRAPPERS = [case for case in _WRAPPERS if case[3] is not None]


def _param_id(case) -> str:
    return case[1]


def _hashing_kv(default_llm_timeout=240) -> SimpleNamespace:
    return SimpleNamespace(
        global_config={
            "llm_model_name": "test-model",
            "default_llm_timeout": default_llm_timeout,
        }
    )


async def _call_wrapper(monkeypatch, case, *, role_timeout=None, **call_kwargs):
    module_name, wrapper_name, inner_name, _ = case
    module = importlib.import_module(module_name)
    calls: list[dict] = []

    async def _record(*args, **kwargs):
        calls.append(kwargs)
        return "ok"

    monkeypatch.setattr(module, inner_name, _record)
    token = _ROLE_LLM_TIMEOUT.set(role_timeout) if role_timeout is not None else None
    try:
        await getattr(module, wrapper_name)(
            "prompt", hashing_kv=_hashing_kv(), **call_kwargs
        )
    finally:
        if token is not None:
            _ROLE_LLM_TIMEOUT.reset(token)

    assert len(calls) == 1
    return calls[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _WRAPPERS, ids=_param_id)
async def test_wrapper_uses_role_timeout_inside_role_call(monkeypatch, case):
    forwarded = await _call_wrapper(monkeypatch, case, role_timeout=77)

    assert forwarded["timeout"] == 77


# Every wrapper calls ``apply_llm_timeout(kwargs)`` before touching its
# kwargs, so the role test above is each wrapper's wiring, and the helper's
# other rows (fallback, explicit argument) run on one wrapper.
_ONE_WRAPPER = [_WRAPPERS[0]]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _ONE_WRAPPER, ids=_param_id)
async def test_wrapper_falls_back_to_default_llm_timeout(monkeypatch, case):
    # A client configuration that is not a mapping carries no timeout of its
    # own, so the fallback still applies.
    forwarded = await _call_wrapper(monkeypatch, case, openai_client_configs=None)

    assert forwarded["timeout"] == 240


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _ONE_WRAPPER, ids=_param_id)
async def test_wrapper_keeps_explicit_timeout(monkeypatch, case):
    forwarded = await _call_wrapper(monkeypatch, case, role_timeout=77, timeout=5)

    assert forwarded["timeout"] == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _CLIENT_CONFIG_WRAPPERS, ids=_param_id)
async def test_wrapper_keeps_client_config_timeout(monkeypatch, case):
    client_configs = {"timeout": 400}
    forwarded = await _call_wrapper(
        monkeypatch, case, role_timeout=77, **{case[3]: client_configs}
    )

    # No top-level timeout, so the client-config value is not overridden.
    assert "timeout" not in forwarded
    assert forwarded[case[3]] == {"timeout": 400}


def test_openai_client_config_timeout_survives_without_top_level_timeout():
    from lightrag.llm.openai import create_openai_async_client

    client = create_openai_async_client(
        api_key="test-key", client_configs={"timeout": 400}, timeout=None
    )

    assert client.timeout == 400


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="no-hashing-kv"),
        pytest.param({"hashing_kv": object()}, id="hashing-kv-without-config"),
    ],
)
def test_apply_llm_timeout_outside_role_without_config_sets_nothing(kwargs):
    apply_llm_timeout(kwargs)

    assert "timeout" not in kwargs
