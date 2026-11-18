"""Which bundled LLM wrappers receive the role's effective LLM timeout.

The role wrapper passes ``timeout`` only to functions marked with
``accepts_llm_timeout``. These tests pin that set and that each marked
wrapper hands the value on to the function that builds its provider client.
"""

import importlib
from types import SimpleNamespace

import pytest

from lightrag.llm_roles import LLM_TIMEOUT_FORWARD_ATTR

pytestmark = pytest.mark.offline


# (module, marked wrapper, inner function the wrapper forwards **kwargs to)
_MARKED_WRAPPERS = [
    ("lightrag.llm.openai", "openai_complete", "openai_complete_if_cache"),
    ("lightrag.llm.openai", "gpt_4o_complete", "openai_complete_if_cache"),
    ("lightrag.llm.openai", "gpt_4o_mini_complete", "openai_complete_if_cache"),
    ("lightrag.llm.openai", "nvidia_openai_complete", "openai_complete_if_cache"),
    (
        "lightrag.llm.openai",
        "azure_openai_complete",
        "azure_openai_complete_if_cache",
    ),
    (
        "lightrag.llm.openai_responses",
        "openai_responses_complete",
        "openai_responses_complete_if_cache",
    ),
    ("lightrag.llm.anthropic", "anthropic_complete", "anthropic_complete_if_cache"),
    (
        "lightrag.llm.anthropic",
        "claude_3_opus_complete",
        "anthropic_complete_if_cache",
    ),
    (
        "lightrag.llm.anthropic",
        "claude_3_sonnet_complete",
        "anthropic_complete_if_cache",
    ),
    (
        "lightrag.llm.anthropic",
        "claude_3_haiku_complete",
        "anthropic_complete_if_cache",
    ),
    ("lightrag.llm.gemini", "gemini_model_complete", "gemini_complete_if_cache"),
    ("lightrag.llm.ollama", "ollama_model_complete", "_ollama_model_if_cache"),
    ("lightrag.llm.lollms", "lollms_model_complete", "lollms_model_if_cache"),
    ("lightrag.llm.zhipu", "zhipu_complete", "zhipu_complete_if_cache"),
    ("lightrag.llm.bedrock", "bedrock_complete", "bedrock_complete_if_cache"),
]

# Local inference or a generation config that rejects unknown kwargs: these
# must never be handed a ``timeout`` they did not ask for.
_UNMARKED_WRAPPERS = [
    ("lightrag.llm.hf", "hf_model_complete"),
    ("lightrag.llm.llama_index_impl", "llama_index_complete"),
    ("lightrag.llm.lmdeploy", "lmdeploy_model_if_cache"),
]


def _param_id(case) -> str:
    return case[1]


@pytest.mark.parametrize("case", _MARKED_WRAPPERS, ids=_param_id)
def test_provider_wrapper_is_marked(case):
    module_name, wrapper_name, _ = case
    wrapper = getattr(importlib.import_module(module_name), wrapper_name)

    assert getattr(wrapper, LLM_TIMEOUT_FORWARD_ATTR, False) is True


@pytest.mark.parametrize("case", _UNMARKED_WRAPPERS, ids=_param_id)
def test_local_or_kwargs_strict_wrapper_is_not_marked(case):
    module_name, wrapper_name = case
    module = pytest.importorskip(module_name)
    wrapper = getattr(module, wrapper_name)

    assert getattr(wrapper, LLM_TIMEOUT_FORWARD_ATTR, False) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _MARKED_WRAPPERS, ids=_param_id)
async def test_marked_wrapper_forwards_timeout_to_client_builder(monkeypatch, case):
    module_name, wrapper_name, inner_name = case
    module = importlib.import_module(module_name)
    calls: list[dict] = []

    async def _record(*args, **kwargs):
        calls.append(kwargs)
        return "ok"

    monkeypatch.setattr(module, inner_name, _record)
    hashing_kv = SimpleNamespace(
        global_config={"llm_model_name": "test-model", "default_llm_timeout": 240}
    )

    await getattr(module, wrapper_name)("prompt", hashing_kv=hashing_kv, timeout=77)

    assert len(calls) == 1
    assert calls[0].get("timeout") == 77
