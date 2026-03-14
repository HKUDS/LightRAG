"""Offline tests for role-specific LLM runtime configuration."""

import asyncio
from argparse import Namespace

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.llm.binding_options import OpenAILLMOptions
from lightrag.utils import EmbeddingFunc, Tokenizer


pytestmark = pytest.mark.offline


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    return np.random.rand(len(texts), 16)


async def _base_llm(*args, **kwargs) -> str:
    return "base"


def _make_rag(tmp_path, **kwargs) -> LightRAG:
    return LightRAG(
        working_dir=str(tmp_path / "role-runtime"),
        workspace="role-runtime",
        llm_model_func=_base_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=16,
            max_token_size=4096,
            func=_mock_embedding,
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_role_functions_are_isolated_and_vlm_present(tmp_path):
    rag = _make_rag(tmp_path)

    funcs = [
        rag.llm_model_func,
        rag.extract_llm_model_func,
        rag.keyword_llm_model_func,
        rag.query_llm_model_func,
        rag.vlm_llm_model_func,
    ]
    assert all(callable(func) for func in funcs)
    assert len({id(func) for func in funcs}) == len(funcs)


@pytest.mark.asyncio
async def test_role_specific_kwargs_and_fallback(tmp_path):
    extract_calls = []
    vlm_calls = []

    async def extract_func(*args, **kwargs):
        extract_calls.append(kwargs)
        return "extract"

    async def vlm_func(*args, **kwargs):
        vlm_calls.append(kwargs)
        return "vlm"

    rag = _make_rag(
        tmp_path,
        llm_model_kwargs={"shared": "base"},
        extract_llm_model_func=extract_func,
        extract_llm_model_kwargs={"shared": "extract", "tag": "extract"},
        vlm_llm_model_func=vlm_func,
        vlm_llm_model_kwargs={"shared": "vlm", "tag": "vlm"},
    )

    await rag.extract_llm_model_func("extract prompt")
    await rag.keyword_llm_model_func("keyword prompt")
    await rag.vlm_llm_model_func("vlm prompt")

    assert extract_calls[-1]["tag"] == "extract"
    assert extract_calls[-1]["shared"] == "extract"
    assert "hashing_kv" in extract_calls[-1]

    # Keyword role falls back to base kwargs when no role kwargs are configured.
    # We do not inspect base function internals, but the call must succeed.
    assert vlm_calls[-1]["tag"] == "vlm"
    assert vlm_calls[-1]["shared"] == "vlm"


@pytest.mark.asyncio
async def test_update_llm_role_config_rewraps_without_double_call(tmp_path):
    call_count = 0
    seen_tags = []

    async def query_func(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        seen_tags.append(kwargs.get("tag"))
        return "query"

    rag = _make_rag(
        tmp_path,
        query_llm_model_func=query_func,
        query_llm_model_kwargs={"tag": "v1"},
    )

    await rag.query_llm_model_func("first")
    assert call_count == 1
    assert seen_tags[-1] == "v1"

    for value in (3, 5, 7):
        rag.update_llm_role_config("query", max_async=value)
        await rag.query_llm_model_func("next")

    rag.update_llm_role_config("query", model_kwargs={"tag": "v2"})
    await rag.query_llm_model_func("final")

    assert call_count == 5
    assert seen_tags[-1] == "v2"
    assert rag.query_llm_model_max_async == 7


@pytest.mark.asyncio
async def test_update_llm_role_config_with_builder_metadata(tmp_path):
    built_calls = []

    def builder(role: str, meta: dict):
        async def built_func(*args, **kwargs):
            built_calls.append({"role": role, "meta": dict(meta), "kwargs": dict(kwargs)})
            return f"{meta['model']}"

        return built_func, {
            "runtime_host": meta["host"],
            "provider_options": meta["provider_options"],
        }

    rag = _make_rag(tmp_path)
    rag.register_role_llm_builder(builder)
    rag.set_role_llm_metadata(
        "query",
        binding="openai",
        model="old-model",
        host="https://old-host",
        api_key="old-key",
        provider_options={"temperature": 0.1},
    )

    rag.update_llm_role_config(
        "query",
        binding="gemini",
        model="gemini-2.0-flash",
        host="https://new-host",
        api_key="new-key",
        provider_options={"temperature": 0.3, "top_k": 8},
    )

    result = await rag.query_llm_model_func("hello")
    assert result == "gemini-2.0-flash"
    assert built_calls[-1]["role"] == "query"
    assert built_calls[-1]["meta"]["binding"] == "gemini"
    assert built_calls[-1]["meta"]["model"] == "gemini-2.0-flash"
    assert built_calls[-1]["kwargs"]["runtime_host"] == "https://new-host"
    assert built_calls[-1]["kwargs"]["provider_options"]["top_k"] == 8


@pytest.mark.asyncio
async def test_update_llm_role_config_rolls_back_on_failure(tmp_path):
    rag = _make_rag(tmp_path, extract_llm_model_kwargs={"tag": "before"})
    original_raw = rag._raw_role_llm_funcs["extract"]
    original_wrapped = rag.extract_llm_model_func
    original_kwargs = dict(rag.extract_llm_model_kwargs)

    def failing_builder(role: str, meta: dict):
        raise RuntimeError("boom")

    rag.register_role_llm_builder(failing_builder)
    rag.set_role_llm_metadata(
        "extract",
        binding="openai",
        model="base-model",
        host="https://base",
        api_key="key",
        provider_options={"temperature": 0.1},
    )

    with pytest.raises(RuntimeError, match="boom"):
        rag.update_llm_role_config(
            "extract",
            binding="gemini",
            provider_options={"temperature": 0.9},
        )

    assert rag._raw_role_llm_funcs["extract"] is original_raw
    assert rag.extract_llm_model_func is original_wrapped
    assert rag.extract_llm_model_kwargs == original_kwargs


def test_options_dict_for_role_inherits_same_provider(monkeypatch):
    args = Namespace(
        openai_llm_temperature=0.2,
        openai_llm_top_p=0.8,
        openai_llm_extra_body={"base": True},
    )
    monkeypatch.setenv("EXTRACT_OPENAI_LLM_TEMPERATURE", "0.7")

    options = OpenAILLMOptions.options_dict_for_role(args, "extract")

    assert options["temperature"] == 0.7
    assert options["top_p"] == 0.8
    assert options["extra_body"] == {"base": True}


def test_options_dict_for_role_resets_cross_provider(monkeypatch):
    args = Namespace(
        openai_llm_temperature=0.2,
        openai_llm_top_p=0.8,
        openai_llm_extra_body={"base": True},
    )
    default_options = OpenAILLMOptions().asdict()
    monkeypatch.setenv("QUERY_OPENAI_LLM_TOP_P", "0.6")

    options = OpenAILLMOptions.options_dict_for_role(
        args, "query", is_cross_provider=True
    )

    assert options["temperature"] == default_options["temperature"]
    assert options["top_p"] == 0.6
    assert options["extra_body"] == default_options["extra_body"]


@pytest.mark.asyncio
async def test_vlm_role_supports_runtime_update(tmp_path):
    vlm_calls = []

    async def vlm_func(*args, **kwargs):
        vlm_calls.append(kwargs)
        return "vlm"

    rag = _make_rag(
        tmp_path,
        vlm_llm_model_func=vlm_func,
        vlm_llm_model_kwargs={"tag": "initial"},
    )

    await rag.vlm_llm_model_func("before")
    rag.update_llm_role_config(
        "vlm",
        model_kwargs={"tag": "updated"},
        max_async=2,
        timeout=240,
    )
    await rag.vlm_llm_model_func("after")

    assert vlm_calls[0]["tag"] == "initial"
    assert vlm_calls[1]["tag"] == "updated"
    assert rag.vlm_llm_model_max_async == 2
    assert rag.vlm_llm_timeout == 240
