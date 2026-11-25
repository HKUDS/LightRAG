"""llama_index_complete() and llama_index_embed(): the two entry points
LightRAG wires up as llm_model_func / embedding_func must actually work
with the kwargs the role LLM wrapper forwards into them.
"""

from __future__ import annotations

import asyncio
from functools import partial

import numpy as np
import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.storage.kvstore import SimpleKVStore
from pydantic import Field

from lightrag.exceptions import APIConnectionError
from lightrag.llm import llama_index_impl
from lightrag.llm.llama_index_impl import (
    llama_index_complete,
    llama_index_complete_if_cache,
    llama_index_embed,
)
from lightrag.utils import EmbeddingFunc

pytestmark = pytest.mark.offline


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChatResponse:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeLLM:
    def __init__(self, content: str = "answer") -> None:
        self.content = content
        self.received_kwargs = None

    async def achat(self, messages, **kwargs):
        self.received_kwargs = kwargs
        return _FakeChatResponse(self.content)


@pytest.mark.asyncio
async def test_llama_index_complete_accepts_llm_instance_kwarg():
    """llm_instance as a kwarg is the pattern the repo's own
    unofficial-sample llama_index demo scripts use, though they pass it to
    llama_index_complete_if_cache directly via their own llm_model_func
    wrapper, not through llama_index_complete. Before the fix, kwargs.get()
    left it in kwargs here too, so it was forwarded a second time into
    llama_index_complete_if_cache, which has no such parameter -- always
    raising TypeError."""
    llm = _FakeLLM(content="hello back")

    result = await llama_index_complete("hi", llm_instance=llm)

    assert result == "hello back"


@pytest.mark.asyncio
async def test_llama_index_complete_accepts_role_wrapper_kwargs():
    """The role LLM wrapper (lightrag/llm_roles.py) injects hashing_kv on
    every call unconditionally, and use_llm_func_with_cache can add
    max_tokens and stream. llama_index_complete_if_cache has no **kwargs
    catch-all, so any of these left in kwargs raises TypeError -- this is
    the actual calling convention, not just the isolated llm_instance case."""
    llm = _FakeLLM(content="hello back")

    result = await llama_index_complete(
        "hi",
        llm_instance=llm,
        hashing_kv=object(),
        max_tokens=256,
        stream=False,
    )

    assert result == "hello back"


@pytest.mark.asyncio
async def test_llama_index_complete_forwards_chat_kwargs():
    llm = _FakeLLM()

    await llama_index_complete("hi", llm_instance=llm, chat_kwargs={"temperature": 0.1})

    assert llm.received_kwargs == {"temperature": 0.1}


@pytest.mark.asyncio
async def test_llama_index_complete_folds_max_tokens_into_chat_kwargs():
    """max_tokens is injected by use_llm_func_with_cache when configured.
    llama_index_complete_if_cache has no top-level parameter for it, but
    forwards chat_kwargs straight into achat() -- so it must be folded in
    there instead of silently discarded."""
    llm = _FakeLLM()

    await llama_index_complete("hi", llm_instance=llm, max_tokens=256)

    assert llm.received_kwargs == {"max_tokens": 256}


@pytest.mark.asyncio
async def test_llama_index_complete_max_tokens_does_not_override_explicit_chat_kwargs():
    llm = _FakeLLM()

    await llama_index_complete(
        "hi",
        llm_instance=llm,
        max_tokens=256,
        chat_kwargs={"max_tokens": 64, "temperature": 0.1},
    )

    assert llm.received_kwargs == {"max_tokens": 64, "temperature": 0.1}


@pytest.mark.asyncio
async def test_llama_index_complete_ignores_settings(monkeypatch):
    """settings is a documented no-op on llama_index_complete: the call runs
    on llm_instance and never reads the global Settings.llm, so a Settings
    object must neither be routed through configure_llama_index() nor leak
    into achat()."""
    from llama_index.core.settings import Settings

    configured = []
    monkeypatch.setattr(
        llama_index_impl,
        "configure_llama_index",
        lambda *args, **kwargs: configured.append((args, kwargs)),
    )
    llm = _FakeLLM(content="hello back")

    result = await llama_index_complete("hi", llm_instance=llm, settings=Settings)

    assert result == "hello back"
    assert configured == []
    assert llm.received_kwargs == {}


class _RecordingEmbedModel(BaseEmbedding):
    vectors: dict[str, list[float]]
    batches: list[list[str]] = Field(default_factory=list)

    def _get_query_embedding(self, query: str) -> list[float]:
        raise AssertionError("Query embedding is not used")

    async def _aget_query_embedding(self, query: str) -> list[float]:
        raise AssertionError("Query embedding is not used")

    def _get_text_embedding(self, text: str) -> list[float]:
        raise AssertionError("Synchronous embedding must not be used")

    async def _aget_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        await asyncio.sleep(0)
        return [self.vectors[text] for text in texts]


@pytest.mark.asyncio
@pytest.mark.parametrize("texts", [["a", "b"], []])
async def test_llama_index_embed_uses_async_batch_method(texts: list[str]) -> None:
    model = _RecordingEmbedModel(vectors={"a": [0.1, 0.2], "b": [0.3, 0.4]})

    result = await llama_index_embed.func(texts, embed_model=model)

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(
        result, np.array([model.vectors[text] for text in texts])
    )


@pytest.mark.asyncio
async def test_llama_index_embed_respects_batch_size_and_order() -> None:
    texts = ["c", "a", "b", "d", "e"]
    model = _RecordingEmbedModel(
        vectors={text: [float(i), 0.5] for i, text in enumerate(texts)},
        embed_batch_size=2,
    )

    result = await llama_index_embed.func(texts, embed_model=model)

    assert sorted(model.batches) == sorted([["c", "a"], ["b", "d"], ["e"]])
    np.testing.assert_array_equal(
        result, np.array([model.vectors[text] for text in texts])
    )


@pytest.mark.asyncio
async def test_llama_index_embed_split_batches_pass_embedding_func_contract() -> None:
    """Wired as the docs show, the re-assembled sub-batches still satisfy
    EmbeddingFunc's one-row-per-text shape check, in input order."""
    texts = ["c", "a", "b", "d", "e"]
    model = _RecordingEmbedModel(
        vectors={text: [float(i), 0.5] for i, text in enumerate(texts)},
        embed_batch_size=2,
    )
    embedding_func = EmbeddingFunc(
        embedding_dim=2,
        max_token_size=64,
        func=partial(llama_index_embed.func, embed_model=model),
    )

    result = await embedding_func(texts)

    assert result.shape == (len(texts), 2)
    assert sorted(len(batch) for batch in model.batches) == [1, 2, 2]
    np.testing.assert_array_equal(
        result, np.array([model.vectors[text] for text in texts])
    )


@pytest.mark.asyncio
async def test_llama_index_embed_reuses_cache_and_preserves_order() -> None:
    model = _RecordingEmbedModel(
        vectors={"a": [0.1, 0.2], "b": [0.3, 0.4], "c": [0.5, 0.6]},
        embeddings_cache=SimpleKVStore(),
    )
    await model.aget_text_embedding_batch(["b"])
    model.batches.clear()
    texts = ["c", "b", "a"]

    result = await llama_index_embed.func(texts, embed_model=model)

    assert model.batches == [["c", "a"]]
    np.testing.assert_array_equal(
        result, np.array([model.vectors[text] for text in texts])
    )
    model.batches.clear()
    repeated = await llama_index_embed.func(texts, embed_model=model)
    assert model.batches == []
    np.testing.assert_array_equal(repeated, result)


@pytest.mark.asyncio
async def test_llama_index_embed_requires_embed_model():
    with pytest.raises(ValueError, match="embed_model must be provided"):
        await llama_index_embed.func(["a"])


class _FailingModel:
    """Stands in for both a llama_index LLM and an embed model; every call
    raises the same exception instance."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    async def achat(self, messages, **kwargs):
        self.calls += 1
        raise self.error

    async def aget_text_embedding_batch(self, texts):
        self.calls += 1
        raise self.error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda m: llama_index_complete_if_cache(m, "hi"), id="complete"),
        pytest.param(
            lambda m: llama_index_embed.func(["a"], embed_model=m), id="embed"
        ),
    ],
)
async def test_llama_index_adds_no_retry_layer(call):
    """Retries belong to the llama_index model, which already retries its
    provider's errors. The removed outer @retry matched only LightRAG's own
    exception types, so it never fired on a real provider error and, when it
    did, hid the error behind tenacity.RetryError. A LightRAG
    APIConnectionError is the input it used to retry: it must now surface
    unchanged after exactly one call."""
    error = APIConnectionError(request=None)
    model = _FailingModel(error)

    with pytest.raises(APIConnectionError) as exc_info:
        await call(model)

    assert exc_info.value is error
    assert model.calls == 1
