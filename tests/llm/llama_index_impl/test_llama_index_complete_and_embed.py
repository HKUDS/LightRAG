"""llama_index_complete() and llama_index_embed(): the two entry points
LightRAG wires up as llm_model_func / embedding_func must actually work
with the kwargs the role LLM wrapper forwards into them.
"""

from __future__ import annotations

import numpy as np
import pytest

from lightrag.llm.llama_index_impl import llama_index_complete, llama_index_embed

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


class _FakeEmbedModel:
    def __init__(self, vectors) -> None:
        self._vectors = vectors

    async def _aget_text_embeddings(self, texts):
        return [self._vectors[t] for t in texts]


@pytest.mark.asyncio
async def test_llama_index_embed_uses_async_batch_method():
    model = _FakeEmbedModel({"a": [0.1, 0.2], "b": [0.3, 0.4]})

    result = await llama_index_embed.func(["a", "b"], embed_model=model)

    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([[0.1, 0.2], [0.3, 0.4]]))


@pytest.mark.asyncio
async def test_llama_index_embed_requires_embed_model():
    with pytest.raises(ValueError, match="embed_model must be provided"):
        await llama_index_embed.func(["a"])
