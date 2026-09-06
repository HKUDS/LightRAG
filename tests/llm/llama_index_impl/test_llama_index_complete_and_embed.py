"""llama_index_complete() and llama_index_embed(): the two documented entry
points must actually work with the calling convention their own docstrings
and the repo's example scripts describe.
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
    """This is the documented calling convention (see the repo's own
    unofficial-sample llama_index demo scripts): llm_instance is passed as a
    keyword argument. Before the fix, kwargs.get() left it in kwargs, so it
    was forwarded a second time into llama_index_complete_if_cache, which
    has no such parameter -- always raising TypeError."""
    llm = _FakeLLM(content="hello back")

    result = await llama_index_complete("hi", llm_instance=llm)

    assert result == "hello back"


@pytest.mark.asyncio
async def test_llama_index_complete_forwards_chat_kwargs():
    llm = _FakeLLM()

    await llama_index_complete("hi", llm_instance=llm, chat_kwargs={"temperature": 0.1})

    assert llm.received_kwargs == {"temperature": 0.1}


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
