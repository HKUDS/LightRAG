"""nvidia_openai_embed(): NVIDIA's embedqa models are asymmetric -- a search
query and an indexed document embed into different regions of the vector
space, distinguished by the API's `input_type` field ("query" / "passage").

Before this fix, `input_type` defaulted to a hardcoded "passage" with no way
for LightRAG's own query/document distinction to reach it: the function had
no `context` parameter, so `wrap_embedding_func_with_attrs`'s auto-detection
left `supports_asymmetric=False`, and `EmbeddingFunc.__call__` never forwards
`context` when that flag is off. Every query embedded through this provider
silently used the wrong (document-tuned) region of the vector space,
degrading retrieval without raising any error.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from lightrag.llm.nvidia_openai import nvidia_openai_embed


class _FakeAsyncOpenAI:
    def __init__(self, *, create):
        self.embeddings = SimpleNamespace(create=create)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _make_response(dim=2048):
    return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1] * dim)])


@pytest.mark.offline
def test_context_makes_this_provider_asymmetric():
    """wrap_embedding_func_with_attrs auto-detects supports_asymmetric from
    the presence of a `context` parameter -- this is what actually gates
    whether EmbeddingFunc.__call__ ever forwards context at all."""
    assert nvidia_openai_embed.supports_asymmetric is True


@pytest.mark.offline
@pytest.mark.asyncio
async def test_query_context_uses_nvidias_query_input_type():
    captured = {}

    async def create(**kwargs):
        captured.update(kwargs)
        return _make_response()

    fake = _FakeAsyncOpenAI(create=create)
    with patch("lightrag.llm.nvidia_openai.AsyncOpenAI", return_value=fake):
        await nvidia_openai_embed(
            ["what is retrieval augmented generation?"], context="query"
        )

    assert captured["extra_body"]["input_type"] == "query"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_document_context_uses_nvidias_passage_input_type():
    captured = {}

    async def create(**kwargs):
        captured.update(kwargs)
        return _make_response()

    fake = _FakeAsyncOpenAI(create=create)
    with patch("lightrag.llm.nvidia_openai.AsyncOpenAI", return_value=fake):
        await nvidia_openai_embed(
            ["some chunk of document content"], context="document"
        )

    assert captured["extra_body"]["input_type"] == "passage"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_no_context_keeps_the_old_passage_default():
    """Backward compatibility: a caller that never passes context (or an
    older LightRAG version that doesn't supply one) must keep behaving
    exactly like before this fix."""
    captured = {}

    async def create(**kwargs):
        captured.update(kwargs)
        return _make_response()

    fake = _FakeAsyncOpenAI(create=create)
    with patch("lightrag.llm.nvidia_openai.AsyncOpenAI", return_value=fake):
        await nvidia_openai_embed(["no context supplied"])

    assert captured["extra_body"]["input_type"] == "passage"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_explicit_input_type_overrides_context():
    """An explicit input_type is a deliberate caller override and must win
    over context-based inference, matching the voyageai_embed precedent."""
    captured = {}

    async def create(**kwargs):
        captured.update(kwargs)
        return _make_response()

    fake = _FakeAsyncOpenAI(create=create)
    with patch("lightrag.llm.nvidia_openai.AsyncOpenAI", return_value=fake):
        await nvidia_openai_embed(["hello"], context="query", input_type="passage")

    assert captured["extra_body"]["input_type"] == "passage"
