"""llama_index_complete_if_cache and llama_index_embed's @retry decorators
must set reraise=True. Without it, tenacity wraps the exhausted-retry
failure in tenacity.RetryError, hiding the provider's own exception (and its
message) from callers -- the same gap already fixed in nvidia_openai.py and
voyageai.py.
"""

from unittest.mock import AsyncMock

import pytest

from lightrag.exceptions import APIConnectionError
from lightrag.llm.llama_index_impl import (
    llama_index_complete_if_cache,
    llama_index_embed,
)

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


class _FailingLLM:
    """A llama_index LLM stand-in whose achat() always raises."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.call_count = 0

    async def achat(self, messages, **kwargs):
        self.call_count += 1
        raise self.error


class _FailingEmbedModel:
    """A llama_index embed-model stand-in whose batch embed always raises."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.call_count = 0

    async def aget_text_embedding_batch(self, texts):
        self.call_count += 1
        raise self.error


async def test_llama_index_complete_reraises_original_error_after_retries(
    monkeypatch,
):
    """Exhausted retries must surface the provider's own error, not
    tenacity.RetryError."""
    monkeypatch.setattr(llama_index_complete_if_cache.retry, "sleep", AsyncMock())
    error = APIConnectionError(request=None)
    llm = _FailingLLM(error)

    with pytest.raises(APIConnectionError) as exc_info:
        await llama_index_complete_if_cache(llm, "hi")

    assert exc_info.value is error
    assert llm.call_count == 3


async def test_llama_index_embed_reraises_original_error_after_retries(monkeypatch):
    """Exhausted retries must surface the provider's own error, not
    tenacity.RetryError."""
    monkeypatch.setattr(llama_index_embed.func.retry, "sleep", AsyncMock())
    error = APIConnectionError(request=None)
    model = _FailingEmbedModel(error)

    with pytest.raises(APIConnectionError) as exc_info:
        await llama_index_embed.func(["hello"], embed_model=model)

    assert exc_info.value is error
    assert model.call_count == 3
