"""Regression tests for nvidia_openai_embed AsyncOpenAI cleanup.

Mirrors the cleanup contract established for the Anthropic backend
(``tests/llm/anthropic_impl/test_anthropic_client_cleanup.py``, PR #3261):
the per-call ``AsyncOpenAI`` client must be released on both the success and
error paths so its httpx connection pool does not leak one pool per call.

``nvidia_openai_embed`` now holds the client in an ``async with`` (matching
``openai_embed``); these tests pin that behavior. The raw function is invoked
via ``.func.__wrapped__`` to bypass the ``EmbeddingFunc`` and tenacity
``@retry`` wrappers.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from lightrag.llm.nvidia_openai import nvidia_openai_embed


class _FakeAsyncOpenAI:
    """Mimics ``AsyncOpenAI`` as an async context manager.

    The real SDK's ``__aexit__`` closes the client; this fake does the same so
    the test can assert on ``close()``.
    """

    def __init__(self, *, create):
        self.embeddings = SimpleNamespace(create=create)
        self.close = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()


def _make_response(embedding=(0.1, 0.2, 0.3)):
    return SimpleNamespace(data=[SimpleNamespace(embedding=list(embedding))])


@pytest.mark.offline
@pytest.mark.asyncio
async def test_client_closed_after_success():
    """Successful embed: client.close() runs after the result is returned."""
    fake = _FakeAsyncOpenAI(create=AsyncMock(return_value=_make_response()))

    with patch("lightrag.llm.nvidia_openai.AsyncOpenAI", return_value=fake):
        result = await nvidia_openai_embed.func.__wrapped__(texts=["hello"])

    assert result.tolist() == [[0.1, 0.2, 0.3]]
    fake.embeddings.create.assert_awaited_once()
    fake.close.assert_awaited()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_client_closed_on_error():
    """API error: client.close() runs before the error propagates."""
    fake = _FakeAsyncOpenAI(create=AsyncMock(side_effect=RuntimeError("boom")))

    with patch("lightrag.llm.nvidia_openai.AsyncOpenAI", return_value=fake):
        with pytest.raises(RuntimeError, match="boom"):
            await nvidia_openai_embed.func.__wrapped__(texts=["hello"])

    fake.embeddings.create.assert_awaited_once()
    fake.close.assert_awaited()
