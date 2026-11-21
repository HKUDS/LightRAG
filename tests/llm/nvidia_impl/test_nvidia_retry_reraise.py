"""nvidia_openai_embed()'s @retry decorator must set reraise=True.

Without it, tenacity's default behavior applies: once stop_after_attempt(3)
is exhausted, the call raises tenacity.RetryError wrapping the last attempt
instead of the attempt's own exception. Every other provider in
lightrag/llm/ (openai.py, gemini.py, ollama.py, zhipu.py, jina.py, ...) sets
reraise=True precisely to avoid this, so that an upstream
`except RateLimitError` / `isinstance(error, RateLimitError)` check keeps
working and any failure summary carries the provider's own message instead
of tenacity's generic "RetryError[<Future ...>]" text.

Concrete failure scenario before the fix: an NVIDIA NIM embedding endpoint
returns HTTP 429 (rate limited) on all 3 attempts. Call sites that catch
RateLimitError to report a quota/rate-limit specific error (or that rely on
isinstance checks against the openai SDK's exception hierarchy) see a bare
tenacity.RetryError instead, losing the error type and the provider's actual
message.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from openai import RateLimitError
from tenacity import RetryError

from lightrag.llm.nvidia_openai import nvidia_openai_embed


def _make_rate_limit_error():
    response = SimpleNamespace(
        request=SimpleNamespace(method="POST", url="https://nvidia.example/v1"),
        status_code=429,
        headers={},
    )
    return RateLimitError(message="rate limit exceeded", response=response, body=None)


class _FakeAsyncOpenAI:
    def __init__(self, *, create):
        self.embeddings = SimpleNamespace(create=create)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.offline
@pytest.mark.asyncio
async def test_exhausted_retries_raise_original_error_not_retry_error(monkeypatch):
    """After all retry attempts fail, the caller must see the provider's own
    exception (RateLimitError), never tenacity.RetryError.

    Goes through ``.func`` (the @retry-decorated callable, skipping only the
    outer EmbeddingFunc/dimension-checking wrapper) so the retry machinery,
    including its stop/reraise behavior, actually runs. ``asyncio.sleep`` is
    patched out so the exponential backoff between the 3 attempts does not
    slow the test down.
    """
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    error = _make_rate_limit_error()
    fake = _FakeAsyncOpenAI(create=AsyncMock(side_effect=error))

    with patch("lightrag.llm.nvidia_openai.AsyncOpenAI", return_value=fake):
        with pytest.raises(RateLimitError):
            await nvidia_openai_embed.func(texts=["hello"])


@pytest.mark.offline
@pytest.mark.asyncio
async def test_exhausted_retries_do_not_raise_retry_error(monkeypatch):
    """Pinning the negative: the bug surfaced as a bare RetryError."""
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    error = _make_rate_limit_error()
    fake = _FakeAsyncOpenAI(create=AsyncMock(side_effect=error))

    with patch("lightrag.llm.nvidia_openai.AsyncOpenAI", return_value=fake):
        try:
            await nvidia_openai_embed.func(texts=["hello"])
        except RetryError:
            pytest.fail(
                "nvidia_openai_embed leaked a tenacity.RetryError instead of "
                "reraising the original RateLimitError"
            )
        except RateLimitError:
            pass
