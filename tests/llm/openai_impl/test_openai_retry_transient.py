"""Regression tests for retrying transient OpenAI failures.

Covers:
  * HTTP 5xx (InternalServerError) is retried on both complete and embed.
  * Transient "could not parse JSON body" 400s are converted to a retryable
    TransientBadRequestError, while genuine 400s fail fast.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import BadRequestError, InternalServerError

from lightrag.llm.openai import (
    TransientBadRequestError,
    openai_complete_if_cache,
    openai_embed,
)


def _retry_exception_types(func) -> set[type]:
    """Collect the exception types a tenacity-decorated func retries on."""
    types: set[type] = set()

    def _walk(retry_obj):
        # retry_any / retry_all expose `.retries`; retry_if_exception_type
        # exposes `.exception_types` (a single type or a tuple of types).
        for child in getattr(retry_obj, "retries", ()):
            _walk(child)
        exc_types = getattr(retry_obj, "exception_types", ())
        if isinstance(exc_types, type):
            exc_types = (exc_types,)
        types.update(exc_types)

    # openai_embed is wrapped by @wrap_embedding_func_with_attrs; the
    # tenacity-decorated callable is on `.func`.
    target = getattr(func, "func", func)
    _walk(target.retry.retry)
    return types


def _make_bad_request(message: str) -> BadRequestError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    response = httpx.Response(status_code=400, request=request)
    return BadRequestError(message, response=response, body=None)


def _make_error_client(error: Exception) -> SimpleNamespace:
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=AsyncMock(side_effect=error))
        ),
        close=AsyncMock(),
    )


@pytest.mark.offline
def test_complete_retries_5xx_and_transient_400():
    retried = _retry_exception_types(openai_complete_if_cache)
    assert InternalServerError in retried
    assert TransientBadRequestError in retried


@pytest.mark.offline
def test_embed_retries_5xx():
    assert InternalServerError in _retry_exception_types(openai_embed)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_transient_json_parse_400_is_wrapped():
    """A 'could not parse JSON body' 400 becomes a retryable wrapper."""
    err = _make_bad_request(
        "Error code: 400 - We could not parse the JSON body of your request."
    )
    fake_client = _make_error_client(err)
    # Call the undecorated coroutine to exercise the handler exactly once
    # (bypasses the tenacity retry loop and its waits).
    with patch(
        "lightrag.llm.openai.create_openai_async_client", return_value=fake_client
    ):
        with pytest.raises(TransientBadRequestError):
            await openai_complete_if_cache.__wrapped__(
                model="gpt-4o-mini", prompt="hello"
            )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_genuine_400_fails_fast():
    """A non-parse 400 (e.g. bad params) is not wrapped and propagates."""
    err = _make_bad_request("Error code: 400 - Invalid value for 'temperature'.")
    fake_client = _make_error_client(err)
    with patch(
        "lightrag.llm.openai.create_openai_async_client", return_value=fake_client
    ):
        with pytest.raises(BadRequestError):
            await openai_complete_if_cache.__wrapped__(
                model="gpt-4o-mini", prompt="hello"
            )
