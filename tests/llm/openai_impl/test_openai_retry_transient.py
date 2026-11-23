"""Regression tests for retrying transient OpenAI failures.

Covers transient "could not parse JSON body" 400s, which are converted to a
retryable TransientBadRequestError while genuine 400s fail fast. HTTP 5xx
retries are counted at the wire in test_openai_permanent_rate_limit.py.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from openai import BadRequestError

from lightrag.llm.openai import (
    TransientBadRequestError,
    openai_complete_if_cache,
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

    _walk(func.retry.retry)
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
def test_complete_retries_transient_400():
    assert TransientBadRequestError in _retry_exception_types(openai_complete_if_cache)


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
    fake_client.close.assert_awaited()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_genuine_400_fails_fast():
    """A non-parse 400 (e.g. bad params) is not wrapped, propagates, and closes the client."""
    err = _make_bad_request("Error code: 400 - Invalid value for 'temperature'.")
    fake_client = _make_error_client(err)
    with patch(
        "lightrag.llm.openai.create_openai_async_client", return_value=fake_client
    ):
        with pytest.raises(BadRequestError):
            await openai_complete_if_cache.__wrapped__(
                model="gpt-4o-mini", prompt="hello"
            )
    # The non-transient 400 path must still close the underlying httpx client
    # to avoid connection leaks in validation-heavy/misconfigured runs.
    fake_client.close.assert_awaited()
