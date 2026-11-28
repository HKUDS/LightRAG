"""Regression coverage for lmdeploy_model_if_cache's retry predicate.

The function is decorated with ``@retry(stop_after_attempt(3), ...)``, but
the predicate used to be ``retry_if_exception_type((RateLimitError,
APIConnectionError, APITimeoutError))`` from ``lightrag.exceptions`` --
exception types this binding never raises. ``lmdeploy``'s own
``Pipeline.generate()`` propagates ``lmdeploy.serve.core.exceptions.
RequestError`` (a ``RuntimeError`` subclass carrying ``status_code``)
uncaught for engine-side failures such as ``ENGINE_UNAVAILABLE`` (503). With
the old predicate, that real transient error never matched, so the retry
loop never fired and the call failed on the very first attempt -- see
lightrag/llm/lmdeploy.py.
"""

from types import SimpleNamespace

import pytest

from lightrag.llm.lmdeploy import lmdeploy_model_if_cache

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


async def test_retries_on_transient_engine_unavailable_error(fake_lmdeploy):
    """A 503 RequestError (ENGINE_UNAVAILABLE) is exactly the transient
    failure the retry decorator exists for, and must be retried rather than
    raised on the first attempt."""
    attempts = {"count": 0}

    async def fake_generate(*_args, **_kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise fake_lmdeploy.RequestError(503)
        yield SimpleNamespace(text="recovered", finish_reason="stop")

    fake_lmdeploy.generate = fake_generate

    result = await lmdeploy_model_if_cache(model="lmdeploy-model", prompt="hello")

    assert result == "recovered"
    assert attempts["count"] == 3


async def test_does_not_retry_permanent_request_error(fake_lmdeploy):
    """A 400 RequestError (e.g. an invalid request) is permanent and must
    fail on the first attempt instead of spending the whole retry budget
    repeating a failure that cannot succeed."""
    attempts = {"count": 0}

    async def fake_generate(*_args, **_kwargs):
        attempts["count"] += 1
        raise fake_lmdeploy.RequestError(400)
        yield  # pragma: no cover - unreachable; keeps this an async generator

    fake_lmdeploy.generate = fake_generate

    with pytest.raises(fake_lmdeploy.RequestError):
        await lmdeploy_model_if_cache(model="lmdeploy-model", prompt="hello")

    assert attempts["count"] == 1


async def test_exhausted_retries_reraise_the_request_error(fake_lmdeploy):
    """A 5xx that persists through every attempt must surface as lmdeploy's
    own RequestError, not as tenacity.RetryError, so the document's
    error_msg keeps lmdeploy's message and callers can still catch it."""
    attempts = {"count": 0}

    async def fake_generate(*_args, **_kwargs):
        attempts["count"] += 1
        raise fake_lmdeploy.RequestError(503)
        yield  # pragma: no cover - unreachable; keeps this an async generator

    fake_lmdeploy.generate = fake_generate

    with pytest.raises(fake_lmdeploy.RequestError, match=r"\(503\)"):
        await lmdeploy_model_if_cache(model="lmdeploy-model", prompt="hello")

    assert attempts["count"] == 3
