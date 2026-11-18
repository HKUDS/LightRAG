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

Imports only from lightrag.llm.lmdeploy directly, mirroring
test_lmdeploy_do_sample.py, for the same reason given there: avoiding a
module-level import of lightrag.llm.ollama that fails without the real
`ollama` package installed.
"""

import asyncio
import sys
from types import SimpleNamespace

import pytest

from lightrag.llm.lmdeploy import lmdeploy_model_if_cache

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


class FakeRequestError(RuntimeError):
    """Stand-in for lmdeploy.serve.core.exceptions.RequestError."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"fake lmdeploy request error ({status_code})")


async def _instant_sleep(_seconds: float) -> None:
    """Replaces asyncio.sleep so the test does not pay the real backoff wait.

    tenacity's AsyncRetrying imports asyncio locally and calls
    asyncio.sleep(seconds) at call time, so patching the asyncio module
    attribute here is visible to it.
    """
    return None


def _install_fake_lmdeploy(monkeypatch, generate) -> None:
    """Install a fake `lmdeploy` module (>= 0.6.0) plus its nested
    `lmdeploy.serve.core.exceptions` module, mirroring the mock setup
    proven in test_lmdeploy_do_sample.py::_install_fake_lmdeploy.

    `lmdeploy.serve.core.exceptions` is faked as its own sys.modules entries
    (not an attribute on the fake top-level module) because
    `_is_retryable_lmdeploy_error` imports it with
    `from lmdeploy.serve.core.exceptions import RequestError`, and Python's
    import machinery resolves each dotted component from sys.modules before
    falling back to a real filesystem import.
    """

    class FakeGenerationConfig:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(
        "lightrag.llm.lmdeploy.initialize_lmdeploy_pipeline",
        lambda **_kwargs: SimpleNamespace(generate=generate),
    )
    monkeypatch.setitem(
        sys.modules,
        "lmdeploy",
        SimpleNamespace(
            __version__="0.6.1",
            version_info=(0, 6, 1),
            GenerationConfig=FakeGenerationConfig,
        ),
    )
    monkeypatch.setitem(sys.modules, "lmdeploy.serve", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "lmdeploy.serve.core", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "lmdeploy.serve.core.exceptions",
        SimpleNamespace(RequestError=FakeRequestError),
    )
    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)


async def test_retries_on_transient_engine_unavailable_error(monkeypatch):
    """A 503 RequestError (ENGINE_UNAVAILABLE) is exactly the transient
    failure the retry decorator exists for, and must be retried rather than
    raised on the first attempt."""
    attempts = {"count": 0}

    async def fake_generate(*_args, **_kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise FakeRequestError(503)
        yield SimpleNamespace(text="recovered", finish_reason="stop")

    _install_fake_lmdeploy(monkeypatch, fake_generate)

    result = await lmdeploy_model_if_cache(model="lmdeploy-model", prompt="hello")

    assert result == "recovered"
    assert attempts["count"] == 3


async def test_does_not_retry_permanent_request_error(monkeypatch):
    """A 400 RequestError (e.g. an invalid request) is permanent and must
    fail on the first attempt instead of spending the whole retry budget
    repeating a failure that cannot succeed."""
    attempts = {"count": 0}

    async def fake_generate(*_args, **_kwargs):
        attempts["count"] += 1
        raise FakeRequestError(400)
        yield  # pragma: no cover - unreachable; keeps this an async generator

    _install_fake_lmdeploy(monkeypatch, fake_generate)

    with pytest.raises(FakeRequestError):
        await lmdeploy_model_if_cache(model="lmdeploy-model", prompt="hello")

    assert attempts["count"] == 1


async def test_exhausted_retries_reraise_the_request_error(monkeypatch):
    """A 5xx that persists through every attempt must surface as lmdeploy's
    own RequestError, not as tenacity.RetryError, so the document's
    error_msg keeps lmdeploy's message and callers can still catch it."""
    attempts = {"count": 0}

    async def fake_generate(*_args, **_kwargs):
        attempts["count"] += 1
        raise FakeRequestError(503)
        yield  # pragma: no cover - unreachable; keeps this an async generator

    _install_fake_lmdeploy(monkeypatch, fake_generate)

    with pytest.raises(FakeRequestError, match=r"\(503\)"):
        await lmdeploy_model_if_cache(model="lmdeploy-model", prompt="hello")

    assert attempts["count"] == 3
