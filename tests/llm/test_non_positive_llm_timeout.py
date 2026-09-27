"""Offline tests for a non-positive ``llm_timeout`` passed to the LLM queue.

``env.example`` documents ``LLM_TIMEOUT=0`` as "no timeout", and the Ollama
binding turns that value into a ``None`` HTTP client timeout. The queue
decorator must agree with both: deriving ``0 * 2`` would hand
``asyncio.wait_for`` a zero-second budget and reject every call before the
wrapped function ever started.
"""

import asyncio
import logging

import pytest

from lightrag.utils import priority_limit_async_func_call


pytestmark = pytest.mark.offline


@pytest.fixture
def lightrag_logger_propagating(monkeypatch):
    """Force the lightrag logger to propagate so caplog can capture records."""
    monkeypatch.setattr(logging.getLogger("lightrag"), "propagate", True)


@pytest.mark.parametrize("llm_timeout", [0, 0.0, -1])
@pytest.mark.asyncio
async def test_non_positive_llm_timeout_still_runs_the_call(llm_timeout):
    async def slow_func(value: str, **_kwargs):
        await asyncio.sleep(0.05)
        return value

    wrapped = priority_limit_async_func_call(
        1, llm_timeout=llm_timeout, queue_name="no-timeout LLM func"
    )(slow_func)
    try:
        assert await wrapped("hello") == "hello"
    finally:
        await wrapped.shutdown(graceful=True, timeout=1)


@pytest.mark.asyncio
async def test_unset_llm_timeout_runs_the_call():
    async def quick_func(value: str, **_kwargs):
        return value

    wrapped = priority_limit_async_func_call(1, queue_name="unset-timeout LLM func")(
        quick_func
    )
    try:
        assert await wrapped("hello") == "hello"
    finally:
        await wrapped.shutdown(graceful=True, timeout=1)


@pytest.mark.asyncio
async def test_non_positive_llm_timeout_is_not_logged_as_enforced(
    lightrag_logger_propagating, caplog
):
    async def quick_func(value: str, **_kwargs):
        return value

    wrapped = priority_limit_async_func_call(
        1, llm_timeout=0, queue_name="zero-timeout LLM func"
    )(quick_func)
    try:
        with caplog.at_level("INFO", logger="lightrag"):
            assert await wrapped("hello") == "hello"
    finally:
        await wrapped.shutdown(graceful=True, timeout=1)

    messages = [record.getMessage() for record in caplog.records]
    init_lines = [m for m in messages if "new workers initialized" in m]
    assert init_lines, messages
    # An enforced 0s budget would be advertised here and would fail the call.
    assert "Func: 0s" not in init_lines[0]
    assert "Worker: 0s" not in init_lines[0]


@pytest.mark.asyncio
async def test_positive_llm_timeout_still_bounds_execution():
    async def stuck_func(value: str, **_kwargs):
        await asyncio.sleep(30)
        return value

    wrapped = priority_limit_async_func_call(
        1, llm_timeout=0.05, queue_name="bounded LLM func"
    )(stuck_func)
    try:
        with pytest.raises(asyncio.TimeoutError):
            await wrapped("hello")
    finally:
        await wrapped.shutdown(graceful=False, timeout=1)
