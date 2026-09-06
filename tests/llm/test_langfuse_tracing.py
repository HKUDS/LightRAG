"""Tests for the shared Langfuse runtime tracing switch."""

import asyncio
import multiprocessing
import sys
from types import ModuleType

import pytest

from lightrag.kg.shared_storage import (
    finalize_share_data,
    get_namespace_data,
    initialize_share_data,
)
from lightrag.llm import langfuse_tracing as tracing


@pytest.fixture(autouse=True)
def reset_tracing_state(monkeypatch):
    monkeypatch.setattr(tracing, "_langfuse_tracing_state", None)
    monkeypatch.setattr(tracing, "_langfuse_client", None)
    monkeypatch.delenv("LANGFUSE_ENABLE_TRACE", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)


def test_initialize_passes_live_span_filter_to_langfuse(monkeypatch):
    state = {}
    captured = {}

    async def fake_get_namespace_data(namespace, workspace=None):
        assert namespace == tracing.LANGFUSE_TRACING_NAMESPACE
        assert workspace == ""
        return state

    class FakeLangfuse:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setattr(tracing, "get_namespace_data", fake_get_namespace_data)
    monkeypatch.setattr(tracing, "is_langfuse_installed", lambda: True)
    fake_langfuse = ModuleType("langfuse")
    fake_langfuse.Langfuse = FakeLangfuse
    monkeypatch.setitem(sys.modules, "langfuse", fake_langfuse)

    asyncio.run(tracing.initialize_langfuse_tracing())

    assert state[tracing.LANGFUSE_TRACING_ENABLED_KEY] is True
    assert captured["should_export_span"] is tracing.should_export_langfuse_span
    assert tracing.should_export_langfuse_span(None) is True

    tracing.set_langfuse_tracing_enabled(False)
    assert tracing.should_export_langfuse_span(None) is False


def _read_tracing_state_in_worker(result_queue):
    async def read_state():
        state = await get_namespace_data(
            tracing.LANGFUSE_TRACING_NAMESPACE,
            workspace="",
        )
        tracing._langfuse_tracing_state = state
        result_queue.put(tracing.should_export_langfuse_span(None))

    asyncio.run(read_state())


def test_runtime_toggle_is_visible_in_another_worker():
    if "fork" not in multiprocessing.get_all_start_methods():
        pytest.skip("cross-worker shared-storage test requires fork")

    finalize_share_data()
    initialize_share_data(workers=2)

    try:
        state = asyncio.run(
            get_namespace_data(tracing.LANGFUSE_TRACING_NAMESPACE, workspace="")
        )
        state[tracing.LANGFUSE_TRACING_ENABLED_KEY] = False

        context = multiprocessing.get_context("fork")
        result_queue = context.Queue()
        process = context.Process(
            target=_read_tracing_state_in_worker,
            args=(result_queue,),
        )
        process.start()
        process.join(timeout=10)

        assert process.exitcode == 0
        assert result_queue.get(timeout=2) is False

        state[tracing.LANGFUSE_TRACING_ENABLED_KEY] = True
        second_queue = context.Queue()
        second_process = context.Process(
            target=_read_tracing_state_in_worker,
            args=(second_queue,),
        )
        second_process.start()
        second_process.join(timeout=10)

        assert second_process.exitcode == 0
        assert second_queue.get(timeout=2) is True

        result_queue.close()
        result_queue.join_thread()
        second_queue.close()
        second_queue.join_thread()
    finally:
        finalize_share_data()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("true", True),
        ("1", True),
        ("off", False),
        ("0", False),
    ],
)
def test_operator_default_is_parsed(monkeypatch, value, expected):
    monkeypatch.setenv("LANGFUSE_ENABLE_TRACE", value)
    assert tracing._configured_default() is expected
