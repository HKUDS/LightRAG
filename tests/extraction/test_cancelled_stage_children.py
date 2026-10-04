"""Cancelled stages must drain children before recovery can start."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

import lightrag.operate as operate
from lightrag.utils import Tokenizer, TokenizerInterface

pytestmark = pytest.mark.offline


class _Tokenizer(TokenizerInterface):
    def encode(self, content):
        return list(content.encode())

    def decode(self, tokens):
        return bytes(tokens).decode()


async def test_cancelled_extraction_drains_running_and_queued_chunks(monkeypatch):
    prior_tasks = asyncio.all_tasks()
    monkeypatch.setenv("MAX_EXTRACT_INPUT_TOKENS", "999999")
    entered = asyncio.Event()
    release = asyncio.Event()
    children = []
    writes = []

    async def llm(*args, **kwargs):
        children.append(asyncio.current_task())
        entered.set()
        await release.wait()
        writes.append("late result")
        return "(entity<|#|>A<|#|>CONCEPT<|#|>Description)<|COMPLETE|>"

    config = {
        "llm_model_func": llm,
        "role_llm_funcs": {"extract": llm},
        "entity_extract_max_gleaning": 0,
        "entity_extract_max_records": 100,
        "entity_extract_max_entities": 40,
        "addon_params": {},
        "tokenizer": Tokenizer("dummy", _Tokenizer()),
        "llm_model_max_async": 1,
    }
    chunks = {
        f"c{i}": {
            "tokens": 4,
            "content": "Text",
            "full_doc_id": "doc",
            "chunk_order_index": i,
            "file_path": "document.txt",
        }
        for i in range(3)
    }
    parent = asyncio.create_task(operate.extract_entities(chunks, config))
    await asyncio.wait_for(entered.wait(), timeout=5)
    parent.cancel()
    with pytest.raises(asyncio.CancelledError):
        await parent
    try:
        assert all(task.done() for task in children)
        release.set()
        await asyncio.sleep(0)
        assert writes == []
        assert len(children) == 1, "queued chunks must not start after cancellation"
    finally:
        release.set()
        remaining = asyncio.all_tasks() - prior_tasks
        for task in remaining:
            task.cancel()
        await asyncio.gather(*remaining, return_exceptions=True)


@pytest.mark.parametrize("phase", ["entity", "relation"])
async def test_merge_cancelled_during_task_creation_drains_writers(monkeypatch, phase):
    prior_tasks = asyncio.all_tasks()
    entered = asyncio.Event()
    writer_entered = asyncio.Event()
    release = asyncio.Event()
    children = []
    writes = []

    @asynccontextmanager
    async def no_lock(*args, **kwargs):
        yield

    async def writer(*args, **kwargs):
        children.append(asyncio.current_task())
        writer_entered.set()
        await release.wait()
        writes.append("late graph write")
        return {} if phase == "entity" else None

    async def scheduling_yield(iteration, every=64):
        if iteration == every and every == (16 if phase == "entity" else 32):
            entered.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(operate, "get_storage_keyed_lock", no_lock)
    monkeypatch.setattr(operate, "_cooperative_yield", scheduling_yield)
    monkeypatch.setattr(operate, "_merge_nodes_then_upsert", writer)
    monkeypatch.setattr(operate, "_merge_edges_then_upsert", writer)
    nodes = {f"E{i}": [{}] for i in range(16)} if phase == "entity" else {}
    edges = {(f"S{i}", f"T{i}"): [{}] for i in range(32)} if phase == "relation" else {}
    parent = asyncio.create_task(
        operate.merge_nodes_and_edges(
            [(nodes, edges)],
            AsyncMock(),
            AsyncMock(),
            AsyncMock(),
            {"llm_model_max_async": 1},
            pipeline_status={"history_messages": []},
            pipeline_status_lock=asyncio.Lock(),
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=5)
    await asyncio.wait_for(writer_entered.wait(), timeout=5)
    parent.cancel()
    with pytest.raises(asyncio.CancelledError):
        await parent
    try:
        assert all(task.done() for task in children)
        release.set()
        await asyncio.sleep(0)
        assert writes == []
    finally:
        release.set()
        remaining = asyncio.all_tasks() - prior_tasks
        for task in remaining:
            task.cancel()
        await asyncio.gather(*remaining, return_exceptions=True)
