"""Phase-1 recovery primitives for issue #3400.

Covers the building blocks later phases depend on:

- ``collect_kg_merge_candidates``: the write-ahead candidate superset a merge
  may touch (entities + relation endpoints + sorted relation pairs).
- ``wait_tasks_with_drain``: sibling-task failure/cancellation must leave no
  background task still writing.
- ``rebuild_knowledge_from_chunks(strict=True)``: missing extraction cache is
  an error, not a silently-logged partial success.
"""

from __future__ import annotations

import asyncio

import pytest

from lightrag.exceptions import KGRebuildCacheMissingError
from lightrag.operate import (
    collect_kg_merge_candidates,
    rebuild_knowledge_from_chunks,
)
from lightrag.utils import wait_tasks_with_drain


# --- collect_kg_merge_candidates -------------------------------------------


def _chunk(nodes: dict, edges: dict):
    return (nodes, edges)


@pytest.mark.offline
def test_candidates_include_entities_and_edge_endpoints():
    """Relation processing can create missing endpoint entities, so every
    endpoint must be in the entity candidate superset even when it was never
    extracted as a standalone entity."""
    chunk_results = [
        _chunk({"ALICE": [{}]}, {("CARL", "ALICE"): [{}]}),
        _chunk({"BOB": [{}]}, {}),
    ]
    entities, relations = collect_kg_merge_candidates(chunk_results)
    assert entities == {"ALICE", "BOB", "CARL"}
    assert relations == {("ALICE", "CARL")}


@pytest.mark.offline
def test_candidates_relation_pairs_are_sorted_and_deduped():
    """Undirected graph: (B, A) and (A, B) are the same candidate pair."""
    chunk_results = [
        _chunk({}, {("B", "A"): [{}]}),
        _chunk({}, {("A", "B"): [{}]}),
    ]
    entities, relations = collect_kg_merge_candidates(chunk_results)
    assert relations == {("A", "B")}
    assert entities == {"A", "B"}


@pytest.mark.offline
def test_candidates_empty_input_yields_empty_sets():
    """Empty candidate sets are meaningful (a doc with no KG contribution
    still needs its — empty — recovery rows written in later phases)."""
    assert collect_kg_merge_candidates([]) == (set(), set())


# --- wait_tasks_with_drain ---------------------------------------------------


@pytest.mark.offline
@pytest.mark.asyncio
async def test_drain_returns_all_results_on_success():
    async def _ok(v):
        return v

    tasks = [asyncio.create_task(_ok(i)) for i in range(5)]
    results = await wait_tasks_with_drain(tasks)
    assert sorted(results) == [0, 1, 2, 3, 4]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_drain_on_failure_cancels_and_drains_all_siblings():
    """After the first failure propagates, NO sibling task may still be
    running — a still-running sibling would keep writing to storage in the
    background (issue #3400, incomplete async failure coordination)."""
    started = asyncio.Event()
    cancelled_flags: list[bool] = []

    async def _fails():
        await started.wait()
        raise RuntimeError("boom")

    async def _slow_writer():
        try:
            await asyncio.sleep(30)
            cancelled_flags.append(False)
        except asyncio.CancelledError:
            cancelled_flags.append(True)
            raise

    tasks = [asyncio.create_task(_slow_writer()) for _ in range(3)]
    tasks.append(asyncio.create_task(_fails()))
    started.set()

    with pytest.raises(RuntimeError, match="boom"):
        await wait_tasks_with_drain(tasks, context="test")

    # Every task has fully finished — none is left running detached.
    assert all(t.done() for t in tasks)
    assert cancelled_flags == [True, True, True]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_drain_raises_first_exception_not_cancellation():
    """The ORIGINAL failure must surface, not the CancelledError of the
    siblings that were cancelled during draining."""

    async def _fails_fast():
        raise ValueError("original failure")

    async def _slow():
        await asyncio.sleep(30)

    tasks = [asyncio.create_task(_slow()), asyncio.create_task(_fails_fast())]
    with pytest.raises(ValueError, match="original failure"):
        await wait_tasks_with_drain(tasks)
    assert all(t.done() for t in tasks)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_drain_external_cancellation_drains_children():
    """Cancelling the waiter itself must also cancel + drain the children
    before CancelledError propagates."""
    entered = asyncio.Event()

    async def _slow():
        entered.set()
        await asyncio.sleep(30)

    children = [asyncio.create_task(_slow()) for _ in range(2)]

    async def _wait():
        await wait_tasks_with_drain(children)

    waiter = asyncio.create_task(_wait())
    await entered.wait()
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert all(t.done() for t in children)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_drain_empty_task_list_is_noop():
    assert await wait_tasks_with_drain([]) == []


# --- rebuild_knowledge_from_chunks strict mode ------------------------------


class _KV:
    def __init__(self, data: dict | None = None):
        self.data = dict(data or {})

    async def get_by_id(self, key):
        return self.data.get(key)

    async def get_by_ids(self, keys):
        return [self.data.get(k) for k in keys]

    async def upsert(self, data):
        self.data.update(data)


def _rebuild_kwargs(text_chunks: _KV, llm_cache: _KV, **overrides):
    kwargs = dict(
        entities_to_rebuild={"ALICE": ["c1"]},
        relationships_to_rebuild={},
        knowledge_graph_inst=None,  # never reached in these tests
        entities_vdb=None,
        relationships_vdb=None,
        text_chunks_storage=text_chunks,
        llm_response_cache=llm_cache,
        global_config={"llm_model_max_async": 1},
    )
    kwargs.update(overrides)
    return kwargs


@pytest.mark.offline
@pytest.mark.asyncio
async def test_strict_rebuild_raises_when_cache_missing():
    """Strict recovery: no cached extraction for a referenced chunk must
    raise, never silently return 'success' (which would let the caller mark
    the document falsely PROCESSED)."""
    text_chunks = _KV({"c1": {"content": "x", "llm_cache_list": []}})
    with pytest.raises(KGRebuildCacheMissingError):
        await rebuild_knowledge_from_chunks(
            **_rebuild_kwargs(text_chunks, _KV(), strict=True)
        )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_strict_rebuild_raises_on_partially_missing_cache():
    """One chunk has cache, one does not: strict mode must still fail —
    rebuilding an aggregate from partial contributions silently degrades it."""
    text_chunks = _KV(
        {
            "c1": {"content": "x", "llm_cache_list": ["k1"]},
            "c2": {"content": "y", "llm_cache_list": []},
        }
    )
    llm_cache = _KV(
        {"k1": {"cache_type": "extract", "chunk_id": "c1", "return": "r", "create_time": 1}}
    )
    with pytest.raises(KGRebuildCacheMissingError):
        await rebuild_knowledge_from_chunks(
            **_rebuild_kwargs(
                text_chunks,
                llm_cache,
                entities_to_rebuild={"ALICE": ["c1", "c2"]},
                strict=True,
            )
        )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_non_strict_rebuild_keeps_best_effort_return():
    """Default (non-strict) behavior is unchanged: missing cache logs a
    warning and returns without raising."""
    text_chunks = _KV({"c1": {"content": "x", "llm_cache_list": []}})
    await rebuild_knowledge_from_chunks(**_rebuild_kwargs(text_chunks, _KV()))
