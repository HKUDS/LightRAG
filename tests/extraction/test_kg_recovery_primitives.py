"""Phase-1 recovery primitives for issue #3400.

Covers the building blocks later phases depend on:

- ``collect_kg_merge_candidates``: the write-ahead candidate superset a merge
  may touch (entities + relation endpoints + sorted relation pairs).
- ``wait_tasks_with_drain``: sibling-task failure/cancellation must leave no
  background task still writing.
- ``rebuild_knowledge_from_chunks(rebuild_policy="rollback")``: missing
  extraction cache is reported as a non-fatal degraded recovery condition.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest
import lightrag.operate as operate_module

from lightrag.operate import (
    collect_kg_merge_candidates,
    rebuild_knowledge_from_chunks,
)
from lightrag.utils import wait_tasks_with_drain


@pytest.fixture(autouse=True)
def _storage_keyed_lock_noop(monkeypatch):
    @asynccontextmanager
    async def _noop_lock(*args, **kwargs):
        yield

    monkeypatch.setattr(operate_module, "get_storage_keyed_lock", _noop_lock)


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
async def test_drain_cancellation_between_wait_and_pending_cancel(monkeypatch):
    """Codex review (PR #3416): cancellation delivered AFTER ``asyncio.wait``
    returned but BEFORE the pending siblings were cancelled — e.g. at the
    cooperative yield while collecting done results, with writers still
    pending after FIRST_EXCEPTION — must still cancel and drain those
    writers before ``CancelledError`` propagates."""
    import lightrag.utils as utils_module

    yield_entered = asyncio.Event()

    async def blocking_yield(iteration: int, every: int = 64) -> None:
        # Deterministically park the waiter INSIDE the done-results loop so
        # the test can cancel it exactly in the reported window.
        if iteration > 0 and iteration % every == 0:
            yield_entered.set()
            await asyncio.Event().wait()  # never set; only a cancel wakes it

    monkeypatch.setattr(utils_module, "_cooperative_yield", blocking_yield)

    async def _ok():
        return 1

    async def _fails():
        raise RuntimeError("boom")

    async def _slow_writer():
        await asyncio.sleep(30)

    # 31 completed + 1 failed = 32 done tasks -> the loop hits the yield at
    # i=32 while the slow writer is still pending (FIRST_EXCEPTION).
    completed = [asyncio.create_task(_ok()) for _ in range(31)]
    failing = asyncio.create_task(_fails())
    await asyncio.wait(completed + [failing])
    sleeper = asyncio.create_task(_slow_writer())
    tasks = completed + [failing, sleeper]

    waiter = asyncio.create_task(wait_tasks_with_drain(tasks))
    await yield_entered.wait()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert sleeper.done(), (
        "pending writer must be cancelled and drained when the waiter is "
        "cancelled between wait() and the pending-cancel section"
    )
    assert sleeper.cancelled()


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


class _AbsentGraph:
    async def get_node(self, _name):
        return None


def _rebuild_kwargs(text_chunks: _KV, llm_cache: _KV, **overrides):
    kwargs = dict(
        entities_to_rebuild={"ALICE": ["c1"]},
        relationships_to_rebuild={},
        knowledge_graph_inst=_AbsentGraph(),
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
async def test_rollback_rebuild_reports_when_cache_missing():
    """Missing recovery material is reported without wedging rollback."""
    text_chunks = _KV({"c1": {"content": "x", "llm_cache_list": []}})
    report = await rebuild_knowledge_from_chunks(
        **_rebuild_kwargs(text_chunks, _KV(), rebuild_policy="rollback")
    )
    assert report.missing_cache_chunk_ids == {"c1"}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_rollback_rebuild_reports_partially_missing_cache():
    """Partial cache loss is visible in the report without failing recovery."""
    text_chunks = _KV(
        {
            "c1": {"content": "x", "llm_cache_list": ["k1"]},
            "c2": {"content": "y", "llm_cache_list": []},
        }
    )
    llm_cache = _KV(
        {
            "k1": {
                "cache_type": "extract",
                "chunk_id": "c1",
                "return": "r",
                "create_time": 1,
            }
        }
    )
    report = await rebuild_knowledge_from_chunks(
        **_rebuild_kwargs(
            text_chunks,
            llm_cache,
            entities_to_rebuild={"ALICE": ["c1", "c2"]},
            rebuild_policy="rollback",
        )
    )
    assert report.missing_cache_chunk_ids == {"c2"}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_default_rebuild_keeps_best_effort_return():
    """Default (non-strict) behavior is unchanged: missing cache logs a
    warning and returns without raising."""
    text_chunks = _KV({"c1": {"content": "x", "llm_cache_list": []}})
    await rebuild_knowledge_from_chunks(**_rebuild_kwargs(text_chunks, _KV()))
