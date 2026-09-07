"""Chunk-tracking cleanup must never outlive-order the graph object it describes.

`adelete_by_entity` / `adelete_by_relation` remove the graph object first and its
tracking rows afterwards. The reverse order -- which the helpers used before this
file existed -- breaks the purge recovery contract: a tracking row is the
authoritative attribution carrier, so dropping it while a transient vector or
graph failure leaves the object alive degrades that object's provenance to the
truncated graph `source_id`, from which `_purge_kg_contributions` can conclude
"no remaining sources" and delete an entity other documents still reference.

`TestFailureLeavesProvenanceIntact` are the fix proofs: they inject a vector-store
failure at the point where the pre-fix code had already deleted the tracking rows
and assert the rows survive alongside the object. They fail behaviourally on the
pre-fix ordering (the row is gone), not by importing a symbol the fix adds.

`TestOrphanRowsConverge` covers the residue of the chosen order -- a row whose
object is already gone -- and pins that a repeat deletion sweeps it, which is what
makes a partial failure recoverable instead of permanent.

`TestDurableCommitOrdering` covers the second half of the same invariant. On the
deferred backends the calls above are all in-memory and `index_done_callback` is
the only durable commit, so sequencing the *calls* proves nothing there: flushing
every store in one `asyncio.gather` leaves the durable order unconstrained, and a
failed GraphML commit next to a successful tracking commit puts the forbidden
state on disk. The flush is therefore split into two ordered phases.

The doubles are deliberately split by write timing, because the two halves of the
invariant fail on different backends: immediate-write (Redis/PG/Mongo KV, Neo4j/PG
graph) for the call ordering, deferred-commit (NetworkX/JSON) for the flush
ordering. tests/pipeline/test_graph_deletion_tracking.py runs the real deferred
stack and is green under every ordering, which is exactly why it cannot stand in
for either group here.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy

import pytest

from lightrag import utils_graph
from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import make_relation_chunk_key

pytestmark = pytest.mark.offline

ENTITY = "ATLAS"
OTHER = "BOREALIS"
RELATION_KEY = make_relation_chunk_key(*sorted([ENTITY, OTHER]))
CHUNKS = {"chunk_ids": ["chunk-1"], "count": 1}


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


class _Boom(RuntimeError):
    """Injected backend failure."""


class _KVStorage:
    """Immediate-write KV double: `delete` takes effect before the next await.

    Every method yields once before doing its work, the way a real async backend
    suspends on I/O. Without that a pending cancellation is never delivered
    inside this code and the cancellation cases below would prove nothing.
    """

    def __init__(self):
        self.records: dict = {}
        self.flushes = 0
        self.fail_delete_times = 0

    async def get_by_id(self, key):
        return deepcopy(self.records.get(key))

    async def upsert(self, data):
        self.records.update(deepcopy(data))

    async def delete(self, ids):
        await asyncio.sleep(0)
        if self.fail_delete_times > 0:
            self.fail_delete_times -= 1
            raise _Boom("tracking delete failed")
        for key in ids:
            self.records.pop(key, None)

    async def index_done_callback(self):
        await asyncio.sleep(0)
        self.flushes += 1


class _VectorStorage:
    def __init__(self, global_config):
        self.global_config = global_config
        self.fail = False
        self.fail_flush = False
        self.flushes = 0

    def _check(self):
        if self.fail:
            raise _Boom("vector backend unavailable")

    async def delete(self, ids):
        self._check()

    async def delete_entity(self, entity_name):
        self._check()

    async def delete_entity_relation(self, entity_name):
        self._check()

    async def index_done_callback(self):
        await asyncio.sleep(0)
        self.flushes += 1
        if self.fail_flush:
            raise _Boom("vector flush failed")


class _Fixture:
    """A real NetworkXStorage so node/edge survival is observed, not simulated."""

    def __init__(self, tmp_path):
        self.global_config = {
            "working_dir": str(tmp_path),
            "workspace": "",
            "embedding_batch_num": 10,
        }
        self.graph = NetworkXStorage(
            namespace="chunk_entity_relation",
            workspace="",
            global_config=self.global_config,
            embedding_func=None,
        )
        self.entities_vdb = _VectorStorage(self.global_config)
        self.relationships_vdb = _VectorStorage(self.global_config)
        self.entity_chunks = _KVStorage()
        self.relation_chunks = _KVStorage()

    async def start(self):
        await self.graph.initialize()
        for name in (ENTITY, OTHER):
            await self.graph.upsert_node(
                name, {"entity_id": name, "description": "d", "source_id": "chunk-1"}
            )
        await self.graph.upsert_edge(
            ENTITY, OTHER, {"description": "d", "weight": 1.0, "source_id": "chunk-1"}
        )
        await self.entity_chunks.upsert({ENTITY: dict(CHUNKS), OTHER: dict(CHUNKS)})
        await self.relation_chunks.upsert({RELATION_KEY: dict(CHUNKS)})
        # Commit the baseline: the persisted graph is what a restart reads back.
        await self.graph.index_done_callback()
        return self

    def persisted_graph(self):
        return NetworkXStorage.load_nx_graph(self.graph._graphml_xml_file)

    async def delete_entity(self):
        return await utils_graph.adelete_by_entity(
            self.graph,
            self.entities_vdb,
            self.relationships_vdb,
            ENTITY,
            entity_chunks_storage=self.entity_chunks,
            relation_chunks_storage=self.relation_chunks,
        )

    async def delete_relation(self):
        return await utils_graph.adelete_by_relation(
            self.graph,
            self.relationships_vdb,
            ENTITY,
            OTHER,
            relation_chunks_storage=self.relation_chunks,
        )


@pytest.fixture
async def rag(tmp_path):
    fixture = await _Fixture(tmp_path).start()
    yield fixture
    await fixture.graph.finalize()


class TestFailureLeavesProvenanceIntact:
    """Fix proofs: a live object must never be left without its tracking row."""

    @pytest.mark.asyncio
    async def test_entity_vector_failure_keeps_all_tracking_rows(self, rag):
        rag.entities_vdb.fail = True

        result = await rag.delete_entity()

        assert result.status == "fail"
        # The object survived the failure, so its provenance must survive too.
        assert await rag.graph.has_node(ENTITY)
        assert rag.entity_chunks.records[ENTITY] == CHUNKS
        assert rag.relation_chunks.records[RELATION_KEY] == CHUNKS

    @pytest.mark.asyncio
    async def test_relation_vector_failure_keeps_tracking_row(self, rag):
        rag.relationships_vdb.fail = True

        result = await rag.delete_relation()

        assert result.status == "fail"
        assert await rag.graph.has_edge(ENTITY, OTHER)
        assert rag.relation_chunks.records[RELATION_KEY] == CHUNKS


class TestOrphanRowsConverge:
    """The residue of this order -- an orphan row -- must be recoverable.

    These use the immediate-write doubles, so the failing step is `delete()`
    itself. The deferred counterpart -- a `delete()` that succeeded in memory
    whose *commit* failed -- is a different failure with a different recovery,
    and lives in `TestFailedCommitsAreRetried`.
    """

    @pytest.mark.asyncio
    async def test_entity_tracking_failure_converges_on_retry(self, rag):
        # Graph node and relation row go first; the entity row delete blows up.
        rag.entity_chunks.fail_delete_times = 1

        first = await rag.delete_entity()

        assert first.status == "fail"
        assert not await rag.graph.has_node(ENTITY)
        assert RELATION_KEY not in rag.relation_chunks.records
        orphan_left_behind = ENTITY in rag.entity_chunks.records
        assert orphan_left_behind

        second = await rag.delete_entity()

        assert second.status == "not_found"
        assert ENTITY not in rag.entity_chunks.records
        # Sweeping the orphan must not touch an unrelated entity's provenance.
        assert rag.entity_chunks.records[OTHER] == CHUNKS

    @pytest.mark.asyncio
    async def test_relation_tracking_failure_converges_on_retry(self, rag):
        rag.relation_chunks.fail_delete_times = 1

        first = await rag.delete_relation()

        assert first.status == "fail"
        assert not await rag.graph.has_edge(ENTITY, OTHER)
        assert RELATION_KEY in rag.relation_chunks.records

        second = await rag.delete_relation()

        assert second.status == "not_found"
        assert RELATION_KEY not in rag.relation_chunks.records

    @pytest.mark.asyncio
    async def test_not_found_for_an_unknown_name_touches_nothing(self, rag):
        result = await utils_graph.adelete_by_entity(
            rag.graph,
            rag.entities_vdb,
            rag.relationships_vdb,
            "NEVER_EXISTED",
            entity_chunks_storage=rag.entity_chunks,
            relation_chunks_storage=rag.relation_chunks,
        )

        assert result.status == "not_found"
        assert set(rag.entity_chunks.records) == {ENTITY, OTHER}
        assert set(rag.relation_chunks.records) == {RELATION_KEY}

    @pytest.mark.asyncio
    async def test_orphan_sweep_handles_a_legacy_shaped_row(self, rag):
        # A partial/legacy row is still stored attribution for a gone object.
        await rag.entity_chunks.upsert({"GHOST": {"count": 0}})

        result = await utils_graph.adelete_by_entity(
            rag.graph,
            rag.entities_vdb,
            rag.relationships_vdb,
            "GHOST",
            entity_chunks_storage=rag.entity_chunks,
            relation_chunks_storage=rag.relation_chunks,
        )

        assert result.status == "not_found"
        assert "GHOST" not in rag.entity_chunks.records


class _DeferredKVStorage:
    """Deferred-commit KV double: `delete` is in-memory, `index_done_callback` commits."""

    def __init__(self, name: str, commit_log: list[str]):
        self.name = name
        self.commit_log = commit_log
        self.records: dict = {}
        self.disk: dict = {}
        self.fail_commit_times = 0

    async def get_by_id(self, key):
        return deepcopy(self.records.get(key))

    async def upsert(self, data):
        self.records.update(deepcopy(data))

    async def delete(self, ids):
        await asyncio.sleep(0)
        for key in ids:
            self.records.pop(key, None)

    async def index_done_callback(self):
        await asyncio.sleep(0)
        self.commit_log.append(self.name)
        if self.fail_commit_times > 0:
            self.fail_commit_times -= 1
            raise _Boom(f"{self.name} commit failed")
        self.disk = deepcopy(self.records)


@pytest.fixture
async def deferred(tmp_path):
    """Real NetworkX graph plus deferred-commit tracking doubles."""
    fixture = _Fixture(tmp_path)
    fixture.commit_log: list[str] = []
    fixture.entity_chunks = _DeferredKVStorage("entity_chunks", fixture.commit_log)
    fixture.relation_chunks = _DeferredKVStorage("relation_chunks", fixture.commit_log)
    await fixture.start()
    # Seed the committed baseline: this is what a restart would read back.
    await fixture.entity_chunks.index_done_callback()
    await fixture.relation_chunks.index_done_callback()
    fixture.commit_log.clear()
    yield fixture
    await fixture.graph.finalize()


def _log_graph_commit(fixture, monkeypatch, *, fail: bool):
    original = fixture.graph.index_done_callback

    async def _commit():
        getattr(fixture, "commit_log", []).append("graph")
        if fail:
            raise _Boom("graph commit failed")
        return await original()

    monkeypatch.setattr(fixture.graph, "index_done_callback", _commit)


class TestDurableCommitOrdering:
    """The graph must reach disk before the tracking rows do."""

    @pytest.mark.asyncio
    async def test_entity_graph_commit_failure_keeps_rows_on_disk(
        self, deferred, monkeypatch
    ):
        _log_graph_commit(deferred, monkeypatch, fail=True)

        result = await deferred.delete_entity()

        assert result.status == "fail"
        # The graph never committed, so on restart the entity is still live --
        # its tracking rows must still be on disk with it.
        assert deferred.entity_chunks.disk[ENTITY] == CHUNKS
        assert deferred.relation_chunks.disk[RELATION_KEY] == CHUNKS
        assert deferred.commit_log == ["graph"]

    @pytest.mark.asyncio
    async def test_relation_graph_commit_failure_keeps_row_on_disk(
        self, deferred, monkeypatch
    ):
        _log_graph_commit(deferred, monkeypatch, fail=True)

        result = await deferred.delete_relation()

        assert result.status == "fail"
        assert deferred.relation_chunks.disk[RELATION_KEY] == CHUNKS
        assert deferred.commit_log == ["graph"]

    @pytest.mark.asyncio
    async def test_successful_entity_delete_commits_graph_first(
        self, deferred, monkeypatch
    ):
        # Stability, not a fix proof: a single gather also happens to start the
        # graph commit first, so only the two failure cases above go red on the
        # unordered flush. This one pins the happy-path order against a future
        # reshuffle of the phases.
        _log_graph_commit(deferred, monkeypatch, fail=False)

        result = await deferred.delete_entity()

        assert result.status == "success"
        assert deferred.commit_log[0] == "graph"
        assert set(deferred.commit_log[1:]) == {"entity_chunks", "relation_chunks"}
        assert ENTITY not in deferred.entity_chunks.disk
        assert RELATION_KEY not in deferred.relation_chunks.disk


class TestMixedBackendDurability:
    """Neither the calls nor the flushes can be ordered in isolation.

    Storage families differ in *when* a mutation becomes durable, so these cases
    mix them the way a real deployment can: a deferred graph (NetworkX) with an
    immediate-write tracking store (Redis/PG), and a deferred graph whose commit
    succeeds while a vector flush fails.
    """

    @pytest.mark.asyncio
    async def test_immediate_kv_rows_survive_a_failed_graph_commit(
        self, rag, monkeypatch
    ):
        # `rag` deliberately pairs the real deferred graph with immediate-write
        # KV doubles: the tracking delete is durable the moment it is called, so
        # it must not happen until the graph commit has succeeded.
        rag.commit_log = []
        _log_graph_commit(rag, monkeypatch, fail=True)

        result = await rag.delete_entity()

        assert result.status == "fail"
        # The GraphML commit failed, so a restart reloads a live entity -- with
        # its authoritative provenance still next to it. In-memory state is not
        # the question here; only what survived to disk is.
        assert rag.persisted_graph().has_node(ENTITY)
        assert rag.entity_chunks.records[ENTITY] == CHUNKS
        assert rag.relation_chunks.records[RELATION_KEY] == CHUNKS

    @pytest.mark.asyncio
    async def test_immediate_kv_relation_row_survives_a_failed_graph_commit(
        self, rag, monkeypatch
    ):
        rag.commit_log = []
        _log_graph_commit(rag, monkeypatch, fail=True)

        result = await rag.delete_relation()

        assert result.status == "fail"
        assert rag.persisted_graph().has_edge(ENTITY, OTHER)
        assert rag.relation_chunks.records[RELATION_KEY] == CHUNKS

    @pytest.mark.asyncio
    async def test_vector_flush_failure_still_clears_tracking(self, deferred):
        # Only the vector flush fails. Bundling it with the graph commit in one
        # gather makes the durable outcome unspecified: the exception propagates
        # while the graph's own commit is still a pending background task, so
        # whether the node's removal ever lands is a matter of scheduling -- and
        # if it does, the tracking callbacks have already been skipped, the rows
        # come back on restart, and a reinsert inherits the old chunk ids (the
        # incident relation rows are not even reachable by the not_found sweep).
        # Committing the graph in its own phase makes both halves definite.
        deferred.entities_vdb.fail_flush = True

        result = await deferred.delete_entity()

        assert result.status == "fail"
        assert not deferred.persisted_graph().has_node(ENTITY)
        assert ENTITY not in deferred.entity_chunks.disk
        assert RELATION_KEY not in deferred.relation_chunks.disk
        # The unrelated entity keeps its provenance through the failure.
        assert deferred.entity_chunks.disk[OTHER] == CHUNKS

    @pytest.mark.asyncio
    async def test_relation_vector_flush_failure_still_clears_tracking(self, deferred):
        deferred.relationships_vdb.fail_flush = True

        result = await deferred.delete_relation()

        assert result.status == "fail"
        assert not deferred.persisted_graph().has_edge(ENTITY, OTHER)
        assert RELATION_KEY not in deferred.relation_chunks.disk


class TestFailedCommitsAreRetried:
    """A retry must be able to commit what an earlier attempt left pending.

    On a deferred backend a failed `index_done_callback` leaves the delete in
    memory and the stale row on disk. Keying the retry's flush off in-memory row
    presence would make that permanent: the row is already invisible in memory,
    so a presence check sees nothing to do and skips the commit that is owed.
    """

    @pytest.mark.asyncio
    async def test_entity_retry_commits_a_failed_tracking_flush(self, deferred):
        deferred.entity_chunks.fail_commit_times = 1

        first = await deferred.delete_entity()

        assert first.status == "fail"
        # In-memory the row is gone; on disk -- what a restart reads -- it is not.
        assert await deferred.entity_chunks.get_by_id(ENTITY) is None
        assert ENTITY in deferred.entity_chunks.disk

        second = await deferred.delete_entity()

        assert second.status == "not_found"
        assert ENTITY not in deferred.entity_chunks.disk
        assert deferred.entity_chunks.disk[OTHER] == CHUNKS

    @pytest.mark.asyncio
    async def test_entity_retry_commits_pending_relation_rows(self, deferred):
        # The incident relation rows are deleted in the same phase; a retry that
        # only ever considers the entity's own row must still flush them.
        deferred.relation_chunks.fail_commit_times = 1

        first = await deferred.delete_entity()

        assert first.status == "fail"
        assert RELATION_KEY in deferred.relation_chunks.disk

        second = await deferred.delete_entity()

        assert second.status == "not_found"
        assert RELATION_KEY not in deferred.relation_chunks.disk

    @pytest.mark.asyncio
    async def test_relation_retry_commits_a_failed_tracking_flush(self, deferred):
        deferred.relation_chunks.fail_commit_times = 1

        first = await deferred.delete_relation()

        assert first.status == "fail"
        assert await deferred.relation_chunks.get_by_id(RELATION_KEY) is None
        assert RELATION_KEY in deferred.relation_chunks.disk

        second = await deferred.delete_relation()

        assert second.status == "not_found"
        assert RELATION_KEY not in deferred.relation_chunks.disk


class TestDeclinedGraphCommitIsAFailure:
    """A graph backend can decline to write and say so by return value.

    `NetworkXStorage.index_done_callback` returns `False` -- without raising --
    when another process committed since this one last read the graph: it
    reloads from disk and discards the in-memory mutation. Reading a normal
    return as proof of a commit would let the deletion proceed to drop the
    tracking rows of a node that is still live, and report success.
    """

    @staticmethod
    def _decline_graph_commit(fixture, monkeypatch):
        async def _declined():
            return False

        monkeypatch.setattr(fixture.graph, "index_done_callback", _declined)

    @pytest.mark.asyncio
    async def test_entity_delete_fails_when_the_graph_declines_to_commit(
        self, rag, monkeypatch
    ):
        self._decline_graph_commit(rag, monkeypatch)

        result = await rag.delete_entity()

        assert result.status == "fail"
        assert "discarded" in result.message
        assert rag.entity_chunks.records[ENTITY] == CHUNKS
        assert rag.relation_chunks.records[RELATION_KEY] == CHUNKS

    @pytest.mark.asyncio
    async def test_relation_delete_fails_when_the_graph_declines_to_commit(
        self, rag, monkeypatch
    ):
        self._decline_graph_commit(rag, monkeypatch)

        result = await rag.delete_relation()

        assert result.status == "fail"
        assert rag.relation_chunks.records[RELATION_KEY] == CHUNKS

    @pytest.mark.asyncio
    async def test_a_backend_returning_none_is_not_treated_as_a_refusal(
        self, rag, monkeypatch
    ):
        # The base signature is `-> None`; only an explicit False means refusal.
        async def _committed_quietly():
            return None

        monkeypatch.setattr(rag.graph, "index_done_callback", _committed_quietly)

        result = await rag.delete_entity()

        assert result.status == "success"
        assert ENTITY not in rag.entity_chunks.records
        assert RELATION_KEY not in rag.relation_chunks.records


class TestFailedGraphSaveDoesNotStrandProvenance:
    """A raised graph save must not turn the retry into a provenance wipe.

    `NetworkXStorage.index_done_callback` used to re-raise without restoring
    `self._graph`, so the node stayed removed in memory while the file still had
    it -- and nothing repaired that (a failed write never sets
    `storage_updated`, so the reload branch never fires). The retry then read
    `has_node` as False, took the not_found branch, and swept the authoritative
    tracking row of a node that is still on disk.
    """

    @pytest.mark.asyncio
    async def test_entity_delete_retry_converges_after_a_failed_save(
        self, rag, monkeypatch
    ):
        original = NetworkXStorage.write_nx_graph
        armed = {"boom": True}

        def _write(graph, file_name, workspace):
            if armed["boom"]:
                armed["boom"] = False
                raise OSError("No space left on device")
            return original(graph, file_name, workspace)

        monkeypatch.setattr(NetworkXStorage, "write_nx_graph", staticmethod(_write))

        first = await rag.delete_entity()

        assert first.status == "fail"
        assert rag.persisted_graph().has_node(ENTITY)
        assert rag.entity_chunks.records[ENTITY] == CHUNKS
        assert rag.relation_chunks.records[RELATION_KEY] == CHUNKS

        second = await rag.delete_entity()

        # The retry deletes for real instead of mistaking a stale in-memory
        # view for a durable removal.
        assert second.status == "success"
        assert not rag.persisted_graph().has_node(ENTITY)
        assert ENTITY not in rag.entity_chunks.records
        assert RELATION_KEY not in rag.relation_chunks.records
        assert rag.entity_chunks.records[OTHER] == CHUNKS

    @pytest.mark.asyncio
    async def test_relation_delete_retry_converges_after_a_failed_save(
        self, rag, monkeypatch
    ):
        original = NetworkXStorage.write_nx_graph
        armed = {"boom": True}

        def _write(graph, file_name, workspace):
            if armed["boom"]:
                armed["boom"] = False
                raise OSError("No space left on device")
            return original(graph, file_name, workspace)

        monkeypatch.setattr(NetworkXStorage, "write_nx_graph", staticmethod(_write))

        first = await rag.delete_relation()

        assert first.status == "fail"
        assert rag.persisted_graph().has_edge(ENTITY, OTHER)
        assert rag.relation_chunks.records[RELATION_KEY] == CHUNKS

        second = await rag.delete_relation()

        assert second.status == "success"
        assert not rag.persisted_graph().has_edge(ENTITY, OTHER)
        assert RELATION_KEY not in rag.relation_chunks.records


class TestCancellationAfterTheCommit:
    """A cancel landing past the graph commit must not skip the cleanup.

    `commit_in_storage_io` deliberately finishes the GraphML write and its commit
    hook before re-raising `CancelledError`, and `CancelledError` is a
    `BaseException`, so the helpers' `except Exception` never sees it. Returning
    at that point leaves the object durably gone with its tracking rows intact --
    and for an entity the incident relation rows are then unreachable by the
    not_found sweep, so recreating that relation inherits the pre-deletion chunk
    ids with no audit line anywhere.
    """

    @staticmethod
    def _cancel_right_after_commit(fixture, monkeypatch, owner):
        original = fixture.graph.index_done_callback

        async def _commit_then_cancel():
            result = await original()
            # The CALLER's task, never `current_task()` and never a bare raise:
            # the owed cleanup runs in a task of its own, so cancelling from the
            # inside models a worker aborting its own work rather than a caller
            # being cancelled -- a different scenario, handled differently (see
            # TestDirectCancellationBeforeTheCommit).
            owner["task"].cancel()
            return result

        monkeypatch.setattr(fixture.graph, "index_done_callback", _commit_then_cancel)

    @staticmethod
    async def _run_cancelled(coro, owner):
        owner["task"] = asyncio.ensure_future(coro)
        with pytest.raises(asyncio.CancelledError):
            await owner["task"]

    @pytest.mark.asyncio
    async def test_entity_tracking_is_cleaned_despite_the_cancel(
        self, rag, monkeypatch
    ):
        owner: dict = {}
        self._cancel_right_after_commit(rag, monkeypatch, owner)

        await self._run_cancelled(rag.delete_entity(), owner)

        # The node is durably gone, so every row it owned must be gone too.
        assert not rag.persisted_graph().has_node(ENTITY)
        assert ENTITY not in rag.entity_chunks.records
        assert RELATION_KEY not in rag.relation_chunks.records
        assert rag.entity_chunks.records[OTHER] == CHUNKS

    @pytest.mark.asyncio
    async def test_relation_tracking_is_cleaned_despite_the_cancel(
        self, rag, monkeypatch
    ):
        owner: dict = {}
        self._cancel_right_after_commit(rag, monkeypatch, owner)

        await self._run_cancelled(rag.delete_relation(), owner)

        assert not rag.persisted_graph().has_edge(ENTITY, OTHER)
        assert RELATION_KEY not in rag.relation_chunks.records


class TestCancellationDuringTheGraphCommit:
    """The real backend delivers the cancel from INSIDE the commit await.

    `commit_in_storage_io` defers a cancellation through the GraphML write and
    the `set_all_update_flags` hook and then re-raises it from that same await
    (`_bounded_submit_impl`). So on the real NetworkX path the caller never
    reaches the statement after the commit -- deferring only *after* the commit
    returned normally protects a window the production backend does not use.

    `TestCancellationAfterTheCommit` monkeypatches `index_done_callback` to
    cancel and RETURN, which delivers the cancel at the following await; that is
    a genuine case (an immediate-write graph backend commits inside its own
    calls) but it is not this one. These cases patch `write_nx_graph`, which
    `index_done_callback` resolves at call time precisely so it can be replaced,
    so the file is really written and the cancel is really raised out of the
    commit.

    The `delete_node` case covers the third delivery point: a cancel before the
    commit leaves the removal sitting in the in-memory graph with the backend
    marked dirty, so the pipeline's next commit publishes it while this cleanup
    never ran at all. All three windows need the one region.
    """

    @staticmethod
    def _cancel_owner_inside_the_graph_write(monkeypatch, owner):
        original = NetworkXStorage.write_nx_graph

        def _write_then_cancel(graph, file_name, workspace="_"):
            original(graph, file_name, workspace)
            # Runs on the storage-io worker thread, so the cancel has to be
            # posted back to the loop that owns the waiting task.
            owner["loop"].call_soon_threadsafe(owner["task"].cancel)

        monkeypatch.setattr(
            NetworkXStorage, "write_nx_graph", staticmethod(_write_then_cancel)
        )

    @staticmethod
    def _cancel_owner_after(fixture, monkeypatch, method_name, owner):
        original = getattr(fixture.graph, method_name)

        async def _work_then_cancel(*args, **kwargs):
            result = await original(*args, **kwargs)
            owner["task"].cancel()
            return result

        monkeypatch.setattr(fixture.graph, method_name, _work_then_cancel)

    @staticmethod
    async def _run_cancelled(coro, owner):
        owner["loop"] = asyncio.get_running_loop()
        owner["task"] = asyncio.ensure_future(coro)
        with pytest.raises(asyncio.CancelledError):
            await owner["task"]

    @pytest.mark.asyncio
    async def test_entity_tracking_survives_a_cancel_raised_by_the_commit(
        self, rag, monkeypatch
    ):
        owner: dict = {}
        self._cancel_owner_inside_the_graph_write(monkeypatch, owner)

        await self._run_cancelled(rag.delete_entity(), owner)

        # The GraphML write landed, so the node is durably gone and every row it
        # owned must be gone with it -- the incident relation row above all, as
        # the not_found sweep can no longer reach it.
        assert not rag.persisted_graph().has_node(ENTITY)
        assert ENTITY not in rag.entity_chunks.records
        assert RELATION_KEY not in rag.relation_chunks.records
        assert rag.entity_chunks.records[OTHER] == CHUNKS

    @pytest.mark.asyncio
    async def test_relation_tracking_survives_a_cancel_raised_by_the_commit(
        self, rag, monkeypatch
    ):
        owner: dict = {}
        self._cancel_owner_inside_the_graph_write(monkeypatch, owner)

        await self._run_cancelled(rag.delete_relation(), owner)

        assert not rag.persisted_graph().has_edge(ENTITY, OTHER)
        assert RELATION_KEY not in rag.relation_chunks.records

    @pytest.mark.asyncio
    async def test_a_cancel_before_the_commit_leaves_no_unpublished_deletion(
        self, rag, monkeypatch
    ):
        owner: dict = {}
        self._cancel_owner_after(rag, monkeypatch, "delete_node", owner)

        await self._run_cancelled(rag.delete_entity(), owner)

        # Either the removal is durable and its rows are gone, or nothing
        # happened; what must not exist is a removal pending in memory whose
        # cleanup was skipped, because the next pipeline commit publishes it.
        assert not rag.persisted_graph().has_node(ENTITY)
        assert not await rag.graph.has_node(ENTITY)
        assert ENTITY not in rag.entity_chunks.records
        assert RELATION_KEY not in rag.relation_chunks.records


class TestDirectCancellationBeforeTheCommit:
    """A cancel with nothing durable yet must not delete the tracking rows.

    The graph mutation, its commit and the tracking cleanup run as a task of
    their own so the CALLER's cancellation cannot cut them apart. That task can
    still be cancelled directly -- the event loop cancels every remaining task at
    shutdown -- and such a cancel is indistinguishable, from the exception alone,
    between two opposite situations: the write was already in flight (durable, so
    the cleanup is owed) and the write was never submitted, because
    `_bounded_submit_impl` leaves the permit wait cancellable precisely so that a
    cancelled caller leaves no work behind.

    Treating both as "the graph committed" is the dangerous direction. With an
    immediate-write tracking store the rows die durably while the node removal is
    only in memory, so the process leaves behind a live on-disk object with no
    provenance -- the state the purge recovery contract forbids, and the one from
    which a later purge concludes "no remaining sources". Giving up the cleanup in
    the durable case instead leaves the residue this staging already documents.
    """

    @staticmethod
    def _cancel_the_region_before_it_commits(fixture, monkeypatch):
        async def _cancel_without_committing():
            # Inside the region, `current_task()` IS the region's own task. The
            # original callback is never called, standing in for a cancellation
            # delivered while waiting for a storage-IO permit: nothing submitted,
            # nothing durable.
            asyncio.current_task().cancel()
            await asyncio.sleep(0)

        monkeypatch.setattr(
            fixture.graph, "index_done_callback", _cancel_without_committing
        )

    @pytest.mark.asyncio
    async def test_entity_rows_survive_a_cancel_with_nothing_committed(
        self, rag, monkeypatch
    ):
        self._cancel_the_region_before_it_commits(rag, monkeypatch)

        with pytest.raises(asyncio.CancelledError):
            await rag.delete_entity()

        # The node never reached disk, so every row describing it must still be
        # there: the deletion is simply retryable.
        assert rag.persisted_graph().has_node(ENTITY)
        assert rag.entity_chunks.records[ENTITY] == CHUNKS
        assert rag.relation_chunks.records[RELATION_KEY] == CHUNKS

    @pytest.mark.asyncio
    async def test_relation_row_survives_a_cancel_with_nothing_committed(
        self, rag, monkeypatch
    ):
        self._cancel_the_region_before_it_commits(rag, monkeypatch)

        with pytest.raises(asyncio.CancelledError):
            await rag.delete_relation()

        assert rag.persisted_graph().has_edge(ENTITY, OTHER)
        assert rag.relation_chunks.records[RELATION_KEY] == CHUNKS
