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
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.exceptions import CommitBookkeepingError
from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import VectorStorageConsistencyError, make_relation_chunk_key

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

    async def upsert(self, data):
        # Creation paths write through the vector store before the commit; the
        # deletion cases never reach this, so it stays a bare success stub.
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


class TestPublishedCommitIsNotAFailure:
    """The three answers a graph commit can give, and the one middle case.

    ``commit_in_storage_io`` runs its bookkeeping hook only after the write
    succeeded, so a hook failure means the mutation is already on disk. It
    arrives as ``CommitBookkeepingError``, which says exactly that, and this
    deletion must go on to retire the tracking rows a durably removed node owes:
    the entity's own row is reachable by the ``not_found`` sweep on a retry, but
    its incident relation rows are not — the edges that named them are gone with
    the node.

    The two neighbours are the contrast, and both keep today's handling:

    * ``_Boom`` — the write never landed, so the object is still live and its
      provenance must stay with it;
    * ``CancelledError`` — carries no evidence either way, so the cleanup is
      deliberately given up rather than guessed at (see ``adelete_by_entity``).

    Fix-proof: drop the ``except CommitBookkeepingError`` from
    ``_commit_graph_or_raise`` and the middle case answers ``fail`` with the
    relation row left behind for a node that is gone.
    """

    @staticmethod
    def _commit_raises(fixture, monkeypatch, exc, *, land_the_write: bool):
        original = fixture.graph.index_done_callback

        async def _commit():
            if land_the_write:
                await original()
            raise exc

        monkeypatch.setattr(fixture.graph, "index_done_callback", _commit)

    @pytest.mark.asyncio
    async def test_a_published_write_that_failed_to_notify_still_cleans_up(
        self, rag, monkeypatch
    ):
        self._commit_raises(
            rag,
            monkeypatch,
            CommitBookkeepingError(
                "the offloaded write landed, but its commit bookkeeping failed",
                result=True,
            ),
            land_the_write=True,
        )

        result = await rag.delete_entity()

        assert result.status == "success"
        assert not rag.persisted_graph().has_node(ENTITY)
        assert ENTITY not in rag.entity_chunks.records
        assert RELATION_KEY not in rag.relation_chunks.records
        # Only this entity's rows: the sweep must not widen on a publication
        # failure any more than on a clean commit.
        assert rag.entity_chunks.records[OTHER] == CHUNKS

    @pytest.mark.asyncio
    async def test_a_write_that_never_landed_keeps_every_row(self, rag, monkeypatch):
        self._commit_raises(
            rag, monkeypatch, _Boom("graph commit failed"), land_the_write=False
        )

        result = await rag.delete_entity()

        assert result.status == "fail"
        assert rag.persisted_graph().has_node(ENTITY)
        assert rag.entity_chunks.records[ENTITY] == CHUNKS
        assert rag.relation_chunks.records[RELATION_KEY] == CHUNKS

    @pytest.mark.asyncio
    async def test_a_cancelled_commit_keeps_every_row(self, rag, monkeypatch):
        self._commit_raises(
            rag, monkeypatch, asyncio.CancelledError(), land_the_write=False
        )

        with pytest.raises(asyncio.CancelledError):
            await rag.delete_entity()

        # Nothing was submitted, so an immediate-write tracking store must not
        # have its row put in the grave while the node survives on disk.
        assert rag.persisted_graph().has_node(ENTITY)
        assert rag.entity_chunks.records[ENTITY] == CHUNKS
        assert rag.relation_chunks.records[RELATION_KEY] == CHUNKS

    @pytest.mark.asyncio
    async def test_a_published_relation_delete_still_cleans_up(self, rag, monkeypatch):
        self._commit_raises(
            rag,
            monkeypatch,
            CommitBookkeepingError(
                "the offloaded write landed, but its commit bookkeeping failed",
                result=True,
            ),
            land_the_write=True,
        )

        result = await rag.delete_relation()

        assert result.status == "success"
        assert not rag.persisted_graph().has_edge(ENTITY, OTHER)
        assert RELATION_KEY not in rag.relation_chunks.records


class TestPublishedCommitSurvivesABrokenSink:
    """The published-commit handler is past the point of no return.

    ``_commit_graph_or_raise`` answers a ``CommitBookkeepingError`` with a log
    line and continues, because the removal is durable. If that log call can
    raise, the exception lands in ``adelete_by_entity``'s generic handler and
    the deletion answers ``fail``/500 with its tracking rows stranded -- exactly
    the misreport the handler exists to prevent, reached through the diagnostic
    about it. ``_persist_graph_updates`` carries the same handler for the
    tracking flush that follows.

    Fix-proof: call ``logger.error`` directly in either handler and both cases
    below report ``fail``.
    """

    @staticmethod
    def _break_the_sink(monkeypatch):
        def log_boom(msg):
            raise RuntimeError("log sink boom")

        monkeypatch.setattr(utils_graph.logger, "error", log_boom)

    @pytest.mark.asyncio
    async def test_entity_delete_still_succeeds_and_cleans_up(self, rag, monkeypatch):
        original = rag.graph.index_done_callback

        async def _commit():
            await original()
            raise CommitBookkeepingError("published, not notified", result=True)

        monkeypatch.setattr(rag.graph, "index_done_callback", _commit)
        self._break_the_sink(monkeypatch)

        result = await rag.delete_entity()

        assert result.status == "success"
        assert not rag.persisted_graph().has_node(ENTITY)
        assert ENTITY not in rag.entity_chunks.records
        assert RELATION_KEY not in rag.relation_chunks.records

    @pytest.mark.asyncio
    async def test_a_tracking_flush_that_only_failed_to_publish_still_succeeds(
        self, rag, monkeypatch
    ):
        original = rag.entity_chunks.index_done_callback

        async def _flush():
            await original()
            raise CommitBookkeepingError("published, not notified", result=None)

        monkeypatch.setattr(rag.entity_chunks, "index_done_callback", _flush)
        self._break_the_sink(monkeypatch)

        result = await rag.delete_entity()

        assert result.status == "success"
        assert ENTITY not in rag.entity_chunks.records


NEW_ENTITY = "CASSIOPEIA"
NEW_RELATION_KEY = make_relation_chunk_key(*sorted([NEW_ENTITY, OTHER]))


@pytest.fixture
async def creating(tmp_path):
    """Real NetworkX graph plus deferred-commit tracking doubles, for creation.

    Same shape as the ``deferred`` fixture, but the object under test does not
    exist yet: creation is the only direction in which a tracking row goes from
    absent to present, which is what makes its commit order load-bearing.
    """
    fixture = _Fixture(tmp_path)
    fixture.commit_log: list[str] = []
    fixture.entity_chunks = _DeferredKVStorage("entity_chunks", fixture.commit_log)
    fixture.relation_chunks = _DeferredKVStorage("relation_chunks", fixture.commit_log)
    await fixture.start()
    await fixture.entity_chunks.index_done_callback()
    await fixture.relation_chunks.index_done_callback()
    fixture.commit_log.clear()
    yield fixture
    await fixture.graph.finalize()


async def _create_entity(fixture, name=NEW_ENTITY):
    return await utils_graph.acreate_entity(
        fixture.graph,
        fixture.entities_vdb,
        fixture.relationships_vdb,
        name,
        {"description": "d", "entity_type": "thing", "source_id": "chunk-9"},
        entity_chunks_storage=fixture.entity_chunks,
        relation_chunks_storage=fixture.relation_chunks,
    )


async def _create_relation(fixture, source=NEW_ENTITY, target=OTHER):
    return await utils_graph.acreate_relation(
        fixture.graph,
        fixture.entities_vdb,
        fixture.relationships_vdb,
        source,
        target,
        {"description": "d", "weight": 1.0, "source_id": "chunk-9"},
        relation_chunks_storage=fixture.relation_chunks,
    )


class TestCreationCommitsTrackingBeforeTheObject:
    """The mirror of `TestDurableCommitOrdering`, for the creation direction.

    Deletion commits the graph first because that is the order which leaves the
    benign residue (a row whose object is gone). Creation must commit the row
    first for exactly the same reason: the forbidden state is an object durable
    without the row that carries its attribution, and on creation the row is the
    half that starts out absent.

    The two order tests and the two "never starts" tests are fix proofs: they go
    red on the previous single `asyncio.gather`, which started every commit at
    once and so left it to the interpreter whether the node or its row reached
    disk first. No concurrency and no co-tenant flush were needed to lose that
    race -- one uncontended `acreate_entity` and a hard exit was enough.

    `test_accepted_residue_is_the_row_without_the_object` is not a fix proof;
    the unordered flush produced that residue too. It pins the state the chosen
    order deliberately keeps, so a later reshuffle cannot quietly trade it for
    its forbidden mirror.
    """

    @pytest.mark.asyncio
    async def test_entity_creation_commits_the_row_first(self, creating, monkeypatch):
        _log_graph_commit(creating, monkeypatch, fail=False)

        await _create_entity(creating)

        assert creating.commit_log[0] == "entity_chunks"
        assert creating.commit_log.index("entity_chunks") < creating.commit_log.index(
            "graph"
        )
        assert NEW_ENTITY in creating.entity_chunks.disk

    @pytest.mark.asyncio
    async def test_relation_creation_commits_the_row_first(self, creating, monkeypatch):
        _log_graph_commit(creating, monkeypatch, fail=False)
        await creating.graph.upsert_node(
            NEW_ENTITY, {"entity_id": NEW_ENTITY, "description": "d", "source_id": "s"}
        )

        await _create_relation(creating)

        assert creating.commit_log.index("relation_chunks") < creating.commit_log.index(
            "graph"
        )
        assert NEW_RELATION_KEY in creating.relation_chunks.disk

    @pytest.mark.asyncio
    async def test_tracking_commit_failure_never_publishes_the_entity(
        self, creating, monkeypatch
    ):
        # Asserting only "the node is not in GraphML" would pass on the pre-fix
        # gather too, for the wrong reason: the tracking commit raises out of
        # the gather before the graph task finishes its write. What actually
        # separates the two is whether the object commit was ever ENTERED --
        # in a real process it is offloaded, so once entered it lands.
        _log_graph_commit(creating, monkeypatch, fail=False)
        creating.entity_chunks.fail_commit_times = 1

        with pytest.raises(_Boom):
            await _create_entity(creating)

        assert "graph" not in creating.commit_log
        assert NEW_ENTITY not in creating.entity_chunks.disk
        persisted = creating.persisted_graph()
        assert persisted.has_node(NEW_ENTITY) is False
        # Skipping this call's own graph commit is not the guarantee: on a
        # deferred backend the node would sit in the process-wide in-memory
        # graph, and the next flush by ANY co-tenant would publish it. The node
        # must never have entered the graph at all.
        await creating.graph.index_done_callback()
        assert creating.persisted_graph().has_node(NEW_ENTITY) is False

    @pytest.mark.asyncio
    async def test_tracking_commit_failure_never_publishes_the_relation(
        self, creating, monkeypatch
    ):
        _log_graph_commit(creating, monkeypatch, fail=False)
        await creating.graph.upsert_node(
            NEW_ENTITY, {"entity_id": NEW_ENTITY, "description": "d", "source_id": "s"}
        )
        creating.relation_chunks.fail_commit_times = 1

        with pytest.raises(_Boom):
            await _create_relation(creating)

        assert "graph" not in creating.commit_log
        assert NEW_RELATION_KEY not in creating.relation_chunks.disk
        persisted = creating.persisted_graph()
        assert persisted.has_edge(NEW_ENTITY, OTHER) is False
        # As above: a later co-tenant flush must find nothing to publish.
        await creating.graph.index_done_callback()
        assert creating.persisted_graph().has_edge(NEW_ENTITY, OTHER) is False

    @pytest.mark.asyncio
    async def test_accepted_residue_is_the_row_without_the_object(
        self, creating, monkeypatch
    ):
        # The mirror residue the order deliberately keeps: the row is durable
        # while the object never became so. Harmless to queries, not inheritable
        # (R1 resets evidence on explicit creation), repairable offline.
        _log_graph_commit(creating, monkeypatch, fail=True)

        with pytest.raises(_Boom):
            await _create_entity(creating)

        assert NEW_ENTITY in creating.entity_chunks.disk
        persisted = creating.persisted_graph()
        assert persisted.has_node(NEW_ENTITY) is False


class TestDeletionOrderIsUnchanged:
    """The deletion direction must keep committing the graph first.

    `TestDurableCommitOrdering` already pins this, but the two-phase split is
    only correct because every deletion path commits the graph itself and then
    calls `_persist_graph_updates` with the tracking storages alone. This pins
    that call-shape contract directly, so a future caller that hands the helper
    a graph and a tracking store together in the removal direction is caught
    here rather than in production.
    """

    @pytest.mark.asyncio
    async def test_persist_helper_orders_tracking_before_graph(self):
        log: list[str] = []

        class _Store:
            def __init__(self, name):
                self.name = name
                self.namespace = name

            async def index_done_callback(self):
                await asyncio.sleep(0)
                log.append(self.name)

        await utils_graph._persist_graph_updates(
            entities_vdb=_Store("entities_vdb"),
            chunk_entity_relation_graph=_Store("graph"),
            entity_chunks_storage=_Store("entity_chunks"),
            relation_chunks_storage=_Store("relation_chunks"),
        )

        assert set(log[:2]) == {"entity_chunks", "relation_chunks"}
        assert set(log[2:]) == {"entities_vdb", "graph"}

    @pytest.mark.asyncio
    async def test_no_deletion_path_passes_graph_and_tracking_together(self, deferred):
        # `adelete_by_entity` commits the graph through `_commit_graph_or_raise`
        # and only then flushes tracking, so the helper never sees both.
        seen: list[dict] = []
        original = utils_graph._persist_graph_updates

        async def _record(**kwargs):
            seen.append({k: v is not None for k, v in kwargs.items()})
            return await original(**kwargs)

        utils_graph._persist_graph_updates = _record
        try:
            await deferred.delete_entity()
        finally:
            utils_graph._persist_graph_updates = original

        assert seen
        for call in seen:
            graph_passed = call.get("chunk_entity_relation_graph", False)
            tracking_passed = call.get("entity_chunks_storage", False) or call.get(
                "relation_chunks_storage", False
            )
            assert not (graph_passed and tracking_passed)


class _ImmediateGraphStorage:
    """Immediate-write graph double: `upsert_*` is durable before the next await.

    Models Neo4j / PostgreSQL, where the mutation itself is the durable write
    and `index_done_callback` is bookkeeping. On such a backend, ordering the
    *flushes* cannot keep a node out of the store -- only ordering the calls
    can, which is what the creation paths do.
    """

    def __init__(self):
        self.nodes: dict = {}
        self.edges: dict = {}
        self.flushes = 0

    async def has_node(self, node_id):
        await asyncio.sleep(0)
        return node_id in self.nodes

    async def get_node(self, node_id):
        return deepcopy(self.nodes.get(node_id))

    async def has_edge(self, source, target):
        await asyncio.sleep(0)
        return (source, target) in self.edges or (target, source) in self.edges

    async def get_edge(self, source, target):
        return deepcopy(
            self.edges.get((source, target)) or self.edges.get((target, source))
        )

    async def upsert_node(self, node_id, node_data):
        await asyncio.sleep(0)
        self.nodes[node_id] = deepcopy(node_data)

    async def upsert_edge(self, source, target, edge_data):
        await asyncio.sleep(0)
        self.edges[(source, target)] = deepcopy(edge_data)

    async def index_done_callback(self):
        await asyncio.sleep(0)
        self.flushes += 1


class TestCreationWritesTheRowBeforeTheGraphCall:
    """Fix proof: the tracking row must precede the graph MUTATION, not only its flush.

    Splitting the flush into two phases orders nothing on an immediate-write
    graph backend -- `upsert_node` is already durable by the time
    `_persist_graph_updates` is reached, so a tracking store that then fails
    leaves exactly the forbidden state (object durable, no attribution row).
    The same hole exists on a deferred backend for a different reason: the node
    sits in the process-wide in-memory graph, and the next flush by any
    co-tenant publishes it.

    These go red on the pre-fix ordering because the node/edge is in the store
    even though the tracking write never landed.
    """

    @pytest.mark.asyncio
    async def test_entity_is_not_written_when_its_row_cannot_be_stored(self, creating):
        graph = _ImmediateGraphStorage()
        creating.entity_chunks.fail_commit_times = 1

        with pytest.raises(_Boom):
            await utils_graph.acreate_entity(
                graph,
                creating.entities_vdb,
                creating.relationships_vdb,
                NEW_ENTITY,
                {"description": "d", "entity_type": "thing", "source_id": "chunk-9"},
                entity_chunks_storage=creating.entity_chunks,
                relation_chunks_storage=creating.relation_chunks,
            )

        assert NEW_ENTITY not in graph.nodes
        assert NEW_ENTITY not in creating.entity_chunks.disk

    @pytest.mark.asyncio
    async def test_relation_is_not_written_when_its_row_cannot_be_stored(
        self, creating
    ):
        graph = _ImmediateGraphStorage()
        for name in (NEW_ENTITY, OTHER):
            await graph.upsert_node(name, {"entity_id": name, "source_id": "chunk-1"})
        creating.relation_chunks.fail_commit_times = 1

        with pytest.raises(_Boom):
            await utils_graph.acreate_relation(
                graph,
                creating.entities_vdb,
                creating.relationships_vdb,
                NEW_ENTITY,
                OTHER,
                {"description": "d", "weight": 1.0, "source_id": "chunk-9"},
                relation_chunks_storage=creating.relation_chunks,
            )

        assert (NEW_ENTITY, OTHER) not in graph.edges
        assert NEW_RELATION_KEY not in creating.relation_chunks.disk


class TestDeclinedCommitFailsTheCreateAndEditPaths:
    """Fix proof: a declined graph commit must surface, not be swallowed.

    `_persist_graph_updates` used to discard `index_done_callback`'s return
    value, so `NetworkXStorage` reloading from disk and DISCARDING the caller's
    mutation looked exactly like a successful commit. The create and edit paths
    commit through that helper (the deletion paths use
    `_commit_graph_or_raise`), so they returned HTTP 200 for a write that never
    reached disk -- while the tracking row phase 1 had just made durable stayed
    behind, describing an object that does not exist.
    """

    @staticmethod
    def _decline_graph_commit(fixture, monkeypatch):
        async def _declined():
            return False

        monkeypatch.setattr(fixture.graph, "index_done_callback", _declined)

    @pytest.mark.asyncio
    async def test_entity_creation_raises_when_the_graph_declines(
        self, creating, monkeypatch
    ):
        self._decline_graph_commit(creating, monkeypatch)

        with pytest.raises(RuntimeError, match="discarded"):
            await _create_entity(creating)

    @pytest.mark.asyncio
    async def test_relation_creation_raises_when_the_graph_declines(
        self, creating, monkeypatch
    ):
        await creating.graph.upsert_node(
            NEW_ENTITY, {"entity_id": NEW_ENTITY, "description": "d", "source_id": "s"}
        )
        self._decline_graph_commit(creating, monkeypatch)

        with pytest.raises(RuntimeError, match="discarded"):
            await _create_relation(creating)

    @pytest.mark.asyncio
    async def test_relation_edit_raises_when_the_graph_declines(
        self, deferred, monkeypatch
    ):
        self._decline_graph_commit(deferred, monkeypatch)

        with pytest.raises(RuntimeError, match="discarded"):
            await utils_graph.aedit_relation(
                deferred.graph,
                deferred.entities_vdb,
                deferred.relationships_vdb,
                ENTITY,
                OTHER,
                {"description": "edited"},
                relation_chunks_storage=deferred.relation_chunks,
            )

    @pytest.mark.asyncio
    async def test_a_declined_commit_outranks_a_failing_vector_flush(
        self, creating, monkeypatch
    ):
        # Both halves of phase 2 fail. The vector store being stale is the
        # rebuildable window; the graph having discarded the write is not, so
        # that is the answer the caller has to get.
        creating.entities_vdb.fail_flush = True
        self._decline_graph_commit(creating, monkeypatch)

        with pytest.raises(RuntimeError, match="discarded"):
            await _create_entity(creating)

    @pytest.mark.asyncio
    async def test_a_backend_returning_none_still_creates(self, creating, monkeypatch):
        # The base signature is `-> None`; only an explicit False means refusal.
        async def _committed_quietly():
            return None

        monkeypatch.setattr(creating.graph, "index_done_callback", _committed_quietly)

        result = await _create_entity(creating)

        assert result["entity_name"] == NEW_ENTITY


class TestRelationEditGrowsBeforeItShrinks:
    """Fix proof: an edit that DROPS evidence must not persist the drop first.

    `aedit_relation`'s tracking update applies a delta in both directions. With
    the row committed unconditionally before the graph, a shrinking edit puts
    the narrowed row on disk while the edge still carries the wider
    `source_id`: a later purge of the dropped chunk's document reads the row,
    concludes "no remaining sources" and deletes a relation another document
    still anchors. The staging is therefore grow-then-shrink -- superset row,
    graph, final row -- so the durable row is never a strict subset of the
    durable evidence.
    """

    SHRINKING_EDIT = {"source_id": "chunk-1"}

    @staticmethod
    async def _seed_two_chunk_relation(fixture):
        await fixture.graph.upsert_edge(
            ENTITY,
            OTHER,
            {
                "description": "d",
                "weight": 2.0,
                "source_id": f"chunk-1{GRAPH_FIELD_SEP}chunk-2",
            },
        )
        await fixture.graph.index_done_callback()
        await fixture.relation_chunks.upsert(
            {RELATION_KEY: {"chunk_ids": ["chunk-1", "chunk-2"], "count": 2}}
        )
        await fixture.relation_chunks.index_done_callback()
        fixture.commit_log.clear()

    async def _edit(self, fixture):
        return await utils_graph.aedit_relation(
            fixture.graph,
            fixture.entities_vdb,
            fixture.relationships_vdb,
            ENTITY,
            OTHER,
            dict(self.SHRINKING_EDIT),
            relation_chunks_storage=fixture.relation_chunks,
        )

    @pytest.mark.asyncio
    async def test_a_failed_graph_commit_leaves_the_wider_row_on_disk(
        self, deferred, monkeypatch
    ):
        await self._seed_two_chunk_relation(deferred)
        _log_graph_commit(deferred, monkeypatch, fail=True)

        with pytest.raises(_Boom):
            await self._edit(deferred)

        # The edge on disk still cites both chunks, so the row must too.
        assert deferred.relation_chunks.disk[RELATION_KEY]["chunk_ids"] == [
            "chunk-1",
            "chunk-2",
        ]

    @pytest.mark.asyncio
    async def test_a_successful_edit_ends_with_the_narrowed_row(self, deferred):
        await self._seed_two_chunk_relation(deferred)

        await self._edit(deferred)

        assert deferred.relation_chunks.disk[RELATION_KEY]["chunk_ids"] == ["chunk-1"]
        assert deferred.relation_chunks.disk[RELATION_KEY]["count"] == 1

    @pytest.mark.asyncio
    async def test_a_purely_growing_edit_commits_the_row_before_the_graph(
        self, deferred, monkeypatch
    ):
        # Stability: the additive direction keeps the original ordering, so the
        # new IDs are durable before the edge that cites them.
        _log_graph_commit(deferred, monkeypatch, fail=False)

        await utils_graph.aedit_relation(
            deferred.graph,
            deferred.entities_vdb,
            deferred.relationships_vdb,
            ENTITY,
            OTHER,
            {"source_id": f"chunk-1{GRAPH_FIELD_SEP}chunk-7", "weight": 2.0},
            relation_chunks_storage=deferred.relation_chunks,
        )

        assert deferred.commit_log.index("relation_chunks") < deferred.commit_log.index(
            "graph"
        )
        assert deferred.relation_chunks.disk[RELATION_KEY]["chunk_ids"] == [
            "chunk-1",
            "chunk-7",
        ]

    @pytest.mark.asyncio
    async def test_a_failed_shrink_keeps_the_edit_and_reports_the_wider_row(
        self, deferred, monkeypatch
    ):
        # The accepted residue of the staging: the edge is durable, the row
        # still names a chunk it no longer cites. Under-deletion, repairable
        # offline -- but NOT silent. `VectorStorageConsistencyError` is the type
        # this codebase already uses for "a step after a durable graph update
        # failed" (its docstring names a chunk-tracking retirement, and the
        # rename branch of `_edit_entity_impl` raises it for the same shape), and
        # the route maps it to a 500. Logging alone would answer 200 for a row
        # only an operator can fix: a retry re-reads the unchanged source_id and
        # skips the tracking block, so nothing else will ever reconcile it.
        await self._seed_two_chunk_relation(deferred)
        calls = {"n": 0}
        original = deferred.relation_chunks.upsert

        async def _upsert(data):
            calls["n"] += 1
            if calls["n"] == 2:
                raise _Boom("shrink write failed")
            return await original(data)

        monkeypatch.setattr(deferred.relation_chunks, "upsert", _upsert)

        with pytest.raises(VectorStorageConsistencyError) as excinfo:
            await self._edit(deferred)

        # The message has to say the edit landed, and name both the row and the
        # only tool that can reconcile it -- that is the whole point of raising
        # this type rather than a bare failure.
        message = str(excinfo.value)
        assert RELATION_KEY in message
        assert "durable" in message
        assert "lightrag-repair-chunk-tracking" in message

        # The edit is durable despite the raise, and the residue is the wider
        # row -- under-deletion, never the over-deleting mirror.
        assert deferred.persisted_graph()[ENTITY][OTHER]["source_id"] == "chunk-1"
        assert deferred.relation_chunks.disk[RELATION_KEY]["chunk_ids"] == [
            "chunk-1",
            "chunk-2",
        ]
