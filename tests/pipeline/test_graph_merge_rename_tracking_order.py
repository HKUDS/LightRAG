"""Merge and rename must not retire a tracking row before its object is gone.

`_merge_entities_impl` and `_edit_entity_impl` migrate chunk tracking rather than
dropping it: the row moves to the surviving key. But the old key was retired
before the graph commit that removes the old object, so on a deferred graph
backend with an immediate-write tracking store every instant in between had the
object on disk with no authoritative provenance -- the state
`docs/design/PurgeRecoveryContract.md` forbids, and the one from which
`_purge_kg_contributions` concludes "no remaining sources" and deletes an entity
other documents still reference. Neither path checked the graph commit result
either, so a declined commit reported success while leaving exactly that.

The fix stages both paths the way `adelete_by_entity` is staged: migrated rows
are written first, the graph commit is confirmed, and only then are the old keys
retired. `TestUpsertStillPrecedesDelete` pins the OTHER half of that ordering,
which came from f86ef93c (#3609): the new row must be written before the old one
is deleted, so a failure can never leave the row under neither key. Both
invariants have to hold at once, and a fix for either one alone re-breaks the
other.

The doubles are immediate-write on purpose (Redis/PG/Mongo semantics): with a
deferred tracking store the deletes would not be durable until the flush and the
ordering bug would be invisible, which is why the real deferred stack cannot
stand in for these cases.
"""

from __future__ import annotations

import asyncio

import pytest

from lightrag import utils_graph
from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import VectorStorageConsistencyError, make_relation_chunk_key

pytestmark = pytest.mark.offline

SOURCE = "ATLAS"
TARGET = "BOREALIS"
OTHER = "CASSINI"
RENAMED = "ATLAS-II"
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
    """Immediate-write KV double: a delete is durable before the next await."""

    def __init__(self, tag):
        self.tag = tag
        self.records: dict = {}
        self.timeline: list = []
        self.delete_error: BaseException | None = None

    async def get_by_id(self, key):
        return self.records.get(key)

    async def get_by_ids(self, keys):
        return [self.records.get(key) for key in keys]

    async def upsert(self, data):
        self.timeline.append(("upsert", self.tag, sorted(data)))
        self.records.update(data)

    async def delete(self, ids):
        # Yield before recording anything: a real RPC-backed store round-trips
        # here, so this is the suspension point at which a pending cancellation
        # is delivered. Without it the double would swallow every cancellation
        # aimed at the retirement step and `TestCancellationDoesNotStrandRows`
        # could not tell a protected cleanup from an unprotected one.
        await asyncio.sleep(0)
        if self.delete_error is not None:
            raise self.delete_error
        self.timeline.append(("delete", self.tag, sorted(ids)))
        for key in ids:
            self.records.pop(key, None)

    async def index_done_callback(self):
        await asyncio.sleep(0)

    async def is_empty(self):
        return not self.records


class _VectorStorage:
    def __init__(self, global_config):
        self.global_config = global_config
        self.delete_error: BaseException | None = None

    async def upsert(self, data):
        pass

    async def delete(self, ids):
        # A real vector store round-trips here; the yield makes this a genuine
        # await between the graph work around it.
        await asyncio.sleep(0)
        if self.delete_error is not None:
            raise self.delete_error

    async def delete_entity(self, entity_name):
        pass

    async def delete_entity_relation(self, entity_name):
        pass

    async def index_done_callback(self):
        await asyncio.sleep(0)


class _DeferredKVStorage(_KVStorage):
    """The default JsonKVStorage's durability: upsert touches memory only.

    ``records`` is the shared in-memory view every process sees; ``persisted``
    is what a restart would find. Only ``index_done_callback`` moves one to the
    other, which is what makes the ordering against the graph commit observable
    at all -- with the immediate-write double above, every write is already on
    "disk" and the window cannot exist.
    """

    def __init__(self, tag):
        super().__init__(tag)
        self.persisted: dict = {}

    async def index_done_callback(self):
        await asyncio.sleep(0)
        self.persisted = {key: dict(value) for key, value in self.records.items()}


class _ImmediateWriteGraph:
    """Neo4j / Memgraph / MongoDB / PostgreSQL semantics for the graph store.

    Those backends persist each mutation as it runs and their graph
    `index_done_callback` is a no-op returning None (which
    `_commit_graph_or_raise` accepts: only an explicit False means "declined").
    So the commit carries no information there, and what decides whether an
    object is still on disk is when the mutation itself ran -- which is why the
    node removal has to be inside the cancellation-deferring region, not before
    it. NetworkX cannot show this: its deletes stay in memory until the commit.
    """

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    async def _write_through(self, coro):
        result = await coro
        await self._inner.index_done_callback()
        return result

    async def upsert_node(self, node_id, node_data):
        return await self._write_through(self._inner.upsert_node(node_id, node_data))

    async def upsert_edge(self, source_node_id, target_node_id, edge_data):
        return await self._write_through(
            self._inner.upsert_edge(source_node_id, target_node_id, edge_data)
        )

    async def delete_node(self, node_id):
        return await self._write_through(self._inner.delete_node(node_id))

    async def index_done_callback(self):
        return None


class _Fixture:
    """A real NetworkXStorage, so "still on disk" is observed, not simulated."""

    def __init__(self, tmp_path, *, immediate: bool = False):
        self.global_config = {
            "working_dir": str(tmp_path),
            "workspace": "",
            "embedding_batch_num": 10,
        }
        graph = NetworkXStorage(
            namespace="chunk_entity_relation",
            workspace="",
            global_config=self.global_config,
            embedding_func=None,
        )
        self.graph = _ImmediateWriteGraph(graph) if immediate else graph
        self.entities_vdb = _VectorStorage(self.global_config)
        self.relationships_vdb = _VectorStorage(self.global_config)
        self.entity_chunks = _KVStorage("entity_chunks")
        self.relation_chunks = _KVStorage("relation_chunks")
        self.timeline: list = []

    async def start(self):
        await self.graph.initialize()
        for name in (SOURCE, TARGET, OTHER):
            await self.graph.upsert_node(
                name, {"entity_id": name, "description": "d", "source_id": "chunk-1"}
            )
        for left in (SOURCE, TARGET):
            await self.graph.upsert_edge(
                left,
                OTHER,
                {"description": "d", "weight": 1.0, "source_id": "chunk-1"},
            )
        await self.entity_chunks.upsert(
            {name: dict(CHUNKS) for name in (SOURCE, TARGET, OTHER)}
        )
        await self.relation_chunks.upsert(
            {
                make_relation_chunk_key(SOURCE, OTHER): dict(CHUNKS),
                make_relation_chunk_key(TARGET, OTHER): dict(CHUNKS),
            }
        )
        await self.graph.index_done_callback()
        # One shared timeline so graph commits and KV writes can be ordered
        # against each other; per-store logs cannot show that interleaving.
        self.entity_chunks.timeline = self.timeline
        self.relation_chunks.timeline = self.timeline
        return self

    def persisted_graph(self):
        return NetworkXStorage.load_nx_graph(self.graph._graphml_xml_file)

    def record_graph_commits(self, monkeypatch):
        original = self.graph.index_done_callback

        async def _logged():
            result = await original()
            self.timeline.append(
                ("graph-commit", sorted(self.persisted_graph().nodes()))
            )
            return result

        monkeypatch.setattr(self.graph, "index_done_callback", _logged)

    def fail_publication(self, monkeypatch, *, after: int = 0):
        """Make the cross-process reload notification fail after a real write.

        `commit_in_storage_io` runs the hook only once the GraphML write
        succeeded, so this models the one failure mode in which the graph
        mutation IS durable while the commit reports trouble.
        """
        calls = {"n": 0}

        async def _flags(namespace, workspace=None):
            calls["n"] += 1
            if calls["n"] > after:
                raise _Boom("shared-storage manager is down")

        monkeypatch.setattr("lightrag.kg.networkx_impl.set_all_update_flags", _flags)

    def use_deferred_tracking(self):
        """Swap in KV doubles that persist only on their commit."""
        for name in ("entity_chunks", "relation_chunks"):
            source = getattr(self, name)
            deferred = _DeferredKVStorage(source.tag)
            deferred.records = dict(source.records)
            deferred.persisted = {k: dict(v) for k, v in source.records.items()}
            deferred.timeline = self.timeline
            setattr(self, name, deferred)
        return self

    def snapshot_tracking_at_each_graph_commit(self, monkeypatch):
        """What a restart would find, sampled at every graph commit."""
        snapshots: list[dict] = []
        original = self.graph.index_done_callback

        async def _sampled():
            # The tracking side is read BEFORE the commit and the graph side
            # AFTER it: that pair is exactly what a process exit right after the
            # commit would leave behind, since the tracking flush has not run
            # yet at that instant.
            on_disk = {
                "entities": {
                    key: dict(row) for key, row in self.entity_chunks.persisted.items()
                },
                "relations": {
                    key: dict(row)
                    for key, row in self.relation_chunks.persisted.items()
                },
            }
            result = await original()
            on_disk["nodes"] = sorted(self.persisted_graph().nodes())
            snapshots.append(on_disk)
            return result

        monkeypatch.setattr(self.graph, "index_done_callback", _sampled)
        return snapshots

    def cancel_caller_on_graph_commit(self, monkeypatch, caller, *, after: int = 0):
        """Cancel ``caller`` from inside a graph commit that really landed.

        Models the shutdown case: the commit finishes, and the cancellation is
        delivered to the task that requested the merge/rename while its tracking
        cleanup is still owed.
        """
        original = self.graph.index_done_callback
        calls = {"n": 0}

        async def _commit():
            result = await original()
            calls["n"] += 1
            if calls["n"] > after:
                caller.cancel()
            return result

        monkeypatch.setattr(self.graph, "index_done_callback", _commit)

    def fail_graph_commit(self, monkeypatch, *, after: int = 0, declined: bool = False):
        original = self.graph.index_done_callback
        calls = {"n": 0}

        async def _commit():
            calls["n"] += 1
            if calls["n"] > after:
                if declined:
                    return False
                raise _Boom("graph save failed")
            return await original()

        monkeypatch.setattr(self.graph, "index_done_callback", _commit)

    async def merge(self):
        return await utils_graph.amerge_entities(
            self.graph,
            self.entities_vdb,
            self.relationships_vdb,
            [SOURCE],
            TARGET,
            None,
            None,
            self.entity_chunks,
            self.relation_chunks,
        )

    async def merge_via_edit(self):
        """The merge reached through `aedit_entity(allow_merge=True)`.

        Renaming onto an existing name is routed into `_merge_entities_impl` by
        `_edit_entity_impl`, and that wrapper is where a post-commit failure can
        be downgraded into a partial-success summary.
        """
        return await utils_graph.aedit_entity(
            self.graph,
            self.entities_vdb,
            self.relationships_vdb,
            SOURCE,
            {"entity_name": TARGET},
            True,
            True,
            self.entity_chunks,
            self.relation_chunks,
        )

    async def rename(self):
        return await utils_graph.aedit_entity(
            self.graph,
            self.entities_vdb,
            self.relationships_vdb,
            SOURCE,
            {"entity_name": RENAMED},
            True,
            False,
            self.entity_chunks,
            self.relation_chunks,
        )


@pytest.fixture
async def rag(tmp_path):
    fixture = await _Fixture(tmp_path).start()
    yield fixture
    await fixture.graph.finalize()


@pytest.fixture
async def immediate_rag(tmp_path):
    """The same fixture on a backend that persists every mutation inline."""
    fixture = await _Fixture(tmp_path, immediate=True).start()
    yield fixture
    await fixture.graph.finalize()


def _live_objects_without_rows(fixture):
    """Every on-disk node/edge whose authoritative tracking row is missing."""
    persisted = fixture.persisted_graph()
    orphans = [n for n in persisted.nodes() if n not in fixture.entity_chunks.records]
    orphans += [
        tuple(sorted(edge))
        for edge in persisted.edges()
        if make_relation_chunk_key(*sorted(edge)) not in fixture.relation_chunks.records
    ]
    return orphans


def _rows_without_live_objects(fixture):
    """Every tracking row whose graph object is no longer on disk.

    The inverse of the check above, and the residue this file's staging trades
    against: an orphan row is dead bookkeeping until its key recurs, at which
    point extraction reads it back as authoritative provenance.
    """
    persisted = fixture.persisted_graph()
    live_edge_keys = {
        make_relation_chunk_key(*sorted(edge)) for edge in persisted.edges()
    }
    orphans = [n for n in fixture.entity_chunks.records if n not in persisted.nodes()]
    orphans += [k for k in fixture.relation_chunks.records if k not in live_edge_keys]
    return orphans


class TestDeclinedCommitIsNotSuccess:
    """A declined graph commit discards the operation; it must not report success.

    `NetworkXStorage.index_done_callback` returns False when another process
    published a newer graph file: it reloads from disk and DROPS the in-memory
    mutation. Both paths took a normal return as proof of a commit, so the
    operation reported success while the rows it had already retired belonged to
    objects still on disk.
    """

    @pytest.mark.asyncio
    async def test_merge_raises_and_keeps_every_row(self, rag, monkeypatch):
        rag.fail_graph_commit(monkeypatch, declined=True)

        with pytest.raises(Exception):
            await rag.merge()

        assert rag.persisted_graph().has_node(SOURCE)
        assert _live_objects_without_rows(rag) == []

    @pytest.mark.asyncio
    async def test_rename_raises_and_keeps_every_row(self, rag, monkeypatch):
        rag.fail_graph_commit(monkeypatch, declined=True)

        with pytest.raises(Exception):
            await rag.rename()

        assert rag.persisted_graph().has_node(SOURCE)
        assert _live_objects_without_rows(rag) == []


class TestFailedCommitKeepsProvenance:
    """A failing commit leaves the old objects live, so their rows must live too."""

    @pytest.mark.asyncio
    async def test_merge_source_removal_failure_keeps_rows(self, rag, monkeypatch):
        # Let the relation redirection commit, then fail the commit that would
        # make the source entity's removal durable.
        rag.fail_graph_commit(monkeypatch, after=1)

        with pytest.raises(Exception) as excinfo:
            await rag.merge()

        assert rag.persisted_graph().has_node(SOURCE)
        assert _live_objects_without_rows(rag) == []
        # The message must describe the state that actually holds: the source
        # entity is still there. Claiming it was removed sends an operator to
        # the vector store while the damage would be in chunk tracking.
        assert "were NOT removed" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_rename_commit_failure_keeps_rows(self, rag, monkeypatch):
        rag.fail_graph_commit(monkeypatch)

        with pytest.raises(Exception):
            await rag.rename()

        assert rag.persisted_graph().has_node(SOURCE)
        assert _live_objects_without_rows(rag) == []


class TestRowsAreRetiredOnlyAfterTheCommit:
    """The ordering pin: no row may be deleted while its object is still on disk."""

    @staticmethod
    def _assert_deletes_follow_the_removal(fixture, gone):
        committed_away = set()
        seen_delete = False
        for event in fixture.timeline:
            if event[0] == "graph-commit":
                committed_away = gone - set(event[1])
            elif event[0] == "delete":
                seen_delete = True
                for key in event[2]:
                    named = set(key.split("<SEP>")) if "<SEP>" in key else {key}
                    assert named & committed_away, (
                        f"deleted {key!r} while its object was still on disk; "
                        f"timeline={fixture.timeline}"
                    )
        assert seen_delete, "no tracking row was retired at all"

    @pytest.mark.asyncio
    async def test_merge_retires_rows_after_the_source_is_gone(self, rag, monkeypatch):
        rag.record_graph_commits(monkeypatch)

        await rag.merge()

        self._assert_deletes_follow_the_removal(rag, {SOURCE})
        assert _live_objects_without_rows(rag) == []
        assert SOURCE not in rag.entity_chunks.records

    @pytest.mark.asyncio
    async def test_rename_retires_rows_after_the_old_name_is_gone(
        self, rag, monkeypatch
    ):
        rag.record_graph_commits(monkeypatch)

        await rag.rename()

        self._assert_deletes_follow_the_removal(rag, {SOURCE})
        assert _live_objects_without_rows(rag) == []
        assert SOURCE not in rag.entity_chunks.records
        assert rag.entity_chunks.records[RENAMED] == CHUNKS


class TestUpsertStillPrecedesDelete:
    """f86ef93c's invariant (#3609): the row must never be under neither key.

    The fix above moves the DELETES later. Moving the UPSERTS later instead would
    satisfy the same ordering assertion while re-opening the bug that commit
    closed, so this pins the other side: with the graph commit failing, the
    migrated row is already written and the old row is still present.
    """

    @pytest.mark.asyncio
    async def test_merge_writes_the_target_row_before_committing(
        self, rag, monkeypatch
    ):
        rag.fail_graph_commit(monkeypatch, after=1)

        with pytest.raises(Exception):
            await rag.merge()

        assert rag.relation_chunks.records[make_relation_chunk_key(TARGET, OTHER)]
        assert rag.entity_chunks.records[SOURCE] == CHUNKS

    @pytest.mark.asyncio
    async def test_rename_writes_the_new_row_before_committing(self, rag, monkeypatch):
        rag.fail_graph_commit(monkeypatch)

        with pytest.raises(Exception):
            await rag.rename()

        assert rag.entity_chunks.records[RENAMED] == CHUNKS
        assert rag.relation_chunks.records[make_relation_chunk_key(RENAMED, OTHER)]
        assert rag.entity_chunks.records[SOURCE] == CHUNKS


class TestCancellationDoesNotStrandRows:
    """A cancellation delivered after the commit must not skip the retirement.

    Once the commit lands, the old node/edges are gone for good and their
    tracking rows describe nothing. `CancelledError` is a `BaseException`, so it
    slips past every `except Exception` on the way out and, unprotected, returns
    with the rows still on disk. For an entity that residue is unreachable: the
    incident relation keys cannot be rediscovered once the node is gone, so a
    later recreation of the same key reads a stale row back as authoritative
    provenance. Both paths therefore run the commit and the retirement inside
    one cancellation-deferring region, exactly as `adelete_by_entity` does.

    The caller is cancelled from inside a commit that really succeeded, which is
    the shutdown case this protects: `commit_in_storage_io` finishes the GraphML
    write before re-raising, so the cancellation surfaces with the cleanup owed.
    """

    @pytest.mark.asyncio
    async def test_merge_retires_rows_even_when_the_caller_is_cancelled(
        self, rag, monkeypatch
    ):
        # after=1: the merge commits twice (the merged target, then the source
        # removal). Only the second one leaves rows owed.
        rag.cancel_caller_on_graph_commit(monkeypatch, asyncio.current_task(), after=1)

        with pytest.raises(asyncio.CancelledError):
            await rag.merge()

        assert SOURCE not in rag.persisted_graph().nodes()
        assert SOURCE not in rag.entity_chunks.records
        assert make_relation_chunk_key(SOURCE, OTHER) not in rag.relation_chunks.records
        assert _live_objects_without_rows(rag) == []

    @pytest.mark.asyncio
    async def test_rename_retires_rows_even_when_the_caller_is_cancelled(
        self, rag, monkeypatch
    ):
        rag.cancel_caller_on_graph_commit(monkeypatch, asyncio.current_task())

        with pytest.raises(asyncio.CancelledError):
            await rag.rename()

        assert SOURCE not in rag.persisted_graph().nodes()
        assert SOURCE not in rag.entity_chunks.records
        assert make_relation_chunk_key(SOURCE, OTHER) not in rag.relation_chunks.records
        assert rag.entity_chunks.records[RENAMED] == CHUNKS
        assert _live_objects_without_rows(rag) == []


class TestMergingEntitiesWithoutRelations:
    """Merging entities that carry no edges must not fail after the commit.

    `stale_relation_keys` is read unconditionally when the source removal is
    made durable, but it was only assigned inside the branch guarded by
    `all_relations`. Entities with no incident edges leave that list empty, so
    the read raised `UnboundLocalError` -- after the commit that had already
    removed the source entities. The merge had landed and the API reported it as
    a failure.
    """

    ISOLATED_SOURCE = "DERELICT"
    ISOLATED_TARGET = "SALVAGE"

    @pytest.mark.asyncio
    async def test_merge_of_edgeless_entities_succeeds(self, rag):
        for name in (self.ISOLATED_SOURCE, self.ISOLATED_TARGET):
            await rag.graph.upsert_node(
                name, {"entity_id": name, "description": "d", "source_id": "chunk-1"}
            )
        await rag.entity_chunks.upsert(
            {
                name: dict(CHUNKS)
                for name in (self.ISOLATED_SOURCE, self.ISOLATED_TARGET)
            }
        )
        await rag.graph.index_done_callback()

        await utils_graph.amerge_entities(
            rag.graph,
            rag.entities_vdb,
            rag.relationships_vdb,
            [self.ISOLATED_SOURCE],
            self.ISOLATED_TARGET,
            None,
            None,
            rag.entity_chunks,
            rag.relation_chunks,
        )

        persisted = rag.persisted_graph()
        assert self.ISOLATED_SOURCE not in persisted.nodes()
        assert self.ISOLATED_TARGET in persisted.nodes()
        assert self.ISOLATED_SOURCE not in rag.entity_chunks.records
        assert _live_objects_without_rows(rag) == []


class TestImmediateWriteBackendsRemoveTheNodeInsideTheRegion:
    """On an inline-persisting graph store the removal must not precede the region.

    Neo4j, Memgraph, MongoDB and PostgreSQL make `delete_node` durable as it
    runs and their graph `index_done_callback` is a no-op, so a removal issued
    before the cancellation-deferring region is already permanent while the
    tracking rows it invalidates are still on disk. Every await in between --
    the vector deletes, the relation rewrites -- can then exit with the object
    gone and its rows stranded, and stranded is where they stay: retrying the
    merge or rename fails its existence check, so nothing rediscovers them.

    Both paths therefore issue the removal inside the region, exactly where
    `adelete_by_entity` issues its own.
    """

    @pytest.mark.asyncio
    async def test_merge_keeps_the_source_when_the_vector_delete_fails(
        self, immediate_rag
    ):
        immediate_rag.entities_vdb.delete_error = _Boom("vector store down")

        with pytest.raises(Exception):
            await immediate_rag.merge()

        assert SOURCE in immediate_rag.persisted_graph().nodes()
        assert immediate_rag.entity_chunks.records[SOURCE] == CHUNKS
        assert _rows_without_live_objects(immediate_rag) == []
        assert _live_objects_without_rows(immediate_rag) == []

    @pytest.mark.asyncio
    async def test_rename_keeps_the_old_name_when_the_vector_delete_fails(
        self, immediate_rag
    ):
        immediate_rag.entities_vdb.delete_error = _Boom("vector store down")

        with pytest.raises(Exception):
            await immediate_rag.rename()

        assert SOURCE in immediate_rag.persisted_graph().nodes()
        assert immediate_rag.entity_chunks.records[SOURCE] == CHUNKS
        assert _rows_without_live_objects(immediate_rag) == []

    @pytest.mark.asyncio
    async def test_merge_retires_rows_when_cancelled_during_the_node_removal(
        self, immediate_rag, monkeypatch
    ):
        caller = asyncio.current_task()
        inner = immediate_rag.graph._inner
        original = inner.delete_node

        async def _delete_then_cancel(node_id):
            await original(node_id)
            caller.cancel()

        monkeypatch.setattr(inner, "delete_node", _delete_then_cancel)

        with pytest.raises(asyncio.CancelledError):
            await immediate_rag.merge()

        assert SOURCE not in immediate_rag.persisted_graph().nodes()
        assert _rows_without_live_objects(immediate_rag) == []
        assert _live_objects_without_rows(immediate_rag) == []


class TestADurableWriteWhoseNotificationFailedIsStillDurable:
    """The retirement is owed whenever the mutation landed, however it reported.

    NetworkX publishes the GraphML file and then tells the other processes to
    reload it. The second step runs only if the write succeeded, so a failure
    there leaves the object durably gone -- and used to surface as a raise, at
    which point both paths reported "the sources were NOT removed" and skipped
    the retirement of rows that no longer describe anything. The backend now
    records that failure instead of raising it, so what a commit failure means
    is once again "the write did not land" and these paths' messages are true.
    """

    @pytest.mark.asyncio
    async def test_merge_retires_rows_when_the_notification_fails(
        self, rag, monkeypatch
    ):
        # after=1: the first commit publishes the merged target, the second one
        # the source removal. Only the second leaves rows owed.
        rag.fail_publication(monkeypatch, after=1)

        await rag.merge()

        assert SOURCE not in rag.persisted_graph().nodes()
        assert SOURCE not in rag.entity_chunks.records
        assert _rows_without_live_objects(rag) == []
        assert _live_objects_without_rows(rag) == []

    @pytest.mark.asyncio
    async def test_rename_retires_rows_when_the_notification_fails(
        self, rag, monkeypatch
    ):
        rag.fail_publication(monkeypatch)

        await rag.rename()

        assert SOURCE not in rag.persisted_graph().nodes()
        assert SOURCE not in rag.entity_chunks.records
        assert rag.entity_chunks.records[RENAMED] == CHUNKS
        assert _rows_without_live_objects(rag) == []
        assert _live_objects_without_rows(rag) == []


class TestADurableMergeIsNeverReportedAsNotHavingHappened:
    """A post-commit cleanup failure must not be downgraded to a partial success.

    `aedit_entity(allow_merge=True)` routes a rename onto an existing name into
    the merge, and its handler re-raises `VectorStorageConsistencyError` while
    folding every other exception into a summary that answers HTTP 200 with
    `final_entity` set to the SOURCE entity. Once the retirement runs, the merge
    is durable and that source is gone -- so a bare failure there reported the
    surviving entity as the one just deleted, and the caller was told the merge
    had not happened. The retirement failures are therefore typed.
    """

    @pytest.mark.asyncio
    async def test_a_failed_entity_row_retirement_is_raised_not_downgraded(self, rag):
        rag.entity_chunks.delete_error = _Boom("tracking store down")

        with pytest.raises(VectorStorageConsistencyError) as excinfo:
            await rag.merge_via_edit()

        message = str(excinfo.value)
        assert "durable" in message
        assert SOURCE in message
        # The merge really did land: the report has to match that.
        assert SOURCE not in rag.persisted_graph().nodes()
        assert TARGET in rag.persisted_graph().nodes()

    @pytest.mark.asyncio
    async def test_a_failed_relation_row_retirement_is_raised_not_downgraded(self, rag):
        rag.relation_chunks.delete_error = _Boom("tracking store down")

        with pytest.raises(VectorStorageConsistencyError):
            await rag.merge_via_edit()

        assert SOURCE not in rag.persisted_graph().nodes()


class TestMigratedRowsAreDurableBeforeTheRemovalCommit:
    """A graph commit may only publish objects whose tracking rows are on disk.

    On the default `JsonKVStorage` an upsert only updates shared memory; the
    rows land on disk in `index_done_callback`. With the tracking flush after
    the graph commit, a process exit in between came back to the new graph with
    only the OLD keys on disk: the surviving entity and its relations had no
    authoritative tracking at all, leaving a purge to read the KEEP-truncated
    graph `source_id` instead.

    Flushing the migrated rows first inverts the residue into the harmless one
    this file already accepts elsewhere: rows on disk for objects that do not
    exist yet, which a retry overwrites.
    """

    @pytest.mark.asyncio
    async def test_merge_persists_the_target_rows_before_removing_the_source(
        self, rag, monkeypatch
    ):
        rag.use_deferred_tracking()
        # Give the source a chunk the target does not have, so "the target's row
        # is on disk" can only be satisfied by the MIGRATED row: the target key
        # already exists from the fixture, and asserting its mere presence would
        # pass without any migration having been persisted at all.
        only_the_source = {"chunk_ids": ["chunk-src"], "count": 1}
        rag.entity_chunks.records[SOURCE] = dict(only_the_source)
        rag.entity_chunks.persisted[SOURCE] = dict(only_the_source)
        source_relation = make_relation_chunk_key(SOURCE, OTHER)
        rag.relation_chunks.records[source_relation] = dict(only_the_source)
        rag.relation_chunks.persisted[source_relation] = dict(only_the_source)

        snapshots = rag.snapshot_tracking_at_each_graph_commit(monkeypatch)

        await rag.merge()

        # EVERY commit, not just the one that removes the source. The merge
        # commits twice, and the first one already publishes the merged target
        # and its redirected relations -- so the rows have to be on disk by
        # then. Checking only the removal commit left that first publication
        # outside the invariant.
        assert len(snapshots) >= 2, f"expected two commits; snapshots={snapshots}"
        for index, snapshot in enumerate(snapshots):
            target_row = snapshot["entities"].get(TARGET, {})
            assert "chunk-src" in target_row.get("chunk_ids", []), (
                f"commit {index} published the merged target while its tracking "
                f"row existed only in memory; snapshot={snapshot}"
            )
            merged_relation = snapshot["relations"].get(
                make_relation_chunk_key(TARGET, OTHER), {}
            )
            assert "chunk-src" in merged_relation.get("chunk_ids", []), (
                f"commit {index} published the redirected relation while its "
                f"merged row existed only in memory; snapshot={snapshot}"
            )
        assert SOURCE not in snapshots[-1]["nodes"]

    @pytest.mark.asyncio
    async def test_rename_persists_the_new_rows_before_removing_the_old_name(
        self, rag, monkeypatch
    ):
        rag.use_deferred_tracking()
        snapshots = rag.snapshot_tracking_at_each_graph_commit(monkeypatch)

        await rag.rename()

        removal = [s for s in snapshots if SOURCE not in s["nodes"]]
        assert removal, f"no commit removed the old name; snapshots={snapshots}"
        at_removal = removal[0]
        assert RENAMED in at_removal["entities"].keys(), (
            "the old name was removed while the renamed entity's tracking row "
            f"existed only in memory; snapshot={at_removal}"
        )
        assert make_relation_chunk_key(RENAMED, OTHER) in at_removal["relations"].keys()
