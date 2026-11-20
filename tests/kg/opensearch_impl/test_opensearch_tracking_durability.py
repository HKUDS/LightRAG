"""A chunk-tracking flush that retains a write is a failure, not a success.

``OpenSearchKVStorage`` keeps a retryable bulk failure (429) in its
per-process buffer and returns from ``index_done_callback`` normally. A
tracking row is a whole row produced by read-modify-write, so a row left there
would later be written OVER whatever another worker committed in between and
erase that writer's attribution. ``_persist_graph_updates`` therefore discards
the retained tracking writes and raises, before the graph is committed.

Runs the real KV flush against a fake server, as
``test_opensearch_config_durability.py`` does for the configuration store.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from lightrag.kg.opensearch_impl import (
    ClientManager,
    OpenSearchKVStorage,
    OpenSearchReferencesIntactError,
)
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils_graph import _persist_graph_updates

pytestmark = pytest.mark.offline


class _mock_lock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


@pytest.fixture(autouse=True)
def _locks_and_shared_storage():
    cache: dict[tuple[str, str], asyncio.Lock] = {}

    def factory(namespace, workspace=None, enable_logging=False):
        return cache.setdefault((namespace, workspace or ""), asyncio.Lock())

    initialize_share_data(workers=1)
    with (
        patch("lightrag.kg.opensearch_impl.get_data_init_lock", side_effect=_mock_lock),
        patch("lightrag.kg.opensearch_impl.get_namespace_lock", side_effect=factory),
        patch("lightrag.kg.opensearch_impl._shard_doc_supported", True),
    ):
        yield
    finalize_share_data()


class _FakeServer:
    """A document map, and a bulk that applies every action or answers each
    one with a retryable 429."""

    def __init__(self):
        self.docs: dict[str, dict] = {}
        self.rate_limited = False
        self.bulk_calls = 0
        self.client = AsyncMock()
        self.client.indices = AsyncMock()
        self.client.indices.exists = AsyncMock(return_value=True)
        self.client.indices.get_mapping = AsyncMock(return_value={})

    async def bulk(self, client, actions, **_):
        self.bulk_calls += 1
        actions = list(actions)
        if self.rate_limited:
            return 0, [
                {action["_op_type"]: {"_id": action["_id"], "status": 429}}
                for action in actions
            ]
        for action in actions:
            if action["_op_type"] == "delete":
                self.docs.pop(action["_id"], None)
            else:
                self.docs[action["_id"]] = dict(action["script"]["params"]["doc"])
        return len(actions), []


class _RecordingGraph:
    def __init__(self):
        self.commits = 0

    async def index_done_callback(self):
        self.commits += 1


async def _tracking_storage(server: _FakeServer) -> OpenSearchKVStorage:
    with patch.object(ClientManager, "get_client", return_value=server.client):
        storage = OpenSearchKVStorage(
            namespace="entity_chunks",
            workspace="tenant",
            global_config={
                "embedding_batch_num": 10,
                "max_graph_nodes": 1000,
                "vector_db_storage_cls_kwargs": {},
            },
            embedding_func=None,
        )
        await storage.initialize()
    return storage


async def test_a_retained_tracking_write_fails_the_commit_and_is_discarded():
    server = _FakeServer()
    storage = await _tracking_storage(server)
    graph = _RecordingGraph()
    with patch(
        "lightrag.kg.opensearch_impl.helpers.async_bulk", side_effect=server.bulk
    ):
        server.rate_limited = True
        await storage.upsert({"Foo": {"chunk_ids": ["old", "custom"], "count": 2}})
        with pytest.raises(RuntimeError, match="not made durable"):
            await _persist_graph_updates(
                entity_chunks_storage=storage,
                chunk_entity_relation_graph=graph,
                context="test",
            )

        assert server.bulk_calls == 1
        # The graph commit is phase 2: it must not run after a failed phase 1.
        assert graph.commits == 0
        assert await storage.has_pending_index_ops(include_deletes=True) is False

        # The discarded row can no longer land over a later commit.
        server.rate_limited = False
        server.docs["Foo"] = {"chunk_ids": ["old", "document"], "count": 2}
        await storage.index_done_callback()

    assert server.docs["Foo"]["chunk_ids"] == ["old", "document"]


async def test_a_tracking_write_that_lands_commits_the_graph():
    server = _FakeServer()
    storage = await _tracking_storage(server)
    graph = _RecordingGraph()
    with patch(
        "lightrag.kg.opensearch_impl.helpers.async_bulk", side_effect=server.bulk
    ):
        await storage.upsert({"Foo": {"chunk_ids": ["old", "custom"], "count": 2}})
        await _persist_graph_updates(
            entity_chunks_storage=storage,
            chunk_entity_relation_graph=graph,
            context="test",
        )

    assert server.docs["Foo"]["chunk_ids"] == ["old", "custom"]
    assert graph.commits == 1


async def test_a_raising_tracking_flush_discards_its_buffer_too():
    """A transport error pops nothing from the buffer. The failed operation's
    row must not be published by a later flush of this process: here a
    timeout, then a 429 on the retry, then a peer commit, then a clean flush."""
    from opensearchpy.exceptions import ConnectionTimeout

    server = _FakeServer()
    storage = await _tracking_storage(server)
    graph = _RecordingGraph()
    await storage.upsert({"Foo": {"chunk_ids": ["old", "custom"], "count": 2}})

    with patch(
        "lightrag.kg.opensearch_impl.helpers.async_bulk",
        side_effect=ConnectionTimeout("TIMEOUT", "timed out", Exception("slow")),
    ):
        with pytest.raises(OpenSearchReferencesIntactError, match="ConnectionTimeout"):
            await _persist_graph_updates(
                entity_chunks_storage=storage,
                chunk_entity_relation_graph=graph,
                context="test",
            )
    assert graph.commits == 0
    assert await storage.has_pending_index_ops(include_deletes=True) is False

    with patch(
        "lightrag.kg.opensearch_impl.helpers.async_bulk", side_effect=server.bulk
    ):
        server.rate_limited = True
        await storage.index_done_callback()  # the caller's final flush
        server.rate_limited = False
        server.docs["Foo"] = {"chunk_ids": ["old", "document"], "count": 2}
        await storage.index_done_callback()  # this worker's next commit

    assert server.docs["Foo"]["chunk_ids"] == ["old", "document"]


async def test_the_synchronous_discard_empties_the_buffer():
    """What the admin-write gate calls on a failure exit: no await, so a second
    cancellation cannot leave the buffer behind."""
    server = _FakeServer()
    storage = await _tracking_storage(server)
    await storage.upsert({"Foo": {"chunk_ids": ["old", "edit"], "count": 2}})
    await storage.delete(["Bar"])

    assert storage.discard_buffered_writes() == 2
    assert await storage.has_pending_index_ops(include_deletes=True) is False

    with patch(
        "lightrag.kg.opensearch_impl.helpers.async_bulk", side_effect=server.bulk
    ):
        server.docs["Foo"] = {"chunk_ids": ["old", "document"], "count": 2}
        await storage.index_done_callback()

    assert server.bulk_calls == 0
    assert server.docs["Foo"]["chunk_ids"] == ["old", "document"]


# ---------------------------------------------------------------------------
# Writers that commit through ``_insert_done``: the document pipeline,
# ``ainsert_custom_chunks`` and ``ainsert_custom_kg`` ask
# ``has_retained_writes`` through ``LightRAG._require_tracking_commit_landed``.
# ---------------------------------------------------------------------------


async def test_has_retained_writes_counts_only_what_a_flush_attempted():
    """A row buffered AFTER a clean flush is pending but not retained: on the
    pipeline that is a sibling document's merge, not a failed commit."""
    server = _FakeServer()
    storage = await _tracking_storage(server)
    with patch(
        "lightrag.kg.opensearch_impl.helpers.async_bulk", side_effect=server.bulk
    ):
        await storage.upsert({"Foo": {"chunk_ids": ["old"], "count": 1}})
        await storage.index_done_callback()
        await storage.upsert({"Bar": {"chunk_ids": ["fresh"], "count": 1}})
        assert await storage.has_pending_index_ops() is True
        assert await storage.has_retained_writes() is False

        server.rate_limited = True
        await storage.index_done_callback()
        assert await storage.has_retained_writes() is True

        server.rate_limited = False
        await storage.index_done_callback()
        assert await storage.has_retained_writes() is False
    assert server.docs["Bar"]["chunk_ids"] == ["fresh"]


@pytest.mark.parametrize("op", ["upsert", "delete"])
async def test_has_retained_writes_counts_a_retained_tombstone(op):
    server = _FakeServer()
    storage = await _tracking_storage(server)
    with patch(
        "lightrag.kg.opensearch_impl.helpers.async_bulk", side_effect=server.bulk
    ):
        if op == "upsert":
            await storage.upsert({"Foo": {"chunk_ids": ["old"], "count": 1}})
        else:
            await storage.delete(["Foo"])
        server.rate_limited = True
        await storage.index_done_callback()
    assert await storage.has_retained_writes() is True
    await storage.drop_pending_index_ops()
    assert await storage.has_retained_writes() is False


def _commit_owner(entity_chunks):
    """The attributes ``_require_tracking_commit_landed`` and
    ``_flush_storages`` read, with no cache pair to chain."""
    from types import SimpleNamespace

    from lightrag.lightrag import LightRAG

    owner = SimpleNamespace(
        entity_chunks=entity_chunks,
        relation_chunks=None,
        text_chunks=None,
        llm_response_cache=None,
    )
    owner._flush_storages = lambda storages, **kw: LightRAG._flush_storages(
        owner, storages, **kw
    )
    return owner, LightRAG._require_tracking_commit_landed


async def test_a_retained_pipeline_tracking_write_is_reflushed_until_it_lands(
    monkeypatch,
):
    monkeypatch.setattr("lightrag.lightrag._TRACKING_RETAINED_FLUSH_BACKOFF_SECONDS", 0)
    server = _FakeServer()
    storage = await _tracking_storage(server)
    owner, require_landed = _commit_owner(storage)
    original_bulk = server.bulk

    async def recovering_bulk(client, actions, **kw):
        result = await original_bulk(client, actions, **kw)
        server.rate_limited = False  # the 429 clears after one refusal
        return result

    with patch(
        "lightrag.kg.opensearch_impl.helpers.async_bulk", side_effect=recovering_bulk
    ):
        await storage.upsert({"Foo": {"chunk_ids": ["old", "document"], "count": 2}})
        server.rate_limited = True
        await storage.index_done_callback()  # the document's _insert_done
        assert await storage.has_retained_writes() is True
        await require_landed(owner)

    assert server.bulk_calls == 2
    assert server.docs["Foo"]["chunk_ids"] == ["old", "document"]
    assert await storage.has_pending_index_ops(include_deletes=True) is False


async def test_a_tracking_write_that_never_lands_fails_without_discarding(
    monkeypatch,
):
    """The helper raises the pipeline's flush-failure type and leaves the
    buffer to the caller's failure path: dropping it here would take rows of
    sibling documents that have not reached their own check yet."""
    from lightrag.exceptions import IndexFlushError
    from lightrag.lightrag import _TRACKING_RETAINED_FLUSH_RETRIES

    monkeypatch.setattr("lightrag.lightrag._TRACKING_RETAINED_FLUSH_BACKOFF_SECONDS", 0)
    server = _FakeServer()
    storage = await _tracking_storage(server)
    owner, require_landed = _commit_owner(storage)
    with patch(
        "lightrag.kg.opensearch_impl.helpers.async_bulk", side_effect=server.bulk
    ):
        await storage.upsert({"Foo": {"chunk_ids": ["old", "document"], "count": 2}})
        server.rate_limited = True
        await storage.index_done_callback()
        with pytest.raises(IndexFlushError, match="not made durable") as exc_info:
            await require_landed(owner)

    assert exc_info.value.storage_name == "OpenSearchKVStorage"
    assert server.bulk_calls == 1 + _TRACKING_RETAINED_FLUSH_RETRIES
    assert await storage.has_retained_writes() is True
    assert "Foo" not in server.docs
