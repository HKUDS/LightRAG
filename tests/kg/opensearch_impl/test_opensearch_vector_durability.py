"""``OpenSearchVectorDBStorage.has_pending_index_ops`` on the real class,
against a server that answers every bulk item with a retryable failure (429).

``_flush_pending_vector_ops`` keeps such items buffered and returns normally,
so ``index_done_callback()`` returning is not proof that a staged vector
reached the server. ``lightrag-rebuild-vdb`` asks this before recording a
target's embedding baseline: recording one over an incomplete index would
claim the target was adopted in the configured space while its vectors are
missing, and nothing later catches that — the startup precheck sees a matching
record and the coverage gate only refuses an EMPTY index. See *Rebuild* in
docs/design/ConfigurationStorageContract.md.

The base ``has_pending_index_ops`` answers ``False``, which is the truth for a
backend that buffers nothing and a wrong answer here; that is what these tests
pin.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from lightrag.kg.opensearch_impl import ClientManager, OpenSearchVectorDBStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import EmbeddingFunc

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
    """A bulk that either applies every action or answers each one with 429."""

    def __init__(self, *, rate_limited: bool):
        self.docs: dict[str, dict] = {}
        self.rate_limited = rate_limited
        self.bulk_calls = 0
        self.client = AsyncMock()
        self.client.indices = AsyncMock()
        self.client.indices.exists = AsyncMock(return_value=True)
        self.client.indices.create = AsyncMock()
        self.client.indices.refresh = AsyncMock()
        self.client.indices.get_mapping = AsyncMock(return_value={})

    async def bulk(self, client, actions, **_):
        self.bulk_calls += 1
        actions = list(actions)
        if self.rate_limited:
            return 0, [
                {
                    action["_op_type"]: {
                        "_id": action["_id"],
                        "status": 429,
                        "error": "rate limited",
                    }
                }
                for action in actions
            ]
        for action in actions:
            if action["_op_type"] == "delete":
                self.docs.pop(action["_id"], None)
            else:
                self.docs[action["_id"]] = {"_id": action["_id"]}
        return len(actions), []


async def _embed(texts, **kwargs):
    return np.ones((len(texts), 8), dtype=np.float32)


async def _storage(server: _FakeServer) -> OpenSearchVectorDBStorage:
    with patch.object(ClientManager, "get_client", return_value=server.client):
        storage = OpenSearchVectorDBStorage(
            namespace="entities",
            workspace="tenant",
            global_config={
                "embedding_batch_num": 10,
                "max_graph_nodes": 1000,
                "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
            },
            embedding_func=EmbeddingFunc(
                embedding_dim=8, max_token_size=1024, func=_embed, model_name="bge-m3"
            ),
            meta_fields={"entity_name"},
        )
        await storage.initialize()
    return storage


async def test_a_rate_limited_flush_leaves_the_vectors_pending():
    """The defect's precondition and the answer that catches it: the flush
    returns normally, and the storage still holds every vector."""
    server = _FakeServer(rate_limited=True)
    storage = await _storage(server)

    with patch("lightrag.kg.opensearch_impl.helpers.async_bulk", new=server.bulk):
        await storage.upsert(
            {"ent-1": {"content": "Alice", "entity_name": "Alice"}},
        )
        await storage.index_done_callback()

        assert server.bulk_calls == 1
        assert server.docs == {}, "the server took nothing"
        assert await storage.has_pending_index_ops() is True
        assert await storage.has_pending_index_ops(include_deletes=True) is True


async def test_a_healthy_flush_leaves_nothing_pending():
    """The other half: a flush the server accepted retains nothing, so asking
    costs a lock and changes no outcome."""
    server = _FakeServer(rate_limited=False)
    storage = await _storage(server)

    with patch("lightrag.kg.opensearch_impl.helpers.async_bulk", new=server.bulk):
        await storage.upsert({"ent-1": {"content": "Alice", "entity_name": "Alice"}})
        await storage.index_done_callback()

        assert server.docs == {"ent-1": {"_id": "ent-1"}}
        assert await storage.has_pending_index_ops() is False
        assert await storage.has_pending_index_ops(include_deletes=True) is False


async def test_a_retained_delete_is_counted_only_when_asked_for():
    """Same split as the KV side: a retained tombstone is excluded by default
    and counted with ``include_deletes=True``."""
    server = _FakeServer(rate_limited=False)
    storage = await _storage(server)

    with patch("lightrag.kg.opensearch_impl.helpers.async_bulk", new=server.bulk):
        await storage.upsert({"ent-1": {"content": "Alice", "entity_name": "Alice"}})
        await storage.index_done_callback()

        server.rate_limited = True
        await storage.delete(["ent-1"])
        await storage.index_done_callback()

        assert await storage.has_pending_index_ops() is False
        assert await storage.has_pending_index_ops(include_deletes=True) is True
