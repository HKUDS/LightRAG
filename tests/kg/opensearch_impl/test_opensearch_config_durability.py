"""The configuration store on the real ``OpenSearchKVStorage``, against a
server that answers every bulk item with a retryable failure (429).

``_flush_pending_kv_ops`` keeps such items buffered and returns normally, and
``get_by_id_strict`` answers from the buffer, so without the pending check the
configuration store would confirm a baseline claim, a rebuild record and a
workspace drop the server never saw. See *A flush that retained anything is a
failed flush here* in docs/design/ConfigurationStorage.md.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from lightrag import config_store as cs
from lightrag.exceptions import ConfigurationStorageError
from opensearchpy import OpenSearchException

from lightrag.kg.opensearch_impl import ClientManager, OpenSearchKVStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import EmbeddingFunc

pytestmark = pytest.mark.offline

WORKSPACE = "tenant"


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
    """Just enough of OpenSearch for the configuration store: a document map,
    an ``mget`` that echoes the requested id, and a bulk that either applies
    every action or answers each one with 429."""

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
        self.client.mget = AsyncMock(side_effect=self._mget)

    async def _mget(self, *, index, body, **_):
        items = []
        for doc_id in body["ids"]:
            source = self.docs.get(doc_id)
            if source is None:
                items.append({"_id": doc_id, "found": False})
            else:
                items.append({"_id": doc_id, "found": True, "_source": dict(source)})
        return {"docs": items}

    async def bulk(self, client, actions, **_):
        self.bulk_calls += 1
        actions = list(actions)
        if self.rate_limited:
            failed = [
                {
                    action["_op_type"]: {
                        "_id": action["_id"],
                        "status": 429,
                        "error": "rate limited",
                    }
                }
                for action in actions
            ]
            return 0, failed
        for action in actions:
            if action["_op_type"] == "delete":
                self.docs.pop(action["_id"], None)
            else:
                self.docs[action["_id"]] = dict(action["script"]["params"]["doc"])
        return len(actions), []


async def _embed(texts, **kwargs):  # pragma: no cover - never called
    return np.zeros((len(texts), 8), dtype=np.float32)


def _embedding():
    return EmbeddingFunc(
        embedding_dim=8, max_token_size=1024, func=_embed, model_name="bge-m3"
    )


async def _config_storage(server: _FakeServer) -> OpenSearchKVStorage:
    with patch.object(ClientManager, "get_client", return_value=server.client):
        storage = cs.create_configuration_storage(
            OpenSearchKVStorage,
            global_config={
                "embedding_batch_num": 10,
                "max_graph_nodes": 1000,
                "vector_db_storage_cls_kwargs": {},
            },
            embedding_func=None,
        )
        await storage.initialize()
    return storage


def _seed_records(server: _FakeServer) -> None:
    for target in cs.EMBEDDING_TARGETS:
        server.docs[cs.embedding_baseline_key(WORKSPACE, target)] = cs.make_config_row(
            scope_workspace=WORKSPACE,
            suffix=cs.embedding_baseline_suffix(target),
            value={"model": "old-model", "dim": 8, "origin": "probe"},
            updated_by="test",
        )


async def test_a_rate_limited_claim_is_a_failure_and_writes_nothing():
    server = _FakeServer(rate_limited=True)
    storage = await _config_storage(server)
    with patch(
        "lightrag.kg.opensearch_impl.helpers.async_bulk", side_effect=server.bulk
    ):
        with pytest.raises(ConfigurationStorageError, match="retained"):
            await cs.claim_embedding_baseline(
                storage,
                workspace=WORKSPACE,
                target="entities",
                candidate=cs.EmbeddingBaseline("bge-m3", 8, "empty"),
                embedding_func=_embedding(),
            )
    assert server.bulk_calls == 1
    assert server.docs == {}, "the server never recorded the claim"
    assert await storage.has_pending_index_ops(include_deletes=True) is False, (
        "the retained operation was dropped, not left to replay at shutdown"
    )


async def test_a_rate_limited_rebuild_record_is_a_failure_and_keeps_the_old_model():
    server = _FakeServer(rate_limited=True)
    _seed_records(server)
    storage = await _config_storage(server)
    with patch(
        "lightrag.kg.opensearch_impl.helpers.async_bulk", side_effect=server.bulk
    ):
        with pytest.raises(ConfigurationStorageError, match="retained"):
            await cs.record_embedding_baseline(
                storage,
                workspace=WORKSPACE,
                target="chunks",
                embedding_func=_embedding(),
            )
    key = cs.embedding_baseline_key(WORKSPACE, "chunks")
    assert server.docs[key]["value"]["model"] == "old-model"


async def test_a_rate_limited_drop_is_a_failure_and_the_rows_survive():
    server = _FakeServer(rate_limited=True)
    _seed_records(server)
    storage = await _config_storage(server)
    with patch(
        "lightrag.kg.opensearch_impl.helpers.async_bulk", side_effect=server.bulk
    ):
        with pytest.raises(ConfigurationStorageError, match="retained"):
            await cs.delete_workspace_configuration(storage, WORKSPACE)
    assert len(server.docs) == 3, "the server still holds every record"
    assert await storage.has_pending_index_ops(include_deletes=True) is False


async def test_a_healthy_server_confirms_the_claim_from_the_server_not_the_buffer():
    server = _FakeServer(rate_limited=False)
    storage = await _config_storage(server)
    with patch(
        "lightrag.kg.opensearch_impl.helpers.async_bulk", side_effect=server.bulk
    ):
        recorded = await cs.claim_embedding_baseline(
            storage,
            workspace=WORKSPACE,
            target="entities",
            candidate=cs.EmbeddingBaseline("bge-m3", 8, "empty"),
            embedding_func=_embedding(),
        )
    assert recorded.model == "bge-m3"
    assert cs.embedding_baseline_key(WORKSPACE, "entities") in server.docs
    assert await storage.has_pending_index_ops(include_deletes=True) is False


async def test_a_refresh_failure_over_a_landed_bulk_still_claims():
    """The mirror of the 429 cases: the bulk LANDED and only
    ``indices.refresh`` failed, so ``index_done_callback`` raises
    ``OpenSearchReferencesIntactError`` over a durable write and an empty
    buffer. Reporting that as a failed claim would refuse a startup, and fail a
    rebuild, over a record the server already has — a durable write reported as
    one that did not happen."""
    server = _FakeServer(rate_limited=False)
    server.client.indices.refresh = AsyncMock(
        side_effect=OpenSearchException("refresh unavailable")
    )
    storage = await _config_storage(server)

    with patch("lightrag.kg.opensearch_impl.helpers.async_bulk", new=server.bulk):
        baseline = await cs.claim_embedding_baseline(
            storage,
            workspace=WORKSPACE,
            target="entities",
            candidate=cs.EmbeddingBaseline("bge-m3", 8, "empty"),
            embedding_func=_embedding(),
        )

    assert baseline.model == "bge-m3"
    key = cs.embedding_baseline_key(WORKSPACE, "entities")
    assert server.docs[key]["value"]["model"] == "bge-m3", "the server has the row"
    assert await storage.has_pending_index_ops(include_deletes=True) is False
