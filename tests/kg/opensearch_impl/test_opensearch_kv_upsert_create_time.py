"""Regression tests: OpenSearchKVStorage preserves create_time server-side.

Issue #3870: ``upsert`` used ``setdefault("create_time", now)``, so an update
whose payload carried business fields only RESET the field to the update time.

The fix does not read the stored row back — that would cost an HTTP round trip
per ``upsert()`` call, and this backend is deliberately called with many small
batches. Instead the flush emits a ``scripted_upsert`` bulk action whose
painless script restores the stored ``create_time`` after replacing the
business value, reproducing Mongo's ``$setOnInsert`` semantics on the server.

These tests pin the two halves that are observable without a cluster: that
``upsert`` reads nothing, and that the flush emits the scripted action. The
script's own semantics are verified against a real OpenSearch in
``test_opensearch_kv_create_time_integration.py``.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightrag.kg.opensearch_impl import (
    _KV_CREATE_TIME_SENTINEL,
    ClientManager,
    OpenSearchKVStorage,
)

pytestmark = pytest.mark.offline


class _DummyEmbeddingFunc:
    embedding_dim = 1
    max_token_size = 1

    async def __call__(self, texts, **kwargs):
        return [[0.0] for _ in texts]


@asynccontextmanager
async def _mock_lock():
    yield


@pytest.fixture
def global_config() -> dict[str, Any]:
    return {
        "embedding_batch_num": 10,
        "max_graph_nodes": 1000,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
    }


@pytest.fixture
def mock_client() -> AsyncMock:
    client = AsyncMock()
    client.indices = AsyncMock()
    client.indices.exists = AsyncMock(return_value=True)
    client.indices.get_mapping = AsyncMock(
        return_value={
            "ct_text_chunks": {
                "mappings": {
                    "_meta": {
                        "lightrag_workspace": "ct",
                        "lightrag_final_namespace": "ct_text_chunks",
                    },
                    "properties": {"__mirrored_id": {"type": "keyword"}},
                }
            }
        }
    )
    client.indices.refresh = AsyncMock(return_value={})
    return client


@pytest.fixture
def storage(global_config, mock_client):
    """A KV storage wired to the mock client, locks stubbed out."""
    with (
        patch.object(ClientManager, "get_client", return_value=mock_client),
        patch("lightrag.kg.opensearch_impl.get_data_init_lock", _mock_lock),
        patch(
            "lightrag.kg.opensearch_impl.get_namespace_lock",
            MagicMock(return_value=_mock_lock()),
        ),
    ):
        s = OpenSearchKVStorage(
            namespace="text_chunks",
            global_config=global_config,
            embedding_func=_DummyEmbeddingFunc(),
            workspace="ct",
        )
        s.client = mock_client
        s._index_ready = True
        s._flush_lock = _MockLock()
        yield s


class _MockLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _upsert_actions(mock_bulk: AsyncMock) -> list[dict[str, Any]]:
    """The index (non-delete) actions of the last flush."""
    actions: list[dict[str, Any]] = []
    for call in mock_bulk.await_args_list:
        for action in call.args[1]:
            if action.get("_op_type") != "delete":
                actions.append(action)
    return actions


@pytest.mark.asyncio
async def test_upsert_reads_nothing(storage, mock_client):
    """The whole point of the scripted upsert: no read on the write path.

    A per-upsert lookup would also have to be held under ``_flush_lock`` (the
    cross-process namespace lock), turning every one of the many small
    ``upsert()`` calls into a network round trip inside that lock.
    """
    with patch("time.time", return_value=1_700_000_000):
        await storage.upsert({"k1": {"content": "a"}})
        await storage.upsert({"k1": {"content": "b"}, "k2": {"content": "c"}})

    mock_client.mget.assert_not_awaited()
    mock_client.get.assert_not_awaited()
    mock_client.search.assert_not_awaited()


@pytest.mark.asyncio
async def test_flush_emits_scripted_upsert_action(storage):
    with patch("time.time", return_value=1_700_000_000):
        await storage.upsert({"k1": {"content": "a", "tokens": 3}})

    with patch(
        "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
    ) as mock_bulk:
        mock_bulk.return_value = (1, [])
        await storage._flush_pending_kv_ops()

    actions = _upsert_actions(mock_bulk)
    assert len(actions) == 1
    action = actions[0]
    assert action["_op_type"] == "update"
    assert action["_id"] == "k1"
    assert action["scripted_upsert"] is True
    assert action["upsert"] == {_KV_CREATE_TIME_SENTINEL: True}
    assert action["retry_on_conflict"] >= 1
    # The replacement value travels as script params, not as _source.
    assert "_source" not in action
    doc = action["script"]["params"]["doc"]
    assert doc["content"] == "a"
    assert doc["tokens"] == 3
    assert doc["__mirrored_id"] == "k1"
    assert doc["update_time"] == 1_700_000_000
    # ...and the script is what restores the stored timestamp.
    source = action["script"]["source"]
    assert "create_time" in source
    assert _KV_CREATE_TIME_SENTINEL in source


@pytest.mark.asyncio
async def test_buffered_create_time_is_the_write_time(storage):
    """Documented residue: a buffered read reports the optimistic estimate.

    ``create_time`` is authoritative only once the flush script has run, so a
    read served from the buffer shows the write time — right for an insert,
    and superseded at flush for a row that already exists on the server.
    """
    with patch("time.time", return_value=1_700_000_000):
        await storage.upsert({"k1": {"content": "a"}})

    buffered = await storage.get_by_id("k1")
    assert buffered["create_time"] == 1_700_000_000
    assert buffered["update_time"] == 1_700_000_000


@pytest.mark.asyncio
async def test_caller_supplied_create_time_is_overwritten(storage):
    """A caller cannot smuggle a create_time past the storage."""
    with patch("time.time", return_value=1_700_000_000):
        await storage.upsert({"k1": {"content": "a", "create_time": 1}})

    with patch(
        "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
    ) as mock_bulk:
        mock_bulk.return_value = (1, [])
        await storage._flush_pending_kv_ops()

    doc = _upsert_actions(mock_bulk)[0]["script"]["params"]["doc"]
    assert doc["create_time"] == 1_700_000_000


@pytest.mark.asyncio
async def test_upsert_after_delete_flushes_only_the_upsert(storage):
    """delete-then-upsert still cancels the tombstone, as before the fix."""
    await storage.delete(["k1"])
    with patch("time.time", return_value=1_700_000_400):
        await storage.upsert({"k1": {"content": "fresh"}})

    with patch(
        "lightrag.kg.opensearch_impl.helpers.async_bulk", new_callable=AsyncMock
    ) as mock_bulk:
        mock_bulk.return_value = (1, [])
        await storage._flush_pending_kv_ops()

    deletes = [
        action
        for call in mock_bulk.await_args_list
        for action in call.args[1]
        if action.get("_op_type") == "delete"
    ]
    assert deletes == []
    actions = _upsert_actions(mock_bulk)
    assert len(actions) == 1
    assert actions[0]["script"]["params"]["doc"]["create_time"] == 1_700_000_400
