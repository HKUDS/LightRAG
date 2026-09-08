"""Regression tests: OpenSearchKVStorage must not reset create_time on update.

Issue #3870 — setdefault(create_time, current_time) on every upsert reset the
field whenever the caller omitted it. Updates must preserve the existing
storage-managed create_time (buffer or index); legacy rows missing the field
keep 0.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip(
    "opensearchpy",
    reason="opensearchpy is required for OpenSearch storage tests",
)

from lightrag.kg.opensearch_impl import ClientManager, OpenSearchKVStorage

pytestmark = pytest.mark.offline


@asynccontextmanager
async def _mock_lock():
    yield


def _mock_lock_factory():
    return _mock_lock()


@pytest.fixture(autouse=True)
def patch_data_init_lock():
    with patch(
        "lightrag.kg.opensearch_impl.get_data_init_lock", side_effect=_mock_lock_factory
    ):
        yield


@pytest.fixture(autouse=True)
def patch_namespace_lock():
    cache: dict[tuple[str, str | None], asyncio.Lock] = {}

    def factory(namespace, workspace=None, enable_logging=False):
        key = (namespace, workspace or "")
        lock = cache.get(key)
        if lock is None:
            lock = asyncio.Lock()
            cache[key] = lock
        return lock

    with patch("lightrag.kg.opensearch_impl.get_namespace_lock", side_effect=factory):
        yield


@pytest.fixture
def embed_func():
    class _Embed:
        embedding_dim = 1
        max_token_size = 1

        async def __call__(self, texts, **kwargs):
            return [[0.0] for _ in texts]

    return _Embed()


@pytest.fixture
def global_config(tmp_path):
    return {
        "working_dir": str(tmp_path),
        "embedding_batch_num": 1,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
    }


async def _mget_not_found(index=None, body=None, **kwargs):
    ids = (body or {}).get("ids") or []
    return {"docs": [{"_id": doc_id, "found": False} for doc_id in ids]}


@pytest.fixture
def mock_client():
    client = AsyncMock()
    client.indices.exists = AsyncMock(return_value=False)
    client.indices.create = AsyncMock(return_value={"acknowledged": True})
    client.indices.get_mapping = AsyncMock(
        return_value={
            "test_text_chunks": {
                "mappings": {
                    "properties": {
                        "content": {"type": "text"},
                        "__mirrored_id": {"type": "keyword"},
                    }
                }
            }
        }
    )
    client.mget = AsyncMock(side_effect=_mget_not_found)
    return client


def _make(global_config, embed_func, workspace="test"):
    return OpenSearchKVStorage(
        namespace="text_chunks",
        global_config=global_config,
        embedding_func=embed_func,
        workspace=workspace,
    )


@pytest.mark.asyncio
async def test_replacement_upsert_preserves_create_time_from_buffer(
    global_config, embed_func, mock_client
):
    with patch.object(ClientManager, "get_client", return_value=mock_client):
        s = _make(global_config, embed_func)
        await s.initialize()

        with patch("time.time", return_value=1_700_000_000):
            await s.upsert({"E": {"chunk_ids": ["c1"], "count": 1}})

        first = s._pending_upserts["E"]
        assert first["create_time"] == 1_700_000_000
        assert first["update_time"] == 1_700_000_000

        # Second upsert in the same buffer window must keep create_time.
        with patch("time.time", return_value=1_700_000_100):
            await s.upsert({"E": {"chunk_ids": ["c1", "c2"], "count": 2}})

        second = s._pending_upserts["E"]
        assert second["create_time"] == 1_700_000_000
        assert second["update_time"] == 1_700_000_100
        assert second["chunk_ids"] == ["c1", "c2"]
        assert second["count"] == 2
        # First insert resolves absence via mget; the replacement must not.
        assert mock_client.mget.await_count == 1


@pytest.mark.asyncio
async def test_replacement_upsert_preserves_create_time_from_index(
    global_config, embed_func, mock_client
):
    mock_client.indices.exists = AsyncMock(return_value=True)
    mock_client.mget = AsyncMock(
        return_value={
            "docs": [
                {
                    "_id": "E",
                    "found": True,
                    "_source": {"create_time": 1_650_000_000},
                }
            ]
        }
    )

    with patch.object(ClientManager, "get_client", return_value=mock_client):
        s = _make(global_config, embed_func)
        await s.initialize()

        with patch("time.time", return_value=1_700_000_100):
            await s.upsert({"E": {"chunk_ids": ["c1", "c2"], "count": 2}})

        pending = s._pending_upserts["E"]
        assert pending["create_time"] == 1_650_000_000
        assert pending["update_time"] == 1_700_000_100
        mock_client.mget.assert_awaited()


@pytest.mark.asyncio
async def test_legacy_index_row_missing_create_time_keeps_zero(
    global_config, embed_func, mock_client
):
    mock_client.indices.exists = AsyncMock(return_value=True)
    mock_client.mget = AsyncMock(
        return_value={
            "docs": [
                {
                    "_id": "legacy",
                    "found": True,
                    "_source": {"chunk_ids": ["c1"], "count": 1},
                }
            ]
        }
    )

    with patch.object(ClientManager, "get_client", return_value=mock_client):
        s = _make(global_config, embed_func)
        await s.initialize()

        with patch("time.time", return_value=1_700_000_200):
            await s.upsert({"legacy": {"chunk_ids": ["c1", "c2"], "count": 2}})

        pending = s._pending_upserts["legacy"]
        assert pending["create_time"] == 0
        assert pending["update_time"] == 1_700_000_200


@pytest.mark.asyncio
async def test_insert_stamps_create_time_when_key_absent(
    global_config, embed_func, mock_client
):
    mock_client.indices.exists = AsyncMock(return_value=True)
    mock_client.mget = AsyncMock(
        return_value={"docs": [{"_id": "new", "found": False}]}
    )

    with patch.object(ClientManager, "get_client", return_value=mock_client):
        s = _make(global_config, embed_func)
        await s.initialize()

        with patch("time.time", return_value=1_700_000_300):
            await s.upsert({"new": {"content": "v1"}})

        pending = s._pending_upserts["new"]
        assert pending["create_time"] == 1_700_000_300
        assert pending["update_time"] == 1_700_000_300


@pytest.mark.asyncio
async def test_delete_then_upsert_stamps_fresh_create_time(
    global_config, embed_func, mock_client
):
    with patch.object(ClientManager, "get_client", return_value=mock_client):
        s = _make(global_config, embed_func)
        await s.initialize()

        with patch("time.time", return_value=1_700_000_000):
            await s.upsert({"E": {"content": "old"}})
        await s.delete(["E"])

        with patch("time.time", return_value=1_700_000_400):
            await s.upsert({"E": {"content": "new"}})

        pending = s._pending_upserts["E"]
        assert pending["create_time"] == 1_700_000_400
        assert pending["update_time"] == 1_700_000_400
