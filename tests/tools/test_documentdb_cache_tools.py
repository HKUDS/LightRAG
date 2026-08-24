"""DocumentDB coverage for storage-facing maintenance tools."""

from unittest.mock import AsyncMock

import pytest

from lightrag.kg.documentdb_impl import DocumentDBKVStorage as StorageClass
from lightrag.tools.clean_llm_query_cache import (
    STORAGE_TYPES as CLEANUP_STORAGE_TYPES,
)
from lightrag.tools.clean_llm_query_cache import (
    CleanupStats,
    CleanupTool,
)
from lightrag.tools.migrate_llm_cache import (
    STORAGE_TYPES as MIGRATION_STORAGE_TYPES,
)
from lightrag.tools.migrate_llm_cache import (
    MigrationTool,
)
from lightrag.tools.rebuild_vdb import enumerate_kv_keys

pytestmark = pytest.mark.offline


class _AsyncCursor:
    def __init__(self, documents):
        self._documents = iter(documents)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._documents)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _Collection:
    def find(self, query, projection):
        assert query == {}
        assert projection == {"_id": 1}
        return _AsyncCursor([{"_id": "chunk-1"}, {"_id": "chunk-2"}])


class DocumentDBKVStorage:
    def __init__(self):
        self._data = _Collection()


def test_documentdb_is_available_to_cache_tools(monkeypatch):
    assert "DocumentDBKVStorage" in CLEANUP_STORAGE_TYPES.values()
    assert "DocumentDBKVStorage" in MIGRATION_STORAGE_TYPES.values()

    monkeypatch.setenv("WORKSPACE", "common")
    monkeypatch.setenv("DOCUMENTDB_WORKSPACE", "documentdb")

    cleanup = CleanupTool()
    migration = MigrationTool()
    assert cleanup.get_workspace_for_storage("DocumentDBKVStorage") == "documentdb"
    assert migration.get_workspace_for_storage("DocumentDBKVStorage") == "documentdb"
    assert cleanup.get_storage_class("DocumentDBKVStorage") is StorageClass
    assert migration.get_storage_class("DocumentDBKVStorage") is StorageClass


def test_documentdb_config_ini_is_detected(tmp_path, monkeypatch):
    (tmp_path / "config.ini").write_text(
        "[documentdb]\nuri=mongodb://localhost:10260/\ndatabase=LightRAG\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    assert CleanupTool().check_config_ini_for_storage("DocumentDBKVStorage")
    assert MigrationTool().check_config_ini_for_storage("DocumentDBKVStorage")


@pytest.mark.asyncio
async def test_rebuild_enumerates_documentdb_keys():
    assert await enumerate_kv_keys(DocumentDBKVStorage()) == ["chunk-1", "chunk-2"]


@pytest.mark.asyncio
async def test_cleanup_routes_documentdb_through_mongo_wire_helpers():
    tool = CleanupTool()
    tool.count_query_caches_mongo = AsyncMock(return_value={"mix": {"query": 1}})
    tool.delete_query_caches_mongo = AsyncMock()
    storage = object()
    stats = CleanupStats()

    result = await tool.count_query_caches(storage, "DocumentDBKVStorage")
    await tool.delete_query_caches(storage, "DocumentDBKVStorage", "all", stats)

    assert result == {"mix": {"query": 1}}
    tool.count_query_caches_mongo.assert_awaited_once_with(storage)
    tool.delete_query_caches_mongo.assert_awaited_once_with(storage, "all", stats)


@pytest.mark.asyncio
async def test_migration_routes_documentdb_through_mongo_wire_helpers():
    tool = MigrationTool()
    tool.get_default_caches_mongo = AsyncMock(return_value={"default:extract:1": {}})
    tool.count_default_caches_mongo = AsyncMock(return_value=1)
    storage = object()

    records = await tool.get_default_caches(storage, "DocumentDBKVStorage")
    count = await tool.count_default_caches(storage, "DocumentDBKVStorage")

    assert records == {"default:extract:1": {}}
    assert count == 1
    tool.get_default_caches_mongo.assert_awaited_once_with(storage)
    tool.count_default_caches_mongo.assert_awaited_once_with(storage)


@pytest.mark.asyncio
async def test_migration_streams_documentdb_through_mongo_wire_helper():
    tool = MigrationTool()
    storage = object()

    async def stream_mongo(received_storage, batch_size):
        assert received_storage is storage
        assert batch_size == 2
        yield {"default:extract:1": {}}

    tool.stream_default_caches_mongo = stream_mongo

    batches = [
        batch
        async for batch in tool.stream_default_caches(
            storage, "DocumentDBKVStorage", batch_size=2
        )
    ]

    assert batches == [{"default:extract:1": {}}]
