from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("pymongo", reason="pymongo is required for DocumentDB tests")

from lightrag.kg.documentdb_impl import (
    DocumentDBClientManager,
    DocumentDBDocStatusStorage,
    DocumentDBGraphStorage,
    DocumentDBKVStorage,
    DocumentDBVectorDBStorage,
)
from lightrag.kg.factory import get_storage_class
from lightrag.kg.mongo_impl import ClientManager

pytestmark = pytest.mark.offline


class _AsyncCursor:
    def __init__(self, rows):
        self.rows = rows

    async def to_list(self, length=None):
        return self.rows


def test_documentdb_backends_resolve_from_storage_factory():
    assert get_storage_class("DocumentDBKVStorage") is DocumentDBKVStorage
    assert get_storage_class("DocumentDBGraphStorage") is DocumentDBGraphStorage
    assert get_storage_class("DocumentDBVectorDBStorage") is DocumentDBVectorDBStorage


def test_documentdb_client_configuration_is_isolated_from_mongodb():
    assert DocumentDBClientManager._instances is not ClientManager._instances
    assert DocumentDBClientManager.uri_env_var == "DOCUMENTDB_URI"
    assert DocumentDBClientManager.database_env_var == "DOCUMENTDB_DATABASE"
    assert DocumentDBKVStorage.client_manager is DocumentDBClientManager


def test_documentdb_workspace_override(monkeypatch):
    monkeypatch.setenv("DOCUMENTDB_WORKSPACE", "documentdb_space")
    monkeypatch.setenv("MONGODB_WORKSPACE", "mongo_space")

    storage = DocumentDBKVStorage(
        namespace="full_docs",
        global_config={},
        embedding_func=None,
        workspace="requested_space",
    )

    assert storage.workspace == "documentdb_space"
    assert storage.final_namespace == "documentdb_space_full_docs"


@pytest.mark.asyncio
async def test_documentdb_file_path_pagination_omits_query_collation():
    cursor = MagicMock()
    cursor.sort.return_value = cursor
    cursor.skip.return_value = cursor
    cursor.limit.return_value = cursor
    cursor.to_list = AsyncMock(return_value=[])

    collection = SimpleNamespace(
        count_documents=AsyncMock(return_value=0),
        find=MagicMock(return_value=cursor),
    )
    storage = DocumentDBDocStatusStorage.__new__(DocumentDBDocStatusStorage)
    storage._data = collection

    documents, total_count = await storage.get_docs_paginated(sort_field="file_path")

    assert documents == []
    assert total_count == 0
    cursor.sort.assert_called_once_with([("file_path", -1)])
    cursor.collation.assert_not_called()


@pytest.mark.asyncio
async def test_documentdb_edge_migration_uses_partial_unique_index():
    storage = DocumentDBGraphStorage.__new__(DocumentDBGraphStorage)
    storage.workspace = "test"
    storage._edge_collection_name = "test_graph_edges"
    storage.edge_collection = SimpleNamespace(
        list_indexes=AsyncMock(return_value=_AsyncCursor([])),
        estimated_document_count=AsyncMock(return_value=2),
        create_index=AsyncMock(),
    )
    storage._dedupe_legacy_edges = AsyncMock(return_value=0)
    storage._backfill_edge_endpoints = AsyncMock(return_value=2)

    await storage.create_edge_indexes_and_migrate_if_not_exists()

    storage.edge_collection.create_index.assert_awaited_once_with(
        [("edge_lo", 1), ("edge_hi", 1)],
        name="test_edge_endpoints_unique",
        unique=True,
        partialFilterExpression={
            "edge_lo": {"$exists": True, "$type": "string"},
            "edge_hi": {"$exists": True, "$type": "string"},
        },
    )


@pytest.mark.asyncio
async def test_documentdb_label_search_goes_directly_to_regex():
    storage = DocumentDBGraphStorage.__new__(DocumentDBGraphStorage)
    storage.collection = SimpleNamespace(
        count_documents=AsyncMock(return_value=1),
    )
    storage._try_atlas_text_search = AsyncMock()
    storage._try_atlas_autocomplete_search = AsyncMock()
    storage._try_atlas_compound_search = AsyncMock()
    storage._fallback_regex_search = AsyncMock(return_value=["Alpha"])

    assert await storage.search_labels(" alpha ", limit=5) == ["Alpha"]

    storage._fallback_regex_search.assert_awaited_once_with("alpha", 5)
    storage._try_atlas_text_search.assert_not_awaited()
    storage._try_atlas_autocomplete_search.assert_not_awaited()
    storage._try_atlas_compound_search.assert_not_awaited()


@pytest.mark.asyncio
async def test_documentdb_vector_index_uses_create_indexes_command():
    storage = DocumentDBVectorDBStorage.__new__(DocumentDBVectorDBStorage)
    storage.workspace = "test"
    storage._collection_name = "test_entities"
    storage._index_name = "vector_knn_index_test_entities"
    storage.embedding_func = SimpleNamespace(embedding_dim=1536)
    storage._data = SimpleNamespace(
        list_indexes=AsyncMock(return_value=_AsyncCursor([]))
    )
    storage.db = SimpleNamespace(command=AsyncMock())

    await storage.create_vector_index_if_not_exists()

    storage.db.command.assert_awaited_once_with(
        {
            "createIndexes": "test_entities",
            "indexes": [
                {
                    "key": {"vector": "cosmosSearch"},
                    "name": "vector_knn_index_test_entities",
                    "cosmosSearchOptions": {
                        "kind": "vector-hnsw",
                        "m": 16,
                        "efConstruction": 64,
                        "similarity": "COS",
                        "dimensions": 1536,
                    },
                }
            ],
        }
    )


@pytest.mark.asyncio
async def test_documentdb_vector_index_rejects_dimension_mismatch():
    storage = DocumentDBVectorDBStorage.__new__(DocumentDBVectorDBStorage)
    storage.workspace = "test"
    storage._collection_name = "test_entities"
    storage._index_name = "vector_knn_index_test_entities"
    storage.embedding_func = SimpleNamespace(embedding_dim=1536)
    storage._data = SimpleNamespace(
        list_indexes=AsyncMock(
            return_value=_AsyncCursor(
                [
                    {
                        "name": storage._index_name,
                        "cosmosSearchOptions": {"dimensions": 768},
                    }
                ]
            )
        )
    )
    storage.db = SimpleNamespace(command=AsyncMock())

    with pytest.raises(ValueError, match="Vector dimension mismatch"):
        await storage.create_vector_index_if_not_exists()

    storage.db.command.assert_not_awaited()


@pytest.mark.asyncio
async def test_documentdb_vector_query_uses_cosmos_search():
    aggregate = AsyncMock(return_value=_AsyncCursor([]))
    storage = DocumentDBVectorDBStorage.__new__(DocumentDBVectorDBStorage)
    storage._data = SimpleNamespace(aggregate=aggregate)
    storage._index_name = "atlas_only_name"
    storage.cosine_better_than_threshold = 0.2

    await storage.query("query", top_k=5, query_embedding=[0.1, 0.2])

    pipeline = aggregate.await_args.args[0]
    assert pipeline[0] == {
        "$search": {
            "cosmosSearch": {
                "path": "vector",
                "vector": [0.1, 0.2],
                "k": 5,
            }
        }
    }
    assert pipeline[1] == {"$addFields": {"score": {"$meta": "searchScore"}}}
