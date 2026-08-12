from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("pymongo", reason="pymongo is required for DocumentDB tests")

from lightrag.kg.documentdb_impl import (
    DocumentDBClientManager,
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
    assert (
        get_storage_class("DocumentDBVectorDBStorage")
        is DocumentDBVectorDBStorage
    )


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
async def test_documentdb_vector_index_uses_create_indexes_command():
    storage = DocumentDBVectorDBStorage.__new__(DocumentDBVectorDBStorage)
    storage.workspace = "test"
    storage._collection_name = "test_entities"
    storage._index_name = "vector_knn_index_test_entities"
    storage.embedding_func = SimpleNamespace(embedding_dim=1536)
    storage._data = SimpleNamespace(list_indexes=AsyncMock(return_value=_AsyncCursor([])))
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
async def test_documentdb_vector_query_omits_atlas_index_name():
    aggregate = AsyncMock(return_value=_AsyncCursor([]))
    storage = DocumentDBVectorDBStorage.__new__(DocumentDBVectorDBStorage)
    storage._data = SimpleNamespace(aggregate=aggregate)
    storage._index_name = "atlas_only_name"
    storage.cosine_better_than_threshold = 0.2

    await storage.query("query", top_k=5, query_embedding=[0.1, 0.2])

    pipeline = aggregate.await_args.args[0]
    assert pipeline[0]["$vectorSearch"] == {
        "path": "vector",
        "queryVector": [0.1, 0.2],
        "numCandidates": 100,
        "limit": 5,
    }