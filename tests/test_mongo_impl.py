import pytest
import os
from unittest.mock import AsyncMock, patch
import numpy as np
from lightrag.kg.mongo_impl import MongoVectorDBStorage
from lightrag.utils import EmbeddingFunc


@pytest.fixture
def mock_embedding_func():
    async def embed_func(texts, **kwargs):
        return np.array([[0.1] * 768 for _ in texts], dtype=np.float32)

    return EmbeddingFunc(
        embedding_dim=768,
        func=embed_func,
        model_name="test_model",
    )


@pytest.fixture
def mongo_vector_storage(mock_embedding_func):
    storage = MongoVectorDBStorage(
        namespace="test_namespace",
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.0,
            },
        },
        embedding_func=mock_embedding_func,
        workspace="test_workspace",
    )
    storage._data = AsyncMock()
    storage._index_name = "test_index"
    return storage


@pytest.mark.asyncio
async def test_mongo_vector_query_basic(mongo_vector_storage):
    # Ensure hybrid search is disabled for this test
    with patch.dict(os.environ, {"ENABLE_HYBRID_SEARCH": "false"}):
        # Mock result from MongoDB
        mock_cursor = AsyncMock()
        mock_cursor.to_list.return_value = [
            {"_id": "doc1", "score": 0.9, "created_at": 12345},
            {"_id": "doc2", "score": 0.8, "created_at": 67890},
        ]
        mongo_vector_storage._data.aggregate.return_value = mock_cursor

        results = await mongo_vector_storage.query("test query", top_k=2)

        assert len(results) == 2
        assert results[0]["id"] == "doc1"
        assert results[0]["distance"] == 0.9
        assert results[0]["created_at"] == 12345

        # Verify aggregation pipeline
        args, kwargs = mongo_vector_storage._data.aggregate.call_args
        pipeline = args[0]
        assert "$vectorSearch" in pipeline[0]
        assert pipeline[0]["$vectorSearch"]["index"] == "test_index"
        assert pipeline[0]["$vectorSearch"]["limit"] == 2


@pytest.mark.asyncio
async def test_mongo_vector_query_hybrid(mongo_vector_storage):
    # Enable hybrid search via env var
    with patch.dict(os.environ, {"ENABLE_HYBRID_SEARCH": "true"}):
        mock_cursor = AsyncMock()
        mock_cursor.to_list.return_value = [
            {"_id": "hybrid1", "score": 0.95, "created_at": 111}
        ]
        mongo_vector_storage._data.aggregate.return_value = mock_cursor

        results = await mongo_vector_storage.query("hybrid query", top_k=1)

        assert len(results) == 1
        assert results[0]["id"] == "hybrid1"

        # Verify aggregation pipeline has hybrid stages
        args, kwargs = mongo_vector_storage._data.aggregate.call_args
        pipeline = args[0]
        # $vectorSearch must be first
        assert "$vectorSearch" in pipeline[0]
        # $match with regex should be after
        assert "$match" in pipeline[2]
        assert pipeline[2]["$match"]["$or"][0]["content"]["$regex"] == "hybrid query"


@pytest.mark.asyncio
async def test_mongo_vector_query_with_embedding(mongo_vector_storage):
    # Ensure hybrid search is disabled for this test
    with patch.dict(os.environ, {"ENABLE_HYBRID_SEARCH": "false"}):
        mock_cursor = AsyncMock()
        mock_cursor.to_list.return_value = []
        mongo_vector_storage._data.aggregate.return_value = mock_cursor

        custom_embedding = np.array([0.5] * 768, dtype=np.float32)
        await mongo_vector_storage.query(
            "ignored", top_k=5, query_embedding=custom_embedding
        )

        args, kwargs = mongo_vector_storage._data.aggregate.call_args
        pipeline = args[0]
        # Check if the custom embedding was used
        assert pipeline[0]["$vectorSearch"]["queryVector"] == custom_embedding.tolist()


@pytest.mark.asyncio
async def test_mongo_vector_init_fail(mongo_vector_storage):
    from pymongo.errors import OperationFailure

    # Mock list_search_indexes to raise an error
    error_response = {
        "ok": 0.0,
        "errmsg": "Error connecting to Search Index Management service",
        "code": 125,
    }
    mongo_vector_storage._data.list_search_indexes.side_effect = OperationFailure(
        error_response["errmsg"], error_response["code"], error_response
    )

    with pytest.raises(SystemExit) as excinfo:
        await mongo_vector_storage.create_vector_index_if_not_exists()

    assert "Failed to initialize MongoDB vector search" in str(excinfo.value)
