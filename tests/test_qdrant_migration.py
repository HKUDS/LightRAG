import os
import pytest
from unittest.mock import MagicMock, patch, AsyncMock, call
import numpy as np
from lightrag.utils import EmbeddingFunc
from lightrag.kg.qdrant_impl import QdrantVectorDBStorage, compute_mdhash_id_for_qdrant

# Mock QdrantClient
@pytest.fixture
def mock_qdrant_client():
    with patch("lightrag.kg.qdrant_impl.QdrantClient") as mock_client_cls:
        client = mock_client_cls.return_value
        client.collection_exists.return_value = False
        client.count.return_value.count = 0
        # Mock payload schema for get_collection
        collection_info = MagicMock()
        collection_info.payload_schema = {}
        client.get_collection.return_value = collection_info
        yield client

# Mock get_data_init_lock to avoid async lock issues in tests
@pytest.fixture(autouse=True)
def mock_data_init_lock():
    with patch("lightrag.kg.qdrant_impl.get_data_init_lock") as mock_lock:
        mock_lock_ctx = AsyncMock()
        mock_lock.return_value = mock_lock_ctx
        yield mock_lock

# Mock Embedding function
@pytest.fixture
def mock_embedding_func():
    async def embed_func(texts, **kwargs):
        return np.array([[0.1] * 768 for _ in texts])
    
    func = EmbeddingFunc(
        embedding_dim=768,
        func=embed_func,
        model_name="test-model"
    )
    return func

@pytest.mark.asyncio
async def test_qdrant_collection_naming(mock_qdrant_client, mock_embedding_func):
    """Test if collection name is correctly generated with model suffix"""
    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {
            "cosine_better_than_threshold": 0.8
        }
    }
    
    storage = QdrantVectorDBStorage(
        namespace="chunks",
        global_config=config,
        embedding_func=mock_embedding_func,
        workspace="test_ws"
    )
    
    # Verify collection name contains model suffix
    expected_suffix = "test_model_768d"
    assert expected_suffix in storage.final_namespace
    assert storage.final_namespace == f"lightrag_vdb_chunks_{expected_suffix}"
    
    # Verify legacy namespace
    assert storage.legacy_namespace == "test_ws_chunks"

@pytest.mark.asyncio
async def test_qdrant_migration_trigger(mock_qdrant_client, mock_embedding_func):
    """Test if migration logic is triggered correctly"""
    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {
            "cosine_better_than_threshold": 0.8
        }
    }
    
    storage = QdrantVectorDBStorage(
        namespace="chunks",
        global_config=config,
        embedding_func=mock_embedding_func,
        workspace="test_ws"
    )
    
    # Setup mocks for migration scenario
    # 1. New collection does not exist
    mock_qdrant_client.collection_exists.side_effect = lambda name: name == storage.legacy_namespace
    
    # 2. Legacy collection exists and has data
    mock_qdrant_client.count.return_value.count = 100
    
    # 3. Mock scroll for data migration
    from qdrant_client import models
    mock_point = MagicMock()
    mock_point.id = "old_id"
    mock_point.vector = [0.1] * 768
    mock_point.payload = {"content": "test"}
    
    # First call returns points, second call returns empty (end of scroll)
    mock_qdrant_client.scroll.side_effect = [
        ([mock_point], "next_offset"), 
        ([], None)
    ]
    
    # Initialize storage (triggers migration)
    await storage.initialize()
    
    # Verify migration steps
    # 1. Legacy count checked
    mock_qdrant_client.count.assert_any_call(
        collection_name=storage.legacy_namespace, 
        exact=True
    )
    
    # 2. New collection created
    mock_qdrant_client.create_collection.assert_called()
    
    # 3. Data scrolled from legacy
    assert mock_qdrant_client.scroll.call_count >= 1
    call_args = mock_qdrant_client.scroll.call_args_list[0]
    assert call_args.kwargs['collection_name'] == storage.legacy_namespace
    assert call_args.kwargs['limit'] == 500
    
    # 4. Data upserted to new
    mock_qdrant_client.upsert.assert_called()
    
    # 5. Payload index created
    mock_qdrant_client.create_payload_index.assert_called()

@pytest.mark.asyncio
async def test_qdrant_no_migration_needed(mock_qdrant_client, mock_embedding_func):
    """Test scenario where new collection already exists"""
    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {
            "cosine_better_than_threshold": 0.8
        }
    }
    
    storage = QdrantVectorDBStorage(
        namespace="chunks",
        global_config=config,
        embedding_func=mock_embedding_func,
        workspace="test_ws"
    )
    
    # New collection exists and Legacy exists (warning case)
    # or New collection exists and Legacy does not exist (normal case)
    # Mocking case where both exist to test logic flow but without migration
    
    # Logic in code: 
    # Case 1: Both exist -> Warning only
    # Case 2: Only new exists -> Ensure index
    
    # Let's test Case 2: Only new collection exists
    mock_qdrant_client.collection_exists.side_effect = lambda name: name == storage.final_namespace
    
    # Initialize
    await storage.initialize()
    
    # Should check index but NOT migrate
    # In Qdrant implementation, Case 2 calls get_collection
    mock_qdrant_client.get_collection.assert_called_with(storage.final_namespace)
    mock_qdrant_client.scroll.assert_not_called()
