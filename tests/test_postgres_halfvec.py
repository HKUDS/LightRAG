import pytest
import numpy as np
from unittest.mock import patch, AsyncMock
from lightrag.utils import EmbeddingFunc
from lightrag.kg.postgres_impl import (
    PGVectorStorage,
)
from lightrag.namespace import NameSpace


# Mock PostgreSQLDB
@pytest.fixture
def mock_pg_db():
    """Mock PostgreSQL database connection"""
    db = AsyncMock()
    db.workspace = "test_workspace"
    db.vector_index_type = None

    # Mock query responses with multirows support
    async def mock_query(sql, params=None, multirows=False, **kwargs):
        return {"exists": False, "count": 0}

    # Mock for execute that mimics PostgreSQLDB.execute() behavior
    async def mock_execute(sql, data=None, **kwargs):
        return None

    db.query = AsyncMock(side_effect=mock_query)
    db.execute = AsyncMock(side_effect=mock_execute)

    return db


# Mock get_data_init_lock to avoid async lock issues in tests
@pytest.fixture(autouse=True)
def mock_data_init_lock():
    with patch("lightrag.kg.postgres_impl.get_data_init_lock") as mock_lock:
        mock_lock_ctx = AsyncMock()
        mock_lock.return_value = mock_lock_ctx
        yield mock_lock


# Mock ClientManager
@pytest.fixture
def mock_client_manager(mock_pg_db):
    with patch("lightrag.kg.postgres_impl.ClientManager") as mock_manager:
        mock_manager.get_client = AsyncMock(return_value=mock_pg_db)
        mock_manager.release_client = AsyncMock()
        yield mock_manager


# Mock Embedding function
@pytest.fixture
def mock_embedding_func():
    async def embed_func(texts, **kwargs):
        return np.array([[0.1] * 768 for _ in texts])

    # Note: EmbeddingFunc in this version of lightrag supports model_name
    func = EmbeddingFunc(embedding_dim=768, func=embed_func, model_name="test_model")
    return func


@pytest.mark.asyncio
async def test_postgres_halfvec_table_creation(
    mock_client_manager, mock_pg_db, mock_embedding_func
):
    """Test if table is created with HALFVEC type when HNSW_HALFVEC is selected"""
    # Set index type to HNSW_HALFVEC
    mock_pg_db.vector_index_type = "HNSW_HALFVEC"
    
    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
    }

    storage = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        global_config=config,
        embedding_func=mock_embedding_func,
        workspace="test_ws",
    )

    # Mock table doesn't exist
    mock_pg_db.check_table_exists = AsyncMock(return_value=False)

    # Initialize storage (should trigger table creation)
    await storage.initialize()

    # Verify table creation SQL contains HALFVEC(768)
    create_table_calls = [
        call for call in mock_pg_db.execute.call_args_list 
        if "CREATE TABLE" in call[0][0]
    ]
    
    assert len(create_table_calls) > 0
    create_sql = create_table_calls[0][0][0]
    assert "HALFVEC(768)" in create_sql
    assert "VECTOR(768)" not in create_sql


@pytest.mark.asyncio
async def test_postgres_vector_table_creation_default(
    mock_client_manager, mock_pg_db, mock_embedding_func
):
    """Test if table is created with default VECTOR type when other index type is selected"""
    # Set index type to HNSW (default)
    mock_pg_db.vector_index_type = "HNSW"
    
    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
    }

    storage = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        global_config=config,
        embedding_func=mock_embedding_func,
        workspace="test_ws",
    )

    # Mock table doesn't exist
    mock_pg_db.check_table_exists = AsyncMock(return_value=False)

    # Initialize storage (should trigger table creation)
    await storage.initialize()

    # Verify table creation SQL contains VECTOR(768)
    create_table_calls = [
        call for call in mock_pg_db.execute.call_args_list 
        if "CREATE TABLE" in call[0][0]
    ]
    
    assert len(create_table_calls) > 0
    create_sql = create_table_calls[0][0][0]
    assert "VECTOR(768)" in create_sql
    assert "HALFVEC(768)" not in create_sql
