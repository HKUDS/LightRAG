import pytest
import numpy as np
from unittest.mock import patch, AsyncMock
from lightrag.utils import EmbeddingFunc
from lightrag.kg.postgres_impl import (
    PGVectorStorage,
    PostgreSQLDB,
    _safe_index_name,
)
from lightrag.exceptions import DataMigrationError
from lightrag.namespace import NameSpace


# Mock PostgreSQLDB
@pytest.fixture
def mock_pg_db():
    """Mock PostgreSQL database connection"""
    db = AsyncMock()
    db.workspace = "test_workspace"
    db.vector_index_type = None

    # Mock query responses: list for search queries (multirows=True), dict for DDL checks
    async def mock_query(sql, params=None, multirows=False, **kwargs):
        if multirows:
            return []
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
        call
        for call in mock_pg_db.execute.call_args_list
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
        call
        for call in mock_pg_db.execute.call_args_list
        if "CREATE TABLE" in call[0][0]
    ]

    assert len(create_table_calls) > 0
    create_sql = create_table_calls[0][0][0]
    assert "VECTOR(768)" in create_sql
    assert "HALFVEC(768)" not in create_sql


# Namespaces that use vector search SQL templates (query path)
QUERY_NAMESPACES = [
    NameSpace.VECTOR_STORE_CHUNKS,
    NameSpace.VECTOR_STORE_ENTITIES,
    NameSpace.VECTOR_STORE_RELATIONSHIPS,
]


@pytest.mark.asyncio
@pytest.mark.parametrize("namespace", QUERY_NAMESPACES)
async def test_query_uses_halfvec_cast_when_hnsw_halfvec(
    mock_client_manager, mock_pg_db, mock_embedding_func, namespace
):
    """When HNSW_HALFVEC is set, generated search SQL uses ::halfvec (not ::vector)."""
    mock_pg_db.vector_index_type = "HNSW_HALFVEC"
    mock_pg_db.check_table_exists = AsyncMock(return_value=True)

    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
    }
    storage = PGVectorStorage(
        namespace=namespace,
        global_config=config,
        embedding_func=mock_embedding_func,
        workspace="test_ws",
    )
    await storage.initialize()

    query_embedding = [0.1] * 768
    await storage.query("test query", top_k=5, query_embedding=query_embedding)

    assert mock_pg_db.query.called
    call_args = mock_pg_db.query.call_args
    sql = call_args[0][0]
    assert "::halfvec" in sql
    assert "::vector" not in sql


@pytest.mark.asyncio
@pytest.mark.parametrize("namespace", QUERY_NAMESPACES)
async def test_query_uses_vector_cast_when_hnsw_default(
    mock_client_manager, mock_pg_db, mock_embedding_func, namespace
):
    """When HNSW (default) is set, generated search SQL uses ::vector (not ::halfvec)."""
    mock_pg_db.vector_index_type = "HNSW"
    mock_pg_db.check_table_exists = AsyncMock(return_value=True)

    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
    }
    storage = PGVectorStorage(
        namespace=namespace,
        global_config=config,
        embedding_func=mock_embedding_func,
        workspace="test_ws",
    )
    await storage.initialize()

    query_embedding = [0.1] * 768
    await storage.query("test query", top_k=5, query_embedding=query_embedding)

    assert mock_pg_db.query.called
    call_args = mock_pg_db.query.call_args
    sql = call_args[0][0]
    assert "::vector" in sql
    assert "::halfvec" not in sql


# ---------------------------------------------------------------------------
# Index switching: old conflicting indexes are dropped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_vector_index_drops_old_indexes_when_switching(mock_pg_db):
    """Switching from HNSW to HNSW_HALFVEC drops the old hnsw_cosine index."""
    mock_pg_db.vector_index_type = "HNSW_HALFVEC"
    mock_pg_db.hnsw_m = 16
    mock_pg_db.hnsw_ef = 64
    mock_pg_db.ivfflat_lists = 100
    mock_pg_db.vchordrq_build_options = ""

    table_name = "lightrag_vdb_chunks_test"

    async def mock_query(sql, params=None, multirows=False, **kwargs):
        if "pg_indexes" in sql:
            return None
        return None

    mock_pg_db.query = AsyncMock(side_effect=mock_query)
    mock_pg_db.execute = AsyncMock()

    # Call the real method with mock_pg_db as self
    await PostgreSQLDB._create_vector_index(mock_pg_db, table_name, 3072)

    execute_calls = [call[0][0] for call in mock_pg_db.execute.call_args_list]

    old_hnsw_name = _safe_index_name(table_name, "hnsw_cosine")
    old_ivfflat_name = _safe_index_name(table_name, "ivfflat_cosine")
    old_vchordrq_name = _safe_index_name(table_name, "vchordrq_cosine")

    drop_calls = [c for c in execute_calls if "DROP INDEX IF EXISTS" in c]
    dropped_names = {c.split("DROP INDEX IF EXISTS ")[1].strip() for c in drop_calls}
    assert old_hnsw_name in dropped_names
    assert old_ivfflat_name in dropped_names
    assert old_vchordrq_name in dropped_names

    new_index_name = _safe_index_name(table_name, "hnsw_halfvec_cosine")
    assert new_index_name not in dropped_names

    alter_calls = [c for c in execute_calls if "ALTER TABLE" in c]
    assert any("HALFVEC(3072)" in c for c in alter_calls)

    create_calls = [c for c in execute_calls if "CREATE INDEX" in c]
    assert any("halfvec_cosine_ops" in c for c in create_calls)


@pytest.mark.asyncio
async def test_create_vector_index_no_drop_when_index_exists(mock_pg_db):
    """If the target index already exists, no DROP or CREATE is issued."""
    mock_pg_db.vector_index_type = "HNSW_HALFVEC"
    mock_pg_db.hnsw_m = 16
    mock_pg_db.hnsw_ef = 64
    mock_pg_db.ivfflat_lists = 100
    mock_pg_db.vchordrq_build_options = ""

    table_name = "lightrag_vdb_chunks_test"

    async def mock_query(sql, params=None, multirows=False, **kwargs):
        if "pg_indexes" in sql:
            return {"?column?": 1}
        return None

    mock_pg_db.query = AsyncMock(side_effect=mock_query)
    mock_pg_db.execute = AsyncMock()

    await PostgreSQLDB._create_vector_index(mock_pg_db, table_name, 3072)

    execute_calls = [call[0][0] for call in mock_pg_db.execute.call_args_list]
    assert not any("DROP INDEX" in c for c in execute_calls)
    assert not any("CREATE INDEX" in c for c in execute_calls)


# ---------------------------------------------------------------------------
# HalfVector dimension detection in setup_table
# ---------------------------------------------------------------------------


class _MockHalfVector:
    """Mimics pgvector.halfvec.HalfVector for testing dimension detection."""

    def __init__(self, dim: int):
        self._dim = dim

    def dimensions(self) -> int:
        return self._dim

    def to_list(self):
        return [0.0] * self._dim


@pytest.mark.asyncio
async def test_setup_table_detects_halfvector_dimension_mismatch(mock_pg_db):
    """DataMigrationError is raised when a HalfVector column has a different dimension."""
    table_name = "lightrag_vdb_chunks_new"
    legacy_table = "lightrag_vdb_chunks"

    mock_pg_db.check_table_exists = AsyncMock(
        side_effect=lambda t: t.lower() == legacy_table.lower()
    )

    call_count = 0

    async def mock_query(sql, params=None, multirows=False, **kwargs):
        nonlocal call_count
        call_count += 1
        if "COUNT(*)" in sql:
            return {"count": 5}
        if "content_vector" in sql:
            return {"content_vector": _MockHalfVector(1024)}
        return None

    mock_pg_db.query = AsyncMock(side_effect=mock_query)
    mock_pg_db.execute = AsyncMock()

    with pytest.raises(DataMigrationError, match="Dimension mismatch"):
        await PGVectorStorage.setup_table(
            db=mock_pg_db,
            table_name=table_name,
            workspace="test_ws",
            embedding_dim=768,
            legacy_table_name=legacy_table,
            base_table=legacy_table,
        )


@pytest.mark.asyncio
async def test_setup_table_accepts_matching_halfvector_dimension(mock_pg_db):
    """No error when HalfVector dimension matches the expected embedding_dim."""
    table_name = "lightrag_vdb_chunks_new"
    legacy_table = "lightrag_vdb_chunks"

    mock_pg_db.check_table_exists = AsyncMock(
        side_effect=lambda t: t.lower() == legacy_table.lower()
    )
    mock_pg_db.vector_index_type = "HNSW_HALFVEC"

    async def mock_query(sql, params=None, multirows=False, **kwargs):
        if "COUNT(*)" in sql:
            return {"count": 5}
        if "content_vector" in sql:
            return {"content_vector": _MockHalfVector(768)}
        if multirows:
            return []
        return None

    mock_pg_db.query = AsyncMock(side_effect=mock_query)
    mock_pg_db.execute = AsyncMock()

    with patch.object(PGVectorStorage, "_pg_create_table", new_callable=AsyncMock):
        await PGVectorStorage.setup_table(
            db=mock_pg_db,
            table_name=table_name,
            workspace="test_ws",
            embedding_dim=768,
            legacy_table_name=legacy_table,
            base_table=legacy_table,
        )
