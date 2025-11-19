"""
E2E Tests for PostgreSQL Vector Storage Model Isolation

These tests use a REAL PostgreSQL database with pgvector extension.
Unlike unit tests, these verify actual database operations, data migration,
and multi-model isolation scenarios.

Prerequisites:
- PostgreSQL with pgvector extension
- Environment variables: POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB
"""

import os
import pytest
import asyncio
import numpy as np
from lightrag.utils import EmbeddingFunc
from lightrag.kg.postgres_impl import PGVectorStorage, PostgreSQLDB, ClientManager
from lightrag.namespace import NameSpace


# E2E test configuration from environment
@pytest.fixture(scope="module")
def pg_config():
    """Real PostgreSQL configuration from environment variables"""
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "user": os.getenv("POSTGRES_USER", "lightrag"),
        "password": os.getenv("POSTGRES_PASSWORD", "lightrag_test_password"),
        "database": os.getenv("POSTGRES_DB", "lightrag_test"),
        "workspace": os.getenv("POSTGRES_WORKSPACE", "e2e_test"),
        "max_connections": 10,
    }


@pytest.fixture(scope="module")
async def real_db(pg_config):
    """Create a real PostgreSQL database connection"""
    db = PostgreSQLDB(pg_config)
    await db.initdb()
    yield db
    # Cleanup: close connection pool
    if db.pool:
        await db.pool.close()


@pytest.fixture
async def cleanup_tables(real_db):
    """Cleanup test tables before and after each test"""
    # Cleanup before test
    tables_to_drop = [
        "LIGHTRAG_VDB_CHUNKS",
        "LIGHTRAG_VDB_CHUNKS_test_model_768d",
        "LIGHTRAG_VDB_CHUNKS_text_embedding_ada_002_1536d",
        "LIGHTRAG_VDB_CHUNKS_bge_small_768d",
        "LIGHTRAG_VDB_CHUNKS_bge_large_1024d",
    ]

    for table in tables_to_drop:
        try:
            await real_db.execute(f"DROP TABLE IF EXISTS {table} CASCADE", None)
        except Exception:
            pass

    yield

    # Cleanup after test
    for table in tables_to_drop:
        try:
            await real_db.execute(f"DROP TABLE IF EXISTS {table} CASCADE", None)
        except Exception:
            pass


@pytest.fixture
def mock_embedding_func():
    """Create a mock embedding function for testing"""
    async def embed_func(texts, **kwargs):
        # Generate fake embeddings with consistent dimension
        return np.array([[0.1] * 768 for _ in texts])

    return EmbeddingFunc(
        embedding_dim=768,
        func=embed_func,
        model_name="test_model"
    )


@pytest.mark.asyncio
async def test_e2e_fresh_installation(real_db, cleanup_tables, mock_embedding_func, pg_config):
    """
    E2E Test: Fresh installation with model_name specified

    Scenario: New workspace, no legacy data
    Expected: Create new table with model suffix, no migration needed
    """
    print("\n[E2E Test] Fresh installation with model_name")

    # Reset ClientManager to use our test config
    ClientManager._instance = None
    ClientManager._client_config = pg_config

    # Create storage with model_name
    storage = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.8
            }
        },
        embedding_func=mock_embedding_func,
        workspace="e2e_test"
    )

    # Initialize storage (should create new table)
    await storage.initialize()

    # Verify table name
    assert "test_model_768d" in storage.table_name
    expected_table = "LIGHTRAG_VDB_CHUNKS_test_model_768d"
    assert storage.table_name == expected_table

    # Verify table exists
    check_query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = $1
        )
    """
    result = await real_db.query(check_query, [expected_table.lower()])
    assert result.get("exists") == True, f"Table {expected_table} should exist"

    # Verify legacy table does NOT exist
    legacy_result = await real_db.query(check_query, ["LIGHTRAG_VDB_CHUNKS".lower()])
    assert legacy_result.get("exists") == False, "Legacy table should not exist"

    print(f"✅ Fresh installation successful: {expected_table} created")

    await storage.finalize()


@pytest.mark.asyncio
async def test_e2e_legacy_migration(real_db, cleanup_tables, pg_config):
    """
    E2E Test: Upgrade from legacy format with automatic migration

    Scenario:
    1. Create legacy table (without model suffix)
    2. Insert test data
    3. Initialize with model_name (triggers migration)
    4. Verify data migrated to new table
    """
    print("\n[E2E Test] Legacy data migration")

    # Step 1: Create legacy table and insert data
    legacy_table = "LIGHTRAG_VDB_CHUNKS"

    create_legacy_sql = f"""
        CREATE TABLE IF NOT EXISTS {legacy_table} (
            workspace VARCHAR(255),
            id VARCHAR(255) PRIMARY KEY,
            content TEXT,
            content_vector vector(1536),
            tokens INTEGER,
            chunk_order_index INTEGER,
            full_doc_id VARCHAR(255),
            file_path TEXT,
            create_time TIMESTAMP,
            update_time TIMESTAMP
        )
    """
    await real_db.execute(create_legacy_sql, None)

    # Insert test data into legacy table
    test_data = [
        ("e2e_test", f"legacy_doc_{i}", f"Legacy content {i}",
         [0.1] * 1536, 100, i, "legacy_doc", "/test/path", "NOW()", "NOW()")
        for i in range(10)
    ]

    for data in test_data:
        insert_sql = f"""
            INSERT INTO {legacy_table}
            (workspace, id, content, content_vector, tokens, chunk_order_index, full_doc_id, file_path, create_time, update_time)
            VALUES ($1, $2, $3, $4::vector, $5, $6, $7, $8, {data[8]}, {data[9]})
        """
        await real_db.execute(insert_sql, {
            "workspace": data[0],
            "id": data[1],
            "content": data[2],
            "content_vector": data[3],
            "tokens": data[4],
            "chunk_order_index": data[5],
            "full_doc_id": data[6],
            "file_path": data[7]
        })

    # Verify legacy data exists
    count_result = await real_db.query(f"SELECT COUNT(*) as count FROM {legacy_table} WHERE workspace=$1", ["e2e_test"])
    legacy_count = count_result.get("count", 0)
    assert legacy_count == 10, f"Expected 10 records in legacy table, got {legacy_count}"
    print(f"✅ Legacy table created with {legacy_count} records")

    # Step 2: Initialize storage with model_name (triggers migration)
    ClientManager._instance = None
    ClientManager._client_config = pg_config

    async def embed_func(texts, **kwargs):
        return np.array([[0.1] * 1536 for _ in texts])

    embedding_func = EmbeddingFunc(
        embedding_dim=1536,
        func=embed_func,
        model_name="text-embedding-ada-002"
    )

    storage = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.8
            }
        },
        embedding_func=embedding_func,
        workspace="e2e_test"
    )

    # Initialize (should trigger migration)
    print("🔄 Starting migration...")
    await storage.initialize()
    print("✅ Migration completed")

    # Step 3: Verify migration
    new_table = storage.table_name
    assert "text_embedding_ada_002_1536d" in new_table

    # Count records in new table
    new_count_result = await real_db.query(f"SELECT COUNT(*) as count FROM {new_table} WHERE workspace=$1", ["e2e_test"])
    new_count = new_count_result.get("count", 0)

    assert new_count == legacy_count, f"Expected {legacy_count} records in new table, got {new_count}"
    print(f"✅ Data migration verified: {new_count}/{legacy_count} records migrated")

    # Verify data content
    sample_result = await real_db.query(f"SELECT id, content FROM {new_table} WHERE workspace=$1 LIMIT 1", ["e2e_test"])
    assert sample_result is not None
    assert "Legacy content" in sample_result.get("content", "")
    print(f"✅ Data integrity verified: {sample_result.get('id')}")

    await storage.finalize()


@pytest.mark.asyncio
async def test_e2e_multi_model_coexistence(real_db, cleanup_tables, pg_config):
    """
    E2E Test: Multiple embedding models coexisting

    Scenario:
    1. Create storage with model A (768d)
    2. Create storage with model B (1024d)
    3. Verify separate tables created
    4. Verify data isolation
    """
    print("\n[E2E Test] Multi-model coexistence")

    ClientManager._instance = None
    ClientManager._client_config = pg_config

    # Model A: 768 dimensions
    async def embed_func_a(texts, **kwargs):
        return np.array([[0.1] * 768 for _ in texts])

    embedding_func_a = EmbeddingFunc(
        embedding_dim=768,
        func=embed_func_a,
        model_name="bge-small"
    )

    storage_a = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.8
            }
        },
        embedding_func=embedding_func_a,
        workspace="e2e_test"
    )

    await storage_a.initialize()
    table_a = storage_a.table_name
    assert "bge_small_768d" in table_a
    print(f"✅ Model A table created: {table_a}")

    # Model B: 1024 dimensions
    async def embed_func_b(texts, **kwargs):
        return np.array([[0.1] * 1024 for _ in texts])

    embedding_func_b = EmbeddingFunc(
        embedding_dim=1024,
        func=embed_func_b,
        model_name="bge-large"
    )

    storage_b = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.8
            }
        },
        embedding_func=embedding_func_b,
        workspace="e2e_test"
    )

    await storage_b.initialize()
    table_b = storage_b.table_name
    assert "bge_large_1024d" in table_b
    print(f"✅ Model B table created: {table_b}")

    # Verify tables are different
    assert table_a != table_b, "Tables should have different names"
    print(f"✅ Table isolation verified: {table_a} != {table_b}")

    # Verify both tables exist
    check_query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = $1
        )
    """
    result_a = await real_db.query(check_query, [table_a.lower()])
    result_b = await real_db.query(check_query, [table_b.lower()])

    assert result_a.get("exists") == True
    assert result_b.get("exists") == True
    print("✅ Both tables exist in database")

    await storage_a.finalize()
    await storage_b.finalize()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
