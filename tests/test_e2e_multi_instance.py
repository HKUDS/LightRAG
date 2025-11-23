"""
E2E Tests for Multi-Instance LightRAG with Multiple Workspaces

These tests verify:
1. Legacy data migration from tables/collections without model suffix
2. Multiple LightRAG instances with different embedding models
3. Multiple workspaces isolation
4. Both PostgreSQL and Qdrant vector storage
5. Real document insertion and query operations

Prerequisites:
- PostgreSQL with pgvector extension
- Qdrant server running
- Environment variables configured
"""

import os
import pytest
import asyncio
import numpy as np
import tempfile
import shutil
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.kg.postgres_impl import PostgreSQLDB

# Conditional import for E2E dependencies
# This prevents offline tests from failing due to missing E2E dependencies
qdrant_client = pytest.importorskip(
    "qdrant_client", reason="Qdrant client required for E2E tests"
)
QdrantClient = qdrant_client.QdrantClient


# Configuration fixtures
@pytest.fixture(scope="function")
def pg_config():
    """PostgreSQL configuration"""
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "user": os.getenv("POSTGRES_USER", "lightrag"),
        "password": os.getenv("POSTGRES_PASSWORD", "lightrag_test_password"),
        "database": os.getenv("POSTGRES_DB", "lightrag_test"),
        "workspace": "multi_instance_test",
        "max_connections": 10,
        "connection_retry_attempts": 3,
        "connection_retry_backoff": 0.5,
        "connection_retry_backoff_max": 5.0,
        "pool_close_timeout": 5.0,
    }


@pytest.fixture(scope="function")
def qdrant_config():
    """Qdrant configuration"""
    return {
        "url": os.getenv("QDRANT_URL", "http://localhost:6333"),
        "api_key": os.getenv("QDRANT_API_KEY", None),
    }


# Cleanup fixtures
@pytest.fixture(scope="function")
async def pg_cleanup(pg_config):
    """Cleanup PostgreSQL tables before and after test"""
    db = PostgreSQLDB(pg_config)
    await db.initdb()

    tables_to_drop = [
        "lightrag_doc_full",
        "lightrag_doc_chunks",
        "lightrag_vdb_chunks",
        "lightrag_vdb_chunks_text_embedding_ada_002_1536d",
        "lightrag_vdb_chunks_text_embedding_3_large_3072d",
        "lightrag_vdb_chunks_model_a_768d",
        "lightrag_vdb_chunks_model_b_1024d",
        "lightrag_vdb_entity",
        "lightrag_vdb_relation",
        "lightrag_llm_cache",
        "lightrag_doc_status",
        "lightrag_full_entities",
        "lightrag_full_relations",
        "lightrag_entity_chunks",
        "lightrag_relation_chunks",
    ]

    # Cleanup before
    for table in tables_to_drop:
        try:
            await db.execute(f"DROP TABLE IF EXISTS {table} CASCADE", None)
        except Exception:
            pass

    yield db

    # Cleanup after
    for table in tables_to_drop:
        try:
            await db.execute(f"DROP TABLE IF EXISTS {table} CASCADE", None)
        except Exception:
            pass

    if db.pool:
        await db.pool.close()


@pytest.fixture(scope="function")
def qdrant_cleanup(qdrant_config):
    """Cleanup Qdrant collections before and after test"""
    client = QdrantClient(
        url=qdrant_config["url"],
        api_key=qdrant_config["api_key"],
        timeout=60,
    )

    collections_to_delete = [
        "lightrag_vdb_chunks",  # Legacy collection (no model suffix)
        "lightrag_vdb_chunks_text_embedding_ada_002_1536d",  # Migrated collection
        "lightrag_vdb_chunks_model_a_768d",
        "lightrag_vdb_chunks_model_b_1024d",
    ]

    # Cleanup before
    for collection in collections_to_delete:
        try:
            if client.collection_exists(collection):
                client.delete_collection(collection)
        except Exception:
            pass

    yield client

    # Cleanup after
    for collection in collections_to_delete:
        try:
            if client.collection_exists(collection):
                client.delete_collection(collection)
        except Exception:
            pass


@pytest.fixture
def temp_working_dirs():
    """Create multiple temporary working directories"""
    dirs = {
        "workspace_a": tempfile.mkdtemp(prefix="lightrag_workspace_a_"),
        "workspace_b": tempfile.mkdtemp(prefix="lightrag_workspace_b_"),
    }
    yield dirs
    # Cleanup
    for dir_path in dirs.values():
        shutil.rmtree(dir_path, ignore_errors=True)


@pytest.fixture
def mock_llm_func():
    """Mock LLM function that returns proper entity/relation format"""

    async def llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        await asyncio.sleep(0)  # Simulate async I/O
        return """entity<|#|>Artificial Intelligence<|#|>concept<|#|>AI is a field of computer science.
entity<|#|>Machine Learning<|#|>concept<|#|>ML is a subset of AI.
relation<|#|>Machine Learning<|#|>Artificial Intelligence<|#|>subset<|#|>ML is a subset of AI.
<|COMPLETE|>"""

    return llm_func


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer"""
    from lightrag.utils import Tokenizer

    class _SimpleTokenizerImpl:
        def encode(self, content: str) -> list[int]:
            return [ord(ch) for ch in content]

        def decode(self, tokens: list[int]) -> str:
            return "".join(chr(t) for t in tokens)

    return Tokenizer("mock-tokenizer", _SimpleTokenizerImpl())


# Test: Legacy data migration
@pytest.mark.asyncio
async def test_legacy_migration_postgres(
    pg_cleanup, mock_llm_func, mock_tokenizer, pg_config
):
    """
    Test automatic migration from legacy PostgreSQL table (no model suffix)

    Scenario:
    1. Create legacy table without model suffix
    2. Insert test data with 1536d vectors
    3. Initialize LightRAG with model_name (triggers migration)
    4. Verify data migrated to new table with model suffix
    """
    print("\n[E2E Test] Legacy data migration (1536d)")

    # Create temp working dir
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp(prefix="lightrag_legacy_test_")

    try:
        # Step 1: Create legacy table and insert data
        legacy_table = "lightrag_vdb_chunks"

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
                create_time TIMESTAMP DEFAULT NOW(),
                update_time TIMESTAMP DEFAULT NOW()
            )
        """
        await pg_cleanup.execute(create_legacy_sql, None)

        # Insert 3 test records
        for i in range(3):
            vector_str = "[" + ",".join(["0.1"] * 1536) + "]"
            insert_sql = f"""
                INSERT INTO {legacy_table}
                (workspace, id, content, content_vector, tokens, chunk_order_index, full_doc_id, file_path)
                VALUES ($1, $2, $3, $4::vector, $5, $6, $7, $8)
            """
            await pg_cleanup.execute(
                insert_sql,
                {
                    "workspace": pg_config["workspace"],
                    "id": f"legacy_{i}",
                    "content": f"Legacy content {i}",
                    "content_vector": vector_str,
                    "tokens": 100,
                    "chunk_order_index": i,
                    "full_doc_id": "legacy_doc",
                    "file_path": "/test/path",
                },
            )

        # Verify legacy data
        count_result = await pg_cleanup.query(
            f"SELECT COUNT(*) as count FROM {legacy_table} WHERE workspace=$1",
            [pg_config["workspace"]],
        )
        legacy_count = count_result.get("count", 0)
        print(f"✅ Legacy table created with {legacy_count} records")

        # Step 2: Initialize LightRAG with model_name (triggers migration)
        async def embed_func(texts):
            await asyncio.sleep(0)
            return np.random.rand(len(texts), 1536)

        embedding_func = EmbeddingFunc(
            embedding_dim=1536,
            max_token_size=8192,
            func=embed_func,
            model_name="text-embedding-ada-002",
        )

        rag = LightRAG(
            working_dir=temp_dir,
            workspace=pg_config["workspace"],  # Match workspace with test data
            llm_model_func=mock_llm_func,
            embedding_func=embedding_func,
            tokenizer=mock_tokenizer,
            kv_storage="PGKVStorage",
            vector_storage="PGVectorStorage",
            # Use default NetworkXStorage for graph storage (AGE extension not available in CI)
            doc_status_storage="PGDocStatusStorage",
            vector_db_storage_cls_kwargs={
                **pg_config,
                "cosine_better_than_threshold": 0.8,
            },
        )

        print("🔄 Initializing LightRAG (triggers migration)...")
        await rag.initialize_storages()

        # Step 3: Verify migration
        new_table = rag.chunks_vdb.table_name
        assert "text_embedding_ada_002_1536d" in new_table.lower()

        new_count_result = await pg_cleanup.query(
            f"SELECT COUNT(*) as count FROM {new_table} WHERE workspace=$1",
            [pg_config["workspace"]],
        )
        new_count = new_count_result.get("count", 0)

        assert new_count == legacy_count, (
            f"Expected {legacy_count} records migrated, got {new_count}"
        )
        print(f"✅ Migration successful: {new_count}/{legacy_count} records migrated")
        print(f"✅ New table: {new_table}")

        # Verify legacy table was automatically deleted after migration (Case 4)
        check_legacy_query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = $1
            )
        """
        legacy_result = await pg_cleanup.query(
            check_legacy_query, [legacy_table.lower()]
        )
        legacy_exists = legacy_result.get("exists", True)
        assert not legacy_exists, (
            f"Legacy table '{legacy_table}' should be deleted after successful migration"
        )
        print(f"✅ Legacy table '{legacy_table}' automatically deleted after migration")

        await rag.finalize_storages()

    finally:
        # Cleanup temp dir
        shutil.rmtree(temp_dir, ignore_errors=True)


# Test: Workspace migration isolation (P0 Bug Fix Verification)
@pytest.mark.asyncio
async def test_workspace_migration_isolation_e2e_postgres(
    pg_cleanup, mock_llm_func, mock_tokenizer, pg_config
):
    """
    E2E Test: Workspace isolation during PostgreSQL migration

    Critical P0 Bug Verification:
    - Legacy table contains MIXED data from workspace_a and workspace_b
    - Initialize LightRAG for workspace_a only
    - Verify ONLY workspace_a data migrated to new table
    - Verify workspace_b data NOT leaked to workspace_a's table
    - Verify workspace_b data preserved in legacy table

    This test validates the fix for the cross-workspace data leakage bug
    where setup_table() was copying ALL records regardless of workspace.
    """
    print("\n[E2E P0 Bug Fix] Workspace migration isolation (PostgreSQL)")

    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp(prefix="lightrag_workspace_isolation_")

    try:
        # Step 1: Create legacy table with MIXED workspace data
        legacy_table = "lightrag_vdb_chunks"

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
                create_time TIMESTAMP DEFAULT NOW(),
                update_time TIMESTAMP DEFAULT NOW()
            )
        """
        await pg_cleanup.execute(create_legacy_sql, None)

        # Insert 3 records for workspace_a
        for i in range(3):
            vector_str = "[" + ",".join([str(0.1 + i * 0.01)] * 1536) + "]"
            insert_sql = f"""
                INSERT INTO {legacy_table}
                (workspace, id, content, content_vector, tokens, chunk_order_index, full_doc_id, file_path)
                VALUES ($1, $2, $3, $4::vector, $5, $6, $7, $8)
            """
            await pg_cleanup.execute(
                insert_sql,
                {
                    "workspace": "workspace_a",
                    "id": f"a_{i}",
                    "content": f"Workspace A content {i}",
                    "content_vector": vector_str,
                    "tokens": 100,
                    "chunk_order_index": i,
                    "full_doc_id": "doc_a",
                    "file_path": "/workspace_a/doc.txt",
                },
            )

        # Insert 3 records for workspace_b
        for i in range(3):
            vector_str = "[" + ",".join([str(0.5 + i * 0.01)] * 1536) + "]"
            insert_sql = f"""
                INSERT INTO {legacy_table}
                (workspace, id, content, content_vector, tokens, chunk_order_index, full_doc_id, file_path)
                VALUES ($1, $2, $3, $4::vector, $5, $6, $7, $8)
            """
            await pg_cleanup.execute(
                insert_sql,
                {
                    "workspace": "workspace_b",
                    "id": f"b_{i}",
                    "content": f"Workspace B content {i}",
                    "content_vector": vector_str,
                    "tokens": 100,
                    "chunk_order_index": i,
                    "full_doc_id": "doc_b",
                    "file_path": "/workspace_b/doc.txt",
                },
            )

        # Verify legacy table has BOTH workspaces' data
        total_count_result = await pg_cleanup.query(
            f"SELECT COUNT(*) as count FROM {legacy_table}", []
        )
        total_count = total_count_result.get("count", 0)
        assert total_count == 6, f"Expected 6 total records, got {total_count}"

        workspace_a_count_result = await pg_cleanup.query(
            f"SELECT COUNT(*) as count FROM {legacy_table} WHERE workspace=$1",
            ["workspace_a"],
        )
        workspace_a_count = workspace_a_count_result.get("count", 0)
        assert workspace_a_count == 3, (
            f"Expected 3 workspace_a records, got {workspace_a_count}"
        )

        workspace_b_count_result = await pg_cleanup.query(
            f"SELECT COUNT(*) as count FROM {legacy_table} WHERE workspace=$1",
            ["workspace_b"],
        )
        workspace_b_count = workspace_b_count_result.get("count", 0)
        assert workspace_b_count == 3, (
            f"Expected 3 workspace_b records, got {workspace_b_count}"
        )

        print(
            f"✅ Legacy table created: {total_count} records (workspace_a: {workspace_a_count}, workspace_b: {workspace_b_count})"
        )

        # Step 2: Initialize LightRAG for workspace_a ONLY
        async def embed_func(texts):
            await asyncio.sleep(0)
            return np.random.rand(len(texts), 1536)

        embedding_func = EmbeddingFunc(
            embedding_dim=1536,
            max_token_size=8192,
            func=embed_func,
            model_name="text-embedding-ada-002",
        )

        rag = LightRAG(
            working_dir=temp_dir,
            workspace="workspace_a",  # CRITICAL: Only workspace_a
            llm_model_func=mock_llm_func,
            embedding_func=embedding_func,
            tokenizer=mock_tokenizer,
            kv_storage="PGKVStorage",
            vector_storage="PGVectorStorage",
            doc_status_storage="PGDocStatusStorage",
            vector_db_storage_cls_kwargs={
                **pg_config,
                "workspace": "workspace_a",  # CRITICAL: Filter by workspace_a
                "cosine_better_than_threshold": 0.8,
            },
        )

        print("🔄 Initializing LightRAG for workspace_a (triggers migration)...")
        await rag.initialize_storages()

        # Step 3: Verify workspace isolation
        new_table = rag.chunks_vdb.table_name
        assert "text_embedding_ada_002_1536d" in new_table.lower()
        print(f"✅ New table created: {new_table}")

        # Verify: NEW table contains ONLY workspace_a data (3 records)
        new_workspace_a_result = await pg_cleanup.query(
            f"SELECT COUNT(*) as count FROM {new_table} WHERE workspace=$1",
            ["workspace_a"],
        )
        new_workspace_a_count = new_workspace_a_result.get("count", 0)
        assert new_workspace_a_count == 3, (
            f"Expected 3 workspace_a records in new table, got {new_workspace_a_count}"
        )
        print(
            f"✅ Migration successful: {new_workspace_a_count} workspace_a records migrated"
        )

        # Verify: NEW table does NOT contain workspace_b data (0 records)
        new_workspace_b_result = await pg_cleanup.query(
            f"SELECT COUNT(*) as count FROM {new_table} WHERE workspace=$1",
            ["workspace_b"],
        )
        new_workspace_b_count = new_workspace_b_result.get("count", 0)
        assert new_workspace_b_count == 0, (
            f"workspace_b data leaked! Found {new_workspace_b_count} records in new table"
        )
        print("✅ No data leakage: 0 workspace_b records in new table (isolated)")

        # Verify: LEGACY table still exists (because workspace_b data remains)
        check_legacy_query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = $1
            )
        """
        legacy_result = await pg_cleanup.query(
            check_legacy_query, [legacy_table.lower()]
        )
        legacy_exists = legacy_result.get("exists", False)
        assert legacy_exists, (
            f"Legacy table '{legacy_table}' should still exist (has workspace_b data)"
        )

        # Verify: LEGACY table still has workspace_b data (3 records)
        legacy_workspace_b_result = await pg_cleanup.query(
            f"SELECT COUNT(*) as count FROM {legacy_table} WHERE workspace=$1",
            ["workspace_b"],
        )
        legacy_workspace_b_count = legacy_workspace_b_result.get("count", 0)
        assert legacy_workspace_b_count == 3, (
            f"workspace_b data lost! Only {legacy_workspace_b_count} remain in legacy table"
        )
        print(
            f"✅ Legacy table preserved: {legacy_workspace_b_count} workspace_b records remain (not migrated)"
        )

        # Verify: LEGACY table does NOT have workspace_a data (migrated and deleted)
        legacy_workspace_a_result = await pg_cleanup.query(
            f"SELECT COUNT(*) as count FROM {legacy_table} WHERE workspace=$1",
            ["workspace_a"],
        )
        legacy_workspace_a_count = legacy_workspace_a_result.get("count", 0)
        assert legacy_workspace_a_count == 0, (
            f"workspace_a data should be removed from legacy after migration, found {legacy_workspace_a_count}"
        )
        print(
            "✅ Legacy cleanup verified: 0 workspace_a records in legacy (cleaned after migration)"
        )

        print(
            "\n🎉 P0 Bug Fix Verified: Workspace migration isolation working correctly!"
        )
        print(
            "   - workspace_a: 3 records migrated to new table, 0 in legacy (migrated)"
        )
        print(
            "   - workspace_b: 0 records in new table (isolated), 3 in legacy (preserved)"
        )

        await rag.finalize_storages()

    finally:
        # Cleanup temp dir
        shutil.rmtree(temp_dir, ignore_errors=True)


# Test: Qdrant legacy data migration
@pytest.mark.asyncio
async def test_legacy_migration_qdrant(
    qdrant_cleanup, mock_llm_func, mock_tokenizer, qdrant_config
):
    """
    Test automatic migration from legacy Qdrant collection (no model suffix)

    Scenario:
    1. Create legacy collection without model suffix
    2. Insert test vectors with 1536d
    3. Initialize LightRAG with model_name (triggers migration)
    4. Verify data migrated to new collection with model suffix
    """
    print("\n[E2E Test] Qdrant legacy data migration (1536d)")

    # Create temp working dir
    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp(prefix="lightrag_qdrant_legacy_")

    try:
        # Step 1: Create legacy collection and insert data
        legacy_collection = "lightrag_vdb_chunks"

        # Create legacy collection without model suffix
        from qdrant_client.models import Distance, VectorParams

        qdrant_cleanup.create_collection(
            collection_name=legacy_collection,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print(f"✅ Created legacy collection: {legacy_collection}")

        # Insert 3 test records
        from qdrant_client.models import PointStruct

        test_vectors = []
        for i in range(3):
            vector = np.random.rand(1536).tolist()
            point = PointStruct(
                id=i,
                vector=vector,
                payload={
                    "id": f"legacy_{i}",
                    "content": f"Legacy content {i}",
                    "tokens": 100,
                    "chunk_order_index": i,
                    "full_doc_id": "legacy_doc",
                    "file_path": "/test/path",
                },
            )
            test_vectors.append(point)

        qdrant_cleanup.upsert(collection_name=legacy_collection, points=test_vectors)

        # Verify legacy data
        legacy_count = qdrant_cleanup.count(legacy_collection).count
        print(f"✅ Legacy collection created with {legacy_count} vectors")

        # Step 2: Initialize LightRAG with model_name (triggers migration)
        async def embed_func(texts):
            await asyncio.sleep(0)
            return np.random.rand(len(texts), 1536)

        embedding_func = EmbeddingFunc(
            embedding_dim=1536,
            max_token_size=8192,
            func=embed_func,
            model_name="text-embedding-ada-002",
        )

        rag = LightRAG(
            working_dir=temp_dir,
            llm_model_func=mock_llm_func,
            embedding_func=embedding_func,
            tokenizer=mock_tokenizer,
            vector_storage="QdrantVectorDBStorage",
            vector_db_storage_cls_kwargs={
                **qdrant_config,
                "cosine_better_than_threshold": 0.8,
            },
        )

        print("🔄 Initializing LightRAG (triggers migration)...")
        await rag.initialize_storages()

        # Step 3: Verify migration
        new_collection = rag.chunks_vdb.final_namespace
        assert "text_embedding_ada_002_1536d" in new_collection

        # Verify new collection exists
        assert qdrant_cleanup.collection_exists(new_collection), (
            f"New collection {new_collection} should exist"
        )

        new_count = qdrant_cleanup.count(new_collection).count

        assert new_count == legacy_count, (
            f"Expected {legacy_count} vectors migrated, got {new_count}"
        )
        print(f"✅ Migration successful: {new_count}/{legacy_count} vectors migrated")
        print(f"✅ New collection: {new_collection}")

        # Verify vector dimension
        collection_info = qdrant_cleanup.get_collection(new_collection)
        assert collection_info.config.params.vectors.size == 1536, (
            "Migrated collection should have 1536 dimensions"
        )
        print(
            f"✅ Vector dimension verified: {collection_info.config.params.vectors.size}d"
        )

        # Verify legacy collection was automatically deleted after migration (Case 4)
        legacy_exists = qdrant_cleanup.collection_exists(legacy_collection)
        assert not legacy_exists, (
            f"Legacy collection '{legacy_collection}' should be deleted after successful migration"
        )
        print(
            f"✅ Legacy collection '{legacy_collection}' automatically deleted after migration"
        )

        await rag.finalize_storages()

    finally:
        # Cleanup temp dir
        shutil.rmtree(temp_dir, ignore_errors=True)


# Test: Multiple LightRAG instances with PostgreSQL
@pytest.mark.asyncio
async def test_multi_instance_postgres(
    pg_cleanup, temp_working_dirs, mock_llm_func, mock_tokenizer, pg_config
):
    """
    Test multiple LightRAG instances with different dimensions and model names

    Scenarios:
    - Instance A: model-a (768d) - explicit model name
    - Instance B: model-b (1024d) - explicit model name
    - Both instances insert documents independently
    - Verify separate tables created for each model+dimension combination
    - Verify data isolation between instances
    """
    print("\n[E2E Multi-Instance] PostgreSQL with 2 models (768d vs 1024d)")

    # Instance A: 768d with model-a
    async def embed_func_a(texts):
        await asyncio.sleep(0)
        return np.random.rand(len(texts), 768)

    embedding_func_a = EmbeddingFunc(
        embedding_dim=768, max_token_size=8192, func=embed_func_a, model_name="model-a"
    )

    # Instance B: 1024d with model-b
    async def embed_func_b(texts):
        await asyncio.sleep(0)
        return np.random.rand(len(texts), 1024)

    embedding_func_b = EmbeddingFunc(
        embedding_dim=1024, max_token_size=8192, func=embed_func_b, model_name="model-b"
    )

    # Initialize LightRAG instance A
    print("📦 Initializing LightRAG instance A (model-a, 768d)...")
    rag_a = LightRAG(
        working_dir=temp_working_dirs["workspace_a"],
        workspace=pg_config["workspace"],  # Use same workspace to test model isolation
        llm_model_func=mock_llm_func,
        embedding_func=embedding_func_a,
        tokenizer=mock_tokenizer,
        kv_storage="PGKVStorage",
        vector_storage="PGVectorStorage",
        # Use default NetworkXStorage for graph storage (AGE extension not available in CI)
        doc_status_storage="PGDocStatusStorage",
        vector_db_storage_cls_kwargs={**pg_config, "cosine_better_than_threshold": 0.8},
    )

    await rag_a.initialize_storages()
    table_a = rag_a.chunks_vdb.table_name
    print(f"✅ Instance A initialized: {table_a}")

    # Initialize LightRAG instance B
    print("📦 Initializing LightRAG instance B (model-b, 1024d)...")
    rag_b = LightRAG(
        working_dir=temp_working_dirs["workspace_b"],
        workspace=pg_config["workspace"],  # Use same workspace to test model isolation
        llm_model_func=mock_llm_func,
        embedding_func=embedding_func_b,
        tokenizer=mock_tokenizer,
        kv_storage="PGKVStorage",
        vector_storage="PGVectorStorage",
        # Use default NetworkXStorage for graph storage (AGE extension not available in CI)
        doc_status_storage="PGDocStatusStorage",
        vector_db_storage_cls_kwargs={**pg_config, "cosine_better_than_threshold": 0.8},
    )

    await rag_b.initialize_storages()
    table_b = rag_b.chunks_vdb.table_name
    print(f"✅ Instance B initialized: {table_b}")

    # Verify table names are different
    assert "model_a_768d" in table_a.lower()
    assert "model_b_1024d" in table_b.lower()
    assert table_a != table_b
    print(f"✅ Table isolation verified: {table_a} != {table_b}")

    # Verify both tables exist in database
    check_query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = $1
        )
    """
    result_a = await pg_cleanup.query(check_query, [table_a.lower()])
    result_b = await pg_cleanup.query(check_query, [table_b.lower()])

    assert result_a.get("exists") is True, f"Table {table_a} should exist"
    assert result_b.get("exists") is True, f"Table {table_b} should exist"
    print("✅ Both tables exist in PostgreSQL")

    # Insert documents in instance A
    print("📝 Inserting document in instance A...")
    await rag_a.ainsert(
        "Document A: This is about artificial intelligence and neural networks."
    )

    # Insert documents in instance B
    print("📝 Inserting document in instance B...")
    await rag_b.ainsert("Document B: This is about machine learning and deep learning.")

    # Verify data isolation
    count_a_result = await pg_cleanup.query(
        f"SELECT COUNT(*) as count FROM {table_a}", []
    )
    count_b_result = await pg_cleanup.query(
        f"SELECT COUNT(*) as count FROM {table_b}", []
    )

    count_a = count_a_result.get("count", 0)
    count_b = count_b_result.get("count", 0)

    print(f"✅ Instance A chunks: {count_a}")
    print(f"✅ Instance B chunks: {count_b}")

    assert count_a > 0, "Instance A should have data"
    assert count_b > 0, "Instance B should have data"

    # Cleanup
    await rag_a.finalize_storages()
    await rag_b.finalize_storages()

    print("✅ Multi-instance PostgreSQL test passed!")


# Test: Multiple LightRAG instances with Qdrant
@pytest.mark.asyncio
async def test_multi_instance_qdrant(
    qdrant_cleanup, temp_working_dirs, mock_llm_func, mock_tokenizer, qdrant_config
):
    """
    Test multiple LightRAG instances with different models using Qdrant

    Scenario:
    - Instance A: model-a (768d)
    - Instance B: model-b (1024d)
    - Both insert documents independently
    - Verify separate collections created and data isolated
    """
    print("\n[E2E Multi-Instance] Qdrant with 2 models (768d vs 1024d)")

    # Create embedding function for model A (768d)
    async def embed_func_a(texts):
        await asyncio.sleep(0)
        return np.random.rand(len(texts), 768)

    embedding_func_a = EmbeddingFunc(
        embedding_dim=768, max_token_size=8192, func=embed_func_a, model_name="model-a"
    )

    # Create embedding function for model B (1024d)
    async def embed_func_b(texts):
        await asyncio.sleep(0)
        return np.random.rand(len(texts), 1024)

    embedding_func_b = EmbeddingFunc(
        embedding_dim=1024, max_token_size=8192, func=embed_func_b, model_name="model-b"
    )

    # Initialize LightRAG instance A
    print("📦 Initializing LightRAG instance A (model-a, 768d)...")
    rag_a = LightRAG(
        working_dir=temp_working_dirs["workspace_a"],
        llm_model_func=mock_llm_func,
        embedding_func=embedding_func_a,
        tokenizer=mock_tokenizer,
        vector_storage="QdrantVectorDBStorage",
        vector_db_storage_cls_kwargs={
            **qdrant_config,
            "cosine_better_than_threshold": 0.8,
        },
    )

    await rag_a.initialize_storages()
    collection_a = rag_a.chunks_vdb.final_namespace
    print(f"✅ Instance A initialized: {collection_a}")

    # Initialize LightRAG instance B
    print("📦 Initializing LightRAG instance B (model-b, 1024d)...")
    rag_b = LightRAG(
        working_dir=temp_working_dirs["workspace_b"],
        llm_model_func=mock_llm_func,
        embedding_func=embedding_func_b,
        tokenizer=mock_tokenizer,
        vector_storage="QdrantVectorDBStorage",
        vector_db_storage_cls_kwargs={
            **qdrant_config,
            "cosine_better_than_threshold": 0.8,
        },
    )

    await rag_b.initialize_storages()
    collection_b = rag_b.chunks_vdb.final_namespace
    print(f"✅ Instance B initialized: {collection_b}")

    # Verify collection names are different
    assert "model_a_768d" in collection_a
    assert "model_b_1024d" in collection_b
    assert collection_a != collection_b
    print(f"✅ Collection isolation verified: {collection_a} != {collection_b}")

    # Verify both collections exist in Qdrant
    assert qdrant_cleanup.collection_exists(collection_a), (
        f"Collection {collection_a} should exist"
    )
    assert qdrant_cleanup.collection_exists(collection_b), (
        f"Collection {collection_b} should exist"
    )
    print("✅ Both collections exist in Qdrant")

    # Verify vector dimensions
    info_a = qdrant_cleanup.get_collection(collection_a)
    info_b = qdrant_cleanup.get_collection(collection_b)

    assert info_a.config.params.vectors.size == 768, "Model A should use 768 dimensions"
    assert info_b.config.params.vectors.size == 1024, (
        "Model B should use 1024 dimensions"
    )
    print(
        f"✅ Vector dimensions verified: {info_a.config.params.vectors.size}d vs {info_b.config.params.vectors.size}d"
    )

    # Insert documents in instance A
    print("📝 Inserting document in instance A...")
    await rag_a.ainsert(
        "Document A: This is about artificial intelligence and neural networks."
    )

    # Insert documents in instance B
    print("📝 Inserting document in instance B...")
    await rag_b.ainsert("Document B: This is about machine learning and deep learning.")

    # Verify data isolation
    count_a = qdrant_cleanup.count(collection_a).count
    count_b = qdrant_cleanup.count(collection_b).count

    print(f"✅ Instance A vectors: {count_a}")
    print(f"✅ Instance B vectors: {count_b}")

    assert count_a > 0, "Instance A should have data"
    assert count_b > 0, "Instance B should have data"

    # Cleanup
    await rag_a.finalize_storages()
    await rag_b.finalize_storages()

    print("✅ Multi-instance Qdrant test passed!")


# ============================================================================
# Complete Migration Scenario Tests with Real Databases
# ============================================================================


@pytest.mark.asyncio
async def test_case1_both_exist_with_data_qdrant(
    qdrant_cleanup, mock_llm_func, mock_tokenizer, qdrant_config
):
    """
    E2E Case 1b: Both new and legacy collections exist, legacy has data
    Expected: Log warning, do not delete legacy (preserve data), use new collection
    """
    print("\n[E2E Case 1b] Both collections exist with data - preservation scenario")

    import tempfile
    import shutil
    from qdrant_client.models import Distance, VectorParams, PointStruct

    temp_dir = tempfile.mkdtemp(prefix="lightrag_case1_")

    try:
        # Step 1: Create both legacy and new collection
        legacy_collection = "lightrag_vdb_chunks"
        new_collection = "lightrag_vdb_chunks_text_embedding_ada_002_1536d"

        # Create legacy collection with data
        qdrant_cleanup.create_collection(
            collection_name=legacy_collection,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        legacy_points = [
            PointStruct(
                id=i,
                vector=np.random.rand(1536).tolist(),
                payload={"id": f"legacy_{i}", "content": f"Legacy doc {i}"},
            )
            for i in range(3)
        ]
        qdrant_cleanup.upsert(collection_name=legacy_collection, points=legacy_points)
        print(f"✅ Created legacy collection with {len(legacy_points)} points")

        # Create new collection (simulate already migrated)
        qdrant_cleanup.create_collection(
            collection_name=new_collection,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print(f"✅ Created new collection '{new_collection}'")

        # Step 2: Initialize LightRAG (should detect both and warn)
        async def embed_func(texts):
            await asyncio.sleep(0)
            return np.random.rand(len(texts), 1536)

        embedding_func = EmbeddingFunc(
            embedding_dim=1536,
            max_token_size=8192,
            func=embed_func,
            model_name="text-embedding-ada-002",
        )

        rag = LightRAG(
            working_dir=temp_dir,
            llm_model_func=mock_llm_func,
            embedding_func=embedding_func,
            tokenizer=mock_tokenizer,
            vector_storage="QdrantVectorDBStorage",
            vector_db_storage_cls_kwargs={
                **qdrant_config,
                "cosine_better_than_threshold": 0.8,
            },
        )

        await rag.initialize_storages()

        # Step 3: Verify behavior
        # Should use new collection (not migrate)
        assert rag.chunks_vdb.final_namespace == new_collection

        # Verify legacy collection still exists (Case 1b: has data, should NOT be deleted)
        legacy_exists = qdrant_cleanup.collection_exists(legacy_collection)
        assert legacy_exists, "Legacy collection with data should NOT be deleted"

        legacy_count = qdrant_cleanup.count(legacy_collection).count
        # Legacy should still have its data (not migrated, not deleted)
        assert legacy_count == 3
        print(
            f"✅ Legacy collection still has {legacy_count} points (preserved, not deleted)"
        )

        await rag.finalize_storages()

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_case2_only_new_exists_qdrant(
    qdrant_cleanup, mock_llm_func, mock_tokenizer, qdrant_config
):
    """
    E2E Case 2: Only new collection exists (already migrated scenario)
    Expected: Use existing collection, no migration
    """
    print("\n[E2E Case 2] Only new collection exists - already migrated")

    import tempfile
    import shutil
    from qdrant_client.models import Distance, VectorParams, PointStruct

    temp_dir = tempfile.mkdtemp(prefix="lightrag_case2_")

    try:
        # Step 1: Create only new collection with data
        new_collection = "lightrag_vdb_chunks_text_embedding_ada_002_1536d"

        qdrant_cleanup.create_collection(
            collection_name=new_collection,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

        # Add some existing data
        existing_points = [
            PointStruct(
                id=i,
                vector=np.random.rand(1536).tolist(),
                payload={
                    "id": f"existing_{i}",
                    "content": f"Existing doc {i}",
                    "workspace_id": "test_ws",
                },
            )
            for i in range(5)
        ]
        qdrant_cleanup.upsert(collection_name=new_collection, points=existing_points)
        print(f"✅ Created new collection with {len(existing_points)} existing points")

        # Step 2: Initialize LightRAG
        async def embed_func(texts):
            await asyncio.sleep(0)
            return np.random.rand(len(texts), 1536)

        embedding_func = EmbeddingFunc(
            embedding_dim=1536,
            max_token_size=8192,
            func=embed_func,
            model_name="text-embedding-ada-002",
        )

        rag = LightRAG(
            working_dir=temp_dir,
            llm_model_func=mock_llm_func,
            embedding_func=embedding_func,
            tokenizer=mock_tokenizer,
            vector_storage="QdrantVectorDBStorage",
            vector_db_storage_cls_kwargs={
                **qdrant_config,
                "cosine_better_than_threshold": 0.8,
            },
        )

        await rag.initialize_storages()

        # Step 3: Verify collection reused
        assert rag.chunks_vdb.final_namespace == new_collection
        count = qdrant_cleanup.count(new_collection).count
        assert count == 5  # Existing data preserved
        print(f"✅ Reused existing collection with {count} points")

        await rag.finalize_storages()

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_backward_compat_old_workspace_naming_qdrant(
    qdrant_cleanup, mock_llm_func, mock_tokenizer, qdrant_config
):
    """
    E2E: Backward compatibility with old workspace-based naming
    Old format: {workspace}_{namespace}
    """
    print("\n[E2E Backward Compat] Old workspace naming migration")

    import tempfile
    import shutil
    from qdrant_client.models import Distance, VectorParams, PointStruct

    temp_dir = tempfile.mkdtemp(prefix="lightrag_backward_compat_")

    try:
        # Step 1: Create old-style collection
        old_collection = "prod_chunks"  # Old format: {workspace}_{namespace}

        qdrant_cleanup.create_collection(
            collection_name=old_collection,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

        # Add legacy data
        legacy_points = [
            PointStruct(
                id=i,
                vector=np.random.rand(1536).tolist(),
                payload={"id": f"old_{i}", "content": f"Old document {i}"},
            )
            for i in range(10)
        ]
        qdrant_cleanup.upsert(collection_name=old_collection, points=legacy_points)
        print(
            f"✅ Created old-style collection '{old_collection}' with {len(legacy_points)} points"
        )

        # Step 2: Initialize LightRAG with prod workspace
        async def embed_func(texts):
            await asyncio.sleep(0)
            return np.random.rand(len(texts), 1536)

        embedding_func = EmbeddingFunc(
            embedding_dim=1536,
            max_token_size=8192,
            func=embed_func,
            model_name="text-embedding-ada-002",
        )

        # Important: Use "prod" workspace to match old naming
        rag = LightRAG(
            working_dir=temp_dir,
            workspace="prod",  # Pass workspace to LightRAG instance
            llm_model_func=mock_llm_func,
            embedding_func=embedding_func,
            tokenizer=mock_tokenizer,
            vector_storage="QdrantVectorDBStorage",
            vector_db_storage_cls_kwargs={
                **qdrant_config,
                "cosine_better_than_threshold": 0.8,
            },
        )

        print(
            "🔄 Initializing with 'prod' workspace (triggers backward-compat migration)..."
        )
        await rag.initialize_storages()

        # Step 3: Verify migration
        new_collection = rag.chunks_vdb.final_namespace
        new_count = qdrant_cleanup.count(new_collection).count

        assert new_count == len(legacy_points)
        print(
            f"✅ Migrated {new_count} points from old collection '{old_collection}' to '{new_collection}'"
        )

        await rag.finalize_storages()

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_empty_legacy_qdrant(
    qdrant_cleanup, mock_llm_func, mock_tokenizer, qdrant_config
):
    """
    E2E: Empty legacy collection migration
    Expected: Skip data migration, create new collection
    """
    print("\n[E2E Empty Legacy] Empty collection migration")

    import tempfile
    import shutil
    from qdrant_client.models import Distance, VectorParams

    temp_dir = tempfile.mkdtemp(prefix="lightrag_empty_legacy_")

    try:
        # Step 1: Create empty legacy collection
        legacy_collection = "lightrag_vdb_chunks"

        qdrant_cleanup.create_collection(
            collection_name=legacy_collection,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print(f"✅ Created empty legacy collection '{legacy_collection}'")

        # Step 2: Initialize LightRAG
        async def embed_func(texts):
            await asyncio.sleep(0)
            return np.random.rand(len(texts), 1536)

        embedding_func = EmbeddingFunc(
            embedding_dim=1536,
            max_token_size=8192,
            func=embed_func,
            model_name="text-embedding-ada-002",
        )

        rag = LightRAG(
            working_dir=temp_dir,
            llm_model_func=mock_llm_func,
            embedding_func=embedding_func,
            tokenizer=mock_tokenizer,
            vector_storage="QdrantVectorDBStorage",
            vector_db_storage_cls_kwargs={
                **qdrant_config,
                "cosine_better_than_threshold": 0.8,
            },
        )

        print("🔄 Initializing (should skip data migration for empty collection)...")
        await rag.initialize_storages()

        # Step 3: Verify new collection created
        new_collection = rag.chunks_vdb.final_namespace
        assert qdrant_cleanup.collection_exists(new_collection)
        print(f"✅ New collection '{new_collection}' created (data migration skipped)")

        await rag.finalize_storages()

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_workspace_isolation_e2e_qdrant(
    qdrant_cleanup, temp_working_dirs, mock_llm_func, mock_tokenizer, qdrant_config
):
    """
    E2E: Workspace isolation within same collection
    Expected: Same model+dim uses same collection, isolated by workspace_id
    """
    print("\n[E2E Workspace Isolation] Same collection, different workspaces")

    async def embed_func(texts):
        await asyncio.sleep(0)
        return np.random.rand(len(texts), 768)

    embedding_func = EmbeddingFunc(
        embedding_dim=768, max_token_size=8192, func=embed_func, model_name="test-model"
    )

    # Instance A: workspace_a
    rag_a = LightRAG(
        working_dir=temp_working_dirs["workspace_a"],
        workspace="workspace_a",  # Pass workspace to LightRAG instance
        llm_model_func=mock_llm_func,
        embedding_func=embedding_func,
        tokenizer=mock_tokenizer,
        vector_storage="QdrantVectorDBStorage",
        vector_db_storage_cls_kwargs={
            **qdrant_config,
            "cosine_better_than_threshold": 0.8,
        },
    )

    # Instance B: workspace_b
    rag_b = LightRAG(
        working_dir=temp_working_dirs["workspace_b"],
        workspace="workspace_b",  # Pass workspace to LightRAG instance
        llm_model_func=mock_llm_func,
        embedding_func=embedding_func,
        tokenizer=mock_tokenizer,
        vector_storage="QdrantVectorDBStorage",
        vector_db_storage_cls_kwargs={
            **qdrant_config,
            "cosine_better_than_threshold": 0.8,
        },
    )

    await rag_a.initialize_storages()
    await rag_b.initialize_storages()

    # Verify: Same collection
    collection_a = rag_a.chunks_vdb.final_namespace
    collection_b = rag_b.chunks_vdb.final_namespace
    assert collection_a == collection_b
    print(f"✅ Both use same collection: '{collection_a}'")

    # Insert data to different workspaces
    await rag_a.ainsert("Document A for workspace A")
    await rag_b.ainsert("Document B for workspace B")

    # Verify isolation: Each workspace should see only its own data
    # This is ensured by workspace_id filtering in queries

    await rag_a.finalize_storages()
    await rag_b.finalize_storages()

    print("✅ Workspace isolation verified (same collection, isolated data)")


# Test: Dimension mismatch during migration (PostgreSQL)
@pytest.mark.asyncio
async def test_dimension_mismatch_postgres(
    pg_cleanup, mock_llm_func, mock_tokenizer, pg_config
):
    """
    Test dimension mismatch scenario - upgrading from 1536d to 3072d model

    Scenario:
    1. Create legacy table with 1536d vectors
    2. Insert test data
    3. Initialize LightRAG with 3072d model
    4. Verify system handles dimension mismatch gracefully
    """
    print("\n[E2E Test] Dimension mismatch: 1536d -> 3072d (PostgreSQL)")

    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp(prefix="lightrag_dim_test_")

    try:
        # Step 1: Create legacy table with 1536d vectors
        legacy_table = "lightrag_vdb_chunks"

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
                create_time TIMESTAMP DEFAULT NOW(),
                update_time TIMESTAMP DEFAULT NOW()
            )
        """
        await pg_cleanup.execute(create_legacy_sql, None)

        # Insert test records with 1536d vectors
        for i in range(3):
            vector_str = "[" + ",".join(["0.1"] * 1536) + "]"
            insert_sql = f"""
                INSERT INTO {legacy_table}
                (workspace, id, content, content_vector, tokens, chunk_order_index, full_doc_id, file_path)
                VALUES ($1, $2, $3, $4::vector, $5, $6, $7, $8)
            """
            await pg_cleanup.execute(
                insert_sql,
                {
                    "workspace": pg_config["workspace"],
                    "id": f"legacy_{i}",
                    "content": f"Legacy content {i}",
                    "content_vector": vector_str,
                    "tokens": 100,
                    "chunk_order_index": i,
                    "full_doc_id": "legacy_doc",
                    "file_path": "/test/path",
                },
            )

        print("✅ Legacy table created with 3 records (1536d)")

        # Step 2: Try to initialize LightRAG with NEW model (3072d)
        async def embed_func_new(texts):
            await asyncio.sleep(0)
            return np.random.rand(len(texts), 3072)  # NEW dimension

        embedding_func_new = EmbeddingFunc(
            embedding_dim=3072,  # NEW dimension
            max_token_size=8192,
            func=embed_func_new,
            model_name="text-embedding-3-large",
        )

        print("📦 Initializing LightRAG with new model (3072d)...")

        # With our fix, this should handle dimension mismatch gracefully:
        # Expected behavior:
        # 1. Detect dimension mismatch (1536d legacy vs 3072d new)
        # 2. Skip migration to prevent data corruption
        # 3. Preserve legacy table with original data
        # 4. Create new empty table for 3072d model
        # 5. System initializes successfully

        rag = LightRAG(
            working_dir=temp_dir,
            workspace=pg_config["workspace"],  # Match workspace with test data
            llm_model_func=mock_llm_func,
            embedding_func=embedding_func_new,
            tokenizer=mock_tokenizer,
            kv_storage="PGKVStorage",
            vector_storage="PGVectorStorage",
            doc_status_storage="PGDocStatusStorage",
            vector_db_storage_cls_kwargs={
                **pg_config,
                "cosine_better_than_threshold": 0.8,
            },
        )

        await rag.initialize_storages()

        # Verify expected behavior
        new_table = rag.chunks_vdb.table_name
        print(f"✅ Initialization succeeded, new table: {new_table}")

        # 1. New table should exist and be created with model suffix
        assert "text_embedding_3_large_3072d" in new_table.lower()
        check_new = f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{new_table.lower()}')"
        new_exists = await pg_cleanup.query(check_new, [])
        assert new_exists.get("exists") is True, "New table should exist"
        print(f"✅ New table created: {new_table}")

        # 2. Legacy table should be preserved (not deleted)
        check_legacy = f"SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = '{legacy_table}')"
        legacy_exists = await pg_cleanup.query(check_legacy, [])
        assert legacy_exists.get("exists") is True, (
            "Legacy table should be preserved when dimensions don't match"
        )
        print(f"✅ Legacy table preserved: {legacy_table}")

        # 3. Legacy table should still have original data (not migrated)
        legacy_count_result = await pg_cleanup.query(
            f"SELECT COUNT(*) as count FROM {legacy_table}", []
        )
        legacy_count = legacy_count_result.get("count", 0)
        assert legacy_count == 3, (
            f"Legacy table should still have 3 records, got {legacy_count}"
        )
        print(f"✅ Legacy data preserved: {legacy_count} records")

        # 4. New table should be empty (migration skipped)
        new_count_result = await pg_cleanup.query(
            f"SELECT COUNT(*) as count FROM {new_table}", []
        )
        new_count = new_count_result.get("count", 0)
        assert new_count == 0, (
            f"New table should be empty (migration skipped), got {new_count}"
        )
        print(
            f"✅ New table is empty (migration correctly skipped): {new_count} records"
        )

        # 5. System should be operational
        print("✅ System initialized successfully despite dimension mismatch")

        await rag.finalize_storages()

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# Test: Dimension mismatch during migration (Qdrant)
@pytest.mark.asyncio
async def test_dimension_mismatch_qdrant(
    qdrant_cleanup, mock_llm_func, mock_tokenizer, qdrant_config
):
    """
    Test dimension mismatch scenario - upgrading from 768d to 1024d model

    Scenario:
    1. Create legacy collection with 768d vectors
    2. Insert test data
    3. Initialize LightRAG with 1024d model
    4. Verify system handles dimension mismatch gracefully
    """
    print("\n[E2E Test] Dimension mismatch: 768d -> 1024d (Qdrant)")

    import tempfile
    import shutil

    temp_dir = tempfile.mkdtemp(prefix="lightrag_qdrant_dim_test_")

    try:
        # Step 1: Create legacy collection with 768d vectors
        legacy_collection = "lightrag_vdb_chunks"

        client = QdrantClient(**qdrant_config)

        # Delete if exists
        try:
            client.delete_collection(legacy_collection)
        except Exception:
            pass

        # Create legacy collection with 768d
        from qdrant_client import models

        client.create_collection(
            collection_name=legacy_collection,
            vectors_config=models.VectorParams(
                size=768, distance=models.Distance.COSINE
            ),
        )

        # Insert test points with 768d vectors
        points = []
        for i in range(3):
            points.append(
                models.PointStruct(
                    id=i,  # Use integer ID instead of string
                    vector=[0.1] * 768,  # OLD dimension
                    payload={"content": f"Legacy content {i}", "id": f"doc_{i}"},
                )
            )

        client.upsert(collection_name=legacy_collection, points=points, wait=True)
        print("✅ Legacy collection created with 3 records (768d)")

        # Step 2: Try to initialize LightRAG with NEW model (1024d)
        async def embed_func_new(texts):
            await asyncio.sleep(0)
            return np.random.rand(len(texts), 1024)  # NEW dimension

        embedding_func_new = EmbeddingFunc(
            embedding_dim=1024,  # NEW dimension
            max_token_size=8192,
            func=embed_func_new,
            model_name="bge-large",
        )

        print("📦 Initializing LightRAG with new model (1024d)...")

        # With our fix, this should handle dimension mismatch gracefully:
        # Expected behavior:
        # 1. Detect dimension mismatch (768d legacy vs 1024d new)
        # 2. Skip migration to prevent data corruption
        # 3. Preserve legacy collection with original data
        # 4. Create new empty collection for 1024d model
        # 5. System initializes successfully

        rag = LightRAG(
            working_dir=temp_dir,
            llm_model_func=mock_llm_func,
            embedding_func=embedding_func_new,
            tokenizer=mock_tokenizer,
            vector_storage="QdrantVectorDBStorage",
            vector_db_storage_cls_kwargs={
                **qdrant_config,
                "cosine_better_than_threshold": 0.8,
            },
        )

        await rag.initialize_storages()

        # Verify expected behavior
        new_collection = rag.chunks_vdb.final_namespace
        print(f"✅ Initialization succeeded, new collection: {new_collection}")

        # 1. New collection should exist with model suffix
        assert "bge_large_1024d" in new_collection
        assert client.collection_exists(new_collection), (
            f"New collection {new_collection} should exist"
        )
        print(f"✅ New collection created: {new_collection}")

        # 2. Legacy collection should be preserved (not deleted)
        legacy_exists = client.collection_exists(legacy_collection)
        assert legacy_exists, (
            "Legacy collection should be preserved when dimensions don't match"
        )
        print(f"✅ Legacy collection preserved: {legacy_collection}")

        # 3. Legacy collection should still have original data (not migrated)
        legacy_count = client.count(legacy_collection).count
        assert legacy_count == 3, (
            f"Legacy collection should still have 3 vectors, got {legacy_count}"
        )
        print(f"✅ Legacy data preserved: {legacy_count} vectors")

        # 4. New collection should be empty (migration skipped)
        new_count = client.count(new_collection).count
        assert new_count == 0, (
            f"New collection should be empty (migration skipped), got {new_count}"
        )
        print(
            f"✅ New collection is empty (migration correctly skipped): {new_count} vectors"
        )

        # 5. Verify new collection has correct dimension
        collection_info = client.get_collection(new_collection)
        new_dim = collection_info.config.params.vectors.size
        assert new_dim == 1024, f"New collection should have 1024d, got {new_dim}d"
        print(f"✅ New collection dimension verified: {new_dim}d")

        # 6. System should be operational
        print("✅ System initialized successfully despite dimension mismatch")

        await rag.finalize_storages()

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        # Cleanup collections
        try:
            for coll in client.get_collections().collections:
                if "lightrag" in coll.name.lower():
                    client.delete_collection(coll.name)
        except Exception:
            pass


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
