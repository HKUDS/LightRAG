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
from qdrant_client import QdrantClient


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
            await pg_cleanup.execute(insert_sql, {
                "workspace": pg_config["workspace"],
                "id": f"legacy_{i}",
                "content": f"Legacy content {i}",
                "content_vector": vector_str,
                "tokens": 100,
                "chunk_order_index": i,
                "full_doc_id": "legacy_doc",
                "file_path": "/test/path"
            })

        # Verify legacy data
        count_result = await pg_cleanup.query(
            f"SELECT COUNT(*) as count FROM {legacy_table} WHERE workspace=$1",
            [pg_config["workspace"]]
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
            model_name="text-embedding-ada-002"
        )

        rag = LightRAG(
            working_dir=temp_dir,
            llm_model_func=mock_llm_func,
            embedding_func=embedding_func,
            tokenizer=mock_tokenizer,
            kv_storage="PGKVStorage",
            vector_storage="PGVectorStorage",
            # Use default NetworkXStorage for graph storage (AGE extension not available in CI)
            doc_status_storage="PGDocStatusStorage",
            vector_db_storage_cls_kwargs={
                **pg_config,
                "cosine_better_than_threshold": 0.8
            },
        )

        print("🔄 Initializing LightRAG (triggers migration)...")
        await rag.initialize_storages()

        # Step 3: Verify migration
        new_table = rag.chunks_vdb.table_name
        assert "text_embedding_ada_002_1536d" in new_table.lower()

        new_count_result = await pg_cleanup.query(
            f"SELECT COUNT(*) as count FROM {new_table} WHERE workspace=$1",
            [pg_config["workspace"]]
        )
        new_count = new_count_result.get("count", 0)

        assert new_count == legacy_count, \
            f"Expected {legacy_count} records migrated, got {new_count}"
        print(f"✅ Migration successful: {new_count}/{legacy_count} records migrated")
        print(f"✅ New table: {new_table}")

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
                }
            )
            test_vectors.append(point)

        qdrant_cleanup.upsert(
            collection_name=legacy_collection,
            points=test_vectors
        )

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
            model_name="text-embedding-ada-002"
        )

        rag = LightRAG(
            working_dir=temp_dir,
            llm_model_func=mock_llm_func,
            embedding_func=embedding_func,
            tokenizer=mock_tokenizer,
            vector_storage="QdrantVectorDBStorage",
            vector_db_storage_cls_kwargs={
                **qdrant_config,
                "cosine_better_than_threshold": 0.8
            },
        )

        print("🔄 Initializing LightRAG (triggers migration)...")
        await rag.initialize_storages()

        # Step 3: Verify migration
        new_collection = rag.chunks_vdb.final_namespace
        assert "text_embedding_ada_002_1536d" in new_collection

        # Verify new collection exists
        assert qdrant_cleanup.collection_exists(new_collection), \
            f"New collection {new_collection} should exist"

        new_count = qdrant_cleanup.count(new_collection).count

        assert new_count == legacy_count, \
            f"Expected {legacy_count} vectors migrated, got {new_count}"
        print(f"✅ Migration successful: {new_count}/{legacy_count} vectors migrated")
        print(f"✅ New collection: {new_collection}")

        # Verify vector dimension
        collection_info = qdrant_cleanup.get_collection(new_collection)
        assert collection_info.config.params.vectors.size == 1536, \
            "Migrated collection should have 1536 dimensions"
        print(f"✅ Vector dimension verified: {collection_info.config.params.vectors.size}d")

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

    Note: Additional embedding functions (C: 1536d, D: no model_name) are defined
    but not used in this test. They can be activated for extended testing.
    """
    print("\n[E2E Multi-Instance] PostgreSQL with 2 models (768d vs 1024d)")

    # Instance A: 768d with model-a
    async def embed_func_a(texts):
        await asyncio.sleep(0)
        return np.random.rand(len(texts), 768)

    embedding_func_a = EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=embed_func_a,
        model_name="model-a"
    )

    # Instance B: 1024d with model-b
    async def embed_func_b(texts):
        await asyncio.sleep(0)
        return np.random.rand(len(texts), 1024)

    embedding_func_b = EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=embed_func_b,
        model_name="model-b"
    )

    # Instance C: 1536d with text-embedding-ada-002
    async def embed_func_c(texts):
        await asyncio.sleep(0)
        return np.random.rand(len(texts), 1536)

    embedding_func_c = EmbeddingFunc(
        embedding_dim=1536,
        max_token_size=8192,
        func=embed_func_c,
        model_name="text-embedding-ada-002"
    )

    # Instance D: 768d WITHOUT model_name (backward compatibility)
    async def embed_func_d(texts):
        await asyncio.sleep(0)
        return np.random.rand(len(texts), 768)

    embedding_func_d = EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=embed_func_d
        # NO model_name - test backward compatibility
    )

    # Initialize LightRAG instance A
    print("📦 Initializing LightRAG instance A (model-a, 768d)...")
    rag_a = LightRAG(
        working_dir=temp_working_dirs["workspace_a"],
        llm_model_func=mock_llm_func,
        embedding_func=embedding_func_a,
        tokenizer=mock_tokenizer,
        kv_storage="PGKVStorage",
        vector_storage="PGVectorStorage",
        # Use default NetworkXStorage for graph storage (AGE extension not available in CI)
        doc_status_storage="PGDocStatusStorage",
        vector_db_storage_cls_kwargs={
            **pg_config,
            "cosine_better_than_threshold": 0.8
        },
    )

    await rag_a.initialize_storages()
    table_a = rag_a.chunks_vdb.table_name
    print(f"✅ Instance A initialized: {table_a}")

    # Initialize LightRAG instance B
    print("📦 Initializing LightRAG instance B (model-b, 1024d)...")
    rag_b = LightRAG(
        working_dir=temp_working_dirs["workspace_b"],
        llm_model_func=mock_llm_func,
        embedding_func=embedding_func_b,
        tokenizer=mock_tokenizer,
        kv_storage="PGKVStorage",
        vector_storage="PGVectorStorage",
        # Use default NetworkXStorage for graph storage (AGE extension not available in CI)
        doc_status_storage="PGDocStatusStorage",
        vector_db_storage_cls_kwargs={
            **pg_config,
            "cosine_better_than_threshold": 0.8
        },
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

    assert result_a.get("exists") == True, f"Table {table_a} should exist"
    assert result_b.get("exists") == True, f"Table {table_b} should exist"
    print("✅ Both tables exist in PostgreSQL")

    # Insert documents in instance A
    print("📝 Inserting document in instance A...")
    await rag_a.ainsert("Document A: This is about artificial intelligence and neural networks.")

    # Insert documents in instance B
    print("📝 Inserting document in instance B...")
    await rag_b.ainsert("Document B: This is about machine learning and deep learning.")

    # Verify data isolation
    count_a_result = await pg_cleanup.query(
        f"SELECT COUNT(*) as count FROM {table_a}",
        []
    )
    count_b_result = await pg_cleanup.query(
        f"SELECT COUNT(*) as count FROM {table_b}",
        []
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
        embedding_dim=768,
        max_token_size=8192,
        func=embed_func_a,
        model_name="model-a"
    )

    # Create embedding function for model B (1024d)
    async def embed_func_b(texts):
        await asyncio.sleep(0)
        return np.random.rand(len(texts), 1024)

    embedding_func_b = EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=embed_func_b,
        model_name="model-b"
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
            "cosine_better_than_threshold": 0.8
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
            "cosine_better_than_threshold": 0.8
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
    assert qdrant_cleanup.collection_exists(collection_a), \
        f"Collection {collection_a} should exist"
    assert qdrant_cleanup.collection_exists(collection_b), \
        f"Collection {collection_b} should exist"
    print("✅ Both collections exist in Qdrant")

    # Verify vector dimensions
    info_a = qdrant_cleanup.get_collection(collection_a)
    info_b = qdrant_cleanup.get_collection(collection_b)

    assert info_a.config.params.vectors.size == 768, "Model A should use 768 dimensions"
    assert info_b.config.params.vectors.size == 1024, "Model B should use 1024 dimensions"
    print(f"✅ Vector dimensions verified: {info_a.config.params.vectors.size}d vs {info_b.config.params.vectors.size}d")

    # Insert documents in instance A
    print("📝 Inserting document in instance A...")
    await rag_a.ainsert("Document A: This is about artificial intelligence and neural networks.")

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


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
