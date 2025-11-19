"""
E2E Tests for Qdrant Vector Storage Model Isolation

These tests use a REAL Qdrant server.
Unlike unit tests, these verify actual collection operations, data migration,
and multi-model isolation scenarios.

Prerequisites:
- Qdrant server running
- Environment variables: QDRANT_URL (optional QDRANT_API_KEY)
"""

import os
import pytest
import asyncio
import numpy as np
from lightrag.utils import EmbeddingFunc
from lightrag.kg.qdrant_impl import QdrantVectorDBStorage
from lightrag.namespace import NameSpace
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


# E2E test configuration from environment
@pytest.fixture(scope="module")
def qdrant_config():
    """Real Qdrant configuration from environment variables"""
    return {
        "url": os.getenv("QDRANT_URL", "http://localhost:6333"),
        "api_key": os.getenv("QDRANT_API_KEY", None),
    }


@pytest.fixture(scope="module")
def qdrant_client(qdrant_config):
    """Create a real Qdrant client"""
    client = QdrantClient(
        url=qdrant_config["url"],
        api_key=qdrant_config["api_key"],
        timeout=60,
    )
    yield client
    # Client auto-closes


@pytest.fixture
async def cleanup_collections(qdrant_client):
    """Cleanup test collections before and after each test"""
    collections_to_delete = [
        "lightrag_vdb_chunks",  # legacy
        "e2e_test_chunks",  # legacy with workspace
        "lightrag_vdb_chunks_test_model_768d",
        "lightrag_vdb_chunks_text_embedding_ada_002_1536d",
        "lightrag_vdb_chunks_bge_small_768d",
        "lightrag_vdb_chunks_bge_large_1024d",
    ]

    # Cleanup before test
    for collection in collections_to_delete:
        try:
            if qdrant_client.collection_exists(collection):
                qdrant_client.delete_collection(collection)
        except Exception:
            pass

    yield

    # Cleanup after test
    for collection in collections_to_delete:
        try:
            if qdrant_client.collection_exists(collection):
                qdrant_client.delete_collection(collection)
        except Exception:
            pass


@pytest.fixture
def mock_embedding_func():
    """Create a mock embedding function for testing"""
    async def embed_func(texts, **kwargs):
        return np.array([[0.1] * 768 for _ in texts])

    return EmbeddingFunc(
        embedding_dim=768,
        func=embed_func,
        model_name="test_model"
    )


@pytest.mark.asyncio
async def test_e2e_qdrant_fresh_installation(qdrant_client, cleanup_collections, mock_embedding_func, qdrant_config):
    """
    E2E Test: Fresh Qdrant installation with model_name specified

    Scenario: New workspace, no legacy collection
    Expected: Create new collection with model suffix, no migration needed
    """
    print("\n[E2E Test] Fresh Qdrant installation with model_name")

    # Create storage with model_name
    storage = QdrantVectorDBStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {
                "url": qdrant_config["url"],
                "api_key": qdrant_config["api_key"],
                "cosine_better_than_threshold": 0.8,
            }
        },
        embedding_func=mock_embedding_func,
        workspace="e2e_test"
    )

    # Initialize storage (should create new collection)
    await storage.initialize()

    # Verify collection name
    assert "test_model_768d" in storage.final_namespace
    expected_collection = "lightrag_vdb_chunks_test_model_768d"
    assert storage.final_namespace == expected_collection

    # Verify collection exists
    assert qdrant_client.collection_exists(expected_collection), \
        f"Collection {expected_collection} should exist"

    # Verify collection properties
    collection_info = qdrant_client.get_collection(expected_collection)
    assert collection_info.vectors_count == 0, "New collection should be empty"
    print(f"✅ Fresh installation successful: {expected_collection} created")

    # Verify legacy collection does NOT exist
    assert not qdrant_client.collection_exists("lightrag_vdb_chunks"), \
        "Legacy collection should not exist"
    assert not qdrant_client.collection_exists("e2e_test_chunks"), \
        "Legacy workspace collection should not exist"

    await storage.finalize()


@pytest.mark.asyncio
async def test_e2e_qdrant_legacy_migration(qdrant_client, cleanup_collections, qdrant_config):
    """
    E2E Test: Upgrade from legacy Qdrant collection with automatic migration

    Scenario:
    1. Create legacy collection (without model suffix)
    2. Insert test data
    3. Initialize with model_name (triggers migration)
    4. Verify data migrated to new collection
    """
    print("\n[E2E Test] Legacy Qdrant collection migration")

    # Step 1: Create legacy collection and insert data
    legacy_collection = "e2e_test_chunks"  # workspace-prefixed legacy name

    qdrant_client.create_collection(
        collection_name=legacy_collection,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

    # Insert test data into legacy collection
    from qdrant_client.models import PointStruct

    test_points = [
        PointStruct(
            id=i,
            vector=[0.1] * 1536,
            payload={
                "workspace_id": "e2e_test",
                "content": f"Legacy content {i}",
                "id": f"legacy_doc_{i}",
            }
        )
        for i in range(10)
    ]

    qdrant_client.upsert(
        collection_name=legacy_collection,
        points=test_points,
        wait=True,
    )

    # Verify legacy data exists
    legacy_info = qdrant_client.get_collection(legacy_collection)
    legacy_count = legacy_info.vectors_count
    assert legacy_count == 10, f"Expected 10 vectors in legacy collection, got {legacy_count}"
    print(f"✅ Legacy collection created with {legacy_count} vectors")

    # Step 2: Initialize storage with model_name (triggers migration)
    async def embed_func(texts, **kwargs):
        return np.array([[0.1] * 1536 for _ in texts])

    embedding_func = EmbeddingFunc(
        embedding_dim=1536,
        func=embed_func,
        model_name="text-embedding-ada-002"
    )

    storage = QdrantVectorDBStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {
                "url": qdrant_config["url"],
                "api_key": qdrant_config["api_key"],
                "cosine_better_than_threshold": 0.8,
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
    new_collection = storage.final_namespace
    assert "text_embedding_ada_002_1536d" in new_collection

    # Verify new collection exists and has data
    assert qdrant_client.collection_exists(new_collection), \
        f"New collection {new_collection} should exist"

    new_info = qdrant_client.get_collection(new_collection)
    new_count = new_info.vectors_count

    assert new_count == legacy_count, \
        f"Expected {legacy_count} vectors in new collection, got {new_count}"
    print(f"✅ Data migration verified: {new_count}/{legacy_count} vectors migrated")

    # Verify data content
    sample_points = qdrant_client.scroll(
        collection_name=new_collection,
        limit=1,
        with_payload=True,
    )[0]

    assert len(sample_points) > 0, "Should have at least one point"
    sample = sample_points[0]
    assert "Legacy content" in sample.payload.get("content", "")
    print(f"✅ Data integrity verified: {sample.payload.get('id')}")

    await storage.finalize()


@pytest.mark.asyncio
async def test_e2e_qdrant_multi_model_coexistence(qdrant_client, cleanup_collections, qdrant_config):
    """
    E2E Test: Multiple embedding models coexisting in Qdrant

    Scenario:
    1. Create storage with model A (768d)
    2. Create storage with model B (1024d)
    3. Verify separate collections created
    4. Verify data isolation
    """
    print("\n[E2E Test] Multi-model coexistence in Qdrant")

    # Model A: 768 dimensions
    async def embed_func_a(texts, **kwargs):
        return np.array([[0.1] * 768 for _ in texts])

    embedding_func_a = EmbeddingFunc(
        embedding_dim=768,
        func=embed_func_a,
        model_name="bge-small"
    )

    storage_a = QdrantVectorDBStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {
                "url": qdrant_config["url"],
                "api_key": qdrant_config["api_key"],
                "cosine_better_than_threshold": 0.8,
            }
        },
        embedding_func=embedding_func_a,
        workspace="e2e_test"
    )

    await storage_a.initialize()
    collection_a = storage_a.final_namespace
    assert "bge_small_768d" in collection_a
    print(f"✅ Model A collection created: {collection_a}")

    # Model B: 1024 dimensions
    async def embed_func_b(texts, **kwargs):
        return np.array([[0.1] * 1024 for _ in texts])

    embedding_func_b = EmbeddingFunc(
        embedding_dim=1024,
        func=embed_func_b,
        model_name="bge-large"
    )

    storage_b = QdrantVectorDBStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {
                "url": qdrant_config["url"],
                "api_key": qdrant_config["api_key"],
                "cosine_better_than_threshold": 0.8,
            }
        },
        embedding_func=embedding_func_b,
        workspace="e2e_test"
    )

    await storage_b.initialize()
    collection_b = storage_b.final_namespace
    assert "bge_large_1024d" in collection_b
    print(f"✅ Model B collection created: {collection_b}")

    # Verify collections are different
    assert collection_a != collection_b, "Collections should have different names"
    print(f"✅ Collection isolation verified: {collection_a} != {collection_b}")

    # Verify both collections exist
    assert qdrant_client.collection_exists(collection_a), \
        f"Collection {collection_a} should exist"
    assert qdrant_client.collection_exists(collection_b), \
        f"Collection {collection_b} should exist"
    print("✅ Both collections exist in Qdrant")

    # Verify vector dimensions
    info_a = qdrant_client.get_collection(collection_a)
    info_b = qdrant_client.get_collection(collection_b)

    # Qdrant stores vector config in config.params.vectors
    assert info_a.config.params.vectors.size == 768, "Model A should use 768 dimensions"
    assert info_b.config.params.vectors.size == 1024, "Model B should use 1024 dimensions"
    print(f"✅ Vector dimensions verified: {info_a.config.params.vectors.size}d vs {info_b.config.params.vectors.size}d")

    await storage_a.finalize()
    await storage_b.finalize()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
