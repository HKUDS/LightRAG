"""
Minimal POC for OpenSearch Storage in LightRAG

This is a proof-of-concept demonstrating basic OpenSearch storage functionality.
Tests KV storage and vector storage with minimal setup.
"""

import asyncio
import numpy as np
from lightrag.kg.opensearch_impl import (
    OpenSearchKVStorage,
    OpenSearchVectorDBStorage,
    ClientManager,
)
from lightrag.kg.shared_storage import initialize_share_data


# Mock embedding function for testing
class MockEmbeddingFunc:
    """Simple mock embedding function for POC"""

    def __init__(self, dim=128):
        self.embedding_dim = dim
        self.max_token_size = 512
        self.model_name = "mock-embedding"

    async def __call__(self, texts, **kwargs):
        """Generate random embeddings for testing"""
        return np.random.rand(len(texts), self.embedding_dim).astype(np.float32)


async def test_kv_storage():
    """Test basic KV storage operations"""
    print("\n=== Testing KV Storage ===")

    # Configuration
    global_config = {"embedding_batch_num": 10}

    # Create storage instance
    storage = OpenSearchKVStorage(
        namespace="test_chunks",
        global_config=global_config,
        embedding_func=MockEmbeddingFunc(),
        workspace="poc_test",
    )

    try:
        # Initialize
        await storage.initialize()
        print("✓ Storage initialized")

        # Test upsert
        test_data = {
            "chunk-001": {
                "content": "This is a test chunk",
                "tokens": 5,
                "chunk_order_index": 0,
            },
            "chunk-002": {
                "content": "Another test chunk",
                "tokens": 4,
                "chunk_order_index": 1,
            },
        }

        await storage.upsert(test_data)
        print("✓ Upserted 2 documents")

        # Wait for indexing
        await asyncio.sleep(1)

        # Test get_by_id
        doc = await storage.get_by_id("chunk-001")
        if doc and doc.get("content") == "This is a test chunk":
            print("✓ Retrieved document by ID")
        else:
            print("✗ Failed to retrieve document")

        # Test get_by_ids
        docs = await storage.get_by_ids(["chunk-001", "chunk-002"])
        if len([d for d in docs if d is not None]) == 2:
            print("✓ Retrieved multiple documents")
        else:
            print("✗ Failed to retrieve multiple documents")

        # Test is_empty
        is_empty = await storage.is_empty()
        if not is_empty:
            print("✓ Storage is not empty")
        else:
            print("✗ Storage should not be empty")

        # Test delete
        await storage.delete(["chunk-001"])
        print("✓ Deleted document")

        # Verify deletion
        await asyncio.sleep(1)
        doc = await storage.get_by_id("chunk-001")
        if doc is None:
            print("✓ Document successfully deleted")
        else:
            print("✗ Document still exists after deletion")

    finally:
        # Cleanup
        await storage.drop()
        await storage.finalize()
        print("✓ Cleanup completed")


async def test_vector_storage():
    """Test basic vector storage operations"""
    print("\n=== Testing Vector Storage ===")

    # Configuration
    global_config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.5},
    }

    # Create storage instance
    storage = OpenSearchVectorDBStorage(
        namespace="test_vectors",
        global_config=global_config,
        embedding_func=MockEmbeddingFunc(dim=128),
        workspace="poc_test",
        meta_fields={"content", "entity_name"},
    )

    try:
        # Initialize
        await storage.initialize()
        print("✓ Vector storage initialized")

        # Test upsert
        test_data = {
            "vec-001": {
                "content": "Apple is a technology company",
                "entity_name": "Apple",
            },
            "vec-002": {
                "content": "Microsoft develops software",
                "entity_name": "Microsoft",
            },
            "vec-003": {
                "content": "Google is a search engine",
                "entity_name": "Google",
            },
        }

        await storage.upsert(test_data)
        print("✓ Upserted 3 vectors")

        # Wait for indexing
        await asyncio.sleep(2)

        # Test query
        results = await storage.query("technology company", top_k=2)
        if len(results) > 0:
            print(f"✓ Query returned {len(results)} results")
            for i, result in enumerate(results):
                print(
                    f"  Result {i+1}: {result.get('entity_name')} (score: {result.get('distance', 0):.3f})"
                )
        else:
            print("✗ Query returned no results")

    finally:
        # Cleanup
        await storage.drop()
        await storage.finalize()
        print("✓ Cleanup completed")


async def test_connection_manager():
    """Test connection manager singleton behavior"""
    print("\n=== Testing Connection Manager ===")

    # Get first client
    client1 = await ClientManager.get_client()
    print("✓ Got first client")

    # Get second client (should be same instance)
    client2 = await ClientManager.get_client()
    print("✓ Got second client")

    if client1 is client2:
        print("✓ Singleton pattern working (same instance)")
    else:
        print("✗ Different instances returned")

    # Release clients
    await ClientManager.release_client(client1)
    await ClientManager.release_client(client2)
    print("✓ Released clients")


async def main():
    """Run all POC tests"""
    print("=" * 60)
    print("OpenSearch Storage POC")
    print("=" * 60)

    initialize_share_data(workers=1)

    try:
        # Test connection manager
        await test_connection_manager()

        # Test KV storage
        await test_kv_storage()

        # Test vector storage
        await test_vector_storage()

        print("\n" + "=" * 60)
        print("POC Tests Completed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error during POC: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
