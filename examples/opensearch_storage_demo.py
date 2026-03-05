"""
Integration test for OpenSearch Storage in LightRAG.

Tests all 4 storage types against a live OpenSearch cluster:
- KV Storage: CRUD, filter_keys
- DocStatus Storage: CRUD, pagination (PIT + search_after), status counts
- Graph Storage: nodes, edges, BFS traversal, search_labels
- Vector Storage: k-NN upsert, query, get/delete

Prerequisites:
    OpenSearch cluster running with k-NN plugin enabled.
    Set env vars: OPENSEARCH_HOSTS, OPENSEARCH_USER, OPENSEARCH_PASSWORD,
                  OPENSEARCH_USE_SSL, OPENSEARCH_VERIFY_CERTS

Usage:
    OPENSEARCH_HOSTS=localhost:9200 OPENSEARCH_USER=admin \
    OPENSEARCH_PASSWORD=<password> OPENSEARCH_USE_SSL=true \
    OPENSEARCH_VERIFY_CERTS=false python examples/opensearch_storage_demo.py
"""

import asyncio
import numpy as np
from lightrag.kg.opensearch_impl import (
    OpenSearchKVStorage,
    OpenSearchDocStatusStorage,
    OpenSearchGraphStorage,
    OpenSearchVectorDBStorage,
    ClientManager,
)
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.base import DocStatus


class MockEmbeddingFunc:
    """Mock embedding function for testing."""

    def __init__(self, dim=128):
        self.embedding_dim = dim
        self.max_token_size = 512
        self.model_name = "mock-embedding"

    async def __call__(self, texts, **kwargs):
        return np.random.rand(len(texts), self.embedding_dim).astype(np.float32)


CONFIG = {
    "embedding_batch_num": 10,
    "max_graph_nodes": 1000,
    "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
}
EMBED = MockEmbeddingFunc()
PASSED = 0
FAILED = 0


def check(condition, msg):
    global PASSED, FAILED
    if condition:
        print(f"  ✓ {msg}")
        PASSED += 1
    else:
        print(f"  ✗ {msg}")
        FAILED += 1


async def test_connection_manager():
    print("\n=== Connection Manager ===")
    client1 = await ClientManager.get_client()
    client2 = await ClientManager.get_client()
    check(client1 is client2, "Singleton pattern (same instance)")
    await ClientManager.release_client(client1)
    await ClientManager.release_client(client2)
    check(True, "Released clients")


async def test_kv_storage():
    print("\n=== KV Storage ===")
    s = OpenSearchKVStorage(
        namespace="integ_kv",
        global_config=CONFIG,
        embedding_func=EMBED,
        workspace="integ",
    )
    await s.initialize()
    try:
        await s.upsert({"k1": {"content": "hello"}, "k2": {"content": "world"}})
        await s.index_done_callback()

        doc = await s.get_by_id("k1")
        check(doc is not None and doc.get("content") == "hello", "get_by_id")

        docs = await s.get_by_ids(["k1", "k2", "missing"])
        check(docs[0] is not None and docs[2] is None, "get_by_ids preserves order")

        missing = await s.filter_keys({"k1", "k99"})
        check(missing == {"k99"}, f"filter_keys: {missing}")

        check(not await s.is_empty(), "is_empty=False")

        await s.delete(["k2"])
        await s.index_done_callback()
        check(await s.get_by_id("k2") is None, "delete + verify")
    finally:
        await s.drop()
        await s.finalize()


async def test_doc_status_storage():
    print("\n=== DocStatus Storage ===")
    s = OpenSearchDocStatusStorage(
        namespace="integ_ds",
        global_config=CONFIG,
        embedding_func=EMBED,
        workspace="integ",
    )
    await s.initialize()
    try:
        # Insert docs
        await s.upsert(
            {
                f"d{i}": {
                    "status": "processed" if i % 2 == 0 else "pending",
                    "file_path": f"/file{i}.txt",
                    "content_summary": f"summary {i}",
                    "content_length": i * 10,
                    "chunks_count": i,
                    "created_at": 1000 + i,
                    "updated_at": 2000 + i,
                }
                for i in range(20)
            }
        )
        await s.index_done_callback()

        # Status counts
        counts = await s.get_all_status_counts()
        check(counts.get("all") == 20, f"all_status_counts: {counts}")
        check(
            counts.get("processed") == 10, f"processed count: {counts.get('processed')}"
        )

        # get_docs_by_status (uses PIT + search_after)
        processed = await s.get_docs_by_status(DocStatus.PROCESSED)
        check(len(processed) == 10, f"get_docs_by_status(processed): {len(processed)}")

        # get_docs_by_track_id (uses PIT + search_after)
        await s.upsert(
            {
                "tracked1": {
                    "status": "processed",
                    "file_path": "/t.txt",
                    "content_summary": "s",
                    "content_length": 1,
                    "chunks_count": 1,
                    "created_at": 100,
                    "updated_at": 200,
                    "track_id": "batch-42",
                }
            }
        )
        await s.index_done_callback()
        tracked = await s.get_docs_by_track_id("batch-42")
        check(len(tracked) == 1, f"get_docs_by_track_id: {len(tracked)}")

        # Paginated (uses PIT + search_after)
        page1, total = await s.get_docs_paginated(page=1, page_size=10)
        check(total == 21, f"paginated total: {total}")
        check(len(page1) == 10, f"page1 size: {len(page1)}")

        page2, _ = await s.get_docs_paginated(page=2, page_size=10)
        check(len(page2) == 10, f"page2 size: {len(page2)}")

        page3, _ = await s.get_docs_paginated(page=3, page_size=10)
        check(len(page3) == 1, f"page3 size: {len(page3)}")

        # With status filter
        filtered, ftotal = await s.get_docs_paginated(
            status_filter=DocStatus.PENDING, page=1, page_size=50
        )
        check(ftotal == 10, f"filtered total: {ftotal}")

        # get_doc_by_file_path
        doc = await s.get_doc_by_file_path("/file0.txt")
        check(doc is not None and doc["_id"] == "d0", "get_doc_by_file_path")
    finally:
        await s.drop()
        await s.finalize()


async def test_graph_storage():
    print("\n=== Graph Storage ===")
    s = OpenSearchGraphStorage(
        namespace="integ_graph",
        global_config=CONFIG,
        embedding_func=EMBED,
        workspace="integ",
    )
    await s.initialize()
    try:
        # Upsert nodes and edges
        await s.upsert_node(
            "Alice", {"entity_type": "person", "description": "A researcher"}
        )
        await s.upsert_node(
            "Bob", {"entity_type": "person", "description": "A developer"}
        )
        await s.upsert_node(
            "Quantum", {"entity_type": "topic", "description": "Quantum computing"}
        )
        await s.upsert_edge(
            "Alice",
            "Bob",
            {"relationship": "knows", "weight": "1.0", "keywords": "collab"},
        )
        await s.upsert_edge(
            "Alice",
            "Quantum",
            {"relationship": "researches", "weight": "2.0", "keywords": "research"},
        )
        await s.upsert_edge(
            "Bob",
            "Quantum",
            {"relationship": "studies", "weight": "0.5", "keywords": "learning"},
        )
        await s.index_done_callback()

        check(await s.has_node("Alice"), "has_node(Alice)")
        check(not await s.has_node("Nobody"), "has_node(Nobody)=False")
        check(await s.has_edge("Alice", "Bob"), "has_edge(Alice,Bob)")

        node = await s.get_node("Alice")
        check(node is not None and node.get("entity_type") == "person", "get_node")
        check(node.get("entity_id") == "Alice", "entity_id field present")

        check(
            await s.node_degree("Alice") == 2,
            f"node_degree(Alice)={await s.node_degree('Alice')}",
        )

        edges = await s.get_node_edges("Alice")
        check(len(edges) == 2, f"get_node_edges: {len(edges)}")

        # Batch ops
        batch = await s.get_nodes_batch(["Alice", "Bob", "Missing"])
        check("Alice" in batch and "Missing" not in batch, "get_nodes_batch")

        degrees = await s.node_degrees_batch(["Alice", "Bob", "Quantum"])
        check(degrees.get("Alice") == 2, f"node_degrees_batch: {degrees}")

        # Knowledge graph (BFS)
        kg = await s.get_knowledge_graph("Alice", max_depth=2)
        check(len(kg.nodes) == 3, f"BFS nodes: {len(kg.nodes)}")
        check(len(kg.edges) == 3, f"BFS edges: {len(kg.edges)}")

        # get_all_labels (uses PIT)
        labels = await s.get_all_labels()
        check("Alice" in labels and "Bob" in labels, f"get_all_labels: {labels}")

        # get_all_nodes (uses PIT)
        all_nodes = await s.get_all_nodes()
        check(len(all_nodes) == 3, f"get_all_nodes: {len(all_nodes)}")

        # get_all_edges (uses PIT)
        all_edges = await s.get_all_edges()
        check(len(all_edges) == 3, f"get_all_edges: {len(all_edges)}")

        # search_labels
        found = await s.search_labels("ali", limit=10)
        check("Alice" in found, f"search_labels('ali'): {found}")

        # popular_labels
        popular = await s.get_popular_labels(limit=10)
        check(len(popular) > 0, f"get_popular_labels: {popular}")

        # Delete node (cascading)
        await s.delete_node("Bob")
        await s.index_done_callback()
        check(not await s.has_node("Bob"), "delete_node cascade")
        check(not await s.has_edge("Alice", "Bob"), "edges removed after delete_node")

        print(f"  (PPL graphlookup: {s._ppl_graphlookup_available})")
    finally:
        await s.drop()
        await s.finalize()


async def test_vector_storage():
    print("\n=== Vector Storage ===")
    s = OpenSearchVectorDBStorage(
        namespace="integ_vec",
        global_config=CONFIG,
        embedding_func=EMBED,
        workspace="integ",
        meta_fields={"content", "entity_name"},
    )
    await s.initialize()
    try:
        await s.upsert(
            {
                "v1": {"content": "apple fruit"},
                "v2": {"content": "banana fruit"},
                "v3": {"content": "quantum physics"},
            }
        )
        await s.index_done_callback()

        results = await s.query("apple", top_k=3)
        check(len(results) > 0, f"query returned {len(results)} results")
        check(all("distance" in r for r in results), "results have distance")

        doc = await s.get_by_id("v1")
        check(doc is not None and doc["id"] == "v1", "get_by_id")

        docs = await s.get_by_ids(["v1", "v2", "missing"])
        check(docs[0] is not None and docs[2] is None, "get_by_ids")

        vecs = await s.get_vectors_by_ids(["v1"])
        check("v1" in vecs and len(vecs["v1"]) == 128, "get_vectors_by_ids")

        await s.delete(["v3"])
        await s.index_done_callback()
        check(await s.get_by_id("v3") is None, "delete + verify")
    finally:
        await s.drop()
        await s.finalize()


async def main():
    print("=" * 60)
    print("OpenSearch Storage Integration Tests")
    print("=" * 60)

    initialize_share_data(workers=1)

    try:
        await test_connection_manager()
        await test_kv_storage()
        await test_doc_status_storage()
        await test_graph_storage()
        await test_vector_storage()
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback

        traceback.print_exc()

    print(f"\n{'=' * 60}")
    print(f"Results: {PASSED} passed, {FAILED} failed")
    print(f"{'=' * 60}")
    if FAILED > 0:
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
