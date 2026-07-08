"""
LanceDB unified storage backend demo for LightRAG.

LanceDB is an embedded database — this demo needs NO external services and
no API keys. It exercises all four storage types against a local directory:

- KV Storage:        CRUD, filter_keys, timestamps
- Vector Storage:    deferred-embedding upsert, cosine query, CJK full-text search
- Graph Storage:     nodes, edges, degrees, BFS subgraph, label search
- DocStatus Storage: CRUD, status counts, pagination

Usage:
    python examples/lightrag_lancedb_demo.py

To use LanceDB as the storage backend of a real LightRAG instance:

    rag = LightRAG(
        working_dir="./rag_storage",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=openai_embed,
        kv_storage="LanceDBKVStorage",
        vector_storage="LanceDBVectorStorage",
        graph_storage="LanceDBGraphStorage",
        doc_status_storage="LanceDBDocStatusStorage",
    )
    await rag.initialize_storages()

or via environment variables for the API server:

    LIGHTRAG_KV_STORAGE=LanceDBKVStorage
    LIGHTRAG_VECTOR_STORAGE=LanceDBVectorStorage
    LIGHTRAG_GRAPH_STORAGE=LanceDBGraphStorage
    LIGHTRAG_DOC_STATUS_STORAGE=LanceDBDocStatusStorage
    LANCEDB_URI=./lancedb_data
"""

import asyncio
import hashlib
import shutil
import tempfile

import numpy as np

from lightrag.base import DocStatus
from lightrag.kg.lancedb_impl import (
    LanceDBDocStatusStorage,
    LanceDBGraphStorage,
    LanceDBKVStorage,
    LanceDBVectorStorage,
)
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import EmbeddingFunc

DIM = 64


async def mock_embed(texts, **kwargs):
    """Deterministic mock embedding (replace with a real model in production)."""

    def vec(text):
        seed = int.from_bytes(hashlib.md5(text.encode()).digest()[:4], "little")
        return np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)

    return np.array([vec(t) for t in texts])


async def main():
    working_dir = tempfile.mkdtemp(prefix="lightrag_lancedb_demo_")
    print(f"LanceDB directory: {working_dir}/lancedb\n")
    initialize_share_data()

    global_config = {
        "working_dir": working_dir,
        "embedding_batch_num": 16,
        "max_graph_nodes": 1000,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
    }
    embedding_func = EmbeddingFunc(
        embedding_dim=DIM, max_token_size=8192, func=mock_embed
    )

    # ------------------------------------------------------------- KV storage
    print("=== KV Storage ===")
    kv = LanceDBKVStorage(
        namespace="text_chunks",
        workspace="demo",
        global_config=global_config,
        embedding_func=embedding_func,
    )
    await kv.initialize()
    await kv.upsert(
        {"chunk-1": {"content": "LightRAG combines knowledge graphs with RAG."}}
    )
    record = await kv.get_by_id("chunk-1")
    print(f"get_by_id -> content={record['content']!r}")
    print(f"filter_keys -> missing={await kv.filter_keys({'chunk-1', 'chunk-2'})}\n")

    # --------------------------------------------------------- Vector storage
    print("=== Vector Storage (with CJK full-text search) ===")
    vdb = LanceDBVectorStorage(
        namespace="chunks",
        workspace="demo",
        global_config=global_config,
        embedding_func=embedding_func,
        meta_fields={"full_doc_id", "content", "file_path"},
    )
    await vdb.initialize()
    contents = {
        "chunk-1": "朱元璋是明朝的開國皇帝，定都南京。",
        "chunk-2": "康熙皇帝是清朝在位時間最長的皇帝。",
        "chunk-3": "LightRAG supports graph-based retrieval augmented generation.",
    }
    await vdb.upsert(
        {
            chunk_id: {"content": text, "full_doc_id": "doc-1", "file_path": "demo.txt"}
            for chunk_id, text in contents.items()
        }
    )
    await vdb.index_done_callback()  # flush: embeds in batches and persists

    hits = await vdb.query(contents["chunk-3"], top_k=3)
    print(f"vector query   -> top hit: {hits[0]['id']} (similarity={hits[0]['distance']:.3f})")
    fts_hits = await vdb.full_text_search("皇帝", top_k=5)
    print(f"FTS query 皇帝 -> {[hit['id'] for hit in fts_hits]}")
    fts_hits = await vdb.full_text_search("graph retrieval", top_k=5)
    print(f"FTS query en   -> {[hit['id'] for hit in fts_hits]}\n")

    # ---------------------------------------------------------- Graph storage
    print("=== Graph Storage ===")
    graph = LanceDBGraphStorage(
        namespace="chunk_entity_relation",
        workspace="demo",
        global_config=global_config,
        embedding_func=embedding_func,
    )
    await graph.initialize()
    await graph.upsert_node(
        "朱元璋",
        {"entity_id": "朱元璋", "entity_type": "person", "description": "明朝開國皇帝"},
    )
    await graph.upsert_node(
        "明朝",
        {"entity_id": "明朝", "entity_type": "organization", "description": "中國朝代"},
    )
    await graph.upsert_edge(
        "朱元璋", "明朝", {"weight": 1.0, "description": "建立", "keywords": "founder"}
    )
    print(f"has_edge(明朝, 朱元璋) [undirected] -> {await graph.has_edge('明朝', '朱元璋')}")
    print(f"node_degree(朱元璋) -> {await graph.node_degree('朱元璋')}")
    print(f"search_labels('朱') -> {await graph.search_labels('朱')}")
    subgraph = await graph.get_knowledge_graph("朱元璋", max_depth=2)
    print(
        f"get_knowledge_graph -> {len(subgraph.nodes)} nodes, {len(subgraph.edges)} edges\n"
    )

    # ------------------------------------------------------ DocStatus storage
    print("=== DocStatus Storage ===")
    doc_status = LanceDBDocStatusStorage(
        namespace="doc_status",
        workspace="demo",
        global_config=global_config,
        embedding_func=None,
    )
    await doc_status.initialize()
    await doc_status.upsert(
        {
            "doc-1": {
                "status": DocStatus.PROCESSED,
                "content_summary": "Ming dynasty history",
                "content_length": 120,
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:01:00+00:00",
                "file_path": "demo.txt",
                "content_hash": "demo-hash",
                "metadata": {},
                "chunks_count": 3,
                "chunks_list": list(contents.keys()),
            }
        }
    )
    print(f"status counts -> {await doc_status.get_all_status_counts()}")
    page, total = await doc_status.get_docs_paginated(page=1, page_size=10)
    print(f"paginated -> total={total}, first={page[0][0]}\n")

    # ----------------------------------------------------------------- Cleanup
    for storage in (kv, vdb, graph, doc_status):
        await storage.finalize()
    finalize_share_data()
    shutil.rmtree(working_dir, ignore_errors=True)
    print("Demo finished successfully.")


if __name__ == "__main__":
    asyncio.run(main())
