"""Live integration smoke test for the official DocumentDB local container.

Run with:

    DOCUMENTDB_URI='mongodb://user:password@localhost:10260/?tls=true&tlsAllowInvalidCertificates=true' \
    DOCUMENTDB_DATABASE=LightRAG \
    ./scripts/test.sh tests/kg/documentdb_impl/test_documentdb_integration.py --run-integration
"""

from __future__ import annotations

import hashlib
import os
import uuid

import numpy as np
import pytest

pytest.importorskip("pymongo", reason="pymongo is required for DocumentDB tests")

from lightrag.base import DocStatus
from lightrag.kg.documentdb_impl import (
    DocumentDBDocStatusStorage,
    DocumentDBGraphStorage,
    DocumentDBKVStorage,
    DocumentDBVectorDBStorage,
)
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import EmbeddingFunc

pytestmark = [pytest.mark.integration, pytest.mark.requires_db]


def _require_documentdb() -> None:
    if not os.getenv("DOCUMENTDB_URI") or not os.getenv("DOCUMENTDB_DATABASE"):
        pytest.skip("DOCUMENTDB_URI and DOCUMENTDB_DATABASE are required")


async def _embed(texts: list[str], **_kwargs) -> np.ndarray:
    vectors = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vector = np.array([digest[0], digest[1], digest[2]], dtype=np.float32)
        norm = np.linalg.norm(vector)
        vectors.append(vector / norm if norm else vector)
    return np.array(vectors, dtype=np.float32)


@pytest.mark.asyncio
async def test_documentdb_all_storage_types() -> None:
    _require_documentdb()
    finalize_share_data()
    initialize_share_data(workers=1)

    workspace = f"docdb_smoke_{uuid.uuid4().hex[:8]}"
    embedding = EmbeddingFunc(
        embedding_dim=3,
        max_token_size=512,
        func=_embed,
    )
    config = {
        "embedding_batch_num": 10,
        "max_graph_nodes": 1000,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": -1.0},
    }
    storages = [
        DocumentDBKVStorage("kv", config, embedding, workspace),
        DocumentDBDocStatusStorage("doc_status", config, embedding, workspace),
        DocumentDBGraphStorage("graph", config, embedding, workspace),
        DocumentDBVectorDBStorage("vectors", config, embedding, workspace),
    ]
    kv, doc_status, graph, vectors = storages
    initialized_storages = []

    try:
        for storage in storages:
            await storage.initialize()
            initialized_storages.append(storage)

        await kv.upsert({"doc-1": {"content": "DocumentDB smoke test"}})
        assert (await kv.get_by_id("doc-1"))["content"] == "DocumentDB smoke test"

        await doc_status.upsert(
            {
                "doc-1": {
                    "status": DocStatus.PROCESSED.value,
                    "file_path": "documentdb-smoke.txt",
                    "content_summary": "smoke test",
                    "content_length": 21,
                    "chunks_count": 1,
                    "created_at": "2026-08-12T00:00:00+00:00",
                    "updated_at": "2026-08-12T00:00:00+00:00",
                    "metadata": {},
                    "error_msg": None,
                }
            }
        )
        status_row = await doc_status.get_by_id("doc-1")
        assert status_row is not None
        assert status_row["status"] == DocStatus.PROCESSED.value
        paginated_docs, total_count = await doc_status.get_docs_paginated(
            sort_field="file_path"
        )
        assert total_count == 1
        assert [doc_id for doc_id, _ in paginated_docs] == ["doc-1"]

        await graph.upsert_node(
            "Alice", {"entity_type": "person", "description": "Researcher"}
        )
        await graph.upsert_node(
            "Bob", {"entity_type": "person", "description": "Developer"}
        )
        await graph.upsert_edge(
            "Alice",
            "Bob",
            {"description": "collaborates", "keywords": "work", "weight": 1.0},
        )
        assert await graph.has_node("Alice")
        assert await graph.has_edge("Alice", "Bob")
        assert await graph.node_degree("Alice") == 1

        await vectors.upsert(
            {
                "vec-1": {"content": "alpha"},
                "vec-2": {"content": "beta"},
            }
        )
        await vectors.index_done_callback()
        vector_row = await vectors.get_by_id("vec-1")
        assert vector_row is not None
        results = await vectors.query("alpha", top_k=2)
        assert {row["id"] for row in results} >= {"vec-1"}
    finally:
        for storage in reversed(initialized_storages):
            try:
                await storage.drop()
            finally:
                await storage.finalize()
        finalize_share_data()
