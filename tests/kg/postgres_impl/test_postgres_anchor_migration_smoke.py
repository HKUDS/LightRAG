"""Real-PostgreSQL smoke test for the legacy recovery-anchor migration.

``tests/test_storage_migrations_tracking.py`` pins the migration's rules on the
JSON / NetworkX backends. This module runs the same startup flow once on the
PostgreSQL backends a production deployment uses, so the strict point reads on
``PGKVStorage`` and the full-graph listing on ``PGTableGraphStorage`` are
exercised for real:

1. seed a legacy workspace: documents, chunks and chunk tracking, a graph whose
   ``source_id`` is capped (it names only the first document's chunk), and no
   ``full_entities`` / ``full_relations`` anchors;
2. run ``check_and_migrate_data`` as server startup does;
3. assert the anchors name every tracked object, including the ones the capped
   ``source_id`` hides, then delete the second document and assert the
   surviving objects no longer track its chunk.

Vectors stay in ``NanoVectorDBStorage`` so the test runs on a plain postgres
image without pgvector, like ``pg-smoke.yml``. The markers and the
``POSTGRES_PASSWORD`` guard follow ``tests/kg/pgtable_impl/test_pgtable_smoke.py``.

Cleanup drops whole workspaces, so the test skips when ``POSTGRES_WORKSPACE``
or ``config.ini``'s ``[postgres] workspace`` would remap the PG backends onto
an existing workspace, and drops nothing unless every PG storage resolved to
the test's own workspace.
"""

from __future__ import annotations

import os
from uuid import uuid4

import numpy as np
import pytest

pytestmark = [pytest.mark.pg_smoke, pytest.mark.integration]

if not os.environ.get("POSTGRES_PASSWORD", ""):
    pytest.skip(
        "POSTGRES_PASSWORD not set — skipping pg_smoke tests",
        allow_module_level=True,
    )

from lightrag import LightRAG  # noqa: E402
from lightrag.base import DocStatus  # noqa: E402
from lightrag.kg.postgres_impl import ClientManager  # noqa: E402
from lightrag.kg.shared_storage import initialize_share_data  # noqa: E402
from lightrag.utils import EmbeddingFunc, make_relation_chunk_key  # noqa: E402

_DOCS = [("doc-one", "chunk-one"), ("doc-two", "chunk-two")]
# entity -> (graph source_id, tracked chunk ids). The SHARED pair is tracked in
# both documents' chunks but its capped source_id names only chunk-one.
_ENTITIES = {
    "SHARED": ("chunk-one", ["chunk-one", "chunk-two"]),
    "SHARED-PEER": ("chunk-one", ["chunk-one", "chunk-two"]),
    "SECOND": ("chunk-two", ["chunk-two"]),
    "SECOND-PEER": ("chunk-two", ["chunk-two"]),
}
_RELATIONS = {
    ("SHARED", "SHARED-PEER"): ("chunk-one", ["chunk-one", "chunk-two"]),
    ("SECOND", "SECOND-PEER"): ("chunk-two", ["chunk-two"]),
}


async def _embed(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=np.float32)


async def _llm(*args, **kwargs) -> str:
    return ""


async def _build_pg_rag(tmp_path, workspace: str) -> LightRAG:
    initialize_share_data(workers=1)
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=workspace,
        kv_storage="PGKVStorage",
        doc_status_storage="PGDocStatusStorage",
        graph_storage="PGTableGraphStorage",
        vector_storage="NanoVectorDBStorage",
        llm_model_func=_llm,
        embedding_func=EmbeddingFunc(embedding_dim=8, max_token_size=8192, func=_embed),
    )
    await rag.initialize_storages()
    return rag


async def _seed_legacy_workspace(rag: LightRAG) -> None:
    for doc, chunk in _DOCS:
        await rag.full_docs.upsert({doc: {"content": doc, "file_path": f"{doc}.txt"}})
        await rag.text_chunks.upsert(
            {
                chunk: {
                    "content": doc,
                    "tokens": 1,
                    "chunk_order_index": 0,
                    "full_doc_id": doc,
                    "file_path": f"{doc}.txt",
                    "llm_cache_list": [],
                }
            }
        )
        await rag.doc_status.upsert(
            {
                doc: {
                    "content_summary": doc,
                    "content_length": len(doc),
                    "file_path": f"{doc}.txt",
                    "status": DocStatus.PROCESSED,
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "chunks_list": [chunk],
                    "metadata": {},
                }
            }
        )
    graph = rag.chunk_entity_relation_graph
    for name, (source_id, tracked) in _ENTITIES.items():
        await graph.upsert_node(
            name,
            {
                "entity_id": name,
                "entity_type": "person",
                "description": name,
                "source_id": source_id,
                "file_path": "legacy.txt",
                "created_at": 1,
            },
        )
        await rag.entity_chunks.upsert(
            {name: {"chunk_ids": tracked, "count": len(tracked)}}
        )
    for (src, tgt), (source_id, tracked) in _RELATIONS.items():
        await graph.upsert_edge(
            src,
            tgt,
            {
                "source_id": source_id,
                "description": "Related",
                "keywords": "related",
                "weight": float(len(tracked)),
                "file_path": "legacy.txt",
            },
        )
        await rag.relation_chunks.upsert(
            {
                make_relation_chunk_key(src, tgt): {
                    "chunk_ids": tracked,
                    "count": len(tracked),
                }
            }
        )
    await rag._insert_done()


def _workspace_storages(rag: LightRAG) -> list:
    """Every PG storage holding the test workspace's rows."""
    return [
        rag.full_docs,
        rag.text_chunks,
        rag.full_entities,
        rag.full_relations,
        rag.entity_chunks,
        rag.relation_chunks,
        rag.llm_response_cache,
        rag.doc_status,
        rag.chunk_entity_relation_graph,
    ]


async def _drop_workspace(rag: LightRAG) -> None:
    errors = []
    for storage in _workspace_storages(rag):
        result = await storage.drop()
        if result.get("status") != "success":
            errors.append((type(storage).__name__, storage.namespace, result))
    assert not errors, f"workspace cleanup failed: {errors}"


@pytest.mark.asyncio
async def test_legacy_anchor_migration_then_delete_on_postgres(tmp_path):
    # Same resolution the PG backends apply in initialize(): env, then config.ini.
    if ClientManager.get_config()["workspace"]:
        pytest.skip(
            "POSTGRES_WORKSPACE / [postgres] workspace would remap the workspace "
            "under test onto an existing one"
        )
    workspace = f"anchor_smoke_{uuid4().hex[:8]}"
    rag = await _build_pg_rag(tmp_path, workspace)
    resolved = {
        type(storage).__name__ + "/" + storage.namespace: storage.workspace
        for storage in _workspace_storages(rag)
    }
    if set(resolved.values()) != {workspace}:
        await rag.finalize_storages()
        pytest.fail(f"PG storages left workspace {workspace!r}: {resolved}")
    try:
        await _seed_legacy_workspace(rag)
        for doc, _ in _DOCS:
            assert await rag.full_entities.get_by_id(doc) is None

        await rag.check_and_migrate_data()

        entities = await rag.full_entities.get_by_id("doc-two")
        relations = await rag.full_relations.get_by_id("doc-two")
        assert entities is not None and relations is not None
        assert set(entities["entity_names"]) == set(_ENTITIES)
        assert {tuple(pair) for pair in relations["relation_pairs"]} == set(_RELATIONS)
        doc_one = await rag.full_entities.get_by_id("doc-one")
        assert set(doc_one["entity_names"]) == {"SHARED", "SHARED-PEER"}

        result = await rag.adelete_by_doc_id("doc-two")
        assert result.status == "success", result

        assert await rag.text_chunks.get_by_id("chunk-two") is None
        assert await rag.text_chunks.get_by_id("chunk-one") is not None
        assert (await rag.entity_chunks.get_by_id("SHARED"))["chunk_ids"] == [
            "chunk-one"
        ]
        shared_key = make_relation_chunk_key("SHARED", "SHARED-PEER")
        assert (await rag.relation_chunks.get_by_id(shared_key))["chunk_ids"] == [
            "chunk-one"
        ]
        graph = rag.chunk_entity_relation_graph
        assert await graph.has_edge("SHARED", "SHARED-PEER")
        assert not await graph.has_node("SECOND")
        assert not await graph.has_node("SECOND-PEER")
    finally:
        try:
            await _drop_workspace(rag)
        finally:
            await rag.finalize_storages()
