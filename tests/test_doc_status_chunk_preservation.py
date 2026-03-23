import asyncio
from datetime import datetime, timezone
from types import MethodType
from uuid import uuid4

import numpy as np
import pytest

import lightrag.lightrag as lightrag_module
from lightrag.base import DocStatus
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.lightrag import LightRAG
from lightrag.utils import (
    EmbeddingFunc,
    Tokenizer,
    compute_mdhash_id,
    make_relation_chunk_key,
)

pytestmark = pytest.mark.offline


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


async def _dummy_llm(*args, **kwargs) -> str:
    return "ok"


def _deterministic_chunking(
    tokenizer,
    content: str,
    split_by_character,
    split_by_character_only: bool,
    chunk_overlap_token_size: int,
    chunk_token_size: int,
) -> list[dict]:
    return [
        {"tokens": 1, "content": f"{content}::chunk1", "chunk_order_index": 0},
        {"tokens": 1, "content": f"{content}::chunk2", "chunk_order_index": 1},
    ]


def _failing_chunking(
    tokenizer,
    content: str,
    split_by_character,
    split_by_character_only: bool,
    chunk_overlap_token_size: int,
    chunk_token_size: int,
) -> list[dict]:
    raise RuntimeError("chunking fail sentinel")


def _status_to_text(status: object) -> str:
    if isinstance(status, DocStatus):
        return status.value
    return str(status).replace("DocStatus.", "").lower()


async def _build_rag(tmp_path, test_name: str, chunking_func) -> LightRAG:
    workspace = f"{test_name}_{uuid4().hex[:8]}"
    rag = LightRAG(
        working_dir=str(tmp_path / test_name),
        workspace=workspace,
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8,
            max_token_size=8192,
            func=_dummy_embedding,
        ),
        tokenizer=Tokenizer("test-tokenizer", _SimpleTokenizerImpl()),
        chunking_func=chunking_func,
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    return rag


async def _seed_chunk_cache_entries(
    rag: LightRAG, chunk_ids: list[str], prefix: str
) -> list[str]:
    updates = {}
    cache_records = {}
    cache_ids: list[str] = []

    for idx, chunk_id in enumerate(chunk_ids):
        chunk_data = await rag.text_chunks.get_by_id(chunk_id)
        assert chunk_data is not None
        cache_id = f"{prefix}-cache-{idx}"
        chunk_data["llm_cache_list"] = [cache_id]
        updates[chunk_id] = chunk_data
        cache_records[cache_id] = {"cache_type": "extract", "return": f"cached-{idx}"}
        cache_ids.append(cache_id)

    await rag.text_chunks.upsert(updates)
    await rag.llm_response_cache.upsert(cache_records)
    return cache_ids


async def _seed_delete_retry_state(
    rag: LightRAG,
    *,
    doc_id: str,
    status_chunk_ids: list[str],
    tracking_chunk_ids: list[str],
    chunk_owners: dict[str, str],
    metadata: dict | None = None,
) -> dict[str, str]:
    entity_a = "ENTITY-A"
    entity_b = "ENTITY-B"
    relation_key = make_relation_chunk_key(entity_a, entity_b)
    source_id = GRAPH_FIELD_SEP.join(tracking_chunk_ids)
    now = datetime.now(timezone.utc).isoformat()

    await rag.full_docs.upsert(
        {doc_id: {"content": "delete retry state doc", "file_path": "delete_retry.txt"}}
    )
    await rag.doc_status.upsert(
        {
            doc_id: {
                "status": DocStatus.PROCESSED,
                "content_summary": "delete retry state",
                "content_length": 22,
                "chunks_count": len(status_chunk_ids),
                "chunks_list": status_chunk_ids,
                "created_at": now,
                "updated_at": now,
                "file_path": "delete_retry.txt",
                "track_id": f"track-{doc_id}",
                "error_msg": "",
                "metadata": metadata or {},
            }
        }
    )

    chunk_payload = {}
    for chunk_id, owner_doc_id in chunk_owners.items():
        chunk_payload[chunk_id] = {
            "content": f"{chunk_id} content",
            "file_path": f"{chunk_id}.txt",
            "full_doc_id": owner_doc_id,
        }

    if chunk_payload:
        await rag.text_chunks.upsert(chunk_payload)
        await rag.chunks_vdb.upsert(chunk_payload)

    await rag.full_entities.upsert({doc_id: {"entity_names": [entity_a, entity_b]}})
    await rag.full_relations.upsert(
        {doc_id: {"relation_pairs": [(entity_a, entity_b)]}}
    )
    await rag.entity_chunks.upsert(
        {
            entity_a: {
                "chunk_ids": tracking_chunk_ids,
                "count": len(tracking_chunk_ids),
            },
            entity_b: {
                "chunk_ids": tracking_chunk_ids,
                "count": len(tracking_chunk_ids),
            },
        }
    )
    await rag.relation_chunks.upsert(
        {
            relation_key: {
                "chunk_ids": tracking_chunk_ids,
                "count": len(tracking_chunk_ids),
            }
        }
    )

    created_at = int(datetime.now(timezone.utc).timestamp())
    for entity_name in [entity_a, entity_b]:
        await rag.chunk_entity_relation_graph.upsert_node(
            entity_name,
            {
                "entity_id": entity_name,
                "source_id": source_id,
                "description": f"{entity_name} description",
                "entity_type": "test",
                "file_path": "delete_retry.txt",
                "created_at": created_at,
                "truncate": "",
            },
        )

    await rag.chunk_entity_relation_graph.upsert_edge(
        entity_a,
        entity_b,
        {
            "source": entity_a,
            "target": entity_b,
            "source_id": source_id,
            "description": "related",
            "keywords": "test",
            "weight": 1.0,
            "file_path": "delete_retry.txt",
        },
    )

    await rag.entities_vdb.upsert(
        {
            compute_mdhash_id(entity_a, prefix="ent-"): {
                "content": f"{entity_a}\n{entity_a} description",
                "entity_name": entity_a,
                "source_id": source_id,
                "description": f"{entity_a} description",
                "entity_type": "test",
                "file_path": "delete_retry.txt",
            },
            compute_mdhash_id(entity_b, prefix="ent-"): {
                "content": f"{entity_b}\n{entity_b} description",
                "entity_name": entity_b,
                "source_id": source_id,
                "description": f"{entity_b} description",
                "entity_type": "test",
                "file_path": "delete_retry.txt",
            },
        }
    )
    await rag.relationships_vdb.upsert(
        {
            compute_mdhash_id(entity_a + entity_b, prefix="rel-"): {
                "content": f"test\t{entity_a}\n{entity_b}\nrelated",
                "src_id": entity_a,
                "tgt_id": entity_b,
                "source_id": source_id,
                "description": "related",
                "keywords": "test",
                "weight": 1.0,
                "file_path": "delete_retry.txt",
            }
        }
    )

    return {
        "entity_a": entity_a,
        "entity_b": entity_b,
        "relation_key": relation_key,
    }


async def _succeed_rebuild_from_remaining_chunks(
    entities_to_rebuild,
    relationships_to_rebuild,
    knowledge_graph_inst,
    entities_vdb,
    relationships_vdb,
    **kwargs,
):
    for entity_name, remaining_chunk_ids in entities_to_rebuild.items():
        node = await knowledge_graph_inst.get_node(entity_name)
        assert node is not None
        updated_node = {
            **node,
            "source_id": GRAPH_FIELD_SEP.join(remaining_chunk_ids),
        }
        await knowledge_graph_inst.upsert_node(entity_name, updated_node)
        await entities_vdb.upsert(
            {
                compute_mdhash_id(entity_name, prefix="ent-"): {
                    "content": f"{entity_name}\n{updated_node['description']}",
                    "entity_name": entity_name,
                    "source_id": updated_node["source_id"],
                    "description": updated_node["description"],
                    "entity_type": updated_node["entity_type"],
                    "file_path": updated_node["file_path"],
                }
            }
        )

    for (src, tgt), remaining_chunk_ids in relationships_to_rebuild.items():
        edge = await knowledge_graph_inst.get_edge(src, tgt)
        assert edge is not None
        updated_edge = {
            **edge,
            "source_id": GRAPH_FIELD_SEP.join(remaining_chunk_ids),
        }
        await knowledge_graph_inst.upsert_edge(src, tgt, updated_edge)
        await relationships_vdb.upsert(
            {
                compute_mdhash_id(src + tgt, prefix="rel-"): {
                    "content": f"{updated_edge['keywords']}\t{src}\n{tgt}\n{updated_edge['description']}",
                    "src_id": src,
                    "tgt_id": tgt,
                    "source_id": updated_edge["source_id"],
                    "description": updated_edge["description"],
                    "keywords": updated_edge["keywords"],
                    "weight": updated_edge["weight"],
                    "file_path": updated_edge["file_path"],
                }
            }
        )


@pytest.mark.asyncio
async def test_extract_failure_preserves_chunks_and_allows_delete_with_cache_cleanup(
    tmp_path, monkeypatch
):
    rag = await _build_rag(tmp_path, "extract_failure_cleanup", _deterministic_chunking)
    try:
        content = "extract failure document"
        file_path = "extract_failure.txt"
        doc_id = compute_mdhash_id(content, prefix="doc-")
        await rag.apipeline_enqueue_documents(input=content, file_paths=file_path)

        async def fail_extract(self, chunks, pipeline_status, pipeline_status_lock):
            raise RuntimeError("extract fail sentinel")

        rag._process_extract_entities = MethodType(fail_extract, rag)

        await rag.apipeline_process_enqueue_documents()

        doc_status = await rag.doc_status.get_by_id(doc_id)
        assert doc_status is not None
        assert _status_to_text(doc_status["status"]) == "failed"
        chunk_ids = doc_status.get("chunks_list", [])
        assert len(chunk_ids) == 2
        assert doc_status.get("chunks_count") == 2

        cache_ids = await _seed_chunk_cache_entries(rag, chunk_ids, "extract")

        result = await rag.adelete_by_doc_id(doc_id, delete_llm_cache=True)
        assert result.status == "success"

        deleted_chunks = await rag.text_chunks.get_by_ids(chunk_ids)
        assert all(item is None for item in deleted_chunks)
        deleted_cache = [
            await rag.llm_response_cache.get_by_id(cid) for cid in cache_ids
        ]
        assert all(item is None for item in deleted_cache)
        assert await rag.doc_status.get_by_id(doc_id) is None
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_extract_failure_before_chunking_preserves_previous_chunk_snapshot(
    tmp_path,
):
    rag = await _build_rag(tmp_path, "extract_failure_pre_chunking", _failing_chunking)
    try:
        content = "chunking failure document"
        file_path = "chunking_failure.txt"
        doc_id = compute_mdhash_id(content, prefix="doc-")
        await rag.apipeline_enqueue_documents(input=content, file_paths=file_path)

        previous_chunks = ["chunk-old-1", "chunk-old-2", "chunk-old-3"]
        existing = await rag.doc_status.get_by_id(doc_id)
        assert existing is not None
        await rag.doc_status.upsert(
            {
                doc_id: {
                    "status": DocStatus.FAILED,
                    "content_summary": existing["content_summary"],
                    "content_length": existing["content_length"],
                    "chunks_count": len(previous_chunks),
                    "chunks_list": previous_chunks,
                    "created_at": existing["created_at"],
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "file_path": existing["file_path"],
                    "track_id": existing["track_id"],
                    "error_msg": "previous failure",
                    "metadata": {"source": "test"},
                }
            }
        )

        await rag.apipeline_process_enqueue_documents()

        failed_status = await rag.doc_status.get_by_id(doc_id)
        assert failed_status is not None
        assert _status_to_text(failed_status["status"]) == "failed"
        assert failed_status.get("chunks_list") == previous_chunks
        assert failed_status.get("chunks_count") == len(previous_chunks)
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_merge_failure_preserves_chunks_and_skip_cache_cleanup_when_disabled(
    tmp_path, monkeypatch
):
    rag = await _build_rag(
        tmp_path, "merge_failure_keep_cache", _deterministic_chunking
    )
    try:
        content = "merge failure document"
        file_path = "merge_failure.txt"
        doc_id = compute_mdhash_id(content, prefix="doc-")
        await rag.apipeline_enqueue_documents(input=content, file_paths=file_path)

        async def ok_extract(self, chunks, pipeline_status, pipeline_status_lock):
            return {"chunk_count": len(chunks)}

        async def fail_merge(**kwargs):
            raise RuntimeError("merge fail sentinel")

        rag._process_extract_entities = MethodType(ok_extract, rag)
        monkeypatch.setattr(lightrag_module, "merge_nodes_and_edges", fail_merge)

        await rag.apipeline_process_enqueue_documents()

        doc_status = await rag.doc_status.get_by_id(doc_id)
        assert doc_status is not None
        assert _status_to_text(doc_status["status"]) == "failed"
        chunk_ids = doc_status.get("chunks_list", [])
        assert len(chunk_ids) == 2
        assert doc_status.get("chunks_count") == 2

        cache_ids = await _seed_chunk_cache_entries(rag, chunk_ids, "merge")
        result = await rag.adelete_by_doc_id(doc_id, delete_llm_cache=False)
        assert result.status == "success"

        remaining_cache = [
            await rag.llm_response_cache.get_by_id(cid) for cid in cache_ids
        ]
        assert all(item is not None for item in remaining_cache)
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_delete_rebuild_failure_prunes_chunk_tracking_before_abort(
    tmp_path, monkeypatch
):
    rag = await _build_rag(
        tmp_path, "delete_rebuild_failure_chunk_tracking", _deterministic_chunking
    )
    try:
        doc_id = "doc-delete-rebuild-failure"
        keep_chunk_id = "chunk-keep"
        drop_chunk_id = "chunk-drop"
        seeded = await _seed_delete_retry_state(
            rag,
            doc_id=doc_id,
            status_chunk_ids=[drop_chunk_id],
            tracking_chunk_ids=[keep_chunk_id, drop_chunk_id],
            chunk_owners={
                keep_chunk_id: "doc-keep",
                drop_chunk_id: doc_id,
            },
        )
        entity_a = seeded["entity_a"]
        entity_b = seeded["entity_b"]
        relation_key = seeded["relation_key"]

        async def fail_rebuild(**kwargs):
            raise RuntimeError("rebuild fail sentinel")

        monkeypatch.setattr(
            lightrag_module, "rebuild_knowledge_from_chunks", fail_rebuild
        )

        result = await rag.adelete_by_doc_id(doc_id)

        entity_tracking = await rag.entity_chunks.get_by_id(entity_a)
        relation_tracking = await rag.relation_chunks.get_by_id(relation_key)
        failed_status = await rag.doc_status.get_by_id(doc_id)

        assert result.status == "fail"
        assert "rebuild fail sentinel" in result.message
        assert await rag.text_chunks.get_by_id(drop_chunk_id) is None
        assert await rag.text_chunks.get_by_id(keep_chunk_id) is not None
        assert failed_status is not None
        assert failed_status["chunks_list"] == [drop_chunk_id]
        assert failed_status["metadata"]["deletion_failed"] is True
        assert (
            failed_status["metadata"]["deletion_failure_stage"]
            == "rebuild_knowledge_graph"
        )
        assert "rebuild fail sentinel" in failed_status["error_msg"]
        assert entity_tracking is not None
        assert entity_tracking["chunk_ids"] == [keep_chunk_id]
        assert entity_tracking["count"] == 1
        assert relation_tracking is not None
        assert relation_tracking["chunk_ids"] == [keep_chunk_id]
        assert relation_tracking["count"] == 1
        assert (
            await rag.chunk_entity_relation_graph.get_edge(entity_a, entity_b)
            is not None
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_delete_retry_succeeds_after_rebuild_failure(tmp_path, monkeypatch):
    rag = await _build_rag(
        tmp_path, "delete_retry_after_failure", _deterministic_chunking
    )
    try:
        doc_id = "doc-delete-retry-success"
        keep_chunk_id = "chunk-keep"
        drop_chunk_id = "chunk-drop"
        seeded = await _seed_delete_retry_state(
            rag,
            doc_id=doc_id,
            status_chunk_ids=[drop_chunk_id],
            tracking_chunk_ids=[keep_chunk_id, drop_chunk_id],
            chunk_owners={
                keep_chunk_id: "doc-keep",
                drop_chunk_id: doc_id,
            },
        )
        entity_a = seeded["entity_a"]
        entity_b = seeded["entity_b"]
        relation_key = seeded["relation_key"]

        async def fail_rebuild(**kwargs):
            raise RuntimeError("rebuild fail sentinel")

        monkeypatch.setattr(
            lightrag_module, "rebuild_knowledge_from_chunks", fail_rebuild
        )
        first_result = await rag.adelete_by_doc_id(doc_id)
        assert first_result.status == "fail"

        monkeypatch.setattr(
            lightrag_module,
            "rebuild_knowledge_from_chunks",
            _succeed_rebuild_from_remaining_chunks,
        )
        second_result = await rag.adelete_by_doc_id(doc_id)

        assert second_result.status == "success"
        assert await rag.doc_status.get_by_id(doc_id) is None
        assert await rag.full_docs.get_by_id(doc_id) is None
        assert await rag.full_entities.get_by_id(doc_id) is None
        assert await rag.full_relations.get_by_id(doc_id) is None
        assert await rag.text_chunks.get_by_id(drop_chunk_id) is None
        assert await rag.text_chunks.get_by_id(keep_chunk_id) is not None

        entity_a_tracking = await rag.entity_chunks.get_by_id(entity_a)
        entity_b_tracking = await rag.entity_chunks.get_by_id(entity_b)
        relation_tracking = await rag.relation_chunks.get_by_id(relation_key)
        edge = await rag.chunk_entity_relation_graph.get_edge(entity_a, entity_b)

        assert entity_a_tracking is not None
        assert entity_a_tracking["chunk_ids"] == [keep_chunk_id]
        assert entity_b_tracking is not None
        assert entity_b_tracking["chunk_ids"] == [keep_chunk_id]
        assert relation_tracking is not None
        assert relation_tracking["chunk_ids"] == [keep_chunk_id]
        assert edge is not None
        assert edge["source_id"] == keep_chunk_id
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_delete_retry_cleans_llm_cache_after_rebuild_failure(
    tmp_path, monkeypatch
):
    rag = await _build_rag(
        tmp_path, "delete_retry_cleans_llm_cache", _deterministic_chunking
    )
    try:
        doc_id = "doc-delete-retry-cache-cleanup"
        keep_chunk_id = "chunk-keep"
        drop_chunk_id = "chunk-drop"
        seeded = await _seed_delete_retry_state(
            rag,
            doc_id=doc_id,
            status_chunk_ids=[drop_chunk_id],
            tracking_chunk_ids=[keep_chunk_id, drop_chunk_id],
            chunk_owners={
                keep_chunk_id: "doc-keep",
                drop_chunk_id: doc_id,
            },
        )
        entity_a = seeded["entity_a"]
        entity_b = seeded["entity_b"]
        cache_ids = await _seed_chunk_cache_entries(rag, [drop_chunk_id], "retry")

        async def fail_rebuild(**kwargs):
            raise RuntimeError("rebuild fail sentinel")

        monkeypatch.setattr(
            lightrag_module, "rebuild_knowledge_from_chunks", fail_rebuild
        )
        first_result = await rag.adelete_by_doc_id(doc_id, delete_llm_cache=True)
        assert first_result.status == "fail"

        failed_status = await rag.doc_status.get_by_id(doc_id)
        assert failed_status is not None
        assert failed_status["metadata"]["deletion_llm_cache_ids"] == cache_ids
        assert await rag.text_chunks.get_by_id(drop_chunk_id) is None

        monkeypatch.setattr(
            lightrag_module,
            "rebuild_knowledge_from_chunks",
            _succeed_rebuild_from_remaining_chunks,
        )
        second_result = await rag.adelete_by_doc_id(doc_id, delete_llm_cache=True)

        assert second_result.status == "success"
        assert await rag.doc_status.get_by_id(doc_id) is None
        assert await rag.full_docs.get_by_id(doc_id) is None
        assert await rag.llm_response_cache.get_by_id(cache_ids[0]) is None
        edge = await rag.chunk_entity_relation_graph.get_edge(entity_a, entity_b)
        assert edge is not None
        assert edge["source_id"] == keep_chunk_id
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_delete_retry_cleans_llm_cache_when_enabled_on_retry(
    tmp_path, monkeypatch
):
    rag = await _build_rag(
        tmp_path, "delete_retry_cache_cleanup_flag_change", _deterministic_chunking
    )
    try:
        doc_id = "doc-delete-retry-cache-flag-change"
        keep_chunk_id = "chunk-keep"
        drop_chunk_id = "chunk-drop"
        await _seed_delete_retry_state(
            rag,
            doc_id=doc_id,
            status_chunk_ids=[drop_chunk_id],
            tracking_chunk_ids=[keep_chunk_id, drop_chunk_id],
            chunk_owners={
                keep_chunk_id: "doc-keep",
                drop_chunk_id: doc_id,
            },
        )
        cache_ids = await _seed_chunk_cache_entries(rag, [drop_chunk_id], "retry-flag")

        async def fail_rebuild(**kwargs):
            raise RuntimeError("rebuild fail sentinel")

        monkeypatch.setattr(
            lightrag_module, "rebuild_knowledge_from_chunks", fail_rebuild
        )
        first_result = await rag.adelete_by_doc_id(doc_id, delete_llm_cache=False)

        assert first_result.status == "fail"
        failed_status = await rag.doc_status.get_by_id(doc_id)
        assert failed_status is not None
        assert failed_status["metadata"]["deletion_llm_cache_ids"] == cache_ids
        assert await rag.text_chunks.get_by_id(drop_chunk_id) is None

        monkeypatch.setattr(
            lightrag_module,
            "rebuild_knowledge_from_chunks",
            _succeed_rebuild_from_remaining_chunks,
        )
        second_result = await rag.adelete_by_doc_id(doc_id, delete_llm_cache=True)

        assert second_result.status == "success"
        assert await rag.doc_status.get_by_id(doc_id) is None
        assert await rag.full_docs.get_by_id(doc_id) is None
        assert await rag.llm_response_cache.get_by_id(cache_ids[0]) is None
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_delete_retry_collects_cache_ids_without_cache_storage(
    tmp_path, monkeypatch
):
    rag = await _build_rag(
        tmp_path,
        "delete_retry_collect_cache_ids_without_storage",
        _deterministic_chunking,
    )
    try:
        doc_id = "doc-delete-retry-collect-cache-ids"
        keep_chunk_id = "chunk-keep"
        drop_chunk_id = "chunk-drop"
        await _seed_delete_retry_state(
            rag,
            doc_id=doc_id,
            status_chunk_ids=[drop_chunk_id],
            tracking_chunk_ids=[keep_chunk_id, drop_chunk_id],
            chunk_owners={
                keep_chunk_id: "doc-keep",
                drop_chunk_id: doc_id,
            },
        )
        cache_ids = await _seed_chunk_cache_entries(
            rag, [drop_chunk_id], "collect-without-storage"
        )

        async def fail_rebuild(**kwargs):
            raise RuntimeError("rebuild fail sentinel")

        cache_storage = rag.llm_response_cache
        rag.llm_response_cache = None
        monkeypatch.setattr(
            lightrag_module, "rebuild_knowledge_from_chunks", fail_rebuild
        )
        first_result = await rag.adelete_by_doc_id(doc_id, delete_llm_cache=False)

        assert first_result.status == "fail"
        failed_status = await rag.doc_status.get_by_id(doc_id)
        assert failed_status is not None
        assert failed_status["metadata"]["deletion_llm_cache_ids"] == cache_ids
        assert await rag.text_chunks.get_by_id(drop_chunk_id) is None

        rag.llm_response_cache = cache_storage
        monkeypatch.setattr(
            lightrag_module,
            "rebuild_knowledge_from_chunks",
            _succeed_rebuild_from_remaining_chunks,
        )
        second_result = await rag.adelete_by_doc_id(doc_id, delete_llm_cache=True)

        assert second_result.status == "success"
        assert await rag.doc_status.get_by_id(doc_id) is None
        assert await rag.full_docs.get_by_id(doc_id) is None
        assert await rag.llm_response_cache.get_by_id(cache_ids[0]) is None
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_delete_retry_succeeds_after_llm_cache_cleanup_failure(
    tmp_path, monkeypatch
):
    rag = await _build_rag(
        tmp_path, "delete_retry_after_cache_cleanup_failure", _deterministic_chunking
    )
    try:
        doc_id = "doc-delete-cache-cleanup-failure"
        drop_chunk_id = "chunk-drop"
        await _seed_delete_retry_state(
            rag,
            doc_id=doc_id,
            status_chunk_ids=[drop_chunk_id],
            tracking_chunk_ids=[drop_chunk_id],
            chunk_owners={drop_chunk_id: doc_id},
        )
        cache_ids = await _seed_chunk_cache_entries(
            rag, [drop_chunk_id], "cache-cleanup-failure"
        )

        async def fail_cache_delete(self, ids):
            raise RuntimeError("llm cache delete fail sentinel")

        monkeypatch.setattr(
            rag.llm_response_cache,
            "delete",
            MethodType(fail_cache_delete, rag.llm_response_cache),
        )
        first_result = await rag.adelete_by_doc_id(doc_id, delete_llm_cache=True)

        assert first_result.status == "fail"
        assert "Failed to delete LLM cache" in first_result.message
        assert await rag.doc_status.get_by_id(doc_id) is not None
        assert await rag.full_docs.get_by_id(doc_id) is not None
        assert await rag.llm_response_cache.get_by_id(cache_ids[0]) is not None

        monkeypatch.undo()
        second_result = await rag.adelete_by_doc_id(doc_id, delete_llm_cache=True)

        assert second_result.status == "success"
        assert await rag.doc_status.get_by_id(doc_id) is None
        assert await rag.full_docs.get_by_id(doc_id) is None
        assert await rag.llm_response_cache.get_by_id(cache_ids[0]) is None
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_delete_retry_succeeds_after_silent_llm_cache_cleanup_failure(
    tmp_path, monkeypatch
):
    rag = await _build_rag(
        tmp_path,
        "delete_retry_after_silent_cache_cleanup_failure",
        _deterministic_chunking,
    )
    try:
        doc_id = "doc-delete-silent-cache-cleanup-failure"
        drop_chunk_id = "chunk-drop"
        await _seed_delete_retry_state(
            rag,
            doc_id=doc_id,
            status_chunk_ids=[drop_chunk_id],
            tracking_chunk_ids=[drop_chunk_id],
            chunk_owners={drop_chunk_id: doc_id},
        )
        cache_ids = await _seed_chunk_cache_entries(
            rag, [drop_chunk_id], "silent-cache-cleanup-failure"
        )

        async def silently_fail_cache_delete(self, ids):
            return None

        monkeypatch.setattr(
            rag.llm_response_cache,
            "delete",
            MethodType(silently_fail_cache_delete, rag.llm_response_cache),
        )
        first_result = await rag.adelete_by_doc_id(doc_id, delete_llm_cache=True)

        assert first_result.status == "fail"
        assert "still exist after delete" in first_result.message
        assert await rag.doc_status.get_by_id(doc_id) is not None
        assert await rag.full_docs.get_by_id(doc_id) is not None
        assert await rag.llm_response_cache.get_by_id(cache_ids[0]) is not None

        monkeypatch.undo()
        second_result = await rag.adelete_by_doc_id(doc_id, delete_llm_cache=True)

        assert second_result.status == "success"
        assert await rag.doc_status.get_by_id(doc_id) is None
        assert await rag.full_docs.get_by_id(doc_id) is None
        assert await rag.llm_response_cache.get_by_id(cache_ids[0]) is None
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_delete_retry_preserves_cache_cleanup_state_when_cache_storage_unavailable(
    tmp_path,
):
    rag = await _build_rag(
        tmp_path, "delete_retry_cache_storage_unavailable", _deterministic_chunking
    )
    try:
        doc_id = "doc-delete-cache-storage-unavailable"
        drop_chunk_id = "chunk-drop"
        await _seed_delete_retry_state(
            rag,
            doc_id=doc_id,
            status_chunk_ids=[drop_chunk_id],
            tracking_chunk_ids=[drop_chunk_id],
            chunk_owners={drop_chunk_id: doc_id},
        )
        cache_ids = await _seed_chunk_cache_entries(
            rag, [drop_chunk_id], "cache-storage-unavailable"
        )

        status_doc = await rag.doc_status.get_by_id(doc_id)
        assert status_doc is not None
        status_doc["metadata"] = {"deletion_llm_cache_ids": cache_ids}
        await rag.doc_status.upsert({doc_id: status_doc})

        cache_storage = rag.llm_response_cache
        rag.llm_response_cache = None

        first_result = await rag.adelete_by_doc_id(doc_id, delete_llm_cache=True)

        assert first_result.status == "fail"
        assert "cache storage is unavailable" in first_result.message
        failed_status = await rag.doc_status.get_by_id(doc_id)
        assert failed_status is not None
        assert failed_status["metadata"]["deletion_llm_cache_ids"] == cache_ids
        assert await rag.full_docs.get_by_id(doc_id) is not None

        rag.llm_response_cache = cache_storage
        second_result = await rag.adelete_by_doc_id(doc_id, delete_llm_cache=True)

        assert second_result.status == "success"
        assert await rag.doc_status.get_by_id(doc_id) is None
        assert await rag.full_docs.get_by_id(doc_id) is None
        assert await rag.llm_response_cache.get_by_id(cache_ids[0]) is None
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_delete_succeeds_when_chunks_list_missing(tmp_path):
    rag = await _build_rag(
        tmp_path, "delete_missing_chunks_list_rejected", _deterministic_chunking
    )
    try:
        doc_id = "doc-delete-missing-chunks-list"
        drop_chunk_id = "chunk-drop-only"
        seeded = await _seed_delete_retry_state(
            rag,
            doc_id=doc_id,
            status_chunk_ids=[],
            tracking_chunk_ids=[drop_chunk_id],
            chunk_owners={drop_chunk_id: doc_id},
        )
        entity_a = seeded["entity_a"]
        entity_b = seeded["entity_b"]
        relation_key = seeded["relation_key"]

        result = await rag.adelete_by_doc_id(doc_id)

        assert result.status == "success"
        assert "without associated chunks" in result.message
        assert await rag.doc_status.get_by_id(doc_id) is None
        assert await rag.full_docs.get_by_id(doc_id) is None
        assert await rag.full_entities.get_by_id(doc_id) is not None
        assert await rag.full_relations.get_by_id(doc_id) is not None
        assert await rag.text_chunks.get_by_id(drop_chunk_id) is not None
        assert await rag.chunks_vdb.get_by_id(drop_chunk_id) is not None
        assert await rag.chunk_entity_relation_graph.get_node(entity_a) is not None
        assert await rag.chunk_entity_relation_graph.get_node(entity_b) is not None
        assert (
            await rag.chunk_entity_relation_graph.get_edge(entity_a, entity_b)
            is not None
        )
        assert await rag.entity_chunks.get_by_id(entity_a) is not None
        assert await rag.entity_chunks.get_by_id(entity_b) is not None
        assert await rag.relation_chunks.get_by_id(relation_key) is not None
        assert (
            await rag.entities_vdb.get_by_id(compute_mdhash_id(entity_a, prefix="ent-"))
            is not None
        )
        assert (
            await rag.entities_vdb.get_by_id(compute_mdhash_id(entity_b, prefix="ent-"))
            is not None
        )
        assert (
            await rag.relationships_vdb.get_by_id(
                compute_mdhash_id(entity_a + entity_b, prefix="rel-")
            )
            is not None
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_delete_ignores_stale_graph_source_ids_when_tracking_exists(tmp_path):
    rag = await _build_rag(
        tmp_path, "delete_ignore_stale_graph_sources", _deterministic_chunking
    )
    try:
        doc_id = "doc-delete-stale-graph-sources"
        drop_chunk_id = "chunk-drop"
        stale_chunk_id = "chunk-stale"
        seeded = await _seed_delete_retry_state(
            rag,
            doc_id=doc_id,
            status_chunk_ids=[drop_chunk_id],
            tracking_chunk_ids=[drop_chunk_id],
            chunk_owners={drop_chunk_id: doc_id},
        )
        entity_a = seeded["entity_a"]
        entity_b = seeded["entity_b"]
        relation_key = seeded["relation_key"]
        stale_source_id = GRAPH_FIELD_SEP.join([stale_chunk_id, drop_chunk_id])

        await rag.chunk_entity_relation_graph.upsert_node(
            entity_a,
            {
                "entity_id": entity_a,
                "source_id": stale_source_id,
                "description": f"{entity_a} description",
                "entity_type": "test",
                "file_path": "delete_retry.txt",
                "created_at": 1,
                "truncate": "",
            },
        )
        await rag.chunk_entity_relation_graph.upsert_node(
            entity_b,
            {
                "entity_id": entity_b,
                "source_id": stale_source_id,
                "description": f"{entity_b} description",
                "entity_type": "test",
                "file_path": "delete_retry.txt",
                "created_at": 1,
                "truncate": "",
            },
        )
        await rag.chunk_entity_relation_graph.upsert_edge(
            entity_a,
            entity_b,
            {
                "source": entity_a,
                "target": entity_b,
                "source_id": stale_source_id,
                "description": "related",
                "keywords": "test",
                "weight": 1.0,
                "file_path": "delete_retry.txt",
            },
        )

        result = await rag.adelete_by_doc_id(doc_id)

        assert result.status == "success"
        assert await rag.doc_status.get_by_id(doc_id) is None
        assert await rag.full_docs.get_by_id(doc_id) is None
        assert await rag.text_chunks.get_by_id(drop_chunk_id) is None
        assert await rag.chunk_entity_relation_graph.get_node(entity_a) is None
        assert await rag.chunk_entity_relation_graph.get_node(entity_b) is None
        assert (
            await rag.chunk_entity_relation_graph.get_edge(entity_a, entity_b) is None
        )
        assert await rag.entity_chunks.get_by_id(entity_a) is None
        assert await rag.entity_chunks.get_by_id(entity_b) is None
        assert await rag.relation_chunks.get_by_id(relation_key) is None
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_validate_and_fix_consistency_preserves_chunks_on_reset(tmp_path):
    rag = await _build_rag(tmp_path, "reset_preserve_chunks", _deterministic_chunking)
    try:
        failed_doc_id = "doc-failed-reset"
        processing_doc_id = "doc-processing-reset"
        inferred_count_doc_id = "doc-inferred-count-reset"

        now = datetime.now(timezone.utc).isoformat()
        await rag.full_docs.upsert(
            {
                failed_doc_id: {"content": "failed doc", "file_path": "failed.txt"},
                processing_doc_id: {
                    "content": "processing doc",
                    "file_path": "processing.txt",
                },
                inferred_count_doc_id: {
                    "content": "inferred count doc",
                    "file_path": "inferred.txt",
                },
            }
        )
        await rag.doc_status.upsert(
            {
                failed_doc_id: {
                    "status": DocStatus.FAILED,
                    "content_summary": "failed",
                    "content_length": 10,
                    "chunks_count": 2,
                    "chunks_list": ["f-1", "f-2"],
                    "created_at": now,
                    "updated_at": now,
                    "file_path": "failed.txt",
                    "track_id": "track-1",
                    "error_msg": "old error",
                    "metadata": {"old": True},
                },
                processing_doc_id: {
                    "status": DocStatus.PROCESSING,
                    "content_summary": "processing",
                    "content_length": 12,
                    "chunks_count": 1,
                    "chunks_list": ["p-1"],
                    "created_at": now,
                    "updated_at": now,
                    "file_path": "processing.txt",
                    "track_id": "track-2",
                    "error_msg": "old error",
                    "metadata": {"old": True},
                },
                inferred_count_doc_id: {
                    "status": DocStatus.FAILED,
                    "content_summary": "inferred",
                    "content_length": 14,
                    "chunks_list": ["i-1", "i-2", "i-3"],
                    "created_at": now,
                    "updated_at": now,
                    "file_path": "inferred.txt",
                    "track_id": "track-3",
                    "error_msg": "old error",
                    "metadata": {"old": True},
                },
            }
        )

        failed_docs = await rag.doc_status.get_docs_by_status(DocStatus.FAILED)
        processing_docs = await rag.doc_status.get_docs_by_status(DocStatus.PROCESSING)
        to_process_docs = {**failed_docs, **processing_docs}

        pipeline_status = {"latest_message": "", "history_messages": []}
        await rag._validate_and_fix_document_consistency(
            to_process_docs=to_process_docs,
            pipeline_status=pipeline_status,
            pipeline_status_lock=asyncio.Lock(),
        )

        failed_reset = await rag.doc_status.get_by_id(failed_doc_id)
        assert failed_reset is not None
        assert _status_to_text(failed_reset["status"]) == "pending"
        assert failed_reset.get("chunks_list") == ["f-1", "f-2"]
        assert failed_reset.get("chunks_count") == 2

        processing_reset = await rag.doc_status.get_by_id(processing_doc_id)
        assert processing_reset is not None
        assert _status_to_text(processing_reset["status"]) == "pending"
        assert processing_reset.get("chunks_list") == ["p-1"]
        assert processing_reset.get("chunks_count") == 1

        inferred_count_reset = await rag.doc_status.get_by_id(inferred_count_doc_id)
        assert inferred_count_reset is not None
        assert _status_to_text(inferred_count_reset["status"]) == "pending"
        assert inferred_count_reset.get("chunks_list") == ["i-1", "i-2", "i-3"]
        assert inferred_count_reset.get("chunks_count") == 3
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_delete_doc_entries_guard_prevents_zombie_record(tmp_path, monkeypatch):
    """When doc_status.delete fails, the guard must not re-create a zombie record.

    The exception handler skips _update_delete_retry_state when deletion_stage is
    "delete_doc_entries". This test confirms: (a) the result is status="fail", and
    (b) no zombie doc_status record is written after full_docs has already been removed.
    """
    rag = await _build_rag(
        tmp_path, "delete_doc_entries_guard", _deterministic_chunking
    )
    try:
        doc_id = "doc-delete-entries-guard"
        drop_chunk_id = "chunk-drop"
        await _seed_delete_retry_state(
            rag,
            doc_id=doc_id,
            status_chunk_ids=[drop_chunk_id],
            tracking_chunk_ids=[drop_chunk_id],
            chunk_owners={drop_chunk_id: doc_id},
        )

        original_delete = rag.doc_status.delete

        call_count = 0

        async def fail_doc_status_delete(ids):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("doc_status delete fail sentinel")

        monkeypatch.setattr(rag.doc_status, "delete", fail_doc_status_delete)

        result = await rag.adelete_by_doc_id(doc_id)

        monkeypatch.setattr(rag.doc_status, "delete", original_delete)

        assert result.status == "fail"
        assert "doc_status delete fail sentinel" in result.message
        # full_docs.delete ran before doc_status.delete; check doc_status was not
        # re-created as a zombie by the exception handler's retry-state write.
        status_record = await rag.doc_status.get_by_id(doc_id)
        # The record may still exist because doc_status.delete was patched to fail,
        # but it must NOT have been re-created with deletion_failed=True metadata
        # (which would indicate the guard fired incorrectly).
        if status_record is not None:
            metadata = status_record.get("metadata", {})
            assert not metadata.get(
                "deletion_failed"
            ), "guard failed: zombie record written with deletion_failed=True"
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_retry_state_write_failure_in_exception_handler_still_returns_fail(
    tmp_path, monkeypatch
):
    """If _update_delete_retry_state itself fails inside the exception handler,
    the caller must still receive a well-formed DeletionResult with status="fail"
    and the original error message — not an unhandled exception.
    """
    rag = await _build_rag(
        tmp_path, "retry_state_write_failure", _deterministic_chunking
    )
    try:
        doc_id = "doc-retry-state-write-fail"
        keep_chunk_id = "chunk-keep"
        drop_chunk_id = "chunk-drop"
        # Include a keep_chunk_id so entities survive and rebuild is triggered
        await _seed_delete_retry_state(
            rag,
            doc_id=doc_id,
            status_chunk_ids=[drop_chunk_id],
            tracking_chunk_ids=[keep_chunk_id, drop_chunk_id],
            chunk_owners={
                keep_chunk_id: "doc-keep",
                drop_chunk_id: doc_id,
            },
        )

        async def fail_rebuild(**kwargs):
            raise RuntimeError("rebuild fail sentinel")

        async def fail_upsert(data):
            raise RuntimeError("doc_status upsert fail sentinel")

        monkeypatch.setattr(
            lightrag_module, "rebuild_knowledge_from_chunks", fail_rebuild
        )
        monkeypatch.setattr(rag.doc_status, "upsert", fail_upsert)

        result = await rag.adelete_by_doc_id(doc_id)

        assert result.status == "fail"
        # The original rebuild error must be in the message, not the status-write error
        assert "rebuild fail sentinel" in result.message
    finally:
        await rag.finalize_storages()
