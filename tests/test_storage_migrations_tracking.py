"""Legacy startup migration must recover all tracked document contributions."""

from uuid import uuid4

import pytest

from lightrag.base import DocStatus
from lightrag.exceptions import StorageCapabilityError
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.utils import make_relation_chunk_key

from tests.pipeline.test_delete_fail_closed import _build_rag

pytestmark = pytest.mark.offline


async def _legacy_rag(tmp_path):
    initialize_share_data(workers=1)
    rag = await _build_rag(tmp_path, f"legacy-{uuid4().hex[:8]}")
    for doc, chunk in [("doc-one", "chunk-one"), ("doc-two", "chunk-two")]:
        await rag.full_docs.upsert({doc: {"content": doc, "file_path": doc + ".txt"}})
        await rag.text_chunks.upsert(
            {
                chunk: {
                    "content": doc,
                    "tokens": 1,
                    "chunk_order_index": 0,
                    "full_doc_id": doc,
                    "file_path": doc + ".txt",
                    "llm_cache_list": [],
                }
            }
        )
        await rag.doc_status.upsert(
            {
                doc: {
                    "content_summary": doc,
                    "content_length": len(doc),
                    "file_path": doc + ".txt",
                    "status": DocStatus.PROCESSED,
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "chunks_list": [chunk],
                    "metadata": {},
                }
            }
        )
    graph = rag.chunk_entity_relation_graph
    for name, source in [
        ("SHARED", "chunk-one"),
        ("SHARED-PEER", "chunk-one"),
        ("SECOND", "chunk-two"),
        ("SECOND-PEER", "chunk-two"),
    ]:
        await graph.upsert_node(
            name,
            {
                "entity_id": name,
                "entity_type": "person",
                "description": name,
                "source_id": source,
                "file_path": "legacy.txt",
                "created_at": 1,
            },
        )
        chunks = (
            ["chunk-one", "chunk-two"] if name.startswith("SHARED") else ["chunk-two"]
        )
        await rag.entity_chunks.upsert(
            {name: {"chunk_ids": chunks, "count": len(chunks)}}
        )
    for src, tgt, source in [
        ("SHARED", "SHARED-PEER", "chunk-one"),
        ("SECOND", "SECOND-PEER", "chunk-two"),
    ]:
        await graph.upsert_edge(
            src,
            tgt,
            {
                "source_id": source,
                "description": "Related",
                "keywords": "related",
                "weight": 2.0 if src == "SHARED" else 1.0,
                "file_path": "legacy.txt",
            },
        )
        chunks = ["chunk-one", "chunk-two"] if src == "SHARED" else ["chunk-two"]
        await rag.relation_chunks.upsert(
            {
                make_relation_chunk_key(src, tgt): {
                    "chunk_ids": chunks,
                    "count": len(chunks),
                }
            }
        )
    await rag._insert_done()
    return rag


@pytest.mark.asyncio
async def test_startup_migration_preserves_tracking_outside_graph_source_window(
    tmp_path,
):
    rag = await _legacy_rag(tmp_path)
    try:
        await rag.check_and_migrate_data()
        entities = await rag.full_entities.get_by_id("doc-two")
        relations = await rag.full_relations.get_by_id("doc-two")
        assert set(entities["entity_names"]) == {
            "SHARED",
            "SHARED-PEER",
            "SECOND",
            "SECOND-PEER",
        }
        assert {tuple(pair) for pair in relations["relation_pairs"]} == {
            ("SHARED", "SHARED-PEER"),
            ("SECOND", "SECOND-PEER"),
        }
        result = await rag.adelete_by_doc_id("doc-two")
        assert result.status == "success"
        assert await rag.text_chunks.get_by_id("chunk-two") is None
        assert (await rag.entity_chunks.get_by_id("SHARED"))["chunk_ids"] == [
            "chunk-one"
        ]
        assert (
            await rag.relation_chunks.get_by_id(
                make_relation_chunk_key("SHARED", "SHARED-PEER")
            )
        )["chunk_ids"] == ["chunk-one"]
        assert await rag.text_chunks.get_by_id("chunk-one") is not None
        assert await rag.chunk_entity_relation_graph.has_edge("SHARED", "SHARED-PEER")
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_present_empty_tracking_is_not_reseeded_from_stale_graph_sources(
    tmp_path,
):
    rag = await _legacy_rag(tmp_path)
    try:
        await rag.entity_chunks.upsert({"SHARED": {"chunk_ids": [], "count": 0}})
        key = make_relation_chunk_key("SHARED", "SHARED-PEER")
        await rag.relation_chunks.upsert({key: {"chunk_ids": [], "count": 0}})
        await rag.check_and_migrate_data()
        assert (await rag.full_entities.get_by_id("doc-one"))["entity_names"] == [
            "SHARED-PEER"
        ]
        assert await rag.full_relations.get_by_id("doc-one") is None
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_absent_tracking_falls_back_to_legacy_graph_provenance(tmp_path):
    rag = await _legacy_rag(tmp_path)
    try:
        await rag.entity_chunks.delete(["SHARED"])
        key = make_relation_chunk_key("SHARED", "SHARED-PEER")
        await rag.relation_chunks.delete([key])
        await rag.check_and_migrate_data()
        assert (
            "SHARED" in (await rag.full_entities.get_by_id("doc-one"))["entity_names"]
        )
        assert (await rag.full_relations.get_by_id("doc-one"))["relation_pairs"] == [
            ["SHARED", "SHARED-PEER"]
        ]
        assert (
            "SHARED"
            not in (await rag.full_entities.get_by_id("doc-two"))["entity_names"]
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
@pytest.mark.parametrize("namespace", ["entity_chunks", "relation_chunks"])
async def test_tracking_read_failure_aborts_before_writing_any_anchors(
    tmp_path, monkeypatch, namespace
):
    rag = await _legacy_rag(tmp_path)

    async def unreadable(key):
        raise OSError("tracking read failed")

    try:
        monkeypatch.setattr(getattr(rag, namespace), "get_by_id_strict", unreadable)
        with pytest.raises(OSError, match="tracking read failed"):
            await rag.check_and_migrate_data()
        for doc in ("doc-one", "doc-two"):
            assert await rag.full_entities.get_by_id(doc) is None
            assert await rag.full_relations.get_by_id(doc) is None
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_unsupported_tracking_read_cannot_be_treated_as_confirmed_absence(
    tmp_path, monkeypatch
):
    rag = await _legacy_rag(tmp_path)
    try:
        monkeypatch.setattr(rag.entity_chunks, "supports_strict_point_reads", False)
        with pytest.raises(StorageCapabilityError, match="strict chunk-tracking"):
            await rag.check_and_migrate_data()
        assert await rag.full_entities.get_by_id("doc-one") is None
        assert await rag.full_relations.get_by_id("doc-one") is None
    finally:
        await rag.finalize_storages()
