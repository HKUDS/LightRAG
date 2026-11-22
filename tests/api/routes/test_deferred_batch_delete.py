"""The batch route finalizes before reporting success or removing source files."""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from lightrag.base import DocStatus, SourceAbsent
from tests.pipeline.test_delete_fail_closed import _build_rag


pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


@pytest.mark.parametrize("failure", [None, "rebuild", "mid_batch", "cancel"])
async def test_batch_route_preserves_recovery_and_source_files(
    deletion_batch, tmp_path, monkeypatch, failure
):
    rag, status, lock = deletion_batch
    with monkeypatch.context() as imports:
        imports.setattr(sys, "argv", [sys.argv[0]])
        routes = importlib.import_module("lightrag.api.routers.document_routes")
    core = importlib.import_module("lightrag.lightrag")
    shared = importlib.import_module("lightrag.kg.shared_storage")
    token = "batch-delete-test"
    status["busy_owner"] = token
    ingress = SimpleNamespace(has_work=lambda: False)
    monkeypatch.setattr(shared, "get_namespace_data", AsyncMock(return_value=status))
    monkeypatch.setattr(shared, "get_namespace_lock", lambda *args, **kwargs: lock)
    monkeypatch.setattr(shared, "get_pipeline_ingress", AsyncMock(return_value=ingress))
    monkeypatch.setattr(core, "get_pipeline_ingress", AsyncMock(return_value=ingress))
    rag.doc_status.resolve_doc_source_strict = AsyncMock(return_value=SourceAbsent())
    for name in ("d1.txt", "d2.txt"):
        (tmp_path / name).write_text(name, encoding="utf-8")

    original_rebuild = core.rebuild_knowledge_from_chunks
    rebuild_calls = []

    async def rebuild(**kwargs):
        rebuild_calls.append(kwargs)
        assert status["busy"] is True
        assert status["destructive_busy"] is True
        assert (tmp_path / "d1.txt").exists()
        assert (tmp_path / "d2.txt").exists()
        assert "d1" in rag.full_entities.data
        if failure == "rebuild":
            raise RuntimeError("injected batch rebuild failure")
        return await original_rebuild(**kwargs)

    monkeypatch.setattr(core, "rebuild_knowledge_from_chunks", rebuild)
    chunk_delete = rag.chunks_vdb.delete
    failed_deletes = []

    async def delete_chunks(ids):
        if failure == "mid_batch" and ids == ["c2"]:
            failed_deletes.append(ids)
            raise RuntimeError("injected chunk deletion failure")
        await chunk_delete(ids)

    monkeypatch.setattr(rag.chunks_vdb, "delete", delete_chunks)
    delete = rag.adelete_by_doc_id

    async def delete_doc(doc_id, **kwargs):
        result = await delete(doc_id, **kwargs)
        if failure == "cancel" and doc_id == "d1":
            status["cancellation_requested"] = True
        return result

    monkeypatch.setattr(rag, "adelete_by_doc_id", delete_doc)
    await routes.background_delete_documents(
        rag,
        routes.DocumentManager(str(tmp_path)),
        ["d1", "d2"],
        delete_file=True,
        token=token,
    )
    assert len(rebuild_calls) == 1
    assert status["busy"] is False
    assert status["destructive_busy"] is False
    if failure == "rebuild":
        assert (tmp_path / "d1.txt").exists()
        assert (tmp_path / "d2.txt").exists()
        assert "0 successful, 2 failed" in status["latest_message"]
        assert rag.doc_status.data["d1"]["status"] == DocStatus.DELETING
        assert (
            "injected batch rebuild failure" in rag.doc_status.data["d2"]["error_msg"]
        )
    elif failure in ("mid_batch", "cancel"):
        assert not (tmp_path / "d1.txt").exists()
        assert (tmp_path / "d2.txt").exists()
        assert "1 successful, 1 failed" in status["latest_message"]
        if failure == "mid_batch":
            assert failed_deletes == [["c2"]]
            assert rag.doc_status.data["d2"]["status"] == DocStatus.DELETING
            assert "c2" in rag.text_chunks.data
            assert rag.entity_chunks.data["A"]["chunk_ids"] == ["c3"]
        else:
            assert rag.doc_status.data["d2"]["status"] == DocStatus.PROCESSED
            assert rag.entity_chunks.data["A"]["chunk_ids"] == ["c2", "c3"]
            history = status["history_messages"]
            assert (
                len(
                    [
                        message
                        for message in history
                        if message.startswith("Deletion cancelled by user")
                    ]
                )
                == 1
            )
            assert not any(
                "Failed to delete" in message and "d2" in message for message in history
            )
    else:
        assert not (tmp_path / "d1.txt").exists()
        assert not (tmp_path / "d2.txt").exists()
        assert "2 successful, 0 failed" in status["latest_message"]
        assert set(rag.doc_status.data) == {"d3"}


async def test_batch_deletion_survives_reopening_persisted_storages(
    tmp_path, monkeypatch
):
    import asyncio
    import json

    with monkeypatch.context() as imports:
        imports.setattr(sys, "argv", [sys.argv[0]])
        routes = importlib.import_module("lightrag.api.routers.document_routes")
    workspace = f"deferred-persisted-{uuid4().hex[:8]}"
    rag = await _build_rag(tmp_path, workspace)
    try:
        for i in range(1, 4):
            await rag.apipeline_enqueue_documents(
                f"alice works at acme {i}", ids=[f"d{i}"], file_paths=[f"d{i}.txt"]
            )
            await rag.apipeline_process_enqueue_documents()
            row = await rag.doc_status.get_by_id(f"d{i}")
            assert row["status"] == DocStatus.PROCESSED
            chunk_id = row["chunks_list"][0]
            chunk = await rag.text_chunks.get_by_id(chunk_id)
            cache_id = f"extract-{chunk_id}"
            chunk["llm_cache_list"] = [cache_id]
            await rag.text_chunks.upsert({chunk_id: chunk})
            await rag.llm_response_cache.upsert(
                {
                    cache_id: {
                        "cache_type": "extract",
                        "chunk_id": chunk_id,
                        "create_time": 1,
                        "return": json.dumps(
                            {
                                "entities": [
                                    {
                                        "name": name,
                                        "type": "PERSON",
                                        "description": name,
                                    }
                                    for name in ("ALICE", "ACME")
                                ],
                                "relationships": [
                                    {
                                        "source": "ALICE",
                                        "target": "ACME",
                                        "description": "works at",
                                        "keywords": "employment",
                                        "weight": 1,
                                    }
                                ],
                            }
                        ),
                    }
                }
            )
        await rag._insert_done()
        surviving_chunk = (await rag.doc_status.get_by_id("d3"))["chunks_list"][0]
        failure = AsyncMock(
            side_effect=RuntimeError("process stopped before finalization")
        )
        monkeypatch.setattr(rag, "afinalize_document_deletions", failure)
        router = routes.create_document_routes(
            rag, routes.DocumentManager(str(tmp_path / "inputs"))
        )
        delete = next(
            route.endpoint
            for route in router.routes
            if route.path == "/documents/delete_document"
        )
        tasks = set()
        response = await delete(
            routes.DeleteDocRequest(doc_ids=["d1", "d2"]), managed_tasks=tasks
        )
        assert response.status == "deletion_started"
        await asyncio.gather(*tasks)
        assert failure.await_count == 1
        assert (await rag.doc_status.get_by_id("d1"))["status"] == DocStatus.DELETING
        await rag.finalize_storages()
        rag = await _build_rag(tmp_path, workspace)
        assert (await rag.doc_status.get_by_id("d2"))["status"] == DocStatus.DELETING
        assert (await rag.adelete_by_doc_id("d1")).status == "success"
        assert (await rag.adelete_by_doc_id("d2")).status == "success"
        assert await rag.doc_status.get_by_id("d1") is None
        assert await rag.doc_status.get_by_id("d2") is None
        assert (await rag.doc_status.get_by_id("d3"))["status"] == DocStatus.PROCESSED
        assert (await rag.chunk_entity_relation_graph.get_node("ALICE"))[
            "source_id"
        ] == surviving_chunk
        assert (await rag.chunk_entity_relation_graph.get_edge("ACME", "ALICE"))[
            "source_id"
        ] == surviving_chunk
    finally:
        await rag.finalize_storages()
