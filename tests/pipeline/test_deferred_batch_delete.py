"""Batch rebuild intent survives failures without discarding recovery anchors."""

import asyncio
import copy
import json
import importlib
from contextlib import asynccontextmanager
from time import perf_counter
from unittest.mock import AsyncMock

import pytest

from lightrag.base import DocStatus
from lightrag.constants import GRAPH_FIELD_SEP, KG_PURGE_PHASE_REBUILD_DEFERRED
from lightrag.utils import make_relation_chunk_key

from .test_purge_primitive import (
    _DocStatus,
    _Graph,
    _KV,
    _edge,
    _make_rag,
    _node,
    _status,
)

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]

core = importlib.import_module("lightrag.lightrag")


@pytest.fixture
def batch(monkeypatch, request):
    @asynccontextmanager
    async def storage_lock(*args, **kwargs):
        yield

    monkeypatch.setattr("lightrag.operate.get_storage_keyed_lock", storage_lock)
    document_count = getattr(request, "param", 3)
    chunks = [f"c{i}" for i in range(1, document_count + 1)]
    rag = _make_rag(
        graph=_Graph(
            nodes={name: _node(name, chunks) for name in ("A", "B")},
            edges={("A", "B"): {**_edge(chunks), "weight": float(len(chunks))}},
        ),
        full_entities=_KV(
            {
                f"d{i}": {"entity_names": ["A", "B"]}
                for i in range(1, document_count + 1)
            }
        ),
        full_relations=_KV(
            {
                f"d{i}": {"relation_pairs": [["A", "B"]]}
                for i in range(1, document_count + 1)
            }
        ),
        entity_chunks=_KV({name: {"chunk_ids": chunks.copy()} for name in ("A", "B")}),
        relation_chunks=_KV(
            {make_relation_chunk_key("A", "B"): {"chunk_ids": chunks.copy()}}
        ),
        text_chunks=_KV(
            {
                f"c{i}": {
                    "content": f"text {i}",
                    "full_doc_id": f"d{i}",
                    "source_id": "custom",
                }
                for i in range(1, document_count + 1)
            }
        ),
        doc_status=_DocStatus(
            {
                f"d{i}": {
                    "status": DocStatus.PROCESSED,
                    "chunks_list": [f"c{i}"],
                    "file_path": f"d{i}.txt",
                    "metadata": {},
                }
                for i in range(1, document_count + 1)
            }
        ),
    )
    rag.workspace = "deferred-test"
    rag.full_docs = _KV(
        {f"d{i}": {"content": f"text {i}"} for i in range(1, document_count + 1)}
    )
    status, lock = _status()
    status.update(busy=True, destructive_busy=True, job_name="Deleting 2 Documents")
    monkeypatch.setattr(core, "get_namespace_data", AsyncMock(return_value=status))
    monkeypatch.setattr(core, "get_namespace_lock", lambda *args, **kwargs: lock)
    return rag, status, lock


async def test_batch_rebuilds_once_and_matches_sequential_deletion(batch, monkeypatch):
    rag, status, lock = batch
    monkeypatch.setattr(core.time, "time", lambda: 1700000000)
    sequential = copy.deepcopy(rag)
    original = core.rebuild_knowledge_from_chunks
    rebuild = AsyncMock(wraps=original)
    monkeypatch.setattr(core, "rebuild_knowledge_from_chunks", rebuild)
    for doc_id in ("d1", "d2"):
        result = await rag.adelete_by_doc_id(doc_id, defer_rebuild=True)
        assert result.status == "deferred"
        assert rag.doc_status.data[doc_id]["status"] == DocStatus.DELETING
        assert doc_id in rag.full_docs.data
        assert doc_id in rag.full_entities.data
    assert rebuild.await_count == 0
    results = await rag.afinalize_document_deletions(["d1", "d2"])
    assert [result.status for result in results] == ["success", "success"]
    assert rebuild.await_count == 1
    assert rebuild.call_args.kwargs["entities_to_rebuild"] == {"A": ["c3"], "B": ["c3"]}
    assert rebuild.call_args.kwargs["relationships_to_rebuild"] == {("A", "B"): ["c3"]}
    assert rag.chunk_entity_relation_graph.nodes["A"]["source_id"] == "c3"
    assert rag.chunk_entity_relation_graph.edges[("A", "B")]["source_id"] == "c3"
    assert set(rag.text_chunks.data) == {"c3"}
    assert set(rag.full_docs.data) == {"d3"}
    assert set(rag.doc_status.data) == {"d3"}
    for doc_id in ("d1", "d2"):
        assert (await sequential.adelete_by_doc_id(doc_id)).status == "success"
    assert rebuild.await_count == 3
    assert (
        rag.chunk_entity_relation_graph.nodes
        == sequential.chunk_entity_relation_graph.nodes
    )
    assert (
        rag.chunk_entity_relation_graph.edges
        == sequential.chunk_entity_relation_graph.edges
    )
    assert rag.entity_chunks.data == sequential.entity_chunks.data
    assert rag.relation_chunks.data == sequential.relation_chunks.data
    assert rag.entities_vdb.data == sequential.entities_vdb.data
    assert rag.relationships_vdb.data == sequential.relationships_vdb.data


@pytest.mark.parametrize("failure_phase", ["rebuild_pending", "rebuild_deferred"])
async def test_crash_after_chunks_disappear_is_recoverable(
    batch, monkeypatch, failure_phase
):
    rag, status, lock = batch
    write_phase = rag._write_kg_purge_phase
    interrupted = []

    async def crash(doc_id, phase, **kwargs):
        if phase == KG_PURGE_PHASE_REBUILD_DEFERRED and not interrupted:
            interrupted.append(phase)
            if failure_phase == "rebuild_deferred":
                await write_phase(doc_id, phase, **kwargs)
            raise RuntimeError("simulated process interruption")
        await write_phase(doc_id, phase, **kwargs)

    monkeypatch.setattr(rag, "_write_kg_purge_phase", crash)
    result = await rag.adelete_by_doc_id("d1", defer_rebuild=True)
    assert result.status == "fail"
    assert interrupted == [KG_PURGE_PHASE_REBUILD_DEFERRED]
    assert "c1" not in rag.text_chunks.data
    assert rag.doc_status.journal("d1")["phase"] == failure_phase
    assert "d1" in rag.full_entities.data
    monkeypatch.setattr(rag, "_write_kg_purge_phase", write_phase)
    # Discard every transient target list: only storage survives this retry.
    result = await rag.adelete_by_doc_id("d1")
    assert result.status == "success"
    assert rag.chunk_entity_relation_graph.nodes["A"][
        "source_id"
    ] == GRAPH_FIELD_SEP.join(["c2", "c3"])


async def test_rebuild_failure_keeps_documents_and_diagnostics_for_retry(
    batch, monkeypatch
):
    rag, status, lock = batch
    for doc_id in ("d1", "d2"):
        assert (
            await rag.adelete_by_doc_id(doc_id, defer_rebuild=True)
        ).status == "deferred"
    original = core.rebuild_knowledge_from_chunks
    rebuild = AsyncMock(side_effect=RuntimeError("LLM temporarily unavailable"))
    monkeypatch.setattr(core, "rebuild_knowledge_from_chunks", rebuild)
    results = await rag.afinalize_document_deletions(["d1", "d2"])
    assert rebuild.await_count == 1
    assert [result.status for result in results] == ["fail", "fail"]
    for doc_id in ("d1", "d2"):
        row = rag.doc_status.data[doc_id]
        assert row["status"] == DocStatus.DELETING
        assert row["metadata"]["deletion_failure_stage"] == "rebuild_knowledge_graph"
        assert "LLM temporarily unavailable" in row["error_msg"]
        assert doc_id in rag.full_entities.data
        assert doc_id in rag.full_docs.data
    monkeypatch.setattr(core, "rebuild_knowledge_from_chunks", original)
    assert [
        result.status for result in await rag.afinalize_document_deletions(["d1", "d2"])
    ] == ["success", "success"]


async def test_batch_deleting_last_owner_skips_removed_graph_objects(
    batch, monkeypatch
):
    rag, status, lock = batch
    for doc_id in ("d1", "d2", "d3"):
        assert (
            await rag.adelete_by_doc_id(doc_id, defer_rebuild=True)
        ).status == "deferred"
    rebuild = AsyncMock(wraps=core.rebuild_knowledge_from_chunks)
    monkeypatch.setattr(core, "rebuild_knowledge_from_chunks", rebuild)
    results = await rag.afinalize_document_deletions(["d1", "d2", "d3"])
    assert [result.status for result in results] == ["success"] * 3
    assert rebuild.await_count == 1
    assert rebuild.call_args.kwargs["entities_to_rebuild"] == {}
    assert rebuild.call_args.kwargs["relationships_to_rebuild"] == {}
    assert rag.chunk_entity_relation_graph.nodes == {}
    assert rag.chunk_entity_relation_graph.edges == {}
    assert rag.doc_status.data == {}


async def test_deferred_payload_cleanup_failure_retains_journal(batch, monkeypatch):
    rag, status, lock = batch
    assert (await rag.adelete_by_doc_id("d1", defer_rebuild=True)).status == "deferred"
    delete = rag.full_docs.delete
    fail = AsyncMock(side_effect=RuntimeError("payload deletion unavailable"))
    monkeypatch.setattr(rag.full_docs, "delete", fail)
    result = (await rag.afinalize_document_deletions(["d1"]))[0]
    assert result.status == "fail"
    assert fail.await_count == 1
    assert rag.doc_status.data["d1"]["status"] == DocStatus.DELETING
    assert rag.doc_status.journal("d1")["phase"] == "completed"
    monkeypatch.setattr(rag.full_docs, "delete", delete)
    assert (await rag.adelete_by_doc_id("d1")).status == "success"
    assert "d1" not in rag.full_docs.data
    assert "d1" not in rag.doc_status.data


async def test_cancelled_prefix_keeps_unreached_document_and_its_contributions(batch):
    rag, status, lock = batch
    untouched = copy.deepcopy(rag.doc_status.data["d2"])
    assert (await rag.adelete_by_doc_id("d1", defer_rebuild=True)).status == "deferred"
    status["cancellation_requested"] = True
    results = await rag.afinalize_document_deletions(["d1"])
    assert results[0].status == "success"
    assert rag.doc_status.data["d2"] == untouched
    assert rag.entity_chunks.data["A"]["chunk_ids"] == ["c2", "c3"]
    assert set(rag.chunk_entity_relation_graph.nodes) == {"A", "B"}


async def test_finalization_filters_dead_tracking_ids(batch, monkeypatch):
    rag, status, lock = batch
    assert (await rag.adelete_by_doc_id("d1", defer_rebuild=True)).status == "deferred"
    rag.entity_chunks.data["A"]["chunk_ids"].append("dead-chunk")
    rebuild = AsyncMock(wraps=core.rebuild_knowledge_from_chunks)
    monkeypatch.setattr(core, "rebuild_knowledge_from_chunks", rebuild)
    assert (await rag.afinalize_document_deletions(["d1"]))[0].status == "success"
    assert rebuild.call_args.kwargs["entities_to_rebuild"]["A"] == ["c2", "c3"]
    assert rag.entity_chunks.data["A"]["chunk_ids"] == ["c2", "c3"]


async def test_orphan_target_retains_anchors_and_fails_finalization(batch):
    rag, status, lock = batch
    assert (await rag.adelete_by_doc_id("d1", defer_rebuild=True)).status == "deferred"
    rag.entity_chunks.data["A"]["chunk_ids"] = ["dead-chunk"]
    results = await rag.afinalize_document_deletions(["d1"])
    assert results[0].status == "fail"
    assert "no live source chunks" in results[0].message
    assert "d1" in rag.full_entities.data
    assert rag.doc_status.data["d1"]["status"] == DocStatus.DELETING


async def test_pre_graph_patch_candidates_survive_deferred_retry(batch):
    rag, status, lock = batch
    rag.full_entities.data.pop("d1")
    rag.full_relations.data.pop("d1")
    rag.doc_status.data["d1"]["metadata"] = {
        "kg_write_state": "pre_graph",
        "custom_chunk_patch": {
            "entity_names": ["A", "B"],
            "relation_pairs": [["A", "B"]],
        },
    }
    assert (await rag.adelete_by_doc_id("d1", defer_rebuild=True)).status == "deferred"
    assert (await rag.adelete_by_doc_id("d1")).status == "success"
    assert rag.entity_chunks.data["A"]["chunk_ids"] == ["c2", "c3"]
    assert rag.chunk_entity_relation_graph.edges[("A", "B")][
        "source_id"
    ] == GRAPH_FIELD_SEP.join(["c2", "c3"])


async def test_failed_rebuild_commit_keeps_pending_intent(batch, monkeypatch):
    rag, status, lock = batch
    assert (await rag.adelete_by_doc_id("d1", defer_rebuild=True)).status == "deferred"
    flush = rag._insert_done
    fail = AsyncMock(side_effect=RuntimeError("rebuilt state could not be committed"))
    monkeypatch.setattr(rag, "_insert_done", fail)
    result = (await rag.afinalize_document_deletions(["d1"]))[0]
    assert result.status == "fail"
    assert fail.await_count == 1
    assert rag.doc_status.journal("d1")["phase"] == KG_PURGE_PHASE_REBUILD_DEFERRED
    assert "d1" in rag.full_entities.data
    monkeypatch.setattr(rag, "_insert_done", flush)
    assert (await rag.adelete_by_doc_id("d1")).status == "success"


async def test_partial_anchor_cleanup_retry_does_not_repeat_rebuild(batch, monkeypatch):
    rag, status, lock = batch
    for doc_id in ("d1", "d2"):
        assert (
            await rag.adelete_by_doc_id(doc_id, defer_rebuild=True)
        ).status == "deferred"
    delete = rag.full_relations.delete
    fail = AsyncMock(side_effect=RuntimeError("anchor cleanup unavailable"))
    monkeypatch.setattr(rag.full_relations, "delete", fail)
    rebuild = AsyncMock(wraps=core.rebuild_knowledge_from_chunks)
    monkeypatch.setattr(core, "rebuild_knowledge_from_chunks", rebuild)
    results = await rag.afinalize_document_deletions(["d1", "d2"])
    assert [result.status for result in results] == ["fail", "fail"]
    assert fail.await_count == 2
    assert rebuild.await_count == 1
    monkeypatch.setattr(rag.full_relations, "delete", delete)
    assert (await rag.adelete_by_doc_id("d1")).status == "success"
    assert (await rag.adelete_by_doc_id("d2")).status == "success"
    assert rebuild.await_count == 1


@pytest.mark.parametrize("batch", [4, 16], indirect=True)
async def test_shared_batch_reduces_rebuild_and_summary_calls(
    batch, monkeypatch, record_property
):
    from lightrag.utils import Tokenizer
    from .test_delete_fail_closed import _SimpleTokenizerImpl

    rag, status, lock = batch
    monkeypatch.setattr(core.time, "time", lambda: 1700000000)
    llm_calls = []

    async def summary_llm(prompt, **kwargs):
        llm_calls.append(prompt)
        await asyncio.sleep(0.001)
        return "Combined description"

    for chunk_id, row in rag.text_chunks.data.items():
        cache_id = f"extract-{chunk_id}"
        row["llm_cache_list"] = [cache_id]
        rag.llm_response_cache.data[cache_id] = {
            "cache_type": "extract",
            "chunk_id": chunk_id,
            "create_time": 1,
            "return": json.dumps(
                {
                    "entities": [
                        {
                            "name": name,
                            "type": "PERSON",
                            "description": f"{name} in {chunk_id}",
                        }
                        for name in ("A", "B")
                    ],
                    "relationships": [
                        {
                            "source": "A",
                            "target": "B",
                            "description": f"link in {chunk_id}",
                            "keywords": "connection",
                            "weight": 1.0,
                        }
                    ],
                }
            ),
        }
    config = rag._build_global_config() | {
        "tokenizer": Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        "summary_context_size": 10000,
        "summary_max_tokens": 10000,
        "summary_length_recommended": 100,
        "force_llm_summary_on_merge": 2,
        "role_llm_funcs": {"extract": summary_llm},
    }
    rag._build_global_config = lambda: config
    rag.llm_response_cache.global_config = {"enable_llm_cache": True}
    sequential = copy.deepcopy(rag)
    doc_ids = list(rag.doc_status.data)[:-2]
    rebuild = AsyncMock(wraps=core.rebuild_knowledge_from_chunks)
    monkeypatch.setattr(core, "rebuild_knowledge_from_chunks", rebuild)
    start = perf_counter()
    for doc_id in doc_ids:
        assert (await sequential.adelete_by_doc_id(doc_id)).status == "success"
    sequential_seconds = perf_counter() - start
    sequential_calls = len(llm_calls)
    assert rebuild.await_count == len(doc_ids)
    rebuild.reset_mock()
    llm_calls.clear()
    start = perf_counter()
    for doc_id in doc_ids:
        assert (
            await rag.adelete_by_doc_id(doc_id, defer_rebuild=True)
        ).status == "deferred"
    assert all(
        result.status == "success"
        for result in await rag.afinalize_document_deletions(doc_ids)
    )
    deferred_seconds = perf_counter() - start
    assert rebuild.await_count == 1
    assert len(llm_calls) == 3
    assert sequential_calls == 3 * len(doc_ids)
    assert (
        rag.chunk_entity_relation_graph.nodes
        == sequential.chunk_entity_relation_graph.nodes
    )
    assert (
        rag.chunk_entity_relation_graph.edges
        == sequential.chunk_entity_relation_graph.edges
    )
    metrics = {
        "deleted_documents": len(doc_ids),
        "sequential_rebuilds": len(doc_ids),
        "deferred_rebuilds": 1,
        "sequential_summary_calls": sequential_calls,
        "deferred_summary_calls": len(llm_calls),
        "sequential_seconds": sequential_seconds,
        "deferred_seconds": deferred_seconds,
    }
    record_property("batch_benchmark", json.dumps(metrics))
    print(json.dumps(metrics))
