"""Batch rebuild intent survives failures without discarding recovery anchors."""

import copy
import importlib
import json
from unittest.mock import AsyncMock

import pytest

from lightrag.base import DocStatus
from lightrag.constants import GRAPH_FIELD_SEP, KG_PURGE_PHASE_REBUILD_DEFERRED
from lightrag.utils import compute_mdhash_id, make_relation_chunk_key, Tokenizer
from .test_delete_fail_closed import _SimpleTokenizerImpl

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]
core = importlib.import_module("lightrag.lightrag")


@pytest.mark.parametrize("deletion_batch", [4], indirect=True)
async def test_batch_rebuilds_once_and_matches_sequential_deletion(
    deletion_batch, monkeypatch
):
    rag, status, lock = deletion_batch
    monkeypatch.setattr(core.time, "time", lambda: 1700000000)
    summary_llm = AsyncMock(return_value="Combined description")
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
    rebuild = AsyncMock(wraps=core.rebuild_knowledge_from_chunks)
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
    assert summary_llm.await_count == 3
    assert rebuild.call_args.kwargs["entities_to_rebuild"] == {
        "A": ["c3", "c4"],
        "B": ["c3", "c4"],
    }
    assert rebuild.call_args.kwargs["relationships_to_rebuild"] == {
        ("A", "B"): ["c3", "c4"]
    }
    assert set(rag.text_chunks.data) == {"c3", "c4"}
    assert set(rag.full_docs.data) == {"d3", "d4"}
    assert set(rag.doc_status.data) == {"d3", "d4"}
    sequential_llm = AsyncMock(return_value="Combined description")
    sequential._build_global_config = lambda: (
        config | {"role_llm_funcs": {"extract": sequential_llm}}
    )
    for doc_id in ("d1", "d2"):
        assert (await sequential.adelete_by_doc_id(doc_id)).status == "success"
    assert rebuild.await_count == 3
    assert sequential_llm.await_count == 6
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
    deletion_batch, monkeypatch, failure_phase
):
    rag, status, lock = deletion_batch
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


@pytest.mark.parametrize("failure_point", ["rebuild", "commit"])
async def test_rebuild_failure_keeps_documents_and_diagnostics_for_retry(
    deletion_batch, monkeypatch, failure_point
):
    rag, status, lock = deletion_batch
    for doc_id in ("d1", "d2"):
        assert (
            await rag.adelete_by_doc_id(doc_id, defer_rebuild=True)
        ).status == "deferred"
    target = core if failure_point == "rebuild" else rag
    attribute = (
        "rebuild_knowledge_from_chunks"
        if failure_point == "rebuild"
        else "_insert_done"
    )
    original = getattr(target, attribute)
    fail = AsyncMock(side_effect=RuntimeError("injected rebuild-stage failure"))
    monkeypatch.setattr(target, attribute, fail)
    results = await rag.afinalize_document_deletions(["d1", "d2"])
    assert fail.await_count == 1
    assert [result.status for result in results] == ["fail", "fail"]
    for doc_id in ("d1", "d2"):
        row = rag.doc_status.data[doc_id]
        assert row["status"] == DocStatus.DELETING
        assert row["metadata"]["deletion_failure_stage"] == "rebuild_knowledge_graph"
        assert "injected rebuild-stage failure" in row["error_msg"]
        assert (
            rag.doc_status.journal(doc_id)["phase"] == KG_PURGE_PHASE_REBUILD_DEFERRED
        )
        assert doc_id in rag.full_entities.data
        assert doc_id in rag.full_docs.data
    monkeypatch.setattr(target, attribute, original)
    assert [
        result.status for result in await rag.afinalize_document_deletions(["d1", "d2"])
    ] == ["success", "success"]


async def test_batch_deleting_last_owner_skips_removed_graph_objects(
    deletion_batch, monkeypatch
):
    rag, status, lock = deletion_batch
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


async def test_finalization_filters_dead_tracking_ids(deletion_batch, monkeypatch):
    rag, status, lock = deletion_batch
    assert (await rag.adelete_by_doc_id("d1", defer_rebuild=True)).status == "deferred"
    rag.entity_chunks.data["A"]["chunk_ids"].append("dead-chunk")
    entity_reads = AsyncMock(wraps=rag.entity_chunks.get_by_ids)
    relation_reads = AsyncMock(wraps=rag.relation_chunks.get_by_ids)
    chunk_reads = AsyncMock(wraps=rag.text_chunks.get_by_ids)
    monkeypatch.setattr(rag.entity_chunks, "get_by_ids", entity_reads)
    monkeypatch.setattr(rag.relation_chunks, "get_by_ids", relation_reads)
    monkeypatch.setattr(rag.text_chunks, "get_by_ids", chunk_reads)
    original_rebuild = core.rebuild_knowledge_from_chunks

    async def check_batch_reads(**kwargs):
        assert entity_reads.await_args_list == [((["A", "B"],), {})]
        assert relation_reads.await_count == 1
        assert chunk_reads.await_count == 1
        assert set(chunk_reads.call_args.args[0]) == {"c2", "c3", "dead-chunk"}
        return await original_rebuild(**kwargs)

    rebuild = AsyncMock(side_effect=check_batch_reads)
    monkeypatch.setattr(core, "rebuild_knowledge_from_chunks", rebuild)
    assert (await rag.afinalize_document_deletions(["d1"]))[0].status == "success"
    assert rebuild.call_args.kwargs["entities_to_rebuild"]["A"] == ["c2", "c3"]
    assert rag.entity_chunks.data["A"]["chunk_ids"] == ["c2", "c3"]


@pytest.mark.parametrize("target_kind", ["entity", "relation", "all"])
async def test_dangling_targets_are_preserved_and_reported_without_blocking_deletion(
    deletion_batch, monkeypatch, target_kind
):
    rag, status, lock = deletion_batch
    entity_names = {"entity": ["A"], "relation": [], "all": ["A", "B"]}[target_kind]
    pairs = [] if target_kind == "entity" else [("A", "B")]
    for name in entity_names:
        rag.entity_chunks.data[name]["chunk_ids"] = ["c1", "dead"]
    for pair in pairs:
        rag.relation_chunks.data[make_relation_chunk_key(*pair)]["chunk_ids"] = [
            "c1",
            "dead",
        ]
    for doc_id in ("d1", "d2"):
        assert (
            await rag.adelete_by_doc_id(doc_id, defer_rebuild=True)
        ).status == "deferred"
    graph = rag.chunk_entity_relation_graph
    nodes_before = copy.deepcopy(graph.nodes)
    edges_before = copy.deepcopy(graph.edges)
    tracking_before = copy.deepcopy(rag.entity_chunks.data)
    relations_before = copy.deepcopy(rag.relation_chunks.data)
    for name in entity_names:
        rag.entities_vdb.data[compute_mdhash_id(name, prefix="ent-")] = {
            "content": "original vector"
        }
    for src, tgt in pairs:
        rag.relationships_vdb.data[compute_mdhash_id(src + tgt, prefix="rel-")] = {
            "content": "original vector"
        }
    graph.upsert_node = AsyncMock(wraps=graph.upsert_node)
    graph.upsert_edge = AsyncMock(wraps=graph.upsert_edge)
    rag.entities_vdb.upsert = AsyncMock(wraps=rag.entities_vdb.upsert)
    rag.relationships_vdb.upsert = AsyncMock(wraps=rag.relationships_vdb.upsert)
    rag.entity_chunks.upsert = AsyncMock(wraps=rag.entity_chunks.upsert)
    rag.relation_chunks.upsert = AsyncMock(wraps=rag.relation_chunks.upsert)
    reports = []
    original = rag._rebuild_deferred_kg_contributions

    async def capture_report(*args, **kwargs):
        report = await original(*args, **kwargs)
        reports.append(report)
        return report

    monkeypatch.setattr(rag, "_rebuild_deferred_kg_contributions", capture_report)
    results = await rag.afinalize_document_deletions(["d1", "d2"])
    assert [result.status for result in results] == ["success", "success"]
    assert len(reports) == 1
    for name in entity_names:
        assert graph.nodes[name] == nodes_before[name]
        assert rag.entity_chunks.data[name] == tracking_before[name]
        vector_id = compute_mdhash_id(name, prefix="ent-")
        assert rag.entities_vdb.data[vector_id] == {"content": "original vector"}
        assert name not in [call.args[0] for call in graph.upsert_node.await_args_list]
        assert all(
            name not in call.args[0]
            for call in rag.entity_chunks.upsert.await_args_list
        )
        assert all(
            vector_id not in call.args[0]
            for call in rag.entities_vdb.upsert.await_args_list
        )
        assert reports[0].degraded_entities[name] == ["dead"]
    for pair in pairs:
        key = make_relation_chunk_key(*pair)
        vector_id = compute_mdhash_id("".join(pair), prefix="rel-")
        assert graph.edges[pair] == edges_before[pair]
        assert rag.relation_chunks.data[key] == relations_before[key]
        assert rag.relationships_vdb.data[vector_id] == {"content": "original vector"}
        assert pair not in [call.args[:2] for call in graph.upsert_edge.await_args_list]
        assert all(
            key not in call.args[0]
            for call in rag.relation_chunks.upsert.await_args_list
        )
        assert all(
            vector_id not in call.args[0]
            for call in rag.relationships_vdb.upsert.await_args_list
        )
        assert reports[0].degraded_relationships[pair] == ["dead"]
    assert set(rag.doc_status.data) == {"d3"}
    assert reports[0].has_warnings
    if target_kind == "all":
        assert "Degraded KG recovery during deletion" in "\n".join(
            status["history_messages"]
        )
    else:
        warnings = rag.doc_status.data["d3"]["metadata"]["kg_recovery_warnings"]
        assert warnings[-1]["code"] == "degraded_document_deletion"
        category = "degraded_entities" if entity_names else "degraded_relationships"
        assert warnings[-1][category]["count"] > 0


async def test_pre_graph_patch_candidates_survive_deferred_retry(deletion_batch):
    rag, status, lock = deletion_batch
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


@pytest.mark.parametrize("storage_name", ["full_docs", "full_relations"])
async def test_cleanup_retry_does_not_repeat_rebuild(
    deletion_batch, monkeypatch, storage_name
):
    rag, status, lock = deletion_batch
    for doc_id in ("d1", "d2"):
        assert (
            await rag.adelete_by_doc_id(doc_id, defer_rebuild=True)
        ).status == "deferred"
    storage = getattr(rag, storage_name)
    delete = storage.delete
    fail = AsyncMock(side_effect=RuntimeError("cleanup unavailable"))
    monkeypatch.setattr(storage, "delete", fail)
    rebuild = AsyncMock(wraps=core.rebuild_knowledge_from_chunks)
    monkeypatch.setattr(core, "rebuild_knowledge_from_chunks", rebuild)
    results = await rag.afinalize_document_deletions(["d1", "d2"])
    assert [result.status for result in results] == ["fail", "fail"]
    assert fail.await_count == 2
    assert rebuild.await_count == 1
    for doc_id in ("d1", "d2"):
        assert rag.doc_status.data[doc_id]["status"] == DocStatus.DELETING
        expected_phase = (
            "completed" if storage_name == "full_docs" else "anchors_pending"
        )
        assert rag.doc_status.journal(doc_id)["phase"] == expected_phase
    monkeypatch.setattr(storage, "delete", delete)
    assert (await rag.adelete_by_doc_id("d1")).status == "success"
    assert (await rag.adelete_by_doc_id("d2")).status == "success"
    assert rebuild.await_count == 1
    assert "d1" not in rag.full_docs.data
    assert "d2" not in rag.doc_status.data
