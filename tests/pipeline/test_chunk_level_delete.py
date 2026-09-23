"""``adelete_chunks_from_doc``: removing part of a processed document.

Drives the real LightRAG object (JSON / NetworkX storages, offline) with
extraction monkeypatched to a deterministic fake — the same harness as
test_custom_chunk_patch.py. Every capitalized word of a chunk is an entity and
the first two are related, so "Alice founded Acme" yields ALICE, ACME and the
ACME--ALICE relation.

Covers the removal half of an incremental update (the addition half is
``ainsert_custom_chunks`` patch mode): exclusive contributions are deleted,
shared ones rebuilt with exact provenance, untouched ones left alone; the
anchors stay a faithful superset; refusals change nothing; a failure part-way
converges on retry; and a later whole-document delete leaves no orphans.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

import lightrag.lightrag as lightrag_module
from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock
from lightrag.utils import EmbeddingFunc, Tokenizer, compute_mdhash_id
from lightrag.utils_pipeline import make_custom_chunk_id

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


def _two_chunks(
    tokenizer,
    content: str,
    split_by_character,
    split_by_character_only: bool,
    chunk_overlap_token_size: int,
    chunk_token_size: int,
) -> list[dict]:
    return [
        {"tokens": 1, "content": f"{content}::one", "chunk_order_index": 0},
        {"tokens": 1, "content": f"{content}::two", "chunk_order_index": 1},
    ]


async def _build_rag(tmp_path, **overrides) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"chunkdel-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        max_parallel_insert=1,
        **overrides,
    )
    await rag.initialize_storages()
    _fake_extraction(rag)
    return rag


def _fake_extraction(rag: LightRAG) -> None:
    async def fake_extract(chunks, *args, **kwargs):
        results = []
        for chunk_id, payload in chunks.items():
            names = [
                word.upper()
                for word in payload["content"].split()
                if word[:1].isupper()
            ]
            nodes = {
                name: [
                    {
                        "entity_name": name,
                        "entity_type": "person",
                        "description": f"{name} description",
                        "source_id": chunk_id,
                        "file_path": "custom",
                        "timestamp": 1,
                    }
                ]
                for name in names
            }
            edges = {}
            if len(names) >= 2:
                src, tgt = sorted(names[:2])
                edges[(src, tgt)] = [
                    {
                        "src_id": src,
                        "tgt_id": tgt,
                        "description": f"{src} and {tgt}",
                        "keywords": "related",
                        "weight": 1.0,
                        "source_id": chunk_id,
                        "file_path": "custom",
                        "timestamp": 1,
                    }
                ]
            results.append((nodes, edges))
        return results

    rag._process_extract_entities = fake_extract


def _cid(doc_id: str, content: str) -> str:
    return make_custom_chunk_id(doc_id, content)


def _status_text(row: dict) -> str:
    raw = row.get("status")
    return raw.value if isinstance(raw, DocStatus) else str(raw)


def _source_ids(obj: dict | None) -> list[str]:
    return [
        cid for cid in (obj or {}).get("source_id", "").split(GRAPH_FIELD_SEP) if cid
    ]


async def _seed_three_chunk_doc(rag: LightRAG, doc_id: str = "doc-1") -> None:
    await rag.ainsert_custom_chunks(
        "base",
        ["Alice founded Acme", "Bob joined Acme", "Carol audits Dana"],
        doc_id=doc_id,
    )


async def _graph_nodes(rag: LightRAG) -> set[str]:
    return {
        node["id"] for node in await rag.chunk_entity_relation_graph.get_all_nodes()
    }


@pytest.mark.asyncio
async def test_removes_exclusive_rebuilds_shared_and_keeps_untouched(tmp_path):
    rag = await _build_rag(tmp_path)
    try:
        await _seed_three_chunk_doc(rag)
        alice, bob, carol = (
            _cid("doc-1", "Alice founded Acme"),
            _cid("doc-1", "Bob joined Acme"),
            _cid("doc-1", "Carol audits Dana"),
        )

        result = await rag.adelete_chunks_from_doc("doc-1", [bob])
        assert (result.status, result.status_code) == ("success", 200), result

        graph = rag.chunk_entity_relation_graph
        # BOB and its relation were fed only by the removed chunk.
        assert await graph.get_node("BOB") is None
        assert await graph.get_edge("ACME", "BOB") is None
        # ACME was shared: rebuilt, and its provenance no longer names the
        # removed chunk, even though the fake writes no extraction cache.
        assert _source_ids(await graph.get_node("ACME")) == [alice]
        assert (await rag.entity_chunks.get_by_id("ACME"))["chunk_ids"] == [alice]
        # Untouched contributions stay exactly as they were.
        assert _source_ids(await graph.get_node("CAROL")) == [carol]
        assert await graph.get_edge("CAROL", "DANA") is not None

        assert await rag.text_chunks.get_by_id(bob) is None
        assert await rag.text_chunks.get_by_id(alice) is not None

        row = await rag.doc_status.get_by_id("doc-1")
        assert _status_text(row) == DocStatus.PROCESSED.value
        assert row["chunks_list"] == [alice, carol]
        assert row["chunks_count"] == 2

        anchors = await rag.full_entities.get_by_id("doc-1")
        assert anchors["entity_names"] == ["ACME", "ALICE", "CAROL", "DANA"]
        relations = await rag.full_relations.get_by_id("doc-1")
        assert sorted(tuple(p) for p in relations["relation_pairs"]) == [
            ("ACME", "ALICE"),
            ("CAROL", "DANA"),
        ]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_object_shared_with_another_document_survives(tmp_path):
    rag = await _build_rag(tmp_path)
    try:
        await rag.ainsert_custom_chunks("one", ["Alice founded Acme"], doc_id="doc-1")
        await rag.ainsert_custom_chunks("two", ["Acme hired Eve"], doc_id="doc-2")
        doc2_chunk = _cid("doc-2", "Acme hired Eve")

        result = await rag.adelete_chunks_from_doc(
            "doc-1", [_cid("doc-1", "Alice founded Acme")]
        )
        assert result.status == "success", result

        assert await rag.chunk_entity_relation_graph.get_node("ALICE") is None
        assert _source_ids(await rag.chunk_entity_relation_graph.get_node("ACME")) == [
            doc2_chunk
        ]
        # doc-1 no longer contributes to ACME, so its anchor drops the name;
        # doc-2's anchors are untouched.
        assert (await rag.full_entities.get_by_id("doc-1"))["entity_names"] == []
        assert (await rag.full_entities.get_by_id("doc-2"))["entity_names"] == [
            "ACME",
            "EVE",
        ]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_ids_the_document_does_not_own_are_never_touched(tmp_path):
    rag = await _build_rag(tmp_path)
    try:
        await rag.ainsert_custom_chunks("one", ["Alice founded Acme"], doc_id="doc-1")
        await rag.ainsert_custom_chunks("two", ["Acme hired Eve"], doc_id="doc-2")
        foreign = _cid("doc-2", "Acme hired Eve")

        result = await rag.adelete_chunks_from_doc("doc-1", [foreign, "chunk-nope"])
        assert (result.status, result.status_code) == ("success", 200)
        assert "owns none" in result.message

        assert await rag.text_chunks.get_by_id(foreign) is not None
        assert await rag.chunk_entity_relation_graph.get_node("EVE") is not None
        row = await rag.doc_status.get_by_id("doc-1")
        assert row["chunks_list"] == [_cid("doc-1", "Alice founded Acme")]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_repeating_a_successful_call_is_a_noop(tmp_path):
    rag = await _build_rag(tmp_path)
    try:
        await _seed_three_chunk_doc(rag)
        bob = _cid("doc-1", "Bob joined Acme")
        assert (await rag.adelete_chunks_from_doc("doc-1", [bob])).status == "success"
        nodes_before = await _graph_nodes(rag)
        row_before = await rag.doc_status.get_by_id("doc-1")

        again = await rag.adelete_chunks_from_doc("doc-1", [bob])
        assert (again.status, again.status_code) == ("success", 200)
        assert await _graph_nodes(rag) == nodes_before
        row_after = await rag.doc_status.get_by_id("doc-1")
        assert row_after["chunks_list"] == row_before["chunks_list"]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_empty_request_and_unknown_document(tmp_path):
    rag = await _build_rag(tmp_path)
    try:
        empty = await rag.adelete_chunks_from_doc("doc-1", ["", ""])
        assert (empty.status, empty.status_code) == ("success", 200)

        missing = await rag.adelete_chunks_from_doc("doc-missing", ["chunk-x"])
        assert (missing.status, missing.status_code) == ("not_found", 404)
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_refuses_a_document_that_is_not_processed(tmp_path):
    rag = await _build_rag(tmp_path)
    try:
        await _seed_three_chunk_doc(rag)
        await rag.doc_status.update_doc_status_fields(
            "doc-1", {"status": DocStatus.FAILED}
        )
        bob = _cid("doc-1", "Bob joined Acme")

        result = await rag.adelete_chunks_from_doc("doc-1", [bob])
        assert (result.status, result.status_code) == ("not_allowed", 409)
        assert await rag.text_chunks.get_by_id(bob) is not None
        assert await rag.chunk_entity_relation_graph.get_node("BOB") is not None
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_refuses_while_a_custom_chunk_operation_is_unfinished(
    tmp_path, monkeypatch
):
    rag = await _build_rag(tmp_path)
    try:
        await _seed_three_chunk_doc(rag)

        async def merge_boom(**kwargs):
            raise RuntimeError("merge boom")

        monkeypatch.setattr(lightrag_module, "merge_nodes_and_edges", merge_boom)
        with pytest.raises(RuntimeError, match="merge boom"):
            await rag.ainsert_custom_chunks("base", ["Frank left"], doc_id="doc-1")
        monkeypatch.undo()

        result = await rag.adelete_chunks_from_doc(
            "doc-1", [_cid("doc-1", "Bob joined Acme")]
        )
        # The journaled patch leaves the row FAILED; either refusal is a 409.
        assert (result.status, result.status_code) == ("not_allowed", 409)
        assert await rag.chunk_entity_relation_graph.get_node("BOB") is not None
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_fails_closed_without_recovery_anchors(tmp_path):
    rag = await _build_rag(tmp_path)
    try:
        await _seed_three_chunk_doc(rag)
        await rag.full_entities.delete(["doc-1"])
        await rag.full_relations.delete(["doc-1"])
        await rag._flush_storages([rag.full_entities, rag.full_relations])
        bob = _cid("doc-1", "Bob joined Acme")

        result = await rag.adelete_chunks_from_doc("doc-1", [bob])
        assert (result.status, result.status_code) == ("fail", 409)
        assert "audit_kg_integrity" in result.message
        # Nothing was deleted.
        assert await rag.text_chunks.get_by_id(bob) is not None
        assert await rag.chunk_entity_relation_graph.get_node("BOB") is not None
        row = await rag.doc_status.get_by_id("doc-1")
        assert bob in row["chunks_list"]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_document_that_never_reached_the_graph(tmp_path):
    """A skip_kg document has no anchors but a pre_graph proof: allowed."""
    rag = await _build_rag(tmp_path, chunking_func=_two_chunks)
    try:
        doc_id = compute_mdhash_id("nokg.txt", prefix="doc-")
        await rag.apipeline_enqueue_documents(
            "text", ids=[doc_id], file_paths=["nokg.txt"], process_options="!"
        )
        await rag.apipeline_process_enqueue_documents()
        chunks = (await rag.doc_status.get_by_id(doc_id))["chunks_list"]
        assert len(chunks) == 2

        result = await rag.adelete_chunks_from_doc(doc_id, [chunks[0]])
        assert result.status == "success", result
        assert (await rag.doc_status.get_by_id(doc_id))["chunks_list"] == [chunks[1]]
        assert await rag.text_chunks.get_by_id(chunks[0]) is None
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_refused_while_the_pipeline_is_busy_and_releases_its_slot(tmp_path):
    rag = await _build_rag(tmp_path)
    try:
        await _seed_three_chunk_doc(rag)
        bob = _cid("doc-1", "Bob joined Acme")
        status = await get_namespace_data("pipeline_status", workspace=rag.workspace)
        lock = get_namespace_lock("pipeline_status", workspace=rag.workspace)

        async with lock:
            status["busy"] = True
        refused = await rag.adelete_chunks_from_doc("doc-1", [bob])
        assert (refused.status, refused.status_code) == ("not_allowed", 403)
        assert await rag.text_chunks.get_by_id(bob) is not None
        async with lock:
            status["busy"] = False

        done = await rag.adelete_chunks_from_doc("doc-1", [bob])
        assert done.status == "success"
        assert status.get("busy") is False
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_document_delete_cannot_join_a_running_chunk_delete(
    tmp_path, monkeypatch
):
    """adelete_by_doc_id joins a busy job named "deleting ... document"; if the
    chunk delete's job matched, a document delete issued mid-way would run
    inside it instead of being refused."""
    rag = await _build_rag(tmp_path)
    try:
        await _seed_three_chunk_doc(rag)
        await rag.ainsert_custom_chunks("two", ["Acme hired Eve"], doc_id="doc-2")
        original_purge = rag._purge_kg_contributions
        seen: dict = {}

        async def purge_with_intruder(*args, **kwargs):
            # Once only: a document delete that wrongly joined would reach this
            # same patched purge and recurse instead of failing the assertion.
            if "intruder" not in seen:
                seen["intruder"] = None
                seen["intruder"] = await rag.adelete_by_doc_id("doc-2")
            return await original_purge(*args, **kwargs)

        monkeypatch.setattr(rag, "_purge_kg_contributions", purge_with_intruder)
        result = await rag.adelete_chunks_from_doc(
            "doc-1", [_cid("doc-1", "Bob joined Acme")]
        )
        assert result.status == "success", result
        assert (seen["intruder"].status, seen["intruder"].status_code) == (
            "not_allowed",
            403,
        )
        assert await rag.doc_status.get_by_id("doc-2") is not None
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_rebuilds_only_what_the_removed_chunks_fed(tmp_path, monkeypatch):
    """The anchors name the whole document; only objects the removed chunks
    feed may be rebuilt, or every delete would rebuild the whole document."""
    rag = await _build_rag(tmp_path)
    try:
        await _seed_three_chunk_doc(rag)
        original_rebuild = lightrag_module.rebuild_knowledge_from_chunks
        rebuilt: dict = {}

        async def recording_rebuild(**kwargs):
            rebuilt["entities"] = set(kwargs["entities_to_rebuild"])
            rebuilt["relations"] = set(kwargs["relationships_to_rebuild"])
            return await original_rebuild(**kwargs)

        monkeypatch.setattr(
            lightrag_module, "rebuild_knowledge_from_chunks", recording_rebuild
        )
        result = await rag.adelete_chunks_from_doc(
            "doc-1", [_cid("doc-1", "Bob joined Acme")]
        )
        assert result.status == "success", result
        assert rebuilt == {"entities": {"ACME"}, "relations": set()}
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_delete_llm_cache_removes_only_the_removed_chunks_rows(tmp_path):
    rag = await _build_rag(tmp_path)
    try:
        await _seed_three_chunk_doc(rag)
        alice = _cid("doc-1", "Alice founded Acme")
        bob = _cid("doc-1", "Bob joined Acme")
        carol = _cid("doc-1", "Carol audits Dana")
        for chunk_id, cache_id in (
            (alice, "default:extract:alice"),
            (bob, "default:extract:bob"),
            (carol, "default:extract:carol"),
        ):
            row = await rag.text_chunks.get_by_id(chunk_id)
            await rag.text_chunks.upsert(
                {chunk_id: {**row, "llm_cache_list": [cache_id]}}
            )
            await rag.llm_response_cache.upsert(
                {cache_id: {"return": "x", "cache_type": "extract"}}
            )
        await rag._flush_storages([rag.text_chunks, rag.llm_response_cache])

        kept = await rag.adelete_chunks_from_doc("doc-1", [carol])
        assert kept.status == "success"
        assert await rag.llm_response_cache.get_by_id("default:extract:carol")

        dropped = await rag.adelete_chunks_from_doc(
            "doc-1", [bob], delete_llm_cache=True
        )
        assert dropped.status == "success", dropped
        assert await rag.llm_response_cache.get_by_id("default:extract:bob") is None
        assert await rag.llm_response_cache.get_by_id("default:extract:alice")
        metadata = (await rag.doc_status.get_by_id("doc-1"))["metadata"]
        assert "deletion_llm_cache_ids" not in metadata
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_failed_rebuild_keeps_the_chunks_and_a_retry_converges(
    tmp_path, monkeypatch
):
    """A rebuild that raises must fail the call (not be logged and skipped),
    keep the removed chunks in place, and leave state a retry can finish —
    even though tracking was already updated while the graph was not."""
    rag = await _build_rag(tmp_path)
    try:
        await _seed_three_chunk_doc(rag)
        alice = _cid("doc-1", "Alice founded Acme")
        bob = _cid("doc-1", "Bob joined Acme")

        original_rebuild = lightrag_module.rebuild_knowledge_from_chunks

        async def rebuild_boom(**kwargs):
            raise RuntimeError("rebuild boom")

        monkeypatch.setattr(
            lightrag_module, "rebuild_knowledge_from_chunks", rebuild_boom
        )
        failed = await rag.adelete_chunks_from_doc("doc-1", [bob])
        assert (failed.status, failed.status_code) == ("fail", 500)
        assert await rag.text_chunks.get_by_id(bob) is not None
        row = await rag.doc_status.get_by_id("doc-1")
        assert _status_text(row) == DocStatus.PROCESSED.value
        assert bob in row["chunks_list"]
        assert row["metadata"]["deletion_failed"] is True
        assert row["metadata"]["deletion_failure_stage"] == "rebuild_knowledge_graph"
        # The half-applied state the retry must see through: tracking dropped
        # the chunk, the graph still names it.
        assert (await rag.entity_chunks.get_by_id("ACME"))["chunk_ids"] == [alice]
        assert bob in _source_ids(
            await rag.chunk_entity_relation_graph.get_node("ACME")
        )

        monkeypatch.setattr(
            lightrag_module, "rebuild_knowledge_from_chunks", original_rebuild
        )
        retried = await rag.adelete_chunks_from_doc("doc-1", [bob])
        assert retried.status == "success", retried
        assert _source_ids(await rag.chunk_entity_relation_graph.get_node("ACME")) == [
            alice
        ]
        assert await rag.text_chunks.get_by_id(bob) is None
        row = await rag.doc_status.get_by_id("doc-1")
        assert bob not in row["chunks_list"]
        assert "deletion_failed" not in row["metadata"]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_failure_before_the_commit_record_converges_on_retry(
    tmp_path, monkeypatch
):
    rag = await _build_rag(tmp_path)
    try:
        await _seed_three_chunk_doc(rag)
        bob = _cid("doc-1", "Bob joined Acme")
        original_commit = rag._commit_chunk_delete_status

        async def commit_boom(*args, **kwargs):
            raise RuntimeError("commit boom")

        monkeypatch.setattr(rag, "_commit_chunk_delete_status", commit_boom)
        failed = await rag.adelete_chunks_from_doc("doc-1", [bob])
        assert (failed.status, failed.status_code) == ("fail", 500)
        # Everything but the commit record landed; the id is still listed.
        assert await rag.text_chunks.get_by_id(bob) is None
        assert bob in (await rag.doc_status.get_by_id("doc-1"))["chunks_list"]

        monkeypatch.setattr(rag, "_commit_chunk_delete_status", original_commit)
        retried = await rag.adelete_chunks_from_doc("doc-1", [bob])
        assert retried.status == "success", retried
        assert bob not in (await rag.doc_status.get_by_id("doc-1"))["chunks_list"]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_incremental_update_then_document_delete_leaves_no_orphans(tmp_path):
    """Delete + patch is an in-place edit, and the anchors it leaves behind are
    still complete enough for a whole-document delete to clean everything."""
    rag = await _build_rag(tmp_path)
    try:
        await _seed_three_chunk_doc(rag)
        bob = _cid("doc-1", "Bob joined Acme")

        assert (await rag.adelete_chunks_from_doc("doc-1", [bob])).status == "success"
        await rag.ainsert_custom_chunks("base", ["Bob rejoined Acme"], doc_id="doc-1")

        row = await rag.doc_status.get_by_id("doc-1")
        assert _cid("doc-1", "Bob rejoined Acme") in row["chunks_list"]
        assert bob not in row["chunks_list"]
        assert await rag.chunk_entity_relation_graph.get_node("BOB") is not None

        deleted = await rag.adelete_by_doc_id("doc-1")
        assert deleted.status == "success", deleted
        assert await _graph_nodes(rag) == set()
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_prune_keeps_untracked_names_and_drops_vanished_ones(tmp_path):
    """Pruning may only over-claim: a name whose ownership cannot be read from
    a tracking row is kept, while a name whose graph object is gone is not."""
    rag = await _build_rag(tmp_path)
    try:
        await _seed_three_chunk_doc(rag)
        await rag.entity_chunks.delete(["CAROL"])
        await rag.chunk_entity_relation_graph.remove_nodes(["DANA"])

        await rag._prune_anchors_after_chunk_delete(
            "doc-1", ["CAROL", "DANA"], [], remaining_chunk_ids=set()
        )
        anchors = await rag.full_entities.get_by_id("doc-1")
        assert "CAROL" in anchors["entity_names"]
        assert "DANA" not in anchors["entity_names"]
        # Names outside the pruned set are never touched.
        assert {"ACME", "ALICE", "BOB"} <= set(anchors["entity_names"])
    finally:
        await rag.finalize_storages()
