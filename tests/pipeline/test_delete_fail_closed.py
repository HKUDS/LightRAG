"""End-to-end fail-closed deletion over real JSON/NetworkX storages (issue #3400).

Reproduces the differential scenario from the issue report: ingest a document,
delete its two recovery anchor rows (simulating pre-#3416 data or an
``ainsert_custom_kg`` document, both of which are documented to have none), then
try to delete or reprocess it.

Before this change, that deleted the chunks and reported success while the
entities and relations stayed in the graph — permanently unattributable, because
the reverse lookup runs graph ``source_id`` -> ``text_chunks`` -> ``full_doc_id``
and the chunks were just removed. ``audit_kg_integrity`` could then only report
them as unrecoverable orphans.

The suite asserts the operator-visible loop is now closed:

    refuse (409, nothing deleted) -> audit_kg_integrity(apply=True) -> retry works
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

import lightrag.pipeline as pipeline_module
from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.constants import (
    KG_PURGE_METADATA_KEY,
    KG_PURGE_PHASE_COMPLETED,
    KG_WRITE_STATE_GRAPH_MUTATION_STARTED,
    KG_WRITE_STATE_METADATA_KEY,
    KG_WRITE_STATE_PRE_GRAPH,
)
from lightrag.tools.kg_integrity_repair import audit_kg_integrity
from lightrag.utils import EmbeddingFunc, Tokenizer, compute_mdhash_id

from .conftest import request_failed_retry

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
    return [{"tokens": 1, "content": f"{content}::chunk1", "chunk_order_index": 0}]


def _wire_fake_extraction(rag: LightRAG) -> None:
    """One ALICE--ACME relation per chunk, so the graph has a node AND an edge."""

    async def fake_extract(chunks, *args, **kwargs):
        results = []
        for chunk_id in chunks:
            nodes = {
                name: [
                    {
                        "entity_name": name,
                        "entity_type": "person",
                        "description": f"{name} description",
                        "source_id": chunk_id,
                        "file_path": "d.txt",
                        "timestamp": 1,
                    }
                ]
                for name in ("ALICE", "ACME")
            }
            edges = {
                ("ACME", "ALICE"): [
                    {
                        "src_id": "ACME",
                        "tgt_id": "ALICE",
                        "description": "works at",
                        "keywords": "employment",
                        "weight": 1.0,
                        "source_id": chunk_id,
                        "file_path": "d.txt",
                        "timestamp": 1,
                    }
                ]
            }
            results.append((nodes, edges))
        return results

    rag._process_extract_entities = fake_extract


async def _build_rag(tmp_path, workspace: str) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=workspace,
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        chunking_func=_deterministic_chunking,
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    _wire_fake_extraction(rag)
    return rag


async def _ingest(rag: LightRAG, file_path: str = "d.txt") -> str:
    doc_id = compute_mdhash_id(file_path, prefix="doc-")
    await rag.apipeline_enqueue_documents(
        "alice works at acme", ids=[doc_id], file_paths=[file_path]
    )
    await rag.apipeline_process_enqueue_documents()
    row = await rag.doc_status.get_by_id(doc_id)
    status = row.get("status")
    status_text = status.value if isinstance(status, DocStatus) else str(status)
    assert status_text == DocStatus.PROCESSED.value, row
    return doc_id


async def _ingest_skip_kg(rag: LightRAG, file_path: str = "nokg.txt") -> str:
    """Ingest with ``process_options='!'`` — extraction and merge are skipped.

    This is a supported mode, and it is the case that produces a document with
    legitimately NO anchor rows: merge never runs, so Phase 0 never writes them.
    """
    doc_id = compute_mdhash_id(file_path, prefix="doc-")
    await rag.apipeline_enqueue_documents(
        "text with no kg", ids=[doc_id], file_paths=[file_path], process_options="!"
    )
    await rag.apipeline_process_enqueue_documents()
    row = await rag.doc_status.get_by_id(doc_id)
    status = row.get("status")
    status_text = status.value if isinstance(status, DocStatus) else str(status)
    assert status_text == DocStatus.PROCESSED.value, row
    assert row["metadata"]["skip_kg"] is True
    assert await rag.full_entities.get_by_id(doc_id) is None
    assert await rag.full_relations.get_by_id(doc_id) is None
    return doc_id


async def _strip_write_state(rag: LightRAG, doc_id: str) -> None:
    """Simulate a document written before the ``kg_write_state`` marker existed."""
    row = await rag.doc_status.get_by_id(doc_id)
    metadata = dict(row.get("metadata") or {})
    metadata.pop(KG_WRITE_STATE_METADATA_KEY, None)
    await rag.doc_status.update_doc_status_fields(doc_id, {"metadata": metadata})
    await rag.doc_status.index_done_callback()


async def _drop_anchors(rag: LightRAG, doc_id: str) -> None:
    """Simulate the pre-#3416 / ainsert_custom_kg state: no recovery anchors."""
    await rag.full_entities.delete([doc_id])
    await rag.full_relations.delete([doc_id])
    await rag.full_entities.index_done_callback()
    await rag.full_relations.index_done_callback()
    assert await rag.full_entities.get_by_id(doc_id) is None
    assert await rag.full_relations.get_by_id(doc_id) is None


async def _kg_snapshot(rag: LightRAG, chunk_ids: list[str]) -> dict:
    return {
        "alice": await rag.chunk_entity_relation_graph.get_node("ALICE"),
        "acme": await rag.chunk_entity_relation_graph.get_node("ACME"),
        "edge": await rag.chunk_entity_relation_graph.get_edge("ACME", "ALICE"),
        "chunks": await rag.text_chunks.get_by_ids(chunk_ids),
        "entity_tracking": await rag.entity_chunks.get_by_id("ALICE"),
    }


async def _chunk_ids(rag: LightRAG, doc_id: str) -> list[str]:
    row = await rag.doc_status.get_by_id(doc_id)
    return list(row.get("chunks_list") or [])


@pytest.mark.asyncio
async def test_delete_refuses_and_preserves_everything_when_anchors_lost(tmp_path):
    """The core scenario: refuse with 409 and leave the document untouched."""
    rag = await _build_rag(tmp_path, f"dfc-{uuid4().hex[:8]}")
    try:
        doc_id = await _ingest(rag)
        chunk_ids = await _chunk_ids(rag, doc_id)
        assert chunk_ids
        await _drop_anchors(rag, doc_id)
        before = await _kg_snapshot(rag, chunk_ids)
        assert before["alice"] is not None and before["edge"] is not None

        result = await rag.adelete_by_doc_id(doc_id)

        assert result.status == "fail"
        assert result.status_code == 409
        assert "recovery anchor row(s) missing" in result.message
        assert "audit_kg_integrity" in result.message

        # Nothing at all was deleted.
        assert await _kg_snapshot(rag, chunk_ids) == before
        assert await rag.full_docs.get_by_id(doc_id) is not None
        row = await rag.doc_status.get_by_id(doc_id)
        assert row is not None
        assert row["metadata"]["deletion_failure_stage"] == "validate_recovery_anchors"
        # Refused on a precondition, so no purge was ever journaled.
        assert KG_PURGE_METADATA_KEY not in row["metadata"]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_audit_repair_unblocks_the_refused_delete(tmp_path):
    """The documented remedy must actually close the loop.

    This is the whole point of failing closed rather than proceeding: while the
    chunks survive, ``audit_kg_integrity`` can still rebuild the anchors from
    their provenance. Deleting first destroys that option forever.
    """
    rag = await _build_rag(tmp_path, f"dfc-{uuid4().hex[:8]}")
    try:
        doc_id = await _ingest(rag)
        await _drop_anchors(rag, doc_id)

        assert (await rag.adelete_by_doc_id(doc_id)).status_code == 409

        # The report is the DIAGNOSIS (what was missing); repaired_docs is what
        # apply=True actually rebuilt from the surviving chunk provenance.
        report = await audit_kg_integrity(rag, apply=True)
        assert doc_id in report["missing_entity_anchors"]
        assert doc_id in report["repaired_docs"]
        assert await rag.full_entities.get_by_id(doc_id) is not None
        assert await rag.full_relations.get_by_id(doc_id) is not None

        result = await rag.adelete_by_doc_id(doc_id)

        assert result.status == "success", result.message
        assert await rag.doc_status.get_by_id(doc_id) is None
        assert await rag.full_docs.get_by_id(doc_id) is None
        # And the KG is clean: no orphans left behind by the deletion.
        final_report = await audit_kg_integrity(rag)
        assert final_report["orphan_entities"] == []
        assert final_report["orphan_relations"] == []
        assert await rag.chunk_entity_relation_graph.get_node("ALICE") is None
        assert await rag.chunk_entity_relation_graph.get_edge("ACME", "ALICE") is None
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_delete_without_anchors_used_to_orphan_the_graph(tmp_path):
    """Differential guard for the exact regression in the issue report.

    Pins the two facts that together made the old behaviour unrecoverable: the
    graph objects survive a purge that skipped them, and their chunks do not.
    If a future change reintroduces the permissive path, this fails.
    """
    rag = await _build_rag(tmp_path, f"dfc-{uuid4().hex[:8]}")
    try:
        doc_id = await _ingest(rag)
        chunk_ids = await _chunk_ids(rag, doc_id)
        await _drop_anchors(rag, doc_id)

        result = await rag.adelete_by_doc_id(doc_id)
        assert result.status == "fail"

        # The chunks are what make the survivors attributable — they must still
        # be here, and the audit must still see the graph as anchored-repairable
        # rather than orphaned.
        assert all(
            chunk is not None for chunk in await rag.text_chunks.get_by_ids(chunk_ids)
        )
        report = await audit_kg_integrity(rag)
        assert report["orphan_entities"] == []
        assert report["orphan_relations"] == []
        assert set(report["missing_entity_anchors"]) == {doc_id}
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_chunkless_document_with_populated_anchors_is_refused(tmp_path):
    """The second route to the same orphan state.

    A document with an empty ``chunks_list`` used to skip the graph entirely and
    report success, and removing its ``doc_status`` row is what destroyed the
    provenance of everything its anchors named.
    """
    rag = await _build_rag(tmp_path, f"dfc-{uuid4().hex[:8]}")
    try:
        doc_id = await _ingest(rag)
        chunk_ids = await _chunk_ids(rag, doc_id)
        # Anchors intact, but the document no longer claims any chunks.
        await rag.doc_status.update_doc_status_fields(
            doc_id, {"chunks_list": [], "chunks_count": 0}
        )
        await rag.doc_status.index_done_callback()

        result = await rag.adelete_by_doc_id(doc_id)

        assert result.status == "fail"
        assert result.status_code == 409
        assert "no chunks to attribute them to" in result.message
        assert await rag.doc_status.get_by_id(doc_id) is not None
        assert await rag.chunk_entity_relation_graph.get_node("ALICE") is not None
        assert all(
            chunk is not None for chunk in await rag.text_chunks.get_by_ids(chunk_ids)
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_chunkless_stub_with_empty_anchors_still_deletes(tmp_path):
    """Fail-closed must not block genuinely empty rows.

    A PROCESSED document that extracted nothing has both anchor rows present and
    empty — a valid proof — so it deletes, and its anchor rows go with it rather
    than being left keyed to a document that no longer exists.
    """
    rag = await _build_rag(tmp_path, f"dfc-{uuid4().hex[:8]}")
    try:
        doc_id = await _ingest(rag)
        chunk_ids = await _chunk_ids(rag, doc_id)
        # First delete the KG contributions the normal way, leaving an empty
        # but PRESENT pair of anchors and no chunks.
        await rag.text_chunks.delete(chunk_ids)
        await rag.chunks_vdb.delete(chunk_ids)
        await rag.full_entities.upsert({doc_id: {"entity_names": [], "count": 0}})
        await rag.full_relations.upsert({doc_id: {"relation_pairs": [], "count": 0}})
        await rag.doc_status.update_doc_status_fields(
            doc_id, {"chunks_list": [], "chunks_count": 0}
        )
        for store in (
            rag.text_chunks,
            rag.chunks_vdb,
            rag.full_entities,
            rag.full_relations,
            rag.doc_status,
        ):
            await store.index_done_callback()

        result = await rag.adelete_by_doc_id(doc_id)

        assert result.status == "success", result.message
        assert await rag.doc_status.get_by_id(doc_id) is None
        assert await rag.full_entities.get_by_id(doc_id) is None
        assert await rag.full_relations.get_by_id(doc_id) is None
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_delete_repairs_graph_before_removing_chunks(tmp_path):
    """Safe destructive ordering, asserted through the public delete API.

    The inline implementation this replaced deleted the chunks first, so a
    failure in between left graph objects pointing at chunks that were gone.
    """
    rag = await _build_rag(tmp_path, f"dfc-{uuid4().hex[:8]}")
    try:
        doc_id = await _ingest(rag)
        chunk_ids = await _chunk_ids(rag, doc_id)
        order: list[str] = []

        original_remove_nodes = rag.chunk_entity_relation_graph.remove_nodes
        original_chunk_delete = rag.text_chunks.delete

        async def spy_remove_nodes(names):
            order.append("graph.remove_nodes")
            return await original_remove_nodes(names)

        async def spy_chunk_delete(ids):
            order.append("text_chunks.delete")
            return await original_chunk_delete(ids)

        rag.chunk_entity_relation_graph.remove_nodes = spy_remove_nodes
        rag.text_chunks.delete = spy_chunk_delete

        assert (await rag.adelete_by_doc_id(doc_id)).status == "success"

        assert order.index("graph.remove_nodes") < order.index("text_chunks.delete")
        assert all(
            chunk is None for chunk in await rag.text_chunks.get_by_ids(chunk_ids)
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_finalization_failure_retries_without_tripping_fail_closed(tmp_path):
    """The deadlock the journal exists to prevent, end to end.

    Purge deletes the anchors LAST, so a failure in the caller's post-purge
    finalization leaves a document with no anchors and no chunks. Without the
    ``completed`` journal the retry would then see "anchors missing" and refuse
    forever — the document would be permanently undeletable.

    The LLM-cache step is the right failure point to exercise: it runs after the
    purge but while ``doc_status`` is still intact, so the retry genuinely
    re-enters the purge rather than short-circuiting on a missing record.
    """
    rag = await _build_rag(tmp_path, f"dfc-{uuid4().hex[:8]}")
    try:
        doc_id = await _ingest(rag)
        # Deterministic fake extraction never populates the LLM cache, so seed a
        # chunk-referenced entry to give the cache step something to fail on.
        chunk_ids = await _chunk_ids(rag, doc_id)
        cache_id = "llm-cache-dfc"
        await rag.llm_response_cache.upsert({cache_id: {"return": "cached"}})
        chunk_row = await rag.text_chunks.get_by_id(chunk_ids[0])
        await rag.text_chunks.upsert(
            {chunk_ids[0]: {**chunk_row, "llm_cache_list": [cache_id]}}
        )
        await rag.llm_response_cache.index_done_callback()
        await rag.text_chunks.index_done_callback()

        original_delete = rag.llm_response_cache.delete
        calls = {"n": 0}

        async def fail_first(ids):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("llm cache delete boom")
            return await original_delete(ids)

        rag.llm_response_cache.delete = fail_first

        first = await rag.adelete_by_doc_id(doc_id, delete_llm_cache=True)
        assert first.status == "fail"
        assert "llm cache delete boom" in first.message
        # The purge itself completed: anchors gone, chunks gone, and the journal
        # records that this is why — not that they were never there.
        assert await rag.full_entities.get_by_id(doc_id) is None
        assert await rag.full_relations.get_by_id(doc_id) is None
        row = await rag.doc_status.get_by_id(doc_id)
        assert row is not None
        assert (
            row["metadata"][KG_PURGE_METADATA_KEY]["phase"] == KG_PURGE_PHASE_COMPLETED
        )

        second = await rag.adelete_by_doc_id(doc_id, delete_llm_cache=True)

        assert second.status == "success", second.message
        assert second.status_code != 409
        assert await rag.doc_status.get_by_id(doc_id) is None
        assert await rag.full_docs.get_by_id(doc_id) is None
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_reprocess_refuses_when_anchors_lost(tmp_path):
    """The resume purge is the third caller, and fails closed the same way.

    Re-processing a document whose content is already extracted purges the
    previous run's chunks and KG contributions first. With the anchors gone that
    purge cannot account for the graph, so the document must fail rather than
    silently shed its old contributions.
    """
    rag = await _build_rag(tmp_path, f"dfc-{uuid4().hex[:8]}")
    try:
        doc_id = await _ingest(rag)
        chunk_ids = await _chunk_ids(rag, doc_id)
        await _drop_anchors(rag, doc_id)

        # Send it back through the pipeline as a resume.
        await rag.doc_status.update_doc_status_fields(
            doc_id, {"status": DocStatus.PENDING}
        )
        await rag.doc_status.index_done_callback()
        await rag.apipeline_process_enqueue_documents()

        row = await rag.doc_status.get_by_id(doc_id)
        status = row.get("status")
        status_text = status.value if isinstance(status, DocStatus) else str(status)
        assert status_text == DocStatus.FAILED.value
        assert "recovery anchor" in (row.get("error_msg") or "")
        # The previous run's data is intact and still repairable.
        assert await rag.chunk_entity_relation_graph.get_node("ALICE") is not None
        assert all(
            chunk is not None for chunk in await rag.text_chunks.get_by_ids(chunk_ids)
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_skip_kg_document_deletes_without_anchors(tmp_path):
    """``skip_kg`` produces a document with legitimately NO anchor rows.

    Extraction and merge are skipped entirely, so Phase 0 never runs. The
    enqueue-time ``kg_write_state=pre_graph`` marker is what carries the proof
    instead — and it must survive to PROCESSED, which is why the PROCESSED
    transition retires only the purge journal and not the marker.
    """
    rag = await _build_rag(tmp_path, f"dfc-{uuid4().hex[:8]}")
    try:
        doc_id = await _ingest_skip_kg(rag)
        row = await rag.doc_status.get_by_id(doc_id)
        assert row["metadata"][KG_WRITE_STATE_METADATA_KEY] == KG_WRITE_STATE_PRE_GRAPH
        chunk_ids = await _chunk_ids(rag, doc_id)
        assert chunk_ids, "skip_kg still produces chunks for naive/mix retrieval"

        result = await rag.adelete_by_doc_id(doc_id)

        assert result.status == "success", result.message
        assert await rag.doc_status.get_by_id(doc_id) is None
        assert all(
            chunk is None for chunk in await rag.text_chunks.get_by_ids(chunk_ids)
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_legacy_skip_kg_document_is_recoverable_via_audit(tmp_path):
    """A ``skip_kg`` document predating the marker must not be undeletable.

    It has no anchors (merge never ran) and no marker (it did not exist yet),
    so it fails closed — correctly, since nothing in the hot path can tell it
    apart from a document whose anchors were lost. The audit is the one place
    that CAN: it enumerates the entire graph, so a document appearing nowhere
    in that scan is proven to own nothing, and empty anchor rows are simply the
    truth about it.

    Without this, the remedy the refusal message names would be a dead end for
    this whole class of document: anchor repair has nothing to rebuild from, so
    ``repaired_docs`` would come back empty and the retry would refuse again.
    """
    rag = await _build_rag(tmp_path, f"dfc-{uuid4().hex[:8]}")
    try:
        doc_id = await _ingest_skip_kg(rag)
        await _strip_write_state(rag, doc_id)

        refused = await rag.adelete_by_doc_id(doc_id)
        assert refused.status_code == 409

        report = await audit_kg_integrity(rag, apply=True)
        assert report["anchorless_docs"] == [doc_id]
        assert doc_id in report["repaired_docs"]
        # Present-and-empty: the normal `anchors` proof for a document that
        # contributed nothing.
        assert (await rag.full_entities.get_by_id(doc_id))["entity_names"] == []
        assert (await rag.full_relations.get_by_id(doc_id))["relation_pairs"] == []

        result = await rag.adelete_by_doc_id(doc_id)
        assert result.status == "success", result.message
        assert await rag.doc_status.get_by_id(doc_id) is None
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_audit_never_certifies_a_document_that_owns_graph_objects(tmp_path):
    """The safety property of the anchorless certification.

    Writing empty anchors for a document that DOES own graph objects would
    manufacture a false proof and hand purge a licence to delete the chunks
    while skipping the graph — precisely the bug this work removes. Absence
    must be established from the completed scan, never assumed from a missing
    anchor row.
    """
    rag = await _build_rag(tmp_path, f"dfc-{uuid4().hex[:8]}")
    try:
        contributing = await _ingest(rag)  # owns ALICE, ACME and an edge
        empty = await _ingest_skip_kg(rag)
        await _drop_anchors(rag, contributing)
        await _strip_write_state(rag, contributing)
        await _strip_write_state(rag, empty)

        report = await audit_kg_integrity(rag, apply=True)

        # Only the genuinely empty one is certified empty.
        assert report["anchorless_docs"] == [empty]
        # The contributing one is repaired with its REAL names, not blanked.
        assert set(report["missing_entity_anchors"][contributing]) == {"ALICE", "ACME"}
        assert sorted(
            (await rag.full_entities.get_by_id(contributing))["entity_names"]
        ) == ["ACME", "ALICE"]
        assert (await rag.full_relations.get_by_id(contributing))["relation_pairs"] == [
            ["ACME", "ALICE"]
        ]

        # And deleting it now really does clean the graph.
        assert (await rag.adelete_by_doc_id(contributing)).status == "success"
        assert await rag.chunk_entity_relation_graph.get_node("ALICE") is None
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_audit_leaves_present_but_empty_anchor_rows_alone(tmp_path):
    """Row presence is the test, so an already-empty pair needs no repair."""
    rag = await _build_rag(tmp_path, f"dfc-{uuid4().hex[:8]}")
    try:
        doc_id = await _ingest_skip_kg(rag)
        await rag.full_entities.upsert({doc_id: {"entity_names": [], "count": 0}})
        await rag.full_relations.upsert({doc_id: {"relation_pairs": [], "count": 0}})
        await rag.full_entities.index_done_callback()
        await rag.full_relations.index_done_callback()

        report = await audit_kg_integrity(rag, apply=True)

        assert report["anchorless_docs"] == []
        assert report["repaired_docs"] == []
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_resume_purge_does_not_falsely_mark_a_pre_graph_document(tmp_path):
    """The resume purge must not advance ``kg_write_state``.

    A document that failed before merge is ``pre_graph`` and owns chunks but no
    graph objects. Its resume purge is allowed precisely BY that marker, so if
    retiring the journal also stamped ``graph_mutation_started``, a second
    pre-merge failure would leave a document demanding anchors it can never
    have — neither reprocessable nor deletable. The marker is monotonic and
    only the anchor-durable hook advances it.
    """
    rag = await _build_rag(tmp_path, f"dfc-{uuid4().hex[:8]}")
    try:
        doc_id = compute_mdhash_id("pg.txt", prefix="doc-")
        await rag.apipeline_enqueue_documents(
            "pre graph doc", ids=[doc_id], file_paths=["pg.txt"]
        )

        # Fail during merge so chunks are written but the marker is still
        # pre_graph (Phase 0's anchor-durable hook never runs).
        async def boom(**kwargs):
            raise RuntimeError("merge boom")

        original_merge = pipeline_module.merge_nodes_and_edges
        pipeline_module.merge_nodes_and_edges = boom
        try:
            await rag.apipeline_process_enqueue_documents()
        finally:
            pipeline_module.merge_nodes_and_edges = original_merge

        row = await rag.doc_status.get_by_id(doc_id)
        assert row["metadata"][KG_WRITE_STATE_METADATA_KEY] == KG_WRITE_STATE_PRE_GRAPH
        assert await rag.full_entities.get_by_id(doc_id) is None

        # The resume purge is permitted by the marker, and must leave it alone.
        await request_failed_retry(rag)
        await rag.apipeline_process_enqueue_documents()

        row = await rag.doc_status.get_by_id(doc_id)
        assert KG_PURGE_METADATA_KEY not in row["metadata"]
        # A completed run advanced it legitimately; what must never happen is
        # the marker moving on while the anchors are still absent.
        if row["metadata"].get(KG_WRITE_STATE_METADATA_KEY) == (
            KG_WRITE_STATE_GRAPH_MUTATION_STARTED
        ):
            assert await rag.full_entities.get_by_id(doc_id) is not None
            assert await rag.full_relations.get_by_id(doc_id) is not None
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_reprocess_clears_stale_chunk_list_after_purge(tmp_path):
    """A successful resume purge must persist the emptied chunk list.

    Leaving the stored ``chunks_list`` pointing at chunks the purge just deleted
    made the row advertise data that no longer existed, and a crash before the
    next write left it that way.
    """
    rag = await _build_rag(tmp_path, f"dfc-{uuid4().hex[:8]}")
    try:
        doc_id = await _ingest(rag)
        first_chunks = await _chunk_ids(rag, doc_id)
        observed: dict = {}

        original_resume_purge = rag._purge_stale_extraction_if_resuming

        async def spy_resume_purge(**kwargs):
            await original_resume_purge(**kwargs)
            # Read straight after the resume purge returns and BEFORE the new
            # run writes any chunks: the reset must already be persisted, not
            # merely applied to the in-memory status object.
            row = await rag.doc_status.get_by_id(kwargs["doc_id"])
            observed["chunks_list"] = list(row.get("chunks_list") or [])
            observed["journal"] = row.get("metadata", {}).get(KG_PURGE_METADATA_KEY)

        rag._purge_stale_extraction_if_resuming = spy_resume_purge

        await rag.doc_status.update_doc_status_fields(
            doc_id, {"status": DocStatus.PENDING}
        )
        await rag.doc_status.index_done_callback()
        await rag.apipeline_process_enqueue_documents()

        assert observed["chunks_list"] == []
        # Journal retired in the same write, so it cannot collide with the next
        # purge's operation id.
        assert observed["journal"] is None
        assert first_chunks
    finally:
        await rag.finalize_storages()
