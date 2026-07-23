"""Fault-injection matrix with restart simulation (issue #3400, Phase 5).

For each persistence boundary in the ingestion saga, inject a failure, then
simulate a process restart (finalize storages, build a FRESH LightRAG over
the same working dir/workspace), retry, and assert convergence:

- the document ends PROCESSED only after a clean run;
- after convergence the KG has no orphan contributions and the recovery
  anchors cover the graph (verified with the offline audit tool);
- no failure point ever yields a durable false PROCESSED.

JSON/NetworkX (buffered) backends — the buffered half of the issue's backend
matrix; immediate-write backends are covered by the mock-based unit suites.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import DocStatus
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
    """Deterministic extraction (instance-level): one ALICE entity per chunk."""

    async def fake_extract(chunks, *args, **kwargs):
        results = []
        for chunk_id in chunks:
            results.append(
                (
                    {
                        "ALICE": [
                            {
                                "entity_name": "ALICE",
                                "entity_type": "person",
                                "description": "ALICE description",
                                "source_id": chunk_id,
                                "file_path": "d.txt",
                                "timestamp": 1,
                            }
                        ]
                    },
                    {},
                )
            )
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


def _status_text(row: dict | None) -> str:
    raw = (row or {}).get("status")
    return raw.value if isinstance(raw, DocStatus) else str(raw or "<missing>")


async def _assert_converged(rag: LightRAG, doc_id: str) -> None:
    row = await rag.doc_status.get_by_id(doc_id)
    assert _status_text(row) == DocStatus.PROCESSED.value

    # Terminal-consistency invariant, checked with the offline audit tool:
    # every graph contribution is anchored and no orphan sources exist.
    report = await audit_kg_integrity(rag)
    assert report["missing_entity_anchors"] == {}
    assert report["missing_relation_anchors"] == {}
    assert report["orphan_entities"] == []
    assert report["orphan_relations"] == []
    assert await rag.chunk_entity_relation_graph.get_node("ALICE") is not None


def _fail_once(monkeypatch, obj, attr: str, exc_message: str):
    """Wrap obj.attr (async) to raise on the FIRST call only."""
    calls = {"n": 0}
    original = getattr(obj, attr)

    async def wrapper(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError(exc_message)
        return await original(*args, **kwargs)

    monkeypatch.setattr(obj, attr, wrapper)
    return calls


# Injection points: (test id, callable(rag, monkeypatch) -> None) applied to
# the FIRST instance before the failing run.
def _inject_anchor_prewrite_flush(rag, monkeypatch):
    _fail_once(
        monkeypatch, rag.full_entities, "index_done_callback", "anchor flush boom"
    )


def _inject_graph_mutation(rag, monkeypatch):
    _fail_once(
        monkeypatch,
        rag.chunk_entity_relation_graph,
        "upsert_node",
        "graph mutation boom",
    )


def _inject_derived_flush(rag, monkeypatch):
    _fail_once(monkeypatch, rag.entities_vdb, "index_done_callback", "vdb flush boom")


def _inject_final_status_write(rag, monkeypatch):
    orig_upsert = rag.doc_status.upsert
    calls = {"boomed": False}

    async def wrapper(data):
        for record in data.values():
            status = record.get("status")
            status_text = status.value if isinstance(status, DocStatus) else str(status)
            if status_text == DocStatus.PROCESSED.value and not calls["boomed"]:
                calls["boomed"] = True
                raise RuntimeError("status write boom")
        return await orig_upsert(data)

    monkeypatch.setattr(rag.doc_status, "upsert", wrapper)


def _fail_once_on(rag, monkeypatch, injector_name):
    {
        "anchor_prewrite_flush": _inject_anchor_prewrite_flush,
        "graph_mutation": _inject_graph_mutation,
        "derived_store_flush": _inject_derived_flush,
        "final_status_write": _inject_final_status_write,
    }[injector_name](rag, monkeypatch)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "injection_point",
    [
        "anchor_prewrite_flush",
        "graph_mutation",
        "derived_store_flush",
        "final_status_write",
    ],
)
async def test_pipeline_failure_then_restart_converges(
    tmp_path, monkeypatch, injection_point
):
    workspace = f"fim-{uuid4().hex[:8]}"
    doc_id = compute_mdhash_id("matrix.txt", prefix="doc-")

    rag1 = await _build_rag(tmp_path, workspace)
    try:
        await rag1.apipeline_enqueue_documents(
            input="matrix doc", file_paths="matrix.txt"
        )
        _fail_once_on(rag1, monkeypatch, injection_point)
        await rag1.apipeline_process_enqueue_documents()

        # No failure point may leave a durable false PROCESSED...
        row = await rag1.doc_status.get_by_id(doc_id)
        assert _status_text(row) != DocStatus.PROCESSED.value, (
            f"{injection_point}: PROCESSED written despite injected failure"
        )
    finally:
        await rag1.finalize_storages()

    # ...and after a restart, an explicit manual retry converges (a FAILED
    # doc re-enters only via the /reprocess_failed semantics; a doc left
    # PENDING/interrupted by the crash is picked up automatically either way).
    rag2 = await _build_rag(tmp_path, workspace)
    try:
        await request_failed_retry(rag2)
        await rag2.apipeline_process_enqueue_documents()
        await _assert_converged(rag2, doc_id)
    finally:
        await rag2.finalize_storages()


@pytest.mark.asyncio
async def test_custom_chunk_failure_then_restart_resume_converges(
    tmp_path, monkeypatch
):
    """Custom-chunk saga across a restart: merge fails, journal survives the
    restart, the same SDK call on a FRESH instance resumes and commits."""
    import lightrag.lightrag as lightrag_module

    workspace = f"fim-cc-{uuid4().hex[:8]}"

    rag1 = await _build_rag(tmp_path, workspace)
    try:
        await rag1.ainsert_custom_chunks("base", ["alice is here"], doc_id="doc-1")

        async def merge_boom(**kwargs):
            raise RuntimeError("merge boom")

        monkeypatch.setattr(lightrag_module, "merge_nodes_and_edges", merge_boom)
        with pytest.raises(RuntimeError, match="merge boom"):
            await rag1.ainsert_custom_chunks("base", ["bob is there"], doc_id="doc-1")
    finally:
        await rag1.finalize_storages()
    monkeypatch.undo()

    rag2 = await _build_rag(tmp_path, workspace)
    try:
        row = await rag2.doc_status.get_by_id("doc-1")
        assert _status_text(row) == DocStatus.FAILED.value
        assert (row.get("metadata") or {}).get("custom_chunk_patch") is not None, (
            "journal must survive a restart"
        )

        await rag2.ainsert_custom_chunks("base", ["bob is there"], doc_id="doc-1")
        row = await rag2.doc_status.get_by_id("doc-1")
        assert _status_text(row) == DocStatus.PROCESSED.value

        report = await audit_kg_integrity(rag2)
        assert report["missing_entity_anchors"] == {}
        assert report["orphan_entities"] == []
    finally:
        await rag2.finalize_storages()


@pytest.mark.asyncio
async def test_custom_chunk_failure_then_restart_rollback_converges(
    tmp_path, monkeypatch
):
    """Same failure, the other recovery choice: scan rollback on a fresh
    instance restores the committed base state."""
    import lightrag.lightrag as lightrag_module

    workspace = f"fim-rb-{uuid4().hex[:8]}"

    rag1 = await _build_rag(tmp_path, workspace)
    try:
        await rag1.ainsert_custom_chunks("base", ["alice is here"], doc_id="doc-1")

        async def merge_boom(**kwargs):
            raise RuntimeError("merge boom")

        monkeypatch.setattr(lightrag_module, "merge_nodes_and_edges", merge_boom)
        with pytest.raises(RuntimeError, match="merge boom"):
            await rag1.ainsert_custom_chunks("base", ["bob is there"], doc_id="doc-1")
    finally:
        await rag1.finalize_storages()
    monkeypatch.undo()

    rag2 = await _build_rag(tmp_path, workspace)
    try:
        result = await rag2.arollback_failed_custom_chunk_patches()
        assert result == {"rolled_back": ["doc-1"], "failed": []}

        row = await rag2.doc_status.get_by_id("doc-1")
        assert _status_text(row) == DocStatus.PROCESSED.value
        assert (row.get("metadata") or {}).get("custom_chunk_patch") is None

        report = await audit_kg_integrity(rag2)
        assert report["missing_entity_anchors"] == {}
        assert report["orphan_entities"] == []
        assert await rag2.chunk_entity_relation_graph.get_node("BOB") is None
    finally:
        await rag2.finalize_storages()


@pytest.mark.asyncio
async def test_audit_tool_detects_and_repairs_missing_anchor(tmp_path):
    """Pre-fix historical data: graph contributions whose anchor row was lost
    are detected and, with apply=True, reconstructed."""
    workspace = f"fim-audit-{uuid4().hex[:8]}"
    doc_id = compute_mdhash_id("audit.txt", prefix="doc-")

    rag = await _build_rag(tmp_path, workspace)
    try:
        await rag.apipeline_enqueue_documents(input="audit doc", file_paths="audit.txt")
        await rag.apipeline_process_enqueue_documents()

        # Simulate a pre-#3400 installation: the anchor row is missing.
        await rag.full_entities.delete([doc_id])

        report = await audit_kg_integrity(rag)
        assert report["missing_entity_anchors"] == {doc_id: ["ALICE"]}

        report = await audit_kg_integrity(rag, apply=True)
        assert report["repaired_docs"] == [doc_id]

        report = await audit_kg_integrity(rag)
        assert report["missing_entity_anchors"] == {}
    finally:
        await rag.finalize_storages()


def _wire_fake_extraction_with_relation(rag: LightRAG) -> None:
    """Test-local extraction override: ALICE, BOB, and an edge between them
    per chunk.

    Overrides ``rag._process_extract_entities`` AFTER ``_build_rag`` has
    already wired the shared ``_wire_fake_extraction`` (single-entity, no
    relations) default. This function is applied only by the test below and
    never mutates ``_wire_fake_extraction`` itself, so every sibling test in
    this file that calls ``_build_rag`` keeps its original one-entity,
    zero-relation wiring and is unaffected.
    """

    async def fake_extract(chunks, *args, **kwargs):
        results = []
        for chunk_id in chunks:
            nodes = {
                "ALICE": [
                    {
                        "entity_name": "ALICE",
                        "entity_type": "person",
                        "description": "ALICE description",
                        "source_id": chunk_id,
                        "file_path": "d.txt",
                        "timestamp": 1,
                    }
                ],
                "BOB": [
                    {
                        "entity_name": "BOB",
                        "entity_type": "person",
                        "description": "BOB description",
                        "source_id": chunk_id,
                        "file_path": "d.txt",
                        "timestamp": 1,
                    }
                ],
            }
            edges = {
                ("ALICE", "BOB"): [
                    {
                        "src_id": "ALICE",
                        "tgt_id": "BOB",
                        "description": "ALICE knows BOB",
                        "keywords": "acquaintance",
                        "source_id": chunk_id,
                        "file_path": "d.txt",
                        "weight": 1.0,
                        "timestamp": 1,
                    }
                ]
            }
            results.append((nodes, edges))
        return results

    rag._process_extract_entities = fake_extract


@pytest.mark.asyncio
async def test_audit_tool_detects_true_orphan_after_anchor_loss_and_purge(tmp_path):
    """Coverage for ``audit_kg_integrity``'s orphan-*detection* path itself,
    for BOTH the entity and relation classification branches.

    This is NOT a regression test for the test above
    (``test_audit_tool_detects_and_repairs_missing_anchor``): that test's
    entity is *unanchored* but still resolvable — its source chunk survives,
    so the audit tool reports it under ``missing_entity_anchors`` and repairs
    it. A "true" orphan (``report["orphan_entities"]`` /
    ``report["orphan_relations"]``) is stronger: a node or edge whose
    ``source_id`` points ONLY at chunks that no longer exist anywhere, so no
    document can be determined at all. Every orphan assertion elsewhere in
    this suite and in ``test_purge_primitive.py`` is ``== []`` — none of them
    ever puts a real orphan in front of the tool. If ``audit_kg_integrity``'s
    orphan-detection logic silently broke for either branch (e.g. stopped
    flagging nodes/edges with an unresolvable ``source_id``, or either the
    ``orphan_entities.append(...)`` or the ``orphan_relations.append(...)``
    call in ``kg_integrity_repair.py`` were deleted), no test would fail.
    This test exists solely to guard both paths at once.

    Construction (matches the tool's own module docstring: "installations
    that ingested documents BEFORE the write-ahead recovery anchors landed
    may hold graph data that full_entities / full_relations do not
    reference"): this test wires a test-local extraction override
    (``_wire_fake_extraction_with_relation``) that produces ALICE, BOB, and
    an edge between them per chunk — unlike the shared
    ``_wire_fake_extraction`` default (ALICE only, no relations) every other
    test in this file uses. After an ordinary ingest, BOTH the
    ``full_entities`` AND ``full_relations`` anchor rows are deleted out
    from under the already-ingested document, then the ordinary
    whole-document purge runs. With both anchors gone,
    ``_purge_kg_contributions``'s candidate discovery
    (``full_entities.get_by_id(doc_id)`` / ``full_relations.get_by_id(doc_id)``)
    resolves to an empty candidate set on both sides, so neither entity-node
    removal nor edge removal happens — but chunk deletion proceeds
    regardless, leaving ALICE, BOB, and the edge between them all pointing
    at a chunk id that no longer resolves to any document.

    IMPORTANT: this test asserts the AUDIT TOOL detects the orphans it was
    built to find — it does NOT assert that purge leaving the nodes/edge
    behind is correct or desired behavior. It is a known, documented gap
    (see the module docstring of ``kg_integrity_repair.py``), and this test
    merely exploits that documented gap to get real orphans on the board so
    both detection paths are exercised. If a future change closes the gap
    (e.g. ``_purge_kg_contributions`` learns to discover entities/relations
    from the graph itself when the anchor row is missing, rather than only
    from the anchor), ALICE/BOB/their edge would stop being orphaned and
    this test's ``orphan_entities`` / ``orphan_relations`` assertions would
    need to be deliberately updated — that is an intentional signal this
    test is designed to raise, not a regression to work around.
    """
    workspace = f"fim-true-orphan-{uuid4().hex[:8]}"
    doc_id = compute_mdhash_id("orphan.txt", prefix="doc-")

    rag = await _build_rag(tmp_path, workspace)
    _wire_fake_extraction_with_relation(rag)
    try:
        await rag.apipeline_enqueue_documents(
            input="orphan doc", file_paths="orphan.txt"
        )
        await rag.apipeline_process_enqueue_documents()

        row = await rag.doc_status.get_by_id(doc_id)
        chunk_ids = list(
            dict.fromkeys(
                c for c in (row.get("chunks_list") or []) if isinstance(c, str) and c
            )
        )
        assert chunk_ids, "fixture must produce at least one chunk to purge"

        # Simulate a pre-#3400 installation: BOTH recovery anchors were
        # never written for this document.
        await rag.full_entities.delete([doc_id])
        await rag.full_relations.delete([doc_id])

        status = {"latest_message": "", "history_messages": []}
        lock = asyncio.Lock()
        await rag._purge_doc_chunks_and_kg(
            doc_id, chunk_ids, pipeline_status=status, pipeline_status_lock=lock
        )

        # The purge "succeeded" (no exception, chunk gone) but never
        # discovered ALICE, BOB, or their edge as delete candidates: both
        # nodes and the edge survive, now pointing at a chunk id that no
        # longer resolves to any document.
        assert await rag.chunk_entity_relation_graph.get_node("ALICE") is not None
        assert await rag.chunk_entity_relation_graph.get_node("BOB") is not None
        assert (
            await rag.chunk_entity_relation_graph.get_edge("ALICE", "BOB") is not None
        )
        assert await rag.text_chunks.get_by_id(chunk_ids[0]) is None

        report = await audit_kg_integrity(rag)
        assert report["orphan_entities"] == ["ALICE", "BOB"]
        assert report["orphan_relations"] == [["ALICE", "BOB"]]
    finally:
        await rag.finalize_storages()
