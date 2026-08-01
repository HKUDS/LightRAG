"""Fail-closed recovery-proof contract for whole-document purge (issue #3400).

A purge learns what a document contributed to the shared knowledge graph from
its write-ahead recovery anchors (``full_entities`` / ``full_relations``). When
those rows were absent, candidate resolution collapsed them to an empty list —
indistinguishable from a legitimately empty anchor — so graph/vector/tracking
cleanup was skipped while the chunks were still deleted. That strands live
entities whose provenance can no longer be reconstructed, because the reverse
lookup runs graph ``source_id`` -> ``text_chunks`` -> ``full_doc_id`` and purge
just removed those chunks.

These tests pin the replacement contract:

- presence of the anchor ROWS is the proof, never the truthiness of the lists
  they hold (an empty row is a document that extracted no entities);
- ``kg_write_state=pre_graph`` is proof on its own, and spends no graph access;
- with no proof, purge raises before its FIRST write — asserted by checking
  every storage double recorded nothing;
- the phase journal makes a partially-failed purge resumable, which is what
  keeps fail-closed from deadlocking on purge's own anchor deletion;
- a journal for a different operation is refused rather than resumed, except a
  ``completed`` one, which is stale bookkeeping rather than a conflict.

In-memory doubles are shared with ``test_purge_primitive`` in spirit but kept
local so each file states the state it depends on.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest
import lightrag.operate as operate_module

from lightrag import LightRAG
from lightrag.constants import (
    GRAPH_FIELD_SEP,
    KG_PURGE_METADATA_KEY,
    KG_PURGE_PHASE_ANCHORS_PENDING,
    KG_PURGE_PHASE_COMPLETED,
    KG_PURGE_PHASE_DERIVED_COMMITTED,
    KG_PURGE_PHASE_PREPARED,
    KG_WRITE_STATE_GRAPH_MUTATION_STARTED,
    KG_WRITE_STATE_METADATA_KEY,
    KG_WRITE_STATE_PRE_GRAPH,
)
from lightrag.exceptions import (
    KGPurgeOperationConflictError,
    RecoveryAnchorMissingError,
    StorageRecordNotFoundError,
)
from lightrag.utils_pipeline import make_kg_purge_operation_id

pytestmark = pytest.mark.offline

DOC = "d1"


@pytest.fixture(autouse=True)
def _storage_keyed_lock_noop(monkeypatch):
    @asynccontextmanager
    async def _noop_lock(*args, **kwargs):
        yield

    monkeypatch.setattr(operate_module, "get_storage_keyed_lock", _noop_lock)


class _KV:
    def __init__(self, data: dict | None = None):
        self.data = dict(data or {})
        self.deleted: list[str] = []
        self.upserted: list[dict] = []

    async def get_by_id(self, key):
        return self.data.get(key)

    async def get_by_ids(self, keys):
        return [self.data.get(k) for k in keys]

    async def upsert(self, data):
        self.upserted.append(dict(data))
        self.data.update(data)

    async def delete(self, ids):
        self.deleted.extend(ids)
        for k in ids:
            self.data.pop(k, None)

    async def index_done_callback(self):
        pass


class _DocStatus(_KV):
    async def update_doc_status_fields(self, doc_id, fields, *, missing_ok=False):
        existing = self.data.get(doc_id)
        if existing is None:
            if missing_ok:
                return
            raise StorageRecordNotFoundError(doc_id)
        self.data[doc_id] = {**existing, **fields}

    def journal(self, doc_id: str = DOC) -> dict | None:
        return (
            (self.data.get(doc_id) or {}).get("metadata", {}).get(KG_PURGE_METADATA_KEY)
        )


class _Vdb:
    def __init__(self):
        self.deleted: list[str] = []
        self.data: dict = {}

    async def delete(self, ids):
        self.deleted.extend(ids)
        for key in ids:
            self.data.pop(key, None)

    async def upsert(self, data):
        self.data.update(data)

    async def index_done_callback(self):
        pass


class _Graph:
    def __init__(self, nodes: dict | None = None, edges: dict | None = None):
        self.nodes = dict(nodes or {})
        self.edges = dict(edges or {})
        self.removed_nodes: list[str] = []
        self.removed_edges: list[tuple[str, str]] = []
        self.reads: list[str] = []

    async def get_nodes_batch(self, names):
        self.reads.append("get_nodes_batch")
        return {n: dict(self.nodes[n]) if n in self.nodes else None for n in names}

    async def get_edges_batch(self, pairs):
        self.reads.append("get_edges_batch")
        out = {}
        for p in pairs:
            s, t = p["src"], p["tgt"]
            edge = self.edges.get((s, t)) or self.edges.get((t, s))
            out[(s, t)] = dict(edge) if edge else None
        return out

    async def get_nodes_edges_batch(self, names):
        self.reads.append("get_nodes_edges_batch")
        return {n: [] for n in names}

    async def remove_nodes(self, names):
        self.removed_nodes.extend(names)
        for n in names:
            self.nodes.pop(n, None)

    async def remove_edges(self, pairs):
        self.removed_edges.extend(pairs)
        for s, t in pairs:
            self.edges.pop((s, t), None)
            self.edges.pop((t, s), None)


def _node(name: str, sources: list[str]) -> dict:
    return {
        "entity_id": name,
        "description": name,
        "source_id": GRAPH_FIELD_SEP.join(sources),
        "entity_type": "X",
        "file_path": "f",
    }


def _status_row(
    *,
    write_state: str | None = None,
    purge_journal: dict | None = None,
    patch_journal: dict | None = None,
) -> dict:
    metadata: dict = {}
    if write_state is not None:
        metadata[KG_WRITE_STATE_METADATA_KEY] = write_state
    if purge_journal is not None:
        metadata[KG_PURGE_METADATA_KEY] = purge_journal
    if patch_journal is not None:
        metadata["custom_chunk_patch"] = patch_journal
    return {"status": "processed", "chunks_list": [], "metadata": metadata}


def _journal(phase: str, chunk_ids: list[str], *, doc_id: str = DOC) -> dict:
    return {
        "schema_version": 1,
        "operation_id": make_kg_purge_operation_id(doc_id, chunk_ids),
        "phase": phase,
        "chunk_count": len(chunk_ids),
        "updated_at": 0,
    }


def _make_rag(
    *,
    graph: _Graph | None = None,
    full_entities: _KV | None = None,
    full_relations: _KV | None = None,
    entity_chunks: _KV | None = None,
    doc_status: _DocStatus | None = None,
) -> LightRAG:
    rag = LightRAG.__new__(LightRAG)
    rag.chunk_entity_relation_graph = graph if graph is not None else _Graph()
    rag.full_entities = full_entities if full_entities is not None else _KV()
    rag.full_relations = full_relations if full_relations is not None else _KV()
    rag.entity_chunks = entity_chunks if entity_chunks is not None else _KV()
    rag.relation_chunks = _KV()
    rag.chunks_vdb = _Vdb()
    rag.entities_vdb = _Vdb()
    rag.relationships_vdb = _Vdb()
    rag.text_chunks = _KV()
    rag.llm_response_cache = _KV()
    rag.doc_status = (
        doc_status if doc_status is not None else _DocStatus({DOC: _status_row()})
    )

    async def _noop_insert_done(*args, **kwargs):
        return None

    rag._insert_done = _noop_insert_done
    rag._build_global_config = lambda: {
        "llm_model_max_async": 1,
        "max_source_ids_per_entity": 100,
        "max_source_ids_per_relation": 100,
        "source_ids_limit_method": "KEEP",
        "max_file_paths": 100,
        "file_path_more_placeholder": "more",
    }
    return rag


def _status():
    return {"latest_message": "", "history_messages": []}, asyncio.Lock()


def _assert_nothing_deleted(rag: LightRAG) -> None:
    """Fail-closed means the raise happened BEFORE the first write."""
    graph = rag.chunk_entity_relation_graph
    assert graph.removed_nodes == []
    assert graph.removed_edges == []
    assert rag.chunks_vdb.deleted == []
    assert rag.entities_vdb.deleted == []
    assert rag.relationships_vdb.deleted == []
    assert rag.text_chunks.deleted == []
    assert rag.entity_chunks.deleted == []
    assert rag.relation_chunks.deleted == []
    assert rag.full_entities.deleted == []
    assert rag.full_relations.deleted == []


async def _purge(rag: LightRAG, chunk_ids: list[str], doc_id: str = DOC):
    status, lock = _status()
    return await rag._purge_kg_contributions(
        doc_id, chunk_ids, pipeline_status=status, pipeline_status_lock=lock
    )


# --------------------------------------------------------------------------
# Proof: the anchor rows
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_anchor_rows_are_a_valid_proof():
    """Present-but-empty anchors mean "extracted nothing", not "unknown".

    Collapsing the two is the original defect, so this is the single most
    important distinction in the contract.
    """
    rag = _make_rag(
        full_entities=_KV({DOC: {"entity_names": [], "count": 0}}),
        full_relations=_KV({DOC: {"relation_pairs": [], "count": 0}}),
    )

    await _purge(rag, ["c1"])

    assert rag.chunks_vdb.deleted == ["c1"]
    assert rag.text_chunks.deleted == ["c1"]
    # Whole-document mode still retires the anchors at the end.
    assert rag.full_entities.deleted == [DOC]
    assert rag.full_relations.deleted == [DOC]


@pytest.mark.parametrize(
    "seed_entities, seed_relations, expected_missing",
    [
        (False, True, ("full_entities",)),
        (True, False, ("full_relations",)),
        (False, False, ("full_entities", "full_relations")),
    ],
    ids=["entities_row_gone", "relations_row_gone", "both_rows_gone"],
)
@pytest.mark.asyncio
async def test_missing_anchor_row_refuses_without_deleting(
    seed_entities, seed_relations, expected_missing
):
    """Either row missing is enough to refuse: both are needed to account for
    a document's contributions, and a half-present pair is not a proof."""
    rag = _make_rag(
        graph=_Graph(nodes={"ALICE": _node("ALICE", ["c1"])}),
        full_entities=_KV(
            {DOC: {"entity_names": ["ALICE"], "count": 1}} if seed_entities else {}
        ),
        full_relations=_KV(
            {DOC: {"relation_pairs": [], "count": 0}} if seed_relations else {}
        ),
        entity_chunks=_KV({"ALICE": {"chunk_ids": ["c1"]}}),
    )

    with pytest.raises(RecoveryAnchorMissingError) as excinfo:
        await _purge(rag, ["c1"])

    assert excinfo.value.reason == RecoveryAnchorMissingError.REASON_MISSING_ANCHOR_ROWS
    assert excinfo.value.doc_id == DOC
    assert excinfo.value.missing_namespaces == expected_missing
    assert "audit_kg_integrity" in str(excinfo.value)
    _assert_nothing_deleted(rag)


@pytest.mark.asyncio
async def test_structurally_broken_anchor_row_is_not_a_proof():
    """A row whose payload is the wrong shape cannot be trusted as a candidate
    list, so it is treated as missing rather than silently read as empty."""
    rag = _make_rag(
        full_entities=_KV({DOC: {"entity_names": "ALICE"}}),  # str, not list
        full_relations=_KV({DOC: {"relation_pairs": [], "count": 0}}),
    )

    with pytest.raises(RecoveryAnchorMissingError) as excinfo:
        await _purge(rag, ["c1"])

    assert excinfo.value.missing_namespaces == ("full_entities",)
    _assert_nothing_deleted(rag)


@pytest.mark.asyncio
async def test_graph_mutation_started_does_not_excuse_missing_anchors():
    """The marker only helps while it still says ``pre_graph``."""
    rag = _make_rag(
        doc_status=_DocStatus(
            {DOC: _status_row(write_state=KG_WRITE_STATE_GRAPH_MUTATION_STARTED)}
        ),
    )

    with pytest.raises(RecoveryAnchorMissingError):
        await _purge(rag, ["c1"])

    _assert_nothing_deleted(rag)


@pytest.mark.asyncio
async def test_unknown_write_state_refuses():
    """A pre-#3416 document has no marker at all; unknown must fail closed."""
    rag = _make_rag(doc_status=_DocStatus({DOC: _status_row()}))

    with pytest.raises(RecoveryAnchorMissingError) as excinfo:
        await _purge(rag, ["c1"])

    assert "kg_write_state=unknown" in str(excinfo.value)
    _assert_nothing_deleted(rag)


@pytest.mark.asyncio
async def test_absent_doc_status_row_refuses():
    """No doc_status row means no marker and no journal — still fail closed."""
    rag = _make_rag(doc_status=_DocStatus({}))

    with pytest.raises(RecoveryAnchorMissingError):
        await _purge(rag, ["c1"])

    _assert_nothing_deleted(rag)


# --------------------------------------------------------------------------
# Proof: kg_write_state=pre_graph
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pre_graph_cleans_staged_chunks_without_touching_the_graph():
    """Proven never to have merged, so the staged chunks can go and the graph
    is not read at all — the candidate set is explicitly empty rather than
    inferred from unreadable anchors."""
    graph = _Graph(nodes={"ALICE": _node("ALICE", ["c1"])})
    rag = _make_rag(
        graph=graph,
        doc_status=_DocStatus({DOC: _status_row(write_state=KG_WRITE_STATE_PRE_GRAPH)}),
    )

    await _purge(rag, ["c1"])

    assert rag.chunks_vdb.deleted == ["c1"]
    assert rag.text_chunks.deleted == ["c1"]
    # No graph access whatsoever, and the unrelated node survives.
    assert graph.reads == []
    assert graph.removed_nodes == []
    assert graph.nodes["ALICE"] is not None


@pytest.mark.asyncio
async def test_pre_graph_still_purges_patch_journal_candidates():
    """A patch-mode merge mutates the graph without writing anchors, so a
    ``pre_graph`` document CAN own graph objects — exactly the ones its
    custom-chunk journal names. Those must still be cleaned."""
    graph = _Graph(nodes={"BOB": _node("BOB", ["c1"])})
    rag = _make_rag(
        graph=graph,
        entity_chunks=_KV({"BOB": {"chunk_ids": ["c1"]}}),
        doc_status=_DocStatus(
            {
                DOC: _status_row(
                    write_state=KG_WRITE_STATE_PRE_GRAPH,
                    patch_journal={"entity_names": ["BOB"], "relation_pairs": []},
                )
            }
        ),
    )

    await _purge(rag, ["c1"])

    assert graph.removed_nodes == ["BOB"]
    assert rag.chunks_vdb.deleted == ["c1"]


# --------------------------------------------------------------------------
# Anchors that name objects the purge cannot classify
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_chunks_but_populated_anchors_refuses():
    """With no chunk ids there is nothing to subtract from those objects'
    source lists, so every one would be kept while the anchors were dropped
    anyway — the same orphan outcome, hence its own refusal reason."""
    rag = _make_rag(
        graph=_Graph(nodes={"ALICE": _node("ALICE", ["c1"])}),
        full_entities=_KV({DOC: {"entity_names": ["ALICE"], "count": 1}}),
        full_relations=_KV({DOC: {"relation_pairs": [], "count": 0}}),
    )

    with pytest.raises(RecoveryAnchorMissingError) as excinfo:
        await _purge(rag, [])

    assert (
        excinfo.value.reason
        == RecoveryAnchorMissingError.REASON_CHUNKLESS_CONTRIBUTIONS
    )
    assert "no chunks to attribute them to" in str(excinfo.value)
    _assert_nothing_deleted(rag)


@pytest.mark.asyncio
async def test_no_chunks_with_empty_anchors_cleans_up_the_stub():
    """Nothing to strand, so the two empty rows are removed rather than left
    keyed to a document that is about to disappear."""
    rag = _make_rag(
        full_entities=_KV({DOC: {"entity_names": [], "count": 0}}),
        full_relations=_KV({DOC: {"relation_pairs": [], "count": 0}}),
    )

    await _purge(rag, [])

    assert rag.full_entities.deleted == [DOC]
    assert rag.full_relations.deleted == [DOC]


# --------------------------------------------------------------------------
# Proof: nothing attribution-bearing would be deleted
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_scope_needs_no_proof():
    """No chunks and no populated anchors: nothing to prove.

    Fail-closed exists to stop a purge destroying the chunk rows and anchor
    rows that are the only record of what a document contributed. An operation
    that removes neither cannot strand anything, whatever the document's
    history — so a legacy row enqueued before ``kg_write_state`` existed, still
    holding no chunks, deletes without a scan or an audit.
    """
    graph = _Graph(nodes={"ALICE": _node("ALICE", ["c1"])})
    rag = _make_rag(
        graph=graph,
        full_entities=_KV({}),
        full_relations=_KV({}),
        doc_status=_DocStatus({DOC: _status_row()}),  # no marker, no journal
    )

    await _purge(rag, [])

    # Nothing was destroyed, and the unrelated graph node is untouched.
    assert graph.reads == []
    assert graph.removed_nodes == []
    assert rag.chunks_vdb.deleted == []
    assert rag.text_chunks.deleted == []


@pytest.mark.asyncio
async def test_empty_scope_does_not_extend_to_a_document_with_chunks():
    """The moment there are chunks to delete, the proof is required again.

    This is the original defect's exact shape: chunks would go while the graph
    was skipped, destroying the provenance that makes the survivors
    attributable.
    """
    rag = _make_rag(
        graph=_Graph(nodes={"ALICE": _node("ALICE", ["c1"])}),
        full_entities=_KV({}),
        full_relations=_KV({}),
        entity_chunks=_KV({"ALICE": {"chunk_ids": ["c1"]}}),
    )

    with pytest.raises(RecoveryAnchorMissingError):
        await _purge(rag, ["c1"])

    _assert_nothing_deleted(rag)


@pytest.mark.asyncio
async def test_empty_scope_does_not_extend_to_a_populated_surviving_anchor():
    """One anchor row missing, the other naming objects, and no chunks.

    Deleting the surviving row would destroy the only record of what those
    objects belong to, so this is a destructive operation and still needs a
    proof — the empty-scope reasoning does not reach it.
    """
    rag = _make_rag(
        graph=_Graph(nodes={"ALICE": _node("ALICE", ["c1"])}),
        full_entities=_KV({DOC: {"entity_names": ["ALICE"], "count": 1}}),
        full_relations=_KV({}),  # missing
    )

    with pytest.raises(RecoveryAnchorMissingError) as excinfo:
        await _purge(rag, [])

    assert excinfo.value.missing_namespaces == ("full_relations",)
    _assert_nothing_deleted(rag)


@pytest.mark.asyncio
async def test_a_false_pre_graph_marker_would_reproduce_the_original_defect():
    """Why ``kg_write_state`` must never be inferred from observable state.

    ``pre_graph`` asserts "this document never touched the graph", which
    licenses deleting its chunks while skipping the graph entirely. That is
    sound only because the marker is written ONCE, at enqueue, when it is
    necessarily true — never derived from a document that already has history.

    This test pins what an inferred marker would cost: a backfill keying off a
    momentarily-empty ``chunks_list`` would stamp a document that does own
    graph objects, and the stamp is durable, so the damage lands later when the
    chunks reappear. The empty-scope proof above is safe precisely because it
    is re-evaluated against live state on every call and grants nothing beyond
    it.
    """
    graph = _Graph(nodes={"ALICE": _node("ALICE", ["c1"])})
    rag = _make_rag(
        graph=graph,
        full_entities=_KV({}),
        full_relations=_KV({}),
        entity_chunks=_KV({"ALICE": {"chunk_ids": ["c1"]}}),
        doc_status=_DocStatus({DOC: _status_row(write_state=KG_WRITE_STATE_PRE_GRAPH)}),
    )

    await _purge(rag, ["c1"])

    # Chunks gone, entity still there: an orphan nothing can attribute.
    assert rag.chunks_vdb.deleted == ["c1"]
    assert graph.removed_nodes == []
    assert graph.nodes["ALICE"] is not None


# --------------------------------------------------------------------------
# Explicit candidates bypass the proof entirely
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_explicit_candidates_need_no_proof_and_write_no_journal():
    """Custom-chunk rollback carries its own operation journal naming the
    complete candidate superset, so it neither consults the anchors nor
    journals a purge phase."""

    class _ExplodingKV(_KV):
        async def get_by_id(self, key):
            raise AssertionError("anchor row must not be read")

    doc_status = _DocStatus({DOC: _status_row()})
    rag = _make_rag(
        graph=_Graph(nodes={"ALICE": _node("ALICE", ["c1"])}),
        full_entities=_ExplodingKV(),
        full_relations=_ExplodingKV(),
        entity_chunks=_KV({"ALICE": {"chunk_ids": ["c1"]}}),
        doc_status=doc_status,
    )
    status, lock = _status()

    await rag._purge_kg_contributions(
        DOC,
        ["c1"],
        candidate_entities=["ALICE"],
        candidate_relations=[],
        patch_only=True,
        pipeline_status=status,
        pipeline_status_lock=lock,
    )

    assert rag.chunk_entity_relation_graph.removed_nodes == ["ALICE"]
    assert doc_status.journal() is None


# --------------------------------------------------------------------------
# The phase journal: resume, and the deadlock it prevents
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_successful_purge_walks_every_phase_in_order():
    doc_status = _DocStatus({DOC: _status_row()})
    phases: list[str] = []
    rag = _make_rag(
        full_entities=_KV({DOC: {"entity_names": [], "count": 0}}),
        full_relations=_KV({DOC: {"relation_pairs": [], "count": 0}}),
        doc_status=doc_status,
    )
    original = rag._write_kg_purge_phase

    async def spy(doc_id, phase, **kwargs):
        phases.append(phase)
        await original(doc_id, phase, **kwargs)

    rag._write_kg_purge_phase = spy

    await _purge(rag, ["c1"])

    assert phases == [
        KG_PURGE_PHASE_PREPARED,
        KG_PURGE_PHASE_DERIVED_COMMITTED,
        KG_PURGE_PHASE_ANCHORS_PENDING,
        KG_PURGE_PHASE_COMPLETED,
    ]
    assert doc_status.journal()["phase"] == KG_PURGE_PHASE_COMPLETED


@pytest.mark.asyncio
async def test_anchors_pending_journal_survives_anchor_loss():
    """The deadlock fail-closed would otherwise create.

    Purge's last step deletes the anchors. A failure after the FIRST anchor
    delete leaves them half-gone, and without the journal every retry would
    see "anchors missing" and refuse forever.
    """
    doc_status = _DocStatus(
        {
            DOC: _status_row(
                purge_journal=_journal(KG_PURGE_PHASE_ANCHORS_PENDING, ["c1"])
            )
        }
    )
    # full_entities already deleted by the failed attempt; only relations left.
    rag = _make_rag(
        full_entities=_KV({}),
        full_relations=_KV({DOC: {"relation_pairs": [], "count": 0}}),
        doc_status=doc_status,
    )

    await _purge(rag, ["c1"])

    assert rag.full_relations.deleted == [DOC]
    assert doc_status.journal()["phase"] == KG_PURGE_PHASE_COMPLETED
    # Chunks were already deleted by the failed attempt; not deleted twice.
    assert rag.chunks_vdb.deleted == []


@pytest.mark.asyncio
async def test_derived_committed_journal_skips_the_expensive_pass():
    """Resume must not re-run candidate analysis or the LLM-cache-backed
    rebuild — it jumps straight to deleting the chunks."""
    graph = _Graph(nodes={"ALICE": _node("ALICE", ["c1"])})
    rag = _make_rag(
        graph=graph,
        full_entities=_KV({}),
        full_relations=_KV({}),
        doc_status=_DocStatus(
            {
                DOC: _status_row(
                    purge_journal=_journal(KG_PURGE_PHASE_DERIVED_COMMITTED, ["c1"])
                )
            }
        ),
    )

    await _purge(rag, ["c1"])

    assert graph.reads == []
    assert graph.removed_nodes == []
    assert rag.chunks_vdb.deleted == ["c1"]


@pytest.mark.asyncio
async def test_prepared_journal_reruns_the_derived_pass():
    """``prepared`` means nothing was deleted yet, so it is NOT a proof on its
    own — the anchors must still be readable."""
    doc_status = _DocStatus(
        {DOC: _status_row(purge_journal=_journal(KG_PURGE_PHASE_PREPARED, ["c1"]))}
    )
    rag = _make_rag(
        full_entities=_KV({}), full_relations=_KV({}), doc_status=doc_status
    )

    with pytest.raises(RecoveryAnchorMissingError):
        await _purge(rag, ["c1"])

    _assert_nothing_deleted(rag)


@pytest.mark.asyncio
async def test_completed_journal_for_the_same_operation_is_a_noop():
    """A retry of the caller's post-purge finalization must not redo the purge
    — nor trip the missing-anchor refusal now that the anchors are gone."""
    graph = _Graph(nodes={"ALICE": _node("ALICE", ["c1"])})
    rag = _make_rag(
        graph=graph,
        full_entities=_KV({}),
        full_relations=_KV({}),
        doc_status=_DocStatus(
            {DOC: _status_row(purge_journal=_journal(KG_PURGE_PHASE_COMPLETED, ["c1"]))}
        ),
    )

    await _purge(rag, ["c1"])

    _assert_nothing_deleted(rag)
    assert graph.reads == []


@pytest.mark.asyncio
async def test_in_flight_journal_for_a_different_operation_is_refused():
    """A changed chunk set means the journaled phase describes work on a
    different set, so resuming from it would skip cleanup for the difference."""
    doc_status = _DocStatus(
        {
            DOC: _status_row(
                purge_journal=_journal(KG_PURGE_PHASE_ANCHORS_PENDING, ["c1"])
            )
        }
    )
    rag = _make_rag(doc_status=doc_status)

    with pytest.raises(KGPurgeOperationConflictError) as excinfo:
        await _purge(rag, ["c1", "c2"])

    assert excinfo.value.doc_id == DOC
    assert excinfo.value.requested_operation_id == make_kg_purge_operation_id(
        DOC, ["c1", "c2"]
    )
    assert excinfo.value.journal_operation_id == make_kg_purge_operation_id(DOC, ["c1"])
    _assert_nothing_deleted(rag)


@pytest.mark.asyncio
async def test_stale_completed_journal_for_a_different_operation_is_ignored():
    """A finished purge holds no resume information worth protecting, and its
    retirement is a separate write a crash can skip — so a stale one must not
    refuse the next purge forever over dead bookkeeping."""
    rag = _make_rag(
        full_entities=_KV({DOC: {"entity_names": [], "count": 0}}),
        full_relations=_KV({DOC: {"relation_pairs": [], "count": 0}}),
        doc_status=_DocStatus(
            {
                DOC: _status_row(
                    purge_journal=_journal(KG_PURGE_PHASE_COMPLETED, ["old"])
                )
            }
        ),
    )

    await _purge(rag, ["c1"])

    assert rag.chunks_vdb.deleted == ["c1"]


@pytest.mark.asyncio
async def test_operation_id_ignores_chunk_order_and_duplicates():
    """The chunk list is assembled from chunks_list unioned with a journal, so
    its order is an artifact; only the SET determines what gets deleted."""
    assert make_kg_purge_operation_id(DOC, ["a", "b"]) == make_kg_purge_operation_id(
        DOC, ["b", "a", "b"]
    )
    assert make_kg_purge_operation_id(DOC, ["a"]) != make_kg_purge_operation_id(
        DOC, ["a", "b"]
    )


@pytest.mark.asyncio
async def test_journal_write_failure_aborts_before_any_deletion():
    """If the journal cannot be persisted, the purge must not proceed: a crash
    later would then be indistinguishable from a never-anchored document."""
    rag = _make_rag(
        full_entities=_KV({DOC: {"entity_names": [], "count": 0}}),
        full_relations=_KV({DOC: {"relation_pairs": [], "count": 0}}),
    )

    async def boom(*args, **kwargs):
        raise RuntimeError("journal write boom")

    rag._write_kg_purge_phase = boom

    with pytest.raises(RuntimeError, match="journal write boom"):
        await _purge(rag, ["c1"])

    _assert_nothing_deleted(rag)
