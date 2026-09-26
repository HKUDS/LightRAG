#!/usr/bin/env python3
"""Offline KG integrity audit / recovery-anchor repair.

Installations that ingested documents BEFORE the write-ahead recovery anchors
landed may hold graph data that ``full_entities`` / ``full_relations`` do not
reference. Such contributions are invisible to the normal purge/retry
discovery path: deleting or reprocessing their document leaves orphan graph
objects behind.

This tool enumerates the whole graph (expensive — deliberately OFFLINE-only;
the ingestion/retry/delete/scan hot paths never do this), maps every node and
edge back to its owning documents via its chunk provenance → ``text_chunks`` →
``full_doc_id``, and:

- reports per-document anchor gaps (graph contributions missing from
  ``full_entities`` / ``full_relations``);
- reports irrecoverable orphans (contributions whose source chunks no longer
  exist, so no document can be determined);
- with ``apply=True``, unions the missing entries into the per-document
  anchor rows (using the same union helper the custom-chunk commit path
  uses) so purge/retry can discover them again. Orphans are only reported —
  removing graph data is left to an operator decision.

Usage (library — works with any configured backend combination):

    rag = LightRAG(...)
    await rag.initialize_storages()
    report = await audit_kg_integrity(rag)              # report only
    report = await audit_kg_integrity(rag, apply=True)  # repair anchors

Usage (CLI — file-based default backends, honoring WORKING_DIR / WORKSPACE /
LIGHTRAG_* storage env vars; server-backend deployments should prefer the
library call from a small script wired to their own LightRAG construction):

    python -m lightrag.tools.kg_integrity_repair [--apply] [--verbose]
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any

from lightrag.base import DocStatus
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.utils import has_chunk_tracking_row, logger, make_relation_chunk_key


def _split_sources(record: dict[str, Any] | None) -> list[str]:
    raw = (record or {}).get("source_id") or ""
    return [chunk_id for chunk_id in raw.split(GRAPH_FIELD_SEP) if chunk_id]


def _union_ids(tracked: list[str] | None, graph_sources: list[str]) -> list[str]:
    """Order-preserving union of a tracking row and a graph ``source_id`` list.

    Both inputs are already flat id lists — the graph half was just split on
    ``GRAPH_FIELD_SEP`` by :func:`_split_sources`, the tracking half is a
    stored list — so this does not re-run the write-path normalization
    (``merge_source_ids``), which re-splits every element with a regex and
    costs several times the whole audit on a large graph. A joined element is
    still split, so a legacy or hand-written row cannot smuggle one in.
    """
    merged: dict[str, None] = {}
    for source in (tracked or (), graph_sources):
        for chunk_id in source:
            if not isinstance(chunk_id, str):
                continue
            parts = (
                chunk_id.split(GRAPH_FIELD_SEP)
                if GRAPH_FIELD_SEP in chunk_id
                else (chunk_id,)
            )
            for part in parts:
                part = part.strip()
                if part:
                    merged[part] = None
    return list(merged)


async def _read_chunk_tracking(
    storage, keys: list[str], batch_size: int
) -> dict[str, list[str]]:
    """Read a chunk-tracking namespace for ``keys``, keyed as given.

    Returns only rows that are present and structurally usable per
    :func:`has_chunk_tracking_row`; an absent or malformed row is omitted,
    leaving the caller with the graph ``source_id`` alone for that object.
    An unconfigured namespace yields an empty mapping.
    """
    if storage is None or not keys:
        return {}

    tracked: dict[str, list[str]] = {}
    for start in range(0, len(keys), batch_size):
        batch = keys[start : start + batch_size]
        rows = await storage.get_by_ids(batch)
        if len(rows) != len(batch):
            raise RuntimeError(
                f"chunk-tracking read returned {len(rows)} rows for "
                f"{len(batch)} keys; refusing to misattribute provenance"
            )
        for key, row in zip(batch, rows):
            if has_chunk_tracking_row(row):
                tracked[key] = row["chunk_ids"]
    return tracked


async def _map_chunks_to_docs(
    rag, chunk_ids: set[str], batch_size: int
) -> dict[str, str]:
    """chunk_id -> full_doc_id for every resolvable chunk."""
    mapping: dict[str, str] = {}
    ordered = sorted(chunk_ids)
    for start in range(0, len(ordered), batch_size):
        batch = ordered[start : start + batch_size]
        rows = await rag.text_chunks.get_by_ids(batch)
        for chunk_id, row in zip(batch, rows):
            if isinstance(row, dict) and row.get("full_doc_id"):
                mapping[chunk_id] = row["full_doc_id"]
    return mapping


async def audit_kg_integrity(
    rag,
    *,
    apply: bool = False,
    batch_size: int = 200,
) -> dict[str, Any]:
    """Audit (and optionally repair) per-document recovery-anchor coverage.

    Returns a report dict:

    - ``entities_total`` / ``relations_total``: graph object counts;
    - ``missing_entity_anchors`` / ``missing_relation_anchors``:
      ``{doc_id: [entity_name | [src, tgt], ...]}`` contributions absent from
      the document's anchor row;
    - ``orphan_entities`` / ``orphan_relations``: contributions with no
      resolvable source chunk (reported, never modified);
    - ``anchorless_docs``: documents with no anchor rows that own nothing in
      the graph — see below;
    - ``repaired_docs``: doc ids whose anchors were updated (``apply=True``).

    **Certifying absence.** A document can legitimately own nothing in the
    graph — ``skip_kg`` (``process_options`` ``'!'``) skips extraction and the
    merge entirely, so no anchor rows are ever written. A purge now
    needs a positive recovery proof, and for such a document written before
    the ``kg_write_state`` marker existed there is none: it has no anchors, and
    it never appears in the graph scan above, so anchor repair has nothing to
    rebuild from and the document would be permanently undeletable.

    This audit is the one place that CAN settle it. It enumerates the whole
    graph — something the hot paths deliberately never do — and resolves each
    object's provenance from its chunk-tracking row unioned with the graph
    ``source_id``, so the scan sees the authoritative chunk list rather than
    the capped view. A document absent from both ``doc_entities`` and
    ``doc_relations`` is therefore not merely unproven but *proven empty*.
    With ``apply=True`` those documents get written empty anchor rows, which
    is simply the truth about them and restores the normal ``anchors`` proof.
    """
    graph = rag.chunk_entity_relation_graph

    all_nodes = await graph.get_all_nodes()
    all_edges = await graph.get_all_edges()

    # Node/edge -> owning docs via their chunk provenance.
    #
    # Chunk tracking outranks the graph ``source_id``: the latter is capped by
    # ``apply_source_ids_limit``, so on any object that accumulated more source
    # chunks than the cap it names a subset. A document whose chunks all fall
    # outside that window would then be absent from the scan and certified as
    # owning nothing, which is the one conclusion this audit must never reach
    # wrongly.
    #
    # UNION rather than the purge's tracking-first-then-fallback: a purge must
    # not act on stale ids, but an audit only reports, and here a miss is the
    # expensive error. The union is a superset of what resolving from
    # ``source_id`` alone produced, so it cannot attribute less than before,
    # and an id no longer resolvable through ``text_chunks`` costs nothing —
    # it is dropped by the chunk-to-doc mapping.
    graph_node_sources: dict[str, list[str]] = {}
    for node in all_nodes:
        name = node.get("entity_id") or node.get("id")
        if not name:
            continue
        graph_node_sources[name] = _split_sources(node)

    graph_edge_sources: dict[tuple[str, str], list[str]] = {}
    for edge in all_edges:
        src, tgt = edge.get("source"), edge.get("target")
        if not src or not tgt:
            continue
        pair = tuple(sorted((src, tgt)))
        if pair in graph_edge_sources:
            continue  # some backends report undirected edges twice
        graph_edge_sources[pair] = _split_sources(edge)

    tracked_entities = await _read_chunk_tracking(
        getattr(rag, "entity_chunks", None), sorted(graph_node_sources), batch_size
    )
    # Keyed BY PAIR, never the reverse: two distinct pairs can share one
    # relation chunk key (an entity name may contain the separator), and a
    # key-to-pair dict would drop one of those edges from the scan entirely.
    edge_keys = {pair: make_relation_chunk_key(*pair) for pair in graph_edge_sources}
    tracked_relations = await _read_chunk_tracking(
        getattr(rag, "relation_chunks", None),
        sorted(set(edge_keys.values())),
        batch_size,
    )

    referenced_chunks: set[str] = set()
    node_sources: dict[str, list[str]] = {}
    for name, graph_sources in graph_node_sources.items():
        sources = _union_ids(tracked_entities.get(name), graph_sources)
        node_sources[name] = sources
        referenced_chunks.update(sources)

    edge_sources: dict[tuple[str, str], list[str]] = {}
    for pair, graph_sources in graph_edge_sources.items():
        sources = _union_ids(tracked_relations.get(edge_keys[pair]), graph_sources)
        edge_sources[pair] = sources
        referenced_chunks.update(sources)

    # The merged lists are what the rest of the audit reads; the tracking rows
    # are the unbounded half, so releasing them here keeps peak memory at or
    # below what resolving from ``source_id`` alone used.
    del tracked_entities, tracked_relations, graph_node_sources, graph_edge_sources

    chunk_to_doc = await _map_chunks_to_docs(rag, referenced_chunks, batch_size)

    doc_entities: dict[str, set[str]] = {}
    orphan_entities: list[str] = []
    for name, sources in node_sources.items():
        docs = {chunk_to_doc[c] for c in sources if c in chunk_to_doc}
        if not docs:
            orphan_entities.append(name)
            continue
        for doc_id in docs:
            doc_entities.setdefault(doc_id, set()).add(name)

    doc_relations: dict[str, set[tuple[str, str]]] = {}
    orphan_relations: list[list[str]] = []
    for pair, sources in edge_sources.items():
        docs = {chunk_to_doc[c] for c in sources if c in chunk_to_doc}
        if not docs:
            orphan_relations.append(list(pair))
            continue
        for doc_id in docs:
            doc_relations.setdefault(doc_id, set()).add(pair)

    # Compare with the stored anchor rows.
    missing_entity_anchors: dict[str, list[str]] = {}
    for doc_id, names in doc_entities.items():
        row = await rag.full_entities.get_by_id(doc_id)
        anchored = set((row or {}).get("entity_names") or [])
        missing = sorted(names - anchored)
        if missing:
            missing_entity_anchors[doc_id] = missing

    missing_relation_anchors: dict[str, list[list[str]]] = {}
    for doc_id, pairs in doc_relations.items():
        row = await rag.full_relations.get_by_id(doc_id)
        anchored = {tuple(p) for p in ((row or {}).get("relation_pairs") or [])}
        missing = sorted(pairs - anchored)
        if missing:
            missing_relation_anchors[doc_id] = [list(p) for p in missing]

    # Documents that own nothing in the graph AND carry no anchor rows. The
    # completed scan above is what makes this a positive statement rather than
    # an absence of evidence: every graph object was visited and none was
    # attributed to these documents.
    anchorless_docs = await _find_anchorless_docs(
        rag, doc_entities, doc_relations, batch_size
    )

    repaired_docs: list[str] = []
    if apply and (missing_entity_anchors or missing_relation_anchors):
        for doc_id in sorted(
            set(missing_entity_anchors) | set(missing_relation_anchors)
        ):
            await rag._union_doc_recovery_anchors(
                doc_id,
                missing_entity_anchors.get(doc_id, []),
                [tuple(p) for p in missing_relation_anchors.get(doc_id, [])],
            )
            repaired_docs.append(doc_id)
        logger.info(
            f"[kg-integrity] Repaired recovery anchors for {len(repaired_docs)} document(s)"
        )

    if apply and anchorless_docs:
        # Empty rows, written through the same union helper: it upserts both
        # namespaces, so a document with no contributions ends up with the
        # present-and-empty pair that IS the normal proof for one.
        for doc_id in anchorless_docs:
            await rag._union_doc_recovery_anchors(doc_id, [], [])
            repaired_docs.append(doc_id)
        logger.info(
            f"[kg-integrity] Wrote empty recovery anchors for "
            f"{len(anchorless_docs)} document(s) with no graph contributions"
        )

    return {
        "entities_total": len(node_sources),
        "relations_total": len(edge_sources),
        "missing_entity_anchors": missing_entity_anchors,
        "missing_relation_anchors": missing_relation_anchors,
        "orphan_entities": sorted(orphan_entities),
        "orphan_relations": sorted(orphan_relations),
        "anchorless_docs": anchorless_docs,
        "repaired_docs": sorted(set(repaired_docs)),
    }


async def _find_anchorless_docs(
    rag,
    doc_entities: dict[str, set[str]],
    doc_relations: dict[str, set[tuple[str, str]]],
    batch_size: int,
) -> list[str]:
    """Documents proven to own nothing in the graph and holding no anchor rows.

    Only documents in a TERMINAL state are considered. A document still moving
    through the pipeline may be about to write its anchors, and while writing
    empty rows for it would be harmless in itself (merge Phase 0 overwrites
    them unconditionally), reporting it as anchorless would be misleading — it
    is unfinished, not empty.

    The doc_status enumeration is strict (complete-or-raise) and failures
    propagate. This function's answer is a PROOF of absence, and a
    best-effort read that silently dropped rows would narrow it: every
    dropped row is a document the audit was asked to certify and quietly
    did not. An empty result must mean "no such documents", never "the scan
    did not run" — so the audit fails loudly rather than reporting a
    certainty it does not have.
    """
    rows = await rag.doc_status.get_docs_by_statuses(
        [DocStatus.PROCESSED, DocStatus.FAILED], strict=True
    )

    candidates = [
        doc_id
        for doc_id in sorted(rows)
        if doc_id not in doc_entities and doc_id not in doc_relations
    ]

    anchorless: list[str] = []
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        entity_rows = await rag.full_entities.get_by_ids(batch)
        relation_rows = await rag.full_relations.get_by_ids(batch)
        for doc_id, entities_row, relations_row in zip(
            batch, entity_rows, relation_rows
        ):
            # Row PRESENCE is the test, matching the purge contract: a present
            # but empty row already is a valid proof and needs no repair.
            if isinstance(entities_row, dict) and isinstance(relations_row, dict):
                continue
            anchorless.append(doc_id)
    return anchorless


def _print_report(report: dict[str, Any], verbose: bool) -> None:
    print(
        f"Graph: {report['entities_total']} entities, "
        f"{report['relations_total']} relations"
    )
    missing_docs = set(report["missing_entity_anchors"]) | set(
        report["missing_relation_anchors"]
    )
    print(f"Documents with anchor gaps: {len(missing_docs)}")
    print(
        f"Orphans (no resolvable source chunk): "
        f"{len(report['orphan_entities'])} entities, "
        f"{len(report['orphan_relations'])} relations"
    )
    if report["repaired_docs"]:
        print(f"Repaired anchors for: {', '.join(report['repaired_docs'])}")
    if verbose:
        for doc_id, names in report["missing_entity_anchors"].items():
            print(f"  {doc_id}: missing entity anchors: {names}")
        for doc_id, pairs in report["missing_relation_anchors"].items():
            print(f"  {doc_id}: missing relation anchors: {pairs}")
        for name in report["orphan_entities"]:
            print(f"  orphan entity: {name}")
        for pair in report["orphan_relations"]:
            print(f"  orphan relation: {pair}")


async def _async_main(apply: bool, verbose: bool) -> bool:
    import numpy as np

    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    async def _noop_llm(*args, **kwargs) -> str:
        raise RuntimeError("kg_integrity_repair never calls the LLM")

    async def _noop_embed(texts: list[str]) -> np.ndarray:
        raise RuntimeError("kg_integrity_repair never embeds")

    rag = LightRAG(
        working_dir=os.getenv("WORKING_DIR", "./rag_storage"),
        workspace=os.getenv("WORKSPACE", ""),
        llm_model_func=_noop_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
            max_token_size=8192,
            func=_noop_embed,
        ),
    )
    await rag.initialize_storages()
    try:
        report = await audit_kg_integrity(rag, apply=apply)
        _print_report(report, verbose)
        return True
    finally:
        await rag.finalize_storages()


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=".env", override=False)
    parser = argparse.ArgumentParser(
        description="Audit (and optionally repair) LightRAG per-document "
        "recovery anchors (full_entities / full_relations)."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Union missing contributions into the anchor rows (default: report only)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-document details"
    )
    args = parser.parse_args()
    ok = asyncio.run(_async_main(apply=args.apply, verbose=args.verbose))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
