"""Derived entity-seeding ground truth, no human annotation.

Derivation rule, quoted verbatim in every report:
an entity is relevant to a query iff the set of ``full_doc_id`` values of its
source chunks intersects the query's qrels-positive corpus ids. Source chunks
come from the ``entity_chunks`` tracking row (the FULL attribution list);
the graph node's ``source_id`` field split on ``GRAPH_FIELD_SEP`` is the
fallback ONLY when no tracking row exists, because the graph field is capped
and would drop truth for entities appearing in more passages than the cap.
Each chunk id resolves to its owning document through the ``text_chunks`` KV
store. Passage ids equal corpus ids because ingest uses ``ids=corpus_id``.

Rules:
- Queries whose derived truth set is empty are returned separately; chunk
  metrics still apply to them, entity hit@k never averages them in.
- A present tracking row is authoritative even when its chunk list is empty;
  an entity with no source chunks contributes no documents and can never be
  relevant under this rule.
- The function takes the graph and KV storages directly so a tiny fake KG can
  pin the mapping offline. ``entity_chunks=None`` means no tracking store:
  every entity falls back to the graph field.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from lightrag.base import BaseGraphStorage, BaseKVStorage
from lightrag.constants import GRAPH_FIELD_SEP


@dataclass(frozen=True, slots=True)
class EntityTruth:
    per_query: dict[str, set[str]]
    zero_truth_query_ids: list[str]
    entity_documents: dict[str, set[str]]


async def derive_entity_truth(
    graph: BaseGraphStorage,
    text_chunks: BaseKVStorage,
    positives_by_query: Mapping[str, Mapping[str, int]],
    entity_chunks: BaseKVStorage | None = None,
) -> EntityTruth:
    """Map every entity to its source documents, then intersect per query."""
    labels = await graph.get_all_labels()
    nodes = await graph.get_nodes_batch(labels) if labels else {}
    tracking_rows = (
        await entity_chunks.get_by_ids(labels)
        if entity_chunks is not None and labels
        else [None] * len(labels)
    )

    chunk_ids: set[str] = set()
    node_chunks: dict[str, list[str]] = {}
    for label, tracking in zip(labels, tracking_rows):
        if isinstance(tracking, dict):
            ids = [cid for cid in tracking.get("chunk_ids", []) if cid]
        else:
            node = nodes.get(label) or {}
            source_id = node.get("source_id") or ""
            ids = [part for part in source_id.split(GRAPH_FIELD_SEP) if part]
        node_chunks[label] = ids
        chunk_ids.update(ids)

    ordered_ids = sorted(chunk_ids)
    chunk_rows = await text_chunks.get_by_ids(ordered_ids) if ordered_ids else []
    # get_by_ids preserves the requested order and yields None for a missing
    # id on every backend, so position is the chunk identity.
    chunk_doc: dict[str, str] = {}
    for chunk_key, row in zip(ordered_ids, chunk_rows):
        if row is None:
            continue
        doc_id = row.get("full_doc_id")
        if isinstance(doc_id, str):
            chunk_doc[chunk_key] = doc_id

    entity_documents: dict[str, set[str]] = {}
    for label, ids in node_chunks.items():
        docs = {chunk_doc[cid] for cid in ids if cid in chunk_doc}
        if docs:
            entity_documents[label] = docs

    per_query: dict[str, set[str]] = {}
    zero_truth: list[str] = []
    for query_id, positives in positives_by_query.items():
        positive_docs = set(positives)
        truth = {
            entity for entity, docs in entity_documents.items() if docs & positive_docs
        }
        if truth:
            per_query[query_id] = truth
        else:
            zero_truth.append(query_id)
    return EntityTruth(
        per_query=per_query,
        zero_truth_query_ids=zero_truth,
        entity_documents=entity_documents,
    )
