"""Retrieval arms: dense, hybrid, lexical-only, and the ablation dispatch.

Rules:
- Dense and hybrid chunk arms go through the REAL retrieval path,
  ``operate._get_vector_context``, with ``QueryParam.enable_hybrid`` selecting
  the dispatch; entity seeding calls ``entities_vdb.query`` /
  ``entities_vdb.hybrid_query`` directly. All arms share one precomputed query
  embedding and the same top_k.
- The lexical-only arm is diagnostic: it executes the SAME lexical SQL
  template ``hybrid_query`` uses, read from ``postgres_impl.SQL_TEMPLATES``
  with the same tokenizer, against the same workspace. PG only; on file
  backends the caller reports it as skipped, never faked.
- Every retrieved chunk must resolve to a corpus id through the ingest
  attribution map; an unmapped chunk is a loud error, not a silent drop.
- The ablation is storage-level dispatch: entity-only = entities hybrid +
  chunks dense; chunk-only = chunks hybrid + entities dense; both = both
  hybrid. No product code changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from lightrag import LightRAG
from lightrag.base import QueryParam
from lightrag.kg.postgres_impl import SQL_TEMPLATES
from lightrag.kg.lexical_tokenizer import tokenize_for_query
from lightrag.operate import _get_vector_context

ARMS = ("dense", "hybrid", "lexical")
B5_COMBINATIONS = {
    "entity_only": (True, False),
    "chunk_only": (False, True),
    "both": (True, True),
}


@dataclass(frozen=True, slots=True)
class ChunkHit:
    chunk_id: str
    doc_id: str
    content: str


def _param(enable_hybrid: bool, top_k: int) -> QueryParam:
    return QueryParam(
        mode="naive",
        only_need_context=True,
        enable_rerank=False,
        enable_hybrid=enable_hybrid,
        stream=False,
        top_k=top_k,
        chunk_top_k=top_k,
    )


async def embed_query(rag: LightRAG, query: str) -> list[float]:
    embeddings = await rag.embedding_func([query], context="query")
    return embeddings[0].tolist()


def _resolve_hits(rows: list[dict], chunk_doc_map: dict[str, str]) -> list[ChunkHit]:
    hits = []
    for row in rows:
        chunk_id = row.get("chunk_id") or row.get("id")
        if chunk_id not in chunk_doc_map:
            raise RuntimeError(
                f"retrieved chunk {chunk_id!r} has no corpus attribution; "
                "the ingest map is incomplete"
            )
        hits.append(
            ChunkHit(
                chunk_id=chunk_id,
                doc_id=chunk_doc_map[chunk_id],
                content=row.get("content", ""),
            )
        )
    return hits


async def retrieve_chunks(
    rag: LightRAG,
    query: str,
    arm: str,
    top_k: int,
    query_embedding: list[float],
    chunk_doc_map: dict[str, str],
) -> list[ChunkHit]:
    """Ordered chunk hits for one arm; order is the retrieval order."""
    if arm in ("dense", "hybrid"):
        rows = await _get_vector_context(
            query,
            rag.chunks_vdb,
            _param(arm == "hybrid", top_k),
            query_embedding=query_embedding,
        )
        return _resolve_hits(rows, chunk_doc_map)
    if arm == "lexical":
        if rag.chunks_vdb.__class__.__name__ != "PGVectorStorage":
            raise ValueError(
                "the lexical-only arm requires PGVectorStorage; report it "
                "as skipped on file backends"
            )
        tsquery = tokenize_for_query(query)
        if not tsquery:
            return []
        rows = await rag.chunks_vdb.db.query(
            SQL_TEMPLATES["lexical_chunks"].format(
                table_name=rag.chunks_vdb.table_name
            ),
            [rag.chunks_vdb.workspace, tsquery, top_k],
            multirows=True,
        )
        return _resolve_hits(rows or [], chunk_doc_map)
    raise ValueError(f"unknown chunk arm {arm!r}")


async def retrieve_entities(
    rag: LightRAG,
    query: str,
    hybrid: bool,
    top_k: int,
    query_embedding: list[float],
) -> list[str]:
    """Entity names in seeding order, dense or hybrid."""
    vdb = rag.entities_vdb
    if hybrid and getattr(vdb, "supports_hybrid_query", False):
        rows = await vdb.hybrid_query(
            query, top_k=top_k, query_embedding=query_embedding
        )
    else:
        rows = await vdb.query(query, top_k=top_k, query_embedding=query_embedding)
    return [row["entity_name"] for row in rows]


def doc_id_sequence(hits: list[ChunkHit]) -> list[str]:
    """One document id per chunk rank; metrics dedup at first occurrence."""
    return [hit.doc_id for hit in hits]
