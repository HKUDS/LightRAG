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
  backends the caller reports it as skipped, never faked. The postgres_impl
  import is lazy so a file-backend run never loads the PG driver stack.
- Every retrieved chunk must resolve to corpus ids through the ingest
  attribution map; an unmapped chunk is a loud error, not a silent drop.
  Chunk ids are content hashes, so one chunk can carry several corpus
  aliases (duplicate-content documents); a hit keeps ALL of them and scoring
  expands them in sorted order at the chunk's rank position.
- The ablation runs the COMBINED mix-mode retrieval path (``operate.kg_query``,
  the same call ``aquery_data`` makes) per combination, with the two storage
  dispatches controlled independently through instance-level capability
  overrides: entity-only = entities hybrid + chunks dense; chunk-only =
  chunks hybrid + entities dense; both = both hybrid. The entity leg can
  therefore change chunk recall, and scoring uses the FINAL context chunks
  and entities, not two independent searches. No product code changes.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

from lightrag import LightRAG
from lightrag.base import QueryParam
from lightrag.kg.lexical_tokenizer import tokenize_for_query
from lightrag.operate import _get_vector_context, kg_query

ARMS = ("dense", "hybrid", "lexical")
B5_COMBINATIONS = {
    "entity_only": (True, False),
    "chunk_only": (False, True),
    "both": (True, True),
}


@dataclass(frozen=True, slots=True)
class ChunkHit:
    chunk_id: str
    doc_ids: tuple[str, ...]
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


def _resolve_hits(
    rows: list[dict], chunk_doc_map: dict[str, list[str]]
) -> list[ChunkHit]:
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
                doc_ids=tuple(sorted(chunk_doc_map[chunk_id])),
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
    chunk_doc_map: dict[str, list[str]],
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
        from lightrag.kg.postgres_impl import SQL_TEMPLATES

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
    """All corpus aliases per chunk rank, sorted; metrics dedup at first
    occurrence. Duplicate-content aliases occupy consecutive ranks."""
    return [doc_id for hit in hits for doc_id in hit.doc_ids]


@contextmanager
def _hybrid_dispatch_overrides(rag: LightRAG, entity_hybrid: bool, chunk_hybrid: bool):
    """Disable hybrid dispatch per storage for one combined retrieval.

    Only ever DISABLES a capability the storage class declares; it never
    enables hybrid on a backend that lacks ``hybrid_query``. The instance
    attribute shadows the class flag for the duration and is removed on
    exit, leaving the class (and every other instance) untouched.
    """
    disabled = []
    for vdb, wanted in (
        (rag.entities_vdb, entity_hybrid),
        (rag.chunks_vdb, chunk_hybrid),
    ):
        if getattr(type(vdb), "supports_hybrid_query", False) and not wanted:
            vdb.supports_hybrid_query = False
            disabled.append(vdb)
    try:
        yield
    finally:
        for vdb in disabled:
            del vdb.supports_hybrid_query


async def retrieve_combined(
    rag: LightRAG,
    query: str,
    entity_hybrid: bool,
    chunk_hybrid: bool,
    top_k: int,
    chunk_doc_map: dict[str, list[str]],
) -> tuple[list[ChunkHit], list[str]]:
    """One ablation combination through the COMBINED mix-mode path.

    Drives ``operate.kg_query`` exactly as ``aquery_data`` does (keywords
    injected, context-only, rerank off), with the entity/chunk dispatches
    controlled independently, and returns the FINAL context chunks and
    entity names for scoring. ``rag._build_global_config`` is the same
    private accessor ``aquery_data`` uses; the public query API copies
    QueryParam and would drop fields this protocol pins.
    """
    param = QueryParam(
        mode="mix",
        only_need_context=True,
        enable_rerank=False,
        enable_hybrid=True,
        stream=False,
        top_k=top_k,
        chunk_top_k=top_k,
        hl_keywords=[query],
        ll_keywords=[query],
    )
    with _hybrid_dispatch_overrides(rag, entity_hybrid, chunk_hybrid):
        result = await kg_query(
            query,
            rag.chunk_entity_relation_graph,
            rag.entities_vdb,
            rag.relationships_vdb,
            rag.text_chunks,
            param,
            rag._build_global_config(),
            hashing_kv=rag.llm_response_cache,
            chunks_vdb=rag.chunks_vdb,
        )
    data: dict = {}
    if result is not None and result.raw_data:
        data = result.raw_data.get("data") or {}
    hits = _resolve_hits(data.get("chunks", []), chunk_doc_map)
    entities = [row["entity_name"] for row in data.get("entities", [])]
    return hits, entities
