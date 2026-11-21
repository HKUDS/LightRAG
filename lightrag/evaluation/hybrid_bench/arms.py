"""Retrieval arms: dense, hybrid, lexical-only, and the ablation dispatch.

Rules:
- Dense and hybrid chunk arms go through the REAL retrieval path,
  ``operate._get_vector_context``, with ``QueryParam.enable_hybrid`` selecting
  the dispatch; the chunk lexical leg stays whole-query AND. Chunk arms share
  one precomputed whole-query embedding and the same top_k.
- Entity seeding is PRODUCTION-FAITHFUL (``ENTITY_SEEDING_PROTOCOL``): one
  keyword-extraction pass per query through the product's
  ``operate.get_keywords_from_query`` (the same call ``kg_query`` makes when
  keywords are not injected), cached within the run. Both entity arms query
  with the joined keyword string and ITS embedding, exactly as
  ``_perform_kg_search`` / ``_get_node_data`` do; the hybrid arm additionally
  passes ``lexical_keywords=keywords_str.split(", ")`` so each keyword seeds
  entities on its own (OR across keywords, the entity-side hybrid fix).
  Omitting ``lexical_keywords`` would measure the legacy whole-query AND path
  and is a protocol defect, not a simplification.
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
from lightrag.operate import _get_vector_context, get_keywords_from_query, kg_query
from lightrag.utils import logger

ARMS = ("dense", "hybrid", "lexical")
#: Report annotation naming the entity-seeding measurement protocol.
ENTITY_SEEDING_PROTOCOL = "production-faithful-multi-keyword"
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


@dataclass(frozen=True, slots=True)
class EntitySeeding:
    """One query's production-style entity search inputs.

    ``keywords_str`` is the joined keyword list (the entity query string in
    ``_perform_kg_search``); ``embedding`` embeds THAT string, not the raw
    query. ``fallback`` names a documented deviation from plain low-level
    seeding: ``"high_level"`` (ll empty, hl stands in, the product's
    ``entities_by_high_level_keywords`` fallback) or ``"zero_keywords"``
    (both lists empty; the query itself becomes the single keyword).
    """

    keywords_str: str
    embedding: list[float]
    fallback: str


async def prepare_entity_seeding(
    rag: LightRAG, query: str, cache: dict[str, EntitySeeding]
) -> EntitySeeding:
    """Extract keywords once per query and form the entity search inputs.

    Rules:
    - Extraction goes through the PRODUCT mechanism ``kg_query`` uses when
      keywords are not injected (``get_keywords_from_query`` with the run's
      configured keyword-role LLM). Provider errors propagate: a failed
      extraction must never silently degrade to whole-query seeding.
    - ``cache`` keys on the query text and lives for the whole run: the bench
      disables the LLM cache, so this dict is what guarantees one extraction
      pass per query shared by every arm.
    - Empty low-level keywords with non-empty high-level ones use the hl list
      (the product's mix-mode entity-store fallback). Both lists empty falls
      back to the query as ONE keyword: production refuses such queries over
      50 chars, but the benchmark must score every query, so the fallback is
      applied at any length, logged, and annotated in the report. A single
      keyword is byte-identical to whole-query AND seeding.
    """
    cached = cache.get(query)
    if cached is not None:
        return cached
    hl_keywords, ll_keywords = await get_keywords_from_query(
        query,
        QueryParam(mode="mix"),
        rag._build_global_config(),
        hashing_kv=rag.llm_response_cache,
    )
    fallback = ""
    keywords = ll_keywords
    if not keywords and hl_keywords:
        keywords = hl_keywords
        fallback = "high_level"
    elif not keywords:
        keywords = [query]
        fallback = "zero_keywords"
    if fallback:
        logger.warning(
            "[hybrid-bench] entity seeding fallback %s for query %r",
            fallback,
            query,
        )
    keywords_str = ", ".join(keywords)
    seeding = EntitySeeding(
        keywords_str=keywords_str,
        embedding=await embed_query(rag, keywords_str),
        fallback=fallback,
    )
    cache[query] = seeding
    return seeding


async def retrieve_entities(
    rag: LightRAG,
    seeding: EntitySeeding,
    hybrid: bool,
    top_k: int,
) -> list[str]:
    """Entity names in seeding order, dense or hybrid.

    Mirrors ``operate._get_node_data``: the query string is the joined
    keyword list, the dense leg embeds that same string, and the hybrid
    lexical leg receives ``lexical_keywords`` recovered by the same
    ``split(", ")`` so each keyword seeds entities on its own (OR across
    keywords). Never drop ``lexical_keywords``: the storage default (None)
    is the legacy whole-query AND path the entity-side hybrid fix replaced.
    """
    vdb = rag.entities_vdb
    if hybrid and getattr(vdb, "supports_hybrid_query", False):
        rows = await vdb.hybrid_query(
            seeding.keywords_str,
            top_k=top_k,
            query_embedding=seeding.embedding,
            lexical_keywords=seeding.keywords_str.split(", "),
        )
    else:
        rows = await vdb.query(
            seeding.keywords_str, top_k=top_k, query_embedding=seeding.embedding
        )
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
