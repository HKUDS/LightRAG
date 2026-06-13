"""Reranker integration for FrameRAG.

Wraps LightRAG's generic_rerank_api with a clean factory pattern.
Supports Jina, Cohere, BGE-reranker (via OpenAI-compatible endpoint),
and any other API that follows the standard rerank format.

Usage:
    from framerag.rerank import make_reranker

    reranker = make_reranker(
        model="BAAI/bge-reranker-v2-m3",
        base_url="http://localhost:8000/rerank",
    )
    rag = FrameRAG(..., rerank_func=reranker, rerank_top_k=50)
"""
from __future__ import annotations

import os
from typing import Callable, Awaitable, Optional

from lightrag.rerank import generic_rerank_api
from lightrag.utils import logger

# Type alias: (query, documents, top_n) → [{"index": int, "relevance_score": float}]
RerankFunc = Callable[[str, list[str], int], Awaitable[list[dict]]]


def make_reranker(
    model: str,
    base_url: str,
    api_key: Optional[str] = None,
    response_format: str = "standard",
    request_format: str = "standard",
    enable_chunking: bool = True,
    max_tokens_per_doc: int = 480,
) -> RerankFunc:
    """Return an async rerank callable for use with FrameRAG.

    Args:
        model:            Model name, e.g. "BAAI/bge-reranker-v2-m3", "jina-reranker-v2-base-multilingual"
        base_url:         Reranker API endpoint, e.g. "https://api.jina.ai/v1/rerank"
        api_key:          API key; falls back to RERANK_API_KEY env var if None
        response_format:  "standard" (Jina/Cohere/BGE) or "aliyun"
        request_format:   "standard" or "aliyun"
        enable_chunking:  Chunk long documents before reranking (recommended)
        max_tokens_per_doc: Max tokens per document chunk
    """
    _api_key = api_key or os.environ.get("RERANK_API_KEY")

    async def _rerank(query: str, documents: list[str], top_n: int) -> list[dict]:
        if not documents:
            return []
        try:
            return await generic_rerank_api(
                query=query,
                documents=documents,
                model=model,
                base_url=base_url,
                api_key=_api_key,
                top_n=top_n,
                response_format=response_format,
                request_format=request_format,
                enable_chunking=enable_chunking,
                max_tokens_per_doc=max_tokens_per_doc,
            )
        except Exception as e:
            logger.warning(f"[FrameRAG] Reranker failed, using diffusion order: {e}")
            return []

    return _rerank


async def rerank_chunk_hits(
    query: str,
    chunk_hits: list[dict],
    chunk_texts: dict[str, str],
    rerank_func: RerankFunc,
    top_n: int,
) -> list[dict]:
    """Re-score chunk_hits with reranker and return top_n reordered.

    Args:
        query:       The user query.
        chunk_hits:  List of {"id": chunk_id, "score": float} from diffusion.
        chunk_texts: Mapping chunk_id → raw text (fetched from storage).
        rerank_func: Callable returned by make_reranker().
        top_n:       How many chunks to keep after reranking.

    Returns:
        Reranked chunk_hits (may be shorter than input if reranker returns fewer).
        Falls back to original diffusion order on failure.
    """
    if not chunk_hits or not rerank_func:
        return chunk_hits[:top_n]

    # Build parallel list: docs[i] corresponds to chunk_hits[i]
    docs: list[str] = []
    valid_hits: list[dict] = []
    for hit in chunk_hits:
        text = chunk_texts.get(hit["id"], "")
        if text:
            docs.append(text)
            valid_hits.append(hit)

    if not docs:
        return chunk_hits[:top_n]

    try:
        results = await rerank_func(query, docs, top_n)
    except Exception as e:
        logger.warning(f"[FrameRAG] rerank_chunk_hits error, using diffusion order: {e}")
        results = []

    if not results:
        # Reranker failed or empty — fall back to diffusion order
        return valid_hits[:top_n]

    reranked: list[dict] = []
    for r in results:
        idx = r.get("index", -1)
        score = r.get("relevance_score", 0.0)
        if 0 <= idx < len(valid_hits):
            hit = dict(valid_hits[idx])
            hit["rerank_score"] = score
            reranked.append(hit)

    return reranked[:top_n]
