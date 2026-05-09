"""Semantic vector chunking — the ``"V"`` strategy.

Wraps LangChain's :class:`SemanticChunker` (from ``langchain-experimental``)
which splits text by sentence embeddings: it first segments the input into
sentences, embeds each sentence (in adjacent windows of ``buffer_size``),
and finds breakpoints where the cosine distance between consecutive
windows crosses a threshold derived from the chosen distribution
(``percentile`` / ``standard_deviation`` / ``interquartile`` /
``gradient``).

The chunker exposed here is ``async`` because LightRAG's
:class:`EmbeddingFunc` is async.  Internally we call SemanticChunker
synchronously inside :func:`asyncio.to_thread` and bridge the embedding
calls back to the main event loop via
:func:`asyncio.run_coroutine_threadsafe`.

Caveats:
  - SemanticChunker does NOT enforce a maximum chunk size; the caller's
    ``chunk_token_size`` is *advisory* here.  Oversized chunks will be
    hard-split before embedding by
    :func:`lightrag.utils.enforce_chunk_token_limit_before_embedding`.
  - When ``embedding_func`` is ``None`` we log a warning and fall back to
    :func:`lightrag.chunker.chunking_by_recursive_character` — V's only
    differentiator is embeddings, and R is the closest structural-only
    alternative.
"""

from __future__ import annotations

import asyncio
from typing import Any

from lightrag.utils import EmbeddingFunc, Tokenizer, logger

try:
    from langchain_core.embeddings import Embeddings
    from langchain_experimental.text_splitter import SemanticChunker

    _LANGCHAIN_EXPERIMENTAL_AVAILABLE = True
except ImportError:
    _LANGCHAIN_EXPERIMENTAL_AVAILABLE = False
    Embeddings = object  # type: ignore[assignment,misc]
    SemanticChunker = None  # type: ignore[assignment]


class _AsyncEmbeddingFuncAdapter(Embeddings):
    """Bridge a LightRAG :class:`EmbeddingFunc` (async) to LangChain's
    sync :class:`Embeddings` interface used by ``SemanticChunker``.

    The adapter must be constructed inside the running event loop so it
    can capture the loop reference; the blocking ``embed_documents`` /
    ``embed_query`` calls are then made from a worker thread (via
    :func:`asyncio.to_thread` in the public chunker) and bounce back to
    the captured loop with :func:`asyncio.run_coroutine_threadsafe`.
    """

    def __init__(
        self,
        embedding_func: EmbeddingFunc,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._embedding_func = embedding_func
        self._loop = loop

    def _run(self, texts: list[str], context: str) -> list[list[float]]:
        future = asyncio.run_coroutine_threadsafe(
            self._embedding_func(texts, context=context),
            self._loop,
        )
        result = future.result()
        return [list(map(float, vec)) for vec in result]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._run(list(texts), context="document")

    def embed_query(self, text: str) -> list[float]:
        return self._run([text], context="query")[0]


async def chunking_by_semantic_vector(
    tokenizer: Tokenizer,
    content: str,
    chunk_token_size: int = 1200,
    *,
    embedding_func: EmbeddingFunc | None = None,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float | None = None,
    buffer_size: int = 1,
) -> list[dict[str, Any]]:
    """Semantic vector chunker — the ``"V"`` chunking strategy.

    Args:
        tokenizer: LightRAG tokenizer (used for output token counts).
        content: Text to split.
        chunk_token_size: Advisory cap (tokens). SemanticChunker does NOT
            enforce a maximum; oversized chunks are hard-split before
            embedding by ``enforce_chunk_token_limit_before_embedding``.
        embedding_func: LightRAG :class:`EmbeddingFunc`. When ``None``
            this chunker logs a warning and falls back to
            :func:`chunking_by_recursive_character`.
        breakpoint_threshold_type: ``percentile`` | ``standard_deviation``
            | ``interquartile`` | ``gradient`` (LangChain default:
            ``percentile``).
        breakpoint_threshold_amount: Threshold magnitude. ``None`` lets
            LangChain pick the per-type default (e.g. 95 for percentile).
        buffer_size: Number of adjacent sentences combined when computing
            distances (LangChain default: 1).

    Returns:
        Ordered list of ``{"tokens", "content", "chunk_order_index"}``
        dicts.
    """
    if not content or not content.strip():
        return []

    if embedding_func is None:
        # V's only differentiator is embeddings — without them the
        # closest neighbour is R's structural splitting.  V chunks are
        # non-overlapping by design (semantic boundaries), so the
        # fallback uses ``chunk_overlap_token_size=0`` to preserve that
        # semantic and avoid LangChain's "overlap > chunk_size" guard
        # for very small ``chunk_token_size``.
        logger.warning(
            "[semantic_vector] embedding_func is None; falling back to "
            "recursive-character chunking."
        )
        from lightrag.chunker.recursive_character import (
            chunking_by_recursive_character,
        )

        return chunking_by_recursive_character(
            tokenizer,
            content,
            chunk_token_size,
            chunk_overlap_token_size=0,
        )

    if not _LANGCHAIN_EXPERIMENTAL_AVAILABLE:
        raise ImportError(
            "langchain-experimental is required for the 'V' chunking "
            "strategy; install with `pip install langchain-experimental>=0.3`."
        )

    loop = asyncio.get_running_loop()
    adapter = _AsyncEmbeddingFuncAdapter(embedding_func, loop)

    chunker_kwargs: dict[str, Any] = {
        "embeddings": adapter,
        "buffer_size": int(buffer_size),
        "breakpoint_threshold_type": breakpoint_threshold_type,
    }
    if breakpoint_threshold_amount is not None:
        chunker_kwargs["breakpoint_threshold_amount"] = float(
            breakpoint_threshold_amount
        )

    splitter = SemanticChunker(**chunker_kwargs)
    pieces = await asyncio.to_thread(splitter.split_text, content)

    results: list[dict[str, Any]] = []
    for piece in pieces:
        body = piece.strip()
        if not body:
            continue
        results.append(
            {
                "tokens": len(tokenizer.encode(body)),
                "content": body,
                "chunk_order_index": len(results),
            }
        )
    return results
