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
import re
from typing import Any

from lightrag.constants import DEFAULT_SENTENCE_SPLIT_REGEX
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


def _sentence_spans(text: str, sentences: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for sentence in sentences:
        if not sentence:
            spans.append((cursor, cursor))
            continue
        start = text.find(sentence, cursor)
        if start < 0:
            start = text.find(sentence)
        if start < 0:
            start = cursor
        end = start + len(sentence)
        spans.append((start, end))
        cursor = end
    return spans


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    start = max(0, min(start, len(text)))
    end = max(start, min(end, len(text)))
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _semantic_groups_with_spans(
    splitter: SemanticChunker,
    text: str,
) -> list[tuple[str, int, int]]:
    """Mirror SemanticChunker grouping while keeping original source spans."""
    single_sentences_list = re.split(splitter.sentence_split_regex, text)
    spans = _sentence_spans(text, single_sentences_list)

    def _group(start_index: int, end_index: int) -> tuple[str, int, int] | None:
        start, _ = spans[start_index]
        _, end = spans[end_index]
        start, end = _trim_span(text, start, end)
        if start >= end:
            return None
        return text[start:end], start, end

    if len(single_sentences_list) == 1:
        group = _group(0, 0)
        return [group] if group else []
    if (
        splitter.breakpoint_threshold_type == "gradient"
        and len(single_sentences_list) == 2
    ):
        return [g for i in range(2) if (g := _group(i, i)) is not None]

    distances, sentences = splitter._calculate_sentence_distances(single_sentences_list)
    if splitter.number_of_chunks is not None:
        breakpoint_distance_threshold = splitter._threshold_from_clusters(distances)
        breakpoint_array = distances
    else:
        breakpoint_distance_threshold, breakpoint_array = (
            splitter._calculate_breakpoint_threshold(distances)
        )

    indices_above_thresh = [
        i for i, x in enumerate(breakpoint_array) if x > breakpoint_distance_threshold
    ]

    chunks: list[tuple[str, int, int]] = []
    start_index = 0
    for index in indices_above_thresh:
        end_index = index
        group_sentences = sentences[start_index : end_index + 1]
        combined_text = " ".join([d["sentence"] for d in group_sentences])
        if (
            splitter.min_chunk_size is not None
            and len(combined_text) < splitter.min_chunk_size
        ):
            continue
        group = _group(start_index, end_index)
        if group is not None:
            chunks.append(group)
        start_index = index + 1

    if start_index < len(sentences):
        group = _group(start_index, len(sentences) - 1)
        if group is not None:
            chunks.append(group)
    return chunks


async def chunking_by_semantic_vector(
    tokenizer: Tokenizer,
    content: str,
    chunk_token_size: int = 1200,
    *,
    embedding_func: EmbeddingFunc | None = None,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float | None = None,
    buffer_size: int = 1,
    sentence_split_regex: str = DEFAULT_SENTENCE_SPLIT_REGEX,
    number_of_chunks: int | None = None,
    min_chunk_size: int | None = None,
) -> list[dict[str, Any]]:
    """Semantic vector chunker — the ``"V"`` chunking strategy.

    Args:
        tokenizer: LightRAG tokenizer (used for output token counts).
        content: Text to split.
        chunk_token_size: Hard upper bound (tokens). SemanticChunker does
            NOT enforce a maximum natively, so any piece that exceeds
            this value is re-split via
            :func:`chunking_by_recursive_character` before being emitted.
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
        sentence_split_regex: Pattern fed to LangChain's
            :class:`SemanticChunker` for the initial sentence split.
            Default extends the upstream English-only pattern with
            Chinese sentence terminators ``。？！`` so mixed-language and
            pure-Chinese inputs split correctly.
        number_of_chunks: Optional target chunk count (LangChain SemanticChunker).
        min_chunk_size: Optional minimum character size for semantic groups.

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
        "sentence_split_regex": sentence_split_regex,
        "number_of_chunks": number_of_chunks,
        "min_chunk_size": min_chunk_size,
    }
    if breakpoint_threshold_amount is not None:
        chunker_kwargs["breakpoint_threshold_amount"] = float(
            breakpoint_threshold_amount
        )

    splitter = SemanticChunker(**chunker_kwargs)
    pieces = await asyncio.to_thread(_semantic_groups_with_spans, splitter, content)

    # SemanticChunker has no internal size cap; oversized pieces here
    # would otherwise rely on the embedding-time hard fallback (which
    # uses ``embedding_token_limit``, not ``chunk_token_size``) to split
    # them.  Enforce ``chunk_token_size`` directly via R for any piece
    # that exceeds it so the user-configured size is actually honored.
    # Lazy import dodges the recursive_character ↔ semantic_vector
    # circular dependency (same pattern as the embedding-None fallback
    # above).
    from lightrag.chunker.recursive_character import (
        chunking_by_recursive_character,
    )

    target_max = max(int(chunk_token_size), 1)
    results: list[dict[str, Any]] = []
    for piece, source_start, source_end in pieces:
        body = piece.strip()
        if not body:
            continue
        piece_tokens = len(tokenizer.encode(body))
        if piece_tokens <= target_max:
            results.append(
                {
                    "tokens": piece_tokens,
                    "content": body,
                    "chunk_order_index": len(results),
                    "_source_span": {
                        "start": source_start,
                        "end": source_end,
                    },
                }
            )
            continue
        # Oversized semantic piece: re-split via R while preserving the
        # surrounding chunk order.  ``chunk_overlap_token_size=0`` keeps
        # V's non-overlapping semantics.
        sub_pieces = chunking_by_recursive_character(
            tokenizer,
            body,
            target_max,
            chunk_overlap_token_size=0,
        )
        for sub in sub_pieces:
            sub_body = sub.get("content", "")
            if not sub_body:
                continue
            sub_span = sub.get("_source_span")
            source_span = None
            if isinstance(sub_span, dict):
                try:
                    source_span = {
                        "start": source_start + int(sub_span["start"]),
                        "end": source_start + int(sub_span["end"]),
                    }
                except (KeyError, TypeError, ValueError):
                    source_span = None
            results.append(
                {
                    "tokens": sub.get("tokens", len(tokenizer.encode(sub_body))),
                    "content": sub_body,
                    "chunk_order_index": len(results),
                    **({"_source_span": source_span} if source_span else {}),
                }
            )
    return results
