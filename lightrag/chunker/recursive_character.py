"""Recursive character chunking — the ``"R"`` strategy.

Wraps LangChain's :class:`RecursiveCharacterTextSplitter` and delivers
output rows in the LightRAG file-chunker schema. The splitter walks the
``separators`` list from longest semantic boundary (``\\n\\n`` by default)
to weakest (the empty string), recursively re-splitting any segment that
still exceeds the token cap.

Token accounting goes through the LightRAG :class:`Tokenizer` via the
``length_function`` plug-in — without that, ``chunk_size`` would be
measured in characters and ``chunk_token_size`` would lose its meaning.

Output cap is *not* enforced internally: oversized segments are produced
when no separator can break them, and
:func:`lightrag.utils.enforce_chunk_token_limit_before_embedding` does the
final hard split before embedding.
"""

from __future__ import annotations

from typing import Any

from lightrag.utils import Tokenizer, logger

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    _LANGCHAIN_TEXT_SPLITTERS_AVAILABLE = True
except ImportError:
    _LANGCHAIN_TEXT_SPLITTERS_AVAILABLE = False
    RecursiveCharacterTextSplitter = None  # type: ignore[assignment]


def _locate_chunk_start(
    content: str,
    body: str,
    tokenizer: Tokenizer,
    *,
    prev_start: int | None,
    prev_body: str | None,
    overlap_tokens: int,
    cursor: int,
) -> int:
    """Return the source-text offset where ``body`` begins, or ``-1``.

    A plain forward :meth:`str.find` is safe for unique text but collapses on
    repeated content: identical overlapping chunks all match the nearest
    repetition, so every span packs into the document head and the tail is left
    without provenance. To avoid that we *predict* this chunk's start from the
    previous one using the token-overlap geometry — ``prev_body`` length minus
    the characters its trailing ``overlap_tokens`` re-share with this chunk —
    and trust the prediction when it lands exactly. Only when it misses (from a
    variable token/char ratio or a stripped separator) do we snap to the
    occurrence nearest the prediction, always staying at or past ``cursor`` (a
    monotonic floor) so spans never move backward.
    """
    if prev_start is None or prev_body is None:
        return content.find(body, cursor)

    prev_tokens = tokenizer.encode(prev_body)
    take = min(len(prev_tokens), overlap_tokens)
    overlap_chars = (
        len(tokenizer.decode(prev_tokens[len(prev_tokens) - take :])) if take else 0
    )
    step = max(1, len(prev_body) - overlap_chars)
    guess = prev_start + step
    if content[guess : guess + len(body)] == body:
        return guess

    # Prediction was off; snap to the occurrence nearest it within a one-chunk
    # window, never before the monotonic floor.
    lo = max(cursor, guess - len(body))
    hi = guess + len(body)
    nearest = -1
    best: int | None = None
    i = content.find(body, lo)
    while i != -1 and i <= hi:
        dist = abs(i - guess)
        if best is None or dist < best:
            best, nearest = dist, i
        i = content.find(body, i + 1)
    if nearest != -1:
        return nearest

    # Nothing near the prediction: fall back to the first match at/after the
    # monotonic floor so a forward-consistent span is still recorded.
    return content.find(body, cursor)


def chunking_by_recursive_character(
    tokenizer: Tokenizer,
    content: str,
    chunk_token_size: int = 1200,
    *,
    chunk_overlap_token_size: int = 100,
    separators: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Recursive character splitter — the ``"R"`` chunking strategy.

    Args:
        tokenizer: LightRAG tokenizer; used as the length function so
            ``chunk_token_size`` and ``chunk_overlap_token_size`` are
            interpreted in tokens, not characters.
        content: Text to split.
        chunk_token_size: Hard target size for each chunk (tokens).
        chunk_overlap_token_size: Token overlap between adjacent chunks.
        separators: Cascade of split candidates. ``None`` defers to
            LangChain's defaults: ``["\\n\\n", "\\n", " ", ""]``.

    Returns:
        Ordered list of ``{"tokens", "content", "chunk_order_index"}``
        dicts.
    """
    if not _LANGCHAIN_TEXT_SPLITTERS_AVAILABLE:
        raise ImportError(
            "langchain-text-splitters is required for the 'R' chunking "
            "strategy; install with `pip install langchain-text-splitters>=0.3`."
        )

    if not content or not content.strip():
        return []

    splitter_kwargs: dict[str, Any] = {
        "chunk_size": max(int(chunk_token_size), 1),
        "chunk_overlap": max(int(chunk_overlap_token_size), 0),
        "length_function": lambda s: len(tokenizer.encode(s)),
        "strip_whitespace": True,
    }
    if separators is not None:
        splitter_kwargs["separators"] = list(separators)

    splitter = RecursiveCharacterTextSplitter(**splitter_kwargs)

    # We deliberately do *not* request LangChain's ``add_start_index``. That
    # offset is computed with a character-vs-token unit mismatch when a
    # token-based ``length_function`` is in play: ``create_documents`` advances
    # its search cursor by ``previous_chunk_len`` (characters) minus
    # ``chunk_overlap`` (tokens), so on overlapping chunks the cursor overshoots
    # each chunk's true start. The result is ``start_index == -1`` for unique
    # text (lost ``_source_span`` → backfill failure under
    # ``require_source_span``) or a match against a later identical run (wrong
    # provenance). Instead we recover spans ourselves via ``_locate_chunk_start``.
    overlap_tokens = max(int(chunk_overlap_token_size), 0)
    docs = splitter.create_documents([content])
    results: list[dict[str, Any]] = []
    prev_start: int | None = None
    prev_body: str | None = None
    cursor = 0
    for doc in docs:
        body = doc.page_content.strip()
        if not body:
            continue
        start_index = _locate_chunk_start(
            content,
            body,
            tokenizer,
            prev_start=prev_start,
            prev_body=prev_body,
            overlap_tokens=overlap_tokens,
            cursor=cursor,
        )
        source_span = None
        if start_index >= 0:
            source_span = {"start": start_index, "end": start_index + len(body)}
            prev_start, prev_body = start_index, body
            cursor = start_index + 1
        results.append(
            {
                "tokens": len(tokenizer.encode(body)),
                "content": body,
                "chunk_order_index": len(results),
                **({"_source_span": source_span} if source_span else {}),
            }
        )

    if not results:
        # Defensive: splitter returned only whitespace fragments. Fall
        # through with a single chunk of stripped content so downstream
        # callers always receive at least one row when input is non-empty.
        logger.warning(
            "[recursive_character] splitter produced no non-empty chunks "
            "for %d-char input; emitting single fallback chunk.",
            len(content),
        )
        body = content.strip()
        if body:
            start = content.find(body)
            results.append(
                {
                    "tokens": len(tokenizer.encode(body)),
                    "content": body,
                    "chunk_order_index": 0,
                    **(
                        {"_source_span": {"start": start, "end": start + len(body)}}
                        if start >= 0
                        else {}
                    ),
                }
            )

    return results
