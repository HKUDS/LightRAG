"""Fixed-size token-window chunking — the LightRAG default strategy.

Chunks the input text into windows of at most ``chunk_token_size`` tokens
with ``chunk_overlap_token_size`` of overlap between adjacent windows.
When ``split_by_character`` is supplied, the splitter first segments on
that delimiter and then either tokenizes each segment as-is
(``split_by_character_only=True``) or further sub-splits any segment
that exceeds the token cap.

Two entry points are exported:

  - :func:`chunking_by_token_size` — the **legacy 6-arg signature**
    used as the default value for :attr:`lightrag.LightRAG.chunking_func`.
    Kept for backward compatibility so externally-supplied chunking
    functions can continue to drop in unchanged.

  - :func:`chunking_by_fixed_token` — the same algorithm exposed under
    the **new file-chunker contract** (standard prefix
    ``(tokenizer, content, chunk_token_size)`` plus keyword-only
    knobs). Used by the file-based chunking dispatcher in
    ``process_single_document`` for ``doc_process_opts.chunking == "F"``.
"""

from __future__ import annotations

from typing import Any

from lightrag.exceptions import ChunkTokenLimitExceededError
from lightrag.utils import Tokenizer, logger


def _trimmed_span(content: str, start: int, end: int) -> tuple[int, int]:
    """Return the source span after applying the chunker's ``.strip()``."""
    start = max(0, min(start, len(content)))
    end = max(start, min(end, len(content)))
    while start < end and content[start].isspace():
        start += 1
    while end > start and content[end - 1].isspace():
        end -= 1
    return start, end


def _source_span(content: str, start: int, end: int) -> dict[str, int] | None:
    start, end = _trimmed_span(content, start, end)
    if start >= end:
        return None
    return {"start": start, "end": end}


def _token_window_source_span(
    tokenizer: Tokenizer,
    content: str,
    tokens: list[int],
    start_token: int,
    end_token: int,
) -> dict[str, int] | None:
    """Map a decoded token window back to its exact source span."""
    window = tokenizer.decode(tokens[start_token:end_token])
    prefix = tokenizer.decode(tokens[:start_token])
    start = len(prefix)
    end = start + len(window)
    if content[start:end] != window:
        found = content.find(
            window,
            max(0, start - 32),
            min(len(content), end + 32 + len(window)),
        )
        if found < 0:
            return None
        start = found
        end = found + len(window)
    return _source_span(content, start, end)


def _make_chunk(
    *,
    content: str,
    tokens: int,
    order: int,
    source_span: dict[str, int] | None,
    emit_source_span: bool,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "tokens": tokens,
        "content": content.strip(),
        "chunk_order_index": order,
    }
    if emit_source_span and source_span is not None:
        item["_source_span"] = source_span
    return item


def chunking_by_token_size(
    tokenizer: Tokenizer,
    content: str,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    chunk_overlap_token_size: int = 100,
    chunk_token_size: int = 1200,
    *,
    _emit_source_span: bool = False,
) -> list[dict[str, Any]]:
    """Legacy 6-arg fixed-token chunker (default for ``LightRAG.chunking_func``).

    Signature is preserved for backward compatibility with externally
    supplied ``chunking_func`` implementations. New file-based chunking
    dispatch uses :func:`chunking_by_fixed_token` instead.
    """
    tokens = tokenizer.encode(content)
    results: list[dict[str, Any]] = []
    if split_by_character:
        raw_chunks = content.split(split_by_character)
        raw_spans: list[tuple[int, int]] = []
        cursor = 0
        for raw_chunk in raw_chunks:
            start = cursor
            end = start + len(raw_chunk)
            raw_spans.append((start, end))
            cursor = end + len(split_by_character)
        new_chunks = []
        if split_by_character_only:
            for chunk, (chunk_start, chunk_end) in zip(raw_chunks, raw_spans):
                _tokens = tokenizer.encode(chunk)
                if len(_tokens) > chunk_token_size:
                    logger.warning(
                        "Chunk split_by_character exceeds token limit: len=%d limit=%d",
                        len(_tokens),
                        chunk_token_size,
                    )
                    raise ChunkTokenLimitExceededError(
                        chunk_tokens=len(_tokens),
                        chunk_token_limit=chunk_token_size,
                        chunk_preview=chunk[:120],
                    )
                span = (
                    _source_span(content, chunk_start, chunk_end)
                    if _emit_source_span
                    else None
                )
                new_chunks.append((len(_tokens), chunk, span))
        else:
            for chunk, (chunk_start, chunk_end) in zip(raw_chunks, raw_spans):
                _tokens = tokenizer.encode(chunk)
                if len(_tokens) > chunk_token_size:
                    for start in range(
                        0, len(_tokens), chunk_token_size - chunk_overlap_token_size
                    ):
                        chunk_content = tokenizer.decode(
                            _tokens[start : start + chunk_token_size]
                        )
                        span = None
                        if _emit_source_span:
                            span = _token_window_source_span(
                                tokenizer,
                                chunk,
                                _tokens,
                                start,
                                min(start + chunk_token_size, len(_tokens)),
                            )
                        if span is not None:
                            span = {
                                "start": chunk_start + span["start"],
                                "end": chunk_start + span["end"],
                            }
                        new_chunks.append(
                            (
                                min(chunk_token_size, len(_tokens) - start),
                                chunk_content,
                                span,
                            )
                        )
                else:
                    span = (
                        _source_span(content, chunk_start, chunk_end)
                        if _emit_source_span
                        else None
                    )
                    new_chunks.append((len(_tokens), chunk, span))
        for index, (_len, chunk, span) in enumerate(new_chunks):
            results.append(
                _make_chunk(
                    content=chunk,
                    tokens=_len,
                    order=index,
                    source_span=span,
                    emit_source_span=_emit_source_span,
                )
            )
    else:
        for index, start in enumerate(
            range(0, len(tokens), chunk_token_size - chunk_overlap_token_size)
        ):
            end = min(start + chunk_token_size, len(tokens))
            chunk_content = tokenizer.decode(tokens[start:end])
            span = (
                _token_window_source_span(tokenizer, content, tokens, start, end)
                if _emit_source_span
                else None
            )
            results.append(
                _make_chunk(
                    content=chunk_content,
                    tokens=min(chunk_token_size, len(tokens) - start),
                    order=index,
                    source_span=span,
                    emit_source_span=_emit_source_span,
                )
            )
    return results


def chunking_by_fixed_token(
    tokenizer: Tokenizer,
    content: str,
    chunk_token_size: int = 1200,
    *,
    chunk_overlap_token_size: int = 100,
    split_by_character: str | None = None,
    split_by_character_only: bool = False,
    _emit_source_span: bool = False,
) -> list[dict[str, Any]]:
    """Fixed-token chunker — file-chunker contract for the ``"F"`` strategy.

    Implements the same fixed-window algorithm as
    :func:`chunking_by_token_size`, exposed under the standard
    file-chunker signature ``(tokenizer, content, chunk_token_size, *,
    <strategy kwargs>)`` so the file-based chunking dispatcher in
    ``process_single_document`` can call every strategy uniformly.
    """
    return chunking_by_token_size(
        tokenizer,
        content,
        split_by_character=split_by_character,
        split_by_character_only=split_by_character_only,
        chunk_overlap_token_size=chunk_overlap_token_size,
        chunk_token_size=chunk_token_size,
        _emit_source_span=_emit_source_span,
    )
