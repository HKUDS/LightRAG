"""Fixed-size token-window chunking — the LightRAG default strategy.

Chunks the input text into windows of at most ``chunk_token_size`` tokens
with ``chunk_overlap_token_size`` of overlap between adjacent windows.
Every window is a slice of the input cut at a character boundary
(:meth:`lightrag.utils.Tokenizer.split_by_token_limit`, the primitive the
pre-embedding hard split also uses), never a decoded token window, so a
window never splits a multi-byte character.
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
from lightrag.utils import Tokenizer, TokenBudgetError, logger


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


def _make_chunk(
    *,
    tokenizer: Tokenizer,
    content: str,
    tokens: int,
    order: int,
    source_span: dict[str, int] | None,
    emit_source_span: bool,
) -> dict[str, Any]:
    """Build one chunk dict whose ``tokens`` counts the stored content.

    ``tokens`` is the count of ``content`` as given; the chunk stores
    ``content.strip()``, so a stripped chunk is counted again. The recount
    may exceed the window's count, and so ``chunk_token_size``, by a token
    or two: BPE attaches a space to the word after it, and a word whose
    leading space is stripped can take more tokens on its own (cl100k:
    ``" bridging"`` is one token, ``"bridging"`` two). The figure reports
    the stored text; the hard limit before embedding recounts on its own.
    """
    stripped = content.strip()
    if stripped != content:
        tokens = len(tokenizer.encode(stripped))
    item: dict[str, Any] = {
        "tokens": tokens,
        "content": stripped,
        "chunk_order_index": order,
    }
    if emit_source_span and source_span is not None:
        item["_source_span"] = source_span
    return item


def _window_step(chunk_token_size: int, chunk_overlap_token_size: int) -> int:
    """Token-window stride for the sliding-window chunk loops.

    When overlap >= size the stride is <= 0, which makes ``range()`` yield an
    empty sequence (dropping the whole segment silently) or raise the opaque
    ``range() arg 3 must not be zero``. Fail closed with the same invariant the
    API-boundary validators enforce.

    Negative overlap is rejected up front in ``chunking_by_token_size()``,
    before this function is ever reached -- see the check there.
    """
    if chunk_overlap_token_size >= chunk_token_size:
        raise ValueError(
            f"chunk_overlap_token_size ({chunk_overlap_token_size}) must be < "
            f"chunk_token_size ({chunk_token_size})"
        )
    return chunk_token_size - chunk_overlap_token_size


def _token_windows(
    tokenizer: Tokenizer,
    text: str,
    chunk_token_size: int,
    chunk_overlap_token_size: int,
) -> list[tuple[int, int, int]]:
    """Cut ``text`` into ``(start, end, token_count)`` character windows.

    Windows cover ``text`` without gaps, overlap by about
    ``chunk_overlap_token_size`` tokens and end at character boundaries.
    The overlap is exact on :class:`~lightrag.utils.TiktokenTokenizer`'s
    offset path and a best-effort estimate on the generic path (see
    :meth:`~lightrag.utils.Tokenizer.split_by_token_limit`).
    ``token_count`` is the count of the window's own text.
    """
    _window_step(chunk_token_size, chunk_overlap_token_size)
    try:
        spans = tokenizer.split_by_token_limit(
            text, chunk_token_size, overlap_tokens=chunk_overlap_token_size
        )
    except TokenBudgetError as exc:
        # Not even one character fits. Unreachable through configuration,
        # which requires MIN_CHUNK_TOKEN_SIZE; a direct call can still ask
        # for a size below one character's token count.
        raise ChunkTokenLimitExceededError(
            chunk_tokens=exc.code_point_token_count,
            chunk_token_limit=chunk_token_size,
            chunk_preview=exc.preview,
        ) from exc
    return [(span.start, span.end, span.token_count) for span in spans]


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
    # Checked up front, before any branching on split_by_character, so a
    # negative chunk_overlap_token_size is always rejected -- including on
    # the split_by_character path where every segment stays under
    # chunk_token_size and _window_step()'s own (upper-bound only) check
    # would otherwise never be reached. The upper-bound check stays lazy,
    # inside _window_step(), since it's only meaningful once windowing is
    # actually about to happen.
    if chunk_overlap_token_size < 0:
        raise ValueError(
            f"chunk_overlap_token_size must be non-negative, got "
            f"{chunk_overlap_token_size}"
        )
    # (token_count, text, start offset in content): the text is always the
    # slice content[start : start + len(text)].
    pieces: list[tuple[int, str, int]] = []
    if split_by_character:
        cursor = 0
        for segment in content.split(split_by_character):
            segment_start = cursor
            cursor += len(segment) + len(split_by_character)
            segment_tokens = len(tokenizer.encode(segment))
            if segment_tokens <= chunk_token_size:
                pieces.append((segment_tokens, segment, segment_start))
                continue
            if split_by_character_only:
                logger.warning(
                    "Chunk split_by_character exceeds token limit: len=%d limit=%d",
                    segment_tokens,
                    chunk_token_size,
                )
                raise ChunkTokenLimitExceededError(
                    chunk_tokens=segment_tokens,
                    chunk_token_limit=chunk_token_size,
                    chunk_preview=segment[:120],
                )
            for start, end, count in _token_windows(
                tokenizer, segment, chunk_token_size, chunk_overlap_token_size
            ):
                pieces.append((count, segment[start:end], segment_start + start))
    else:
        for start, end, count in _token_windows(
            tokenizer, content, chunk_token_size, chunk_overlap_token_size
        ):
            pieces.append((count, content[start:end], start))
    return [
        _make_chunk(
            tokenizer=tokenizer,
            content=text,
            tokens=count,
            order=index,
            source_span=(
                _source_span(content, start, start + len(text))
                if _emit_source_span
                else None
            ),
            emit_source_span=_emit_source_span,
        )
        for index, (count, text, start) in enumerate(pieces)
    ]


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
