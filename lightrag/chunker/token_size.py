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
from lightrag.utils import _REPLACEMENT_CHAR, Tokenizer, logger


def _recover_boundary_window(
    content: str,
    window: str,
    prefix: str,
    prefix_start: int,
) -> tuple[int, int] | None:
    """Locate a window whose edges split a multi-byte character.

    ``prefix`` is the decode of the tokens between a verified anchor at char
    ``prefix_start`` and the window. Such a window decodes with leading and/or
    trailing U+FFFD, so it is not a slice of ``content``; the text between
    them is. Returns the ``(start, end)`` of that text, or ``None``.

    The offset is computed, never searched for: text that repeats (an emoji
    run, a repeated phrase) matches in several places, and only one of them is
    this window. The start is the characters wholly before the window plus the
    one its first token splits, counted from ``prefix`` with its trailing
    U+FFFD dropped -- so the result does not depend on how many U+FFFD a
    decoder emits for one broken character.

    Call it only for ``content`` that holds no U+FFFD of its own. With one
    anywhere in it, a window's edge U+FFFD cannot be told from split
    characters. :func:`_token_window_source_span` declines ambiguous spans
    instead and enforces this restriction on recovery.
    """
    trimmed = window.strip(_REPLACEMENT_CHAR)
    if not trimmed or trimmed == window:
        return None
    whole = prefix.rstrip(_REPLACEMENT_CHAR)
    start = prefix_start + len(whole) + (1 if len(whole) != len(prefix) else 0)
    end = start + len(trimmed)
    if not content.startswith(trimmed, start):
        return None
    return start, end


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


def _origin_anchor(
    tokenizer: Tokenizer, content: str, tokens: list[int]
) -> tuple[int, int]:
    """The verified ``(0, char)`` anchor that token 0 decodes from.

    A decoder may drop whitespace at the sequence start (Metaspace), so token
    0 can decode from after the segment's leading whitespace. A prefix match
    cannot locate it inside a run of whitespace; the whole-segment decode is
    an exact suffix of ``content`` and does. Costs one decode per segment,
    only when ``content`` starts with whitespace.
    """
    if not content[:1].isspace():
        return 0, 0
    full = tokenizer.decode(tokens)
    skip = len(content) - len(full)
    if skip > 0 and content.endswith(full) and content[:skip].isspace():
        return 0, skip
    return 0, 0


def _token_window_source_span(
    tokenizer: Tokenizer,
    content: str,
    tokens: list[int],
    start_token: int,
    end_token: int,
    *,
    anchor: tuple[int, int] | None,
    recover_boundary: bool | None = None,
) -> tuple[dict[str, int] | None, tuple[int, int] | None]:
    """Map a token window from a verified ``(token, char)`` anchor.

    Decode through the window from that anchor to establish its end, rather
    than searching for text near an estimated start. In particular, decoding
    a prefix separately can insert U+FFFD or drop a partial character; equality
    with repeated source text does not prove that estimated position.

    An exact window must be a suffix of the joint decode, and that decode must
    match the source at the anchor. The exact suffix also verifies the window's
    own start, even if decoding it in isolation drops a leading space or
    incomplete bytes that the joint decode retained.

    With literal U+FFFD in the source, an edge U+FFFD needs a token round trip
    before it can be accepted as literal text. Otherwise decline the span.
    Without source U+FFFD, the existing boundary recovery can safely locate the
    interior text. Callers precompute ``recover_boundary`` to avoid rescanning.

    Decoding covers only the delta and window, keeping the normal path O(N).
    Runs of ambiguous mid-character starts may retain the last verified anchor.
    If both the joint decode and the prefix (ignoring trailing boundary
    placeholders) disagree with the source, a lossy tokenizer may have changed
    earlier text.
    A dead anchor (None) omits provenance for the rest of this segment without
    affecting chunk text or repeatedly decoding from the stale anchor.
    """
    if anchor is None:
        return None, None
    anchor_token, anchor_char = anchor
    if start_token < anchor_token:  # non-monotonic caller: use the source origin
        anchor_token, anchor_char = 0, 0
    window_tokens = tokens[start_token:end_token]
    window = tokenizer.decode(window_tokens)
    joint = tokenizer.decode(tokens[anchor_token:end_token])
    if recover_boundary is None:
        recover_boundary = _REPLACEMENT_CHAR not in content

    ambiguous_replacement = (
        not recover_boundary
        and (window.startswith(_REPLACEMENT_CHAR) or window.endswith(_REPLACEMENT_CHAR))
        and tokenizer.encode(window) != window_tokens
    )
    joint_matches = content.startswith(joint, anchor_char)
    if (
        window
        and not ambiguous_replacement
        and joint.endswith(window)
        and joint_matches
    ):
        end = anchor_char + len(joint)
        start = end - len(window)
        return _source_span(content, start, end), (start_token, start)

    prefix = tokenizer.decode(tokens[anchor_token:start_token])
    if not joint_matches and not content.startswith(
        prefix.rstrip(_REPLACEMENT_CHAR), anchor_char
    ):
        return None, None

    if ambiguous_replacement:
        # A text match cannot distinguish a literal U+FFFD from an artifact.
        return None, anchor

    recovered = (
        _recover_boundary_window(content, window, prefix, anchor_char)
        if recover_boundary
        else None
    )
    if recovered is None:
        return None, anchor
    start, end = recovered
    if window.startswith(_REPLACEMENT_CHAR):
        return _source_span(content, start, end), anchor
    return _source_span(content, start, end), (start_token, start)


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
                    # Anchor is chunk-relative (offsets are shifted by chunk_start
                    # below), so it resets per split-by-character segment.
                    anchor = (
                        _origin_anchor(tokenizer, chunk, _tokens)
                        if _emit_source_span
                        else (0, 0)
                    )
                    recover_boundary = _REPLACEMENT_CHAR not in chunk
                    for start in range(
                        0,
                        len(_tokens),
                        _window_step(chunk_token_size, chunk_overlap_token_size),
                    ):
                        end_token = min(start + chunk_token_size, len(_tokens))
                        chunk_content = tokenizer.decode(_tokens[start:end_token])
                        span = None
                        if _emit_source_span:
                            span, anchor = _token_window_source_span(
                                tokenizer,
                                chunk,
                                _tokens,
                                start,
                                end_token,
                                anchor=anchor,
                                recover_boundary=recover_boundary,
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
                        # Same stop rule as the fixed-window loop below.
                        if end_token >= len(_tokens):
                            break
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
        # Only the fixed-window path needs the whole-document token list;
        # the split_by_character path re-encodes each segment anyway (BPE
        # boundaries differ on substrings), so the full encode is deferred
        # here rather than paid on every call.
        tokens = tokenizer.encode(content)
        anchor = (
            _origin_anchor(tokenizer, content, tokens) if _emit_source_span else (0, 0)
        )
        recover_boundary = _REPLACEMENT_CHAR not in content
        for index, start in enumerate(
            range(
                0,
                len(tokens),
                _window_step(chunk_token_size, chunk_overlap_token_size),
            )
        ):
            end = min(start + chunk_token_size, len(tokens))
            chunk_content = tokenizer.decode(tokens[start:end])
            span = None
            if _emit_source_span:
                span, anchor = _token_window_source_span(
                    tokenizer,
                    content,
                    tokens,
                    start,
                    end,
                    anchor=anchor,
                    recover_boundary=recover_boundary,
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
            # Once a window reaches the last token, every later start falls
            # inside this window's overlap, so those windows would only repeat
            # its tail: no new text, but each still costs an extraction call
            # and an embedding.
            if end >= len(tokens):
                break
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
