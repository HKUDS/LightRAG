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

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from lightrag.constants import MAX_R_SEPARATOR_CHARS, MAX_R_SEPARATORS
from lightrag.utils import Tokenizer, logger

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    _LANGCHAIN_TEXT_SPLITTERS_AVAILABLE = True
except ImportError:
    _LANGCHAIN_TEXT_SPLITTERS_AVAILABLE = False
    RecursiveCharacterTextSplitter = None  # type: ignore[assignment]


_SpanPiece = tuple[str, int, int]


@dataclass(frozen=True)
class RSeparatorNormalization:
    """The bounded R-separator cascade plus any correction that was needed.

    The normalizer is deliberately silent: it is also the final safety boundary
    for direct SDK calls and old persisted document snapshots, where a warning
    per document would turn one historic bad configuration into a log storm.
    Configuration ingress points inspect this result and emit a single warning
    when they cache a corrected value.
    """

    separators: list[str] | None
    dropped: int = 0
    truncated_from: int | None = None

    @property
    def changed(self) -> bool:
        return self.dropped > 0 or self.truncated_from is not None


def _split_text_with_regex_spans(
    text: str,
    separator_pattern: str,
    *,
    keep_separator: bool | str,
    base_offset: int,
) -> list[_SpanPiece]:
    """Mirror LangChain's regex split while retaining source offsets."""
    if not separator_pattern:
        return [
            (char, base_offset + index, base_offset + index + 1)
            for index, char in enumerate(text)
            if char
        ]

    matches = list(re.finditer(separator_pattern, text))
    if not matches:
        return [(text, base_offset, base_offset + len(text))] if text else []

    pieces: list[_SpanPiece] = []
    if keep_separator:
        if keep_separator == "end":
            cursor = 0
            for match in matches:
                if match.end() > cursor:
                    pieces.append(
                        (
                            text[cursor : match.end()],
                            base_offset + cursor,
                            base_offset + match.end(),
                        )
                    )
                cursor = match.end()
            if cursor < len(text):
                pieces.append(
                    (text[cursor:], base_offset + cursor, base_offset + len(text))
                )
        else:
            first = matches[0]
            if first.start() > 0:
                pieces.append(
                    (text[: first.start()], base_offset, base_offset + first.start())
                )
            for index, match in enumerate(matches):
                end = (
                    matches[index + 1].start()
                    if index + 1 < len(matches)
                    else len(text)
                )
                if end > match.start():
                    pieces.append(
                        (
                            text[match.start() : end],
                            base_offset + match.start(),
                            base_offset + end,
                        )
                    )
    else:
        cursor = 0
        for match in matches:
            if match.start() > cursor:
                pieces.append(
                    (
                        text[cursor : match.start()],
                        base_offset + cursor,
                        base_offset + match.start(),
                    )
                )
            cursor = match.end()
        if cursor < len(text):
            pieces.append(
                (text[cursor:], base_offset + cursor, base_offset + len(text))
            )

    return [piece for piece in pieces if piece[0]]


def _join_span_pieces(
    pieces: list[_SpanPiece],
    separator: str,
    *,
    strip_whitespace: bool,
) -> _SpanPiece | None:
    """Join split pieces exactly as LangChain does and compute the trimmed span."""
    if not pieces:
        return None

    chars: list[str] = []
    char_offsets: list[int] = []
    for index, (fragment, start, end) in enumerate(pieces):
        if index > 0 and separator:
            previous_end = pieces[index - 1][2]
            for sep_index, sep_char in enumerate(separator):
                chars.append(sep_char)
                char_offsets.append(previous_end + sep_index)
        chars.extend(fragment)
        char_offsets.extend(range(start, end))

    text = "".join(chars)
    if strip_whitespace:
        left = 0
        right = len(text)
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
    else:
        left, right = 0, len(text)

    if left >= right:
        return None
    return text[left:right], char_offsets[left], char_offsets[right - 1] + 1


def _merge_splits_with_spans(
    splits: Sequence[_SpanPiece],
    separator: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    length_function: Callable[[str], int],
    strip_whitespace: bool,
) -> list[_SpanPiece]:
    """Mirror ``TextSplitter._merge_splits`` while preserving source spans."""
    separator_len = length_function(separator)
    docs: list[_SpanPiece] = []
    current_doc: list[_SpanPiece] = []
    total = 0

    for split in splits:
        split_len = length_function(split[0])
        if (
            total + split_len + (separator_len if len(current_doc) > 0 else 0)
            > chunk_size
        ):
            if total > chunk_size:
                logger.warning(
                    "Created a chunk of size %d, which is longer than the specified %d",
                    total,
                    chunk_size,
                )
            if len(current_doc) > 0:
                doc = _join_span_pieces(
                    current_doc,
                    separator,
                    strip_whitespace=strip_whitespace,
                )
                if doc is not None:
                    docs.append(doc)
                while total > chunk_overlap or (
                    total + split_len + (separator_len if len(current_doc) > 0 else 0)
                    > chunk_size
                    and total > 0
                ):
                    total -= length_function(current_doc[0][0]) + (
                        separator_len if len(current_doc) > 1 else 0
                    )
                    current_doc = current_doc[1:]
        current_doc.append(split)
        total += split_len + (separator_len if len(current_doc) > 1 else 0)

    doc = _join_span_pieces(
        current_doc,
        separator,
        strip_whitespace=strip_whitespace,
    )
    if doc is not None:
        docs.append(doc)
    return docs


def inspect_r_separators(
    separators: Sequence[str] | None,
) -> RSeparatorNormalization:
    """Bound a separator cascade and report whether it was corrected.

    ``None`` passes straight through: it is the documented way to ask
    :func:`chunking_by_recursive_character` for LangChain's own default cascade
    (``["\\n\\n", "\\n", " ", ""]``), which is NOT
    :data:`~lightrag.constants.DEFAULT_R_SEPARATORS` — turning one into the other
    here would change how a direct SDK caller's CJK text splits.

    Only bounding happens here. What to do about an empty result differs per
    caller and is therefore the caller's decision. This function does not log:
    it runs in document-processing hot paths as well as at configuration ingress.

    Args:
        separators: cascade to bound, or ``None``.

    Returns:
        The bounded cascade, possibly empty, and correction diagnostics.
    """
    if separators is None:
        return RSeparatorNormalization(None)

    bounded: list[str] = []
    dropped = 0
    for candidate in separators:
        if len(candidate) > MAX_R_SEPARATOR_CHARS:
            dropped += 1
            continue
        bounded.append(candidate)
    truncated_from: int | None = None
    if len(bounded) > MAX_R_SEPARATORS:
        # Truncation must preserve a terminating sentinel. The empty string is
        # the char-level fallback: ``_split_text_with_spans`` takes the
        # ``if not candidate`` branch on it and splits anything. Dropping it
        # exhausts ``new_separators`` early, so a segment that has no ordinary
        # separator in it is emitted whole instead of being split — a data-quality
        # regression introduced by a security fix. Only re-append a sentinel the
        # caller actually had: ``load_chunk_separators`` strips it on purpose.
        sentinel = "" if any(not candidate for candidate in bounded) else None
        keep = MAX_R_SEPARATORS - (1 if sentinel is not None else 0)
        truncated = [candidate for candidate in bounded[:keep] if candidate]
        if sentinel is not None:
            truncated.append(sentinel)
        truncated_from = len(bounded)
        bounded = truncated

    return RSeparatorNormalization(
        bounded, dropped=dropped, truncated_from=truncated_from
    )


def log_r_separator_normalization(
    result: RSeparatorNormalization,
    *,
    context: str,
) -> None:
    """Log corrections made while caching a separator configuration.

    Callers must invoke this only at a configuration ingress boundary. The
    runtime normalizer intentionally remains silent so already-persisted bad
    snapshots and direct SDK calls cannot emit one warning per document.
    """
    prefix = f"[{context}] " if context else ""
    if result.dropped:
        logger.warning(
            f"{prefix}dropped {result.dropped} separator(s) longer than "
            f"{MAX_R_SEPARATOR_CHARS} characters"
        )
    if result.truncated_from is not None:
        logger.warning(
            f"{prefix}separator cascade of {result.truncated_from} entries "
            f"truncated to {len(result.separators or [])} "
            f"(limit {MAX_R_SEPARATORS})"
        )


def normalize_r_separators(separators: Sequence[str] | None) -> list[str] | None:
    """Return a bounded separator cascade without logging.

    This is the runtime backstop shared by direct SDK calls and per-document
    snapshots. Warnings belong to the configuration ingress points, which use
    :func:`inspect_r_separators` plus :func:`log_r_separator_normalization`
    instead — see :class:`RSeparatorNormalization`.
    """
    return inspect_r_separators(separators).separators


def _split_text_with_spans(
    text: str,
    *,
    base_offset: int,
    separators: Sequence[str],
    chunk_size: int,
    chunk_overlap: int,
    length_function: Callable[[str], int],
    keep_separator: bool | str,
    is_separator_regex: bool,
    strip_whitespace: bool,
) -> list[_SpanPiece]:
    """Mirror ``RecursiveCharacterTextSplitter._split_text`` with offsets.

    One departure from a literal mirror, for cost only: a separator that matches
    but does not actually divide the text is skipped in place instead of via a
    recursive call. With ``keep_separator`` truthy, a separator occurring exactly
    once at offset 0 yields a single piece byte-identical to the input, so the
    parent would call ``length_function`` on it — a whole-text token encode — and
    recurse with the same text. A cascade of such separators therefore costs one
    full encode per level while splitting nothing, which is what made
    ``O(len(separators) x len(text))`` reachable from a single request
    (GHSA-26pm-px5v-8c4w).

    The loop below reaches the same state the recursion would, minus the wasted
    encodes, so the output is unchanged.
    """
    remaining: Sequence[str] = separators
    while True:
        separator = remaining[-1]
        new_separators: Sequence[str] = []
        for index, candidate in enumerate(remaining):
            separator_pattern = (
                candidate if is_separator_regex else re.escape(candidate)
            )
            if not candidate:
                separator = candidate
                break
            if re.search(separator_pattern, text):
                separator = candidate
                new_separators = remaining[index + 1 :]
                break

        separator_pattern = separator if is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex_spans(
            text,
            separator_pattern,
            keep_separator=keep_separator,
            base_offset=base_offset,
        )

        # A no-op split: one piece, identical to the input. Recursing on it would
        # re-encode the whole text to discover it is still oversized and then land
        # exactly here with the next separator, so advance in place instead. When
        # no separators remain the loop exits and the piece is emitted whole, as
        # the recursion would have.
        if (
            new_separators
            and len(splits) == 1
            and splits[0][0] == text
            and splits[0][1] == base_offset
        ):
            remaining = new_separators
            continue
        break

    final_chunks: list[_SpanPiece] = []
    good_splits: list[_SpanPiece] = []
    merge_separator = "" if keep_separator else separator
    for split in splits:
        if length_function(split[0]) < chunk_size:
            good_splits.append(split)
        else:
            if good_splits:
                final_chunks.extend(
                    _merge_splits_with_spans(
                        good_splits,
                        merge_separator,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        length_function=length_function,
                        strip_whitespace=strip_whitespace,
                    )
                )
                good_splits = []
            if not new_separators:
                final_chunks.append(split)
            else:
                final_chunks.extend(
                    _split_text_with_spans(
                        split[0],
                        base_offset=split[1],
                        separators=new_separators,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        length_function=length_function,
                        keep_separator=keep_separator,
                        is_separator_regex=is_separator_regex,
                        strip_whitespace=strip_whitespace,
                    )
                )
    if good_splits:
        final_chunks.extend(
            _merge_splits_with_spans(
                good_splits,
                merge_separator,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=length_function,
                strip_whitespace=strip_whitespace,
            )
        )
    return final_chunks


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
            LangChain's defaults: ``["\\n\\n", "\\n", " ", ""]``. Entries over
            :data:`~lightrag.constants.MAX_R_SEPARATOR_CHARS` are dropped and a
            cascade over :data:`~lightrag.constants.MAX_R_SEPARATORS` is
            truncated; if nothing survives, the call behaves as ``None``.
            Bounding is silent — see :func:`inspect_r_separators` to obtain the
            correction detail yourself.

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

    def length_function(text: str) -> int:
        return len(tokenizer.encode(text))

    # Final silent boundary for direct SDK calls and persisted snapshots that
    # predate configuration-time normalization.
    separators = normalize_r_separators(separators)
    if separators is not None and not separators:
        # Everything was dropped as over-long. Falling back to this function's own
        # default (LangChain's cascade) rather than to DEFAULT_R_SEPARATORS keeps
        # the behaviour identical to ``separators=None``; substituting the
        # repo-wide cascade here would silently change how CJK text splits for a
        # caller who never asked for it. An empty list would be worse still —
        # ``_split_text_with_spans`` reads ``separators[-1]`` on its first line.
        #
        # No warning: this branch is reachable only from a direct SDK call. Every
        # configured path resolves the same emptiness earlier — the env and
        # addon_params ingress points report it once when they cache the value,
        # and the pipeline drops the ``separators`` key from a stale snapshot
        # before calling here, so this function receives ``None``. Warning here
        # would therefore only fire per call, for one unchanging argument, which
        # is exactly the amplification the ingress cache exists to remove. A
        # caller that wants the diagnostic can ask for it: ``inspect_r_separators``
        # returns the same correction detail without splitting anything.
        separators = None

    splitter_kwargs: dict[str, Any] = {
        "chunk_size": max(int(chunk_token_size), 1),
        "chunk_overlap": max(int(chunk_overlap_token_size), 0),
        "length_function": length_function,
        "strip_whitespace": True,
    }
    if separators is not None:
        splitter_kwargs["separators"] = list(separators)

    splitter = RecursiveCharacterTextSplitter(**splitter_kwargs)

    # We deliberately do *not* request LangChain's ``add_start_index``. That
    # offset is computed with a character-vs-token unit mismatch when a
    # token-based ``length_function`` is in play, and text-search recovery is
    # ambiguous for repeated blocks. Instead we mirror LangChain's split/merge
    # control flow while carrying each split unit's source offsets through it.
    pieces = _split_text_with_spans(
        content,
        base_offset=0,
        separators=list(splitter._separators),
        chunk_size=int(splitter._chunk_size),
        chunk_overlap=int(splitter._chunk_overlap),
        length_function=length_function,
        keep_separator=splitter._keep_separator,
        is_separator_regex=bool(splitter._is_separator_regex),
        strip_whitespace=bool(splitter._strip_whitespace),
    )
    results: list[dict[str, Any]] = []
    for raw_body, start_index, end_index in pieces:
        left = 0
        right = len(raw_body)
        while left < right and raw_body[left].isspace():
            left += 1
        while right > left and raw_body[right - 1].isspace():
            right -= 1
        body = raw_body[left:right]
        if not body:
            continue
        start_index += left
        end_index -= len(raw_body) - right
        results.append(
            {
                "tokens": len(tokenizer.encode(body)),
                "content": body,
                "chunk_order_index": len(results),
                "_source_span": {"start": start_index, "end": end_index},
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
