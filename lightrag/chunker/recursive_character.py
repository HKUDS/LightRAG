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
from typing import Any

from lightrag.utils import Tokenizer, logger

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    _LANGCHAIN_TEXT_SPLITTERS_AVAILABLE = True
except ImportError:
    _LANGCHAIN_TEXT_SPLITTERS_AVAILABLE = False
    RecursiveCharacterTextSplitter = None  # type: ignore[assignment]


_SpanPiece = tuple[str, int, int]


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
    """Mirror ``RecursiveCharacterTextSplitter._split_text`` with offsets."""
    separator = separators[-1]
    new_separators: Sequence[str] = []
    for index, candidate in enumerate(separators):
        separator_pattern = candidate if is_separator_regex else re.escape(candidate)
        if not candidate:
            separator = candidate
            break
        if re.search(separator_pattern, text):
            separator = candidate
            new_separators = separators[index + 1 :]
            break

    separator_pattern = separator if is_separator_regex else re.escape(separator)
    splits = _split_text_with_regex_spans(
        text,
        separator_pattern,
        keep_separator=keep_separator,
        base_offset=base_offset,
    )

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

    def length_function(text: str) -> int:
        return len(tokenizer.encode(text))

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
