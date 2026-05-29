"""Unit tests for :func:`lightrag.sidecar.backfill.backfill_chunk_sidecars`.

These exercise the F/R/V sidecar backfill in isolation: a small ``blocks.jsonl``
is written to ``tmp_path`` and hand-built chunk lists are matched against it, so
no real chunker, tokenizer, or embedding is needed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lightrag.exceptions import ChunkBlockMatchError
from lightrag.sidecar import backfill_chunk_sidecars

# The merged text the chunker would have received is
# "\n\n".join(content for content rows with content.strip()).
_BLOCK_SEPARATOR = "\n\n"


def _write_blocks(tmp_path: Path, blocks: list[tuple[str, str]]) -> str:
    """Write a blocks.jsonl with a meta header + ``(blockid, content)`` rows."""
    path = tmp_path / "doc.blocks.jsonl"
    lines = [json.dumps({"type": "meta", "format": "lightrag", "version": "1.0"})]
    for blockid, content in blocks:
        lines.append(
            json.dumps(
                {
                    "type": "content",
                    "blockid": blockid,
                    "format": "plain_text",
                    "content": content,
                    "heading": "",
                    "parent_headings": [],
                    "level": 1,
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _chunk(content: str, order: int) -> dict:
    return {"content": content, "tokens": len(content), "chunk_order_index": order}


def _refs(chunk: dict) -> list[str]:
    return [r["id"] for r in chunk["sidecar"]["refs"]]


@pytest.mark.offline
def test_chunk_spanning_two_blocks_lists_both_refs(tmp_path: Path) -> None:
    blocks_path = _write_blocks(
        tmp_path, [("b1", "Alpha paragraph."), ("b2", "Beta paragraph.")]
    )
    merged = _BLOCK_SEPARATOR.join(["Alpha paragraph.", "Beta paragraph."])
    chunks = [_chunk(merged, 0)]

    backfill_chunk_sidecars(chunks, blocks_path)

    assert chunks[0]["sidecar"]["type"] == "block"
    assert chunks[0]["sidecar"]["id"] == "b1"
    assert _refs(chunks[0]) == ["b1", "b2"]


@pytest.mark.offline
def test_chunk_equal_to_single_block(tmp_path: Path) -> None:
    blocks_path = _write_blocks(
        tmp_path, [("b1", "Alpha paragraph."), ("b2", "Beta paragraph.")]
    )
    chunks = [_chunk("Beta paragraph.", 0)]

    backfill_chunk_sidecars(chunks, blocks_path)

    assert chunks[0]["sidecar"]["id"] == "b2"
    assert _refs(chunks[0]) == ["b2"]


@pytest.mark.offline
def test_whitespace_normalized_match_for_v_style(tmp_path: Path) -> None:
    # Block content has internal newlines; the V chunker rejoins sentences with
    # single spaces, so the chunk body is not byte-verbatim.
    block = "First sentence.\nSecond sentence.\nThird sentence."
    blocks_path = _write_blocks(tmp_path, [("b1", block)])
    chunks = [_chunk("First sentence. Second sentence. Third sentence.", 0)]

    backfill_chunk_sidecars(chunks, blocks_path)

    assert _refs(chunks[0]) == ["b1"]


@pytest.mark.offline
def test_v_inserts_space_between_adjacent_sentences_still_matches(
    tmp_path: Path,
) -> None:
    # The source block has two sentences with NO whitespace at the boundary
    # ("sentence.Second"); the V SemanticChunker rejoins sentences with a single
    # space, inserting whitespace the source never had. Collapse-to-space matching
    # would mismatch and raise; whitespace-stripped matching must still locate it.
    blocks_path = _write_blocks(tmp_path, [("b1", "First sentence.Second sentence.")])
    chunks = [_chunk("First sentence. Second sentence.", 0)]

    backfill_chunk_sidecars(chunks, blocks_path)

    assert _refs(chunks[0]) == ["b1"]


@pytest.mark.offline
def test_v_reflow_across_blocks_keeps_boundary_refs(tmp_path: Path) -> None:
    # V reflows internal newlines to spaces and the chunk spans the block
    # boundary. Whitespace-stripped matching must still attribute both blocks.
    blocks_path = _write_blocks(
        tmp_path,
        [("b1", "Alpha line one.\nAlpha line two."), ("b2", "Beta block body.")],
    )
    chunks = [_chunk("Alpha line one. Alpha line two. Beta block body.", 0)]

    backfill_chunk_sidecars(chunks, blocks_path)

    assert _refs(chunks[0]) == ["b1", "b2"]


@pytest.mark.offline
def test_repeated_identical_blocks_map_to_distinct_blocks(tmp_path: Path) -> None:
    blocks_path = _write_blocks(
        tmp_path, [("b1", "Repeated text."), ("b2", "Repeated text.")]
    )
    chunks = [_chunk("Repeated text.", 0), _chunk("Repeated text.", 1)]

    backfill_chunk_sidecars(chunks, blocks_path)

    assert _refs(chunks[0]) == ["b1"]
    assert _refs(chunks[1]) == ["b2"]


@pytest.mark.offline
def test_phrase_recurring_inside_previous_chunk_resolves_forward(
    tmp_path: Path,
) -> None:
    # "common" appears inside b1; the next chunk's true home is b2. The forward
    # cursor must not re-match the occurrence inside the previous chunk.
    blocks_path = _write_blocks(
        tmp_path,
        [("b1", "common prefix and common middle"), ("b2", "common tail block")],
    )
    chunks = [
        _chunk("common prefix and common middle", 0),
        _chunk("common tail block", 1),
    ]

    backfill_chunk_sidecars(chunks, blocks_path)

    assert _refs(chunks[0]) == ["b1"]
    assert _refs(chunks[1]) == ["b2"]


@pytest.mark.offline
def test_overlapping_chunks_match_inside_previous_span(tmp_path: Path) -> None:
    # Simulate F/R token overlap: chunk 2 begins inside chunk 1's span. The
    # forward-from-start cursor resolves the overlap position directly.
    blocks_path = _write_blocks(
        tmp_path, [("b1", "one two three four five six seven eight")]
    )
    chunks = [
        _chunk("one two three four five", 0),
        _chunk("four five six seven eight", 1),
    ]

    backfill_chunk_sidecars(chunks, blocks_path)

    assert _refs(chunks[0]) == ["b1"]
    assert _refs(chunks[1]) == ["b1"]


@pytest.mark.offline
def test_overlap_chunk_with_later_duplicate_resolves_to_overlap(
    tmp_path: Path,
) -> None:
    # Regression: an overlapping chunk whose normalized text recurs LATER in the
    # document. The overlap chunk extends past the previous chunk's end (as real F/R
    # overlap chunks always do) AND its text reappears in a later block. The cursor
    # must resolve it to the *leftmost* end-advancing occurrence (the overlap inside
    # b1), not jump to the later b2 duplicate and strand the following chunk.
    #
    # Merged text (normalized): "xxyyzz" + "yyzz" = "xxyyzzyyzz".
    #   "yyzz" appears at the b1 overlap (offset 2) AND as b2 (offset 6).
    blocks_path = _write_blocks(tmp_path, [("b1", "xx yy zz"), ("b2", "yy zz")])
    chunks = [
        _chunk("xx yy", 0),  # b1
        _chunk("yy zz", 1),  # overlaps chunk 0 (shares "yy", adds "zz") -> still b1
        _chunk("yy zz", 2),  # the genuine b2 occurrence
    ]

    backfill_chunk_sidecars(chunks, blocks_path)

    assert _refs(chunks[0]) == ["b1"]
    assert _refs(chunks[1]) == ["b1"]
    assert _refs(chunks[2]) == ["b2"]


@pytest.mark.offline
def test_overlap_window_on_stripped_separator_advances_into_next_block(
    tmp_path: Path,
) -> None:
    # Regression for the reviewer's second case: b1="a", b2="abab", chunk_size=2,
    # overlap=1 yields (non-empty) chunks ["a", "a", "ab", "ba", "ab", "b"]. The token
    # overlap repeatedly falls on the stripped separator, so consecutive chunks can
    # share a normalized START — only the END advances. A start-anchored cursor
    # strands "ba" (its true position sits at/under the previous start) and raises;
    # the end-anchored cursor places every tail chunk inside b2.
    blocks_path = _write_blocks(tmp_path, [("b1", "a"), ("b2", "abab")])
    chunks = [
        _chunk("a", 0),  # b1
        _chunk("a", 1),  # window crossed the stripped separator -> b2
        _chunk("ab", 2),
        _chunk("ba", 3),
        _chunk("ab", 4),
        _chunk("b", 5),
    ]

    backfill_chunk_sidecars(chunks, blocks_path)

    assert _refs(chunks[0]) == ["b1"]
    for ch in chunks[1:]:
        assert _refs(ch) == ["b2"]


@pytest.mark.offline
def test_cross_block_artifact_from_stripping_is_rejected(tmp_path: Path) -> None:
    # Regression: identical adjacent blocks b1="aa", b2="aa" with no overlap yield
    # chunks ["aa", "aa"]. In the stripped projection "aaaa", "aa" also occurs at
    # offset 1, spanning b1's tail + b2's head across the removed separator — an
    # artifact of stripping, not a real chunk. The second "aa" must map to b2 alone,
    # so matching must prefer the single-block occurrence over the cross-block one.
    blocks_path = _write_blocks(tmp_path, [("b1", "aa"), ("b2", "aa")])
    chunks = [_chunk("aa", 0), _chunk("aa", 1)]

    backfill_chunk_sidecars(chunks, blocks_path)

    assert _refs(chunks[0]) == ["b1"]
    assert _refs(chunks[1]) == ["b2"]


@pytest.mark.offline
def test_genuine_cross_block_chunk_still_lists_both(tmp_path: Path) -> None:
    # A chunk whose content genuinely spans the boundary (its window held the
    # separator) has no single-block occurrence, so the cross-block fallback keeps it
    # and both blocks are listed — the single-block preference must not break this.
    blocks_path = _write_blocks(tmp_path, [("b1", "alpha"), ("b2", "beta")])
    merged = _BLOCK_SEPARATOR.join(["alpha", "beta"])
    chunks = [_chunk(merged, 0)]

    backfill_chunk_sidecars(chunks, blocks_path)

    assert _refs(chunks[0]) == ["b1", "b2"]


@pytest.mark.offline
def test_source_span_preferred_over_ambiguous_text(tmp_path: Path) -> None:
    # "ab\n\ncd" and "abcd" collapse to the same whitespace-stripped text,
    # but the chunk's private source span is authoritative.
    blocks_path = _write_blocks(tmp_path, [("b1", "ab"), ("b2", "cd"), ("b3", "abcd")])
    chunks = [
        {
            **_chunk("ab\n\ncd", 0),
            "_source_span": {"start": 0, "end": 6},
        },
        {
            **_chunk("abcd", 1),
            "_source_span": {"start": 8, "end": 12},
        },
    ]

    backfill_chunk_sidecars(chunks, blocks_path, require_source_span=True)

    assert _refs(chunks[0]) == ["b1", "b2"]
    assert _refs(chunks[1]) == ["b3"]


@pytest.mark.offline
def test_require_source_span_rejects_missing_span(tmp_path: Path) -> None:
    blocks_path = _write_blocks(tmp_path, [("b1", "Present content.")])
    chunks = [_chunk("Present content.", 4)]

    with pytest.raises(ChunkBlockMatchError) as exc:
        backfill_chunk_sidecars(chunks, blocks_path, require_source_span=True)

    assert exc.value.chunk_order_index == 4


@pytest.mark.offline
def test_require_source_span_rejects_invalid_span(tmp_path: Path) -> None:
    blocks_path = _write_blocks(tmp_path, [("b1", "Present content.")])
    chunks = [
        {
            **_chunk("Present content.", 5),
            "_source_span": {"start": 0, "end": 10_000},
        }
    ]

    with pytest.raises(ChunkBlockMatchError) as exc:
        backfill_chunk_sidecars(chunks, blocks_path, require_source_span=True)

    assert exc.value.chunk_order_index == 5


@pytest.mark.offline
def test_require_source_span_rejects_mismatched_span_text(tmp_path: Path) -> None:
    blocks_path = _write_blocks(tmp_path, [("b1", "Alpha."), ("b2", "Beta.")])
    chunks = [
        {
            **_chunk("Beta.", 6),
            "_source_span": {"start": 0, "end": 6},
        }
    ]

    with pytest.raises(ChunkBlockMatchError) as exc:
        backfill_chunk_sidecars(chunks, blocks_path, require_source_span=True)

    assert exc.value.chunk_order_index == 6


@pytest.mark.offline
def test_unmatchable_chunk_raises(tmp_path: Path) -> None:
    blocks_path = _write_blocks(tmp_path, [("b1", "Present content.")])
    chunks = [_chunk("Absent content not in any block.", 3)]

    with pytest.raises(ChunkBlockMatchError) as exc:
        backfill_chunk_sidecars(chunks, blocks_path)
    assert exc.value.chunk_order_index == 3


@pytest.mark.offline
def test_empty_blocks_path_is_noop(tmp_path: Path) -> None:
    chunks = [_chunk("anything", 0)]
    backfill_chunk_sidecars(chunks, "")
    assert "sidecar" not in chunks[0]


@pytest.mark.offline
def test_meta_only_file_is_noop(tmp_path: Path) -> None:
    blocks_path = _write_blocks(tmp_path, [])
    chunks = [_chunk("anything", 0)]
    backfill_chunk_sidecars(chunks, blocks_path)
    assert "sidecar" not in chunks[0]


@pytest.mark.offline
def test_unreadable_path_is_noop(tmp_path: Path) -> None:
    chunks = [_chunk("anything", 0)]
    backfill_chunk_sidecars(chunks, str(tmp_path / "does_not_exist.blocks.jsonl"))
    assert "sidecar" not in chunks[0]


@pytest.mark.offline
def test_existing_sidecar_is_preserved(tmp_path: Path) -> None:
    blocks_path = _write_blocks(tmp_path, [("b1", "Alpha."), ("b2", "Beta.")])
    # A P/mm-style chunk whose content would NOT match block order; it must be
    # left untouched because it already carries a valid sidecar.
    pre = {
        "content": "Beta.",
        "tokens": 1,
        "chunk_order_index": 0,
        "sidecar": {
            "type": "drawing",
            "id": "im-xyz-0001",
            "refs": [{"type": "drawing", "id": "im-xyz-0001"}],
        },
    }
    chunks = [pre]
    backfill_chunk_sidecars(chunks, blocks_path)
    assert chunks[0]["sidecar"]["type"] == "drawing"
    assert chunks[0]["sidecar"]["id"] == "im-xyz-0001"


@pytest.mark.offline
def test_multimodal_tag_block_matches_verbatim(tmp_path: Path) -> None:
    block = 'See: <table id="tb-abc-0001" format="json">[[1,2]]</table> done.'
    blocks_path = _write_blocks(tmp_path, [("b1", block)])
    chunks = [_chunk(block, 0)]

    backfill_chunk_sidecars(chunks, blocks_path)

    assert _refs(chunks[0]) == ["b1"]


@pytest.mark.offline
def test_whitespace_only_block_excluded_from_refs(tmp_path: Path) -> None:
    # The middle row is whitespace-only -> not part of the merged text and never
    # contributes a ref. A chunk spanning b1..b3 lists only b1 and b3.
    blocks_path = _write_blocks(
        tmp_path, [("b1", "First."), ("bws", "   "), ("b3", "Third.")]
    )
    merged = _BLOCK_SEPARATOR.join(["First.", "Third."])
    chunks = [_chunk(merged, 0)]

    backfill_chunk_sidecars(chunks, blocks_path)

    assert _refs(chunks[0]) == ["b1", "b3"]


@pytest.mark.offline
def test_empty_chunk_skipped_then_following_matches(tmp_path: Path) -> None:
    blocks_path = _write_blocks(tmp_path, [("b1", "Real content.")])
    chunks = [_chunk("", 0), _chunk("Real content.", 1)]

    backfill_chunk_sidecars(chunks, blocks_path)

    assert "sidecar" not in chunks[0]
    assert _refs(chunks[1]) == ["b1"]


@pytest.mark.offline
def test_replacement_char_chunk_skipped_not_failed(tmp_path: Path) -> None:
    # A multi-byte UTF-8 char split at a token-window boundary decodes to U+FFFD in
    # both the chunk content and its span probe, so the chunk is unlocatable by any
    # means. Under require_source_span it must be SKIPPED (no sidecar), not raise and
    # FAIL the whole document; the following clean chunk still maps correctly.
    b1, b2 = "Status update 🎉 done.", "Clean tail block."
    blocks_path = _write_blocks(tmp_path, [("b1", b1), ("b2", b2)])
    merged = _BLOCK_SEPARATOR.join([b1, b2])
    b2_start = merged.index(b2)
    chunks = [
        # First chunk lost a byte at the emoji boundary -> U+FFFD, no usable span.
        _chunk("Status update � done.", 0),
        # The clean chunk carries a valid span (as the real chunker would emit), so
        # the strict contract is preserved for it.
        {
            **_chunk(b2, 1),
            "_source_span": {"start": b2_start, "end": b2_start + len(b2)},
        },
    ]

    # Must not raise even with the strict span contract.
    backfill_chunk_sidecars(chunks, blocks_path, require_source_span=True)

    assert "sidecar" not in chunks[0]  # provenance degraded for the corrupt chunk
    assert _refs(chunks[1]) == ["b2"]  # clean chunk still resolves


@pytest.mark.offline
def test_replacement_char_chunk_skipped_in_text_fallback(tmp_path: Path) -> None:
    # Same degradation in the non-required (text-matching) path: a U+FFFD chunk is
    # skipped rather than raising ChunkBlockMatchError.
    blocks_path = _write_blocks(tmp_path, [("b1", "Alpha 🚀 beta.")])
    chunks = [_chunk("Alpha � beta.", 0)]

    backfill_chunk_sidecars(chunks, blocks_path)

    assert "sidecar" not in chunks[0]
