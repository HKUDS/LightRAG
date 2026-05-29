"""Unit tests for :func:`lightrag.sidecar.backfill.backfill_chunk_sidecars`.

These exercise the F/R/V sidecar backfill in isolation: a small ``blocks.jsonl``
is written to ``tmp_path`` and hand-built chunk lists are matched against it, so
no real chunker, tokenizer, or embedding is needed.

Backfill is span-first: each F/R/V chunk carries a private ``_source_span`` (char
offsets into the reconstructed merged text) and is mapped to the block(s) that span
overlaps. A chunk without a usable span FAILs the document, except the inherently
unlocatable replacement-char case, which degrades to no-sidecar.
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
    """A sidecar-less chunk WITHOUT a source span (FAILs unless it is unlocatable)."""
    return {"content": content, "tokens": len(content), "chunk_order_index": order}


def _span_chunk(content: str, order: int, start: int, end: int) -> dict:
    """A chunk carrying the ``_source_span`` a real F/R/V chunker would emit."""
    return {**_chunk(content, order), "_source_span": {"start": start, "end": end}}


def _refs(chunk: dict) -> list[str]:
    return [r["id"] for r in chunk["sidecar"]["refs"]]


@pytest.mark.offline
def test_span_within_single_block_maps_to_it(tmp_path: Path) -> None:
    blocks_path = _write_blocks(
        tmp_path, [("b1", "Alpha paragraph."), ("b2", "Beta paragraph.")]
    )
    merged = _BLOCK_SEPARATOR.join(["Alpha paragraph.", "Beta paragraph."])
    start = merged.index("Beta paragraph.")
    chunks = [_span_chunk("Beta paragraph.", 0, start, start + len("Beta paragraph."))]

    backfill_chunk_sidecars(chunks, blocks_path)

    assert chunks[0]["sidecar"]["id"] == "b2"
    assert _refs(chunks[0]) == ["b2"]


@pytest.mark.offline
def test_span_spanning_two_blocks_lists_both_refs(tmp_path: Path) -> None:
    blocks_path = _write_blocks(
        tmp_path, [("b1", "Alpha paragraph."), ("b2", "Beta paragraph.")]
    )
    merged = _BLOCK_SEPARATOR.join(["Alpha paragraph.", "Beta paragraph."])
    chunks = [_span_chunk(merged, 0, 0, len(merged))]

    backfill_chunk_sidecars(chunks, blocks_path)

    assert chunks[0]["sidecar"]["type"] == "block"
    assert chunks[0]["sidecar"]["id"] == "b1"
    assert _refs(chunks[0]) == ["b1", "b2"]


@pytest.mark.offline
def test_v_reflowed_content_matches_span_via_whitespace_strip(tmp_path: Path) -> None:
    # The block has internal newlines; the V chunker rejoins sentences with single
    # spaces, so the chunk body is not byte-verbatim against its source span. Span
    # validation must accept the whitespace-stripped match.
    block = "First sentence.\nSecond sentence.\nThird sentence."
    blocks_path = _write_blocks(tmp_path, [("b1", block)])
    chunks = [
        _span_chunk(
            "First sentence. Second sentence. Third sentence.", 0, 0, len(block)
        )
    ]

    backfill_chunk_sidecars(chunks, blocks_path)

    assert _refs(chunks[0]) == ["b1"]


@pytest.mark.offline
def test_v_inserts_space_between_adjacent_sentences_still_matches(
    tmp_path: Path,
) -> None:
    # The source block has two sentences with NO whitespace at the boundary
    # ("sentence.Second"); V rejoins them with a single space, inserting whitespace
    # the source never had. Collapse-to-single-space validation would mismatch;
    # whitespace-stripped validation must still accept the span.
    block = "First sentence.Second sentence."
    blocks_path = _write_blocks(tmp_path, [("b1", block)])
    chunks = [_span_chunk("First sentence. Second sentence.", 0, 0, len(block))]

    backfill_chunk_sidecars(chunks, blocks_path)

    assert _refs(chunks[0]) == ["b1"]


@pytest.mark.offline
def test_v_reflow_across_blocks_keeps_boundary_refs(tmp_path: Path) -> None:
    # V reflows internal newlines to spaces and the chunk spans the block boundary;
    # the span covers both blocks (and the separator), so both are attributed.
    blocks_path = _write_blocks(
        tmp_path,
        [("b1", "Alpha line one.\nAlpha line two."), ("b2", "Beta block body.")],
    )
    merged = _BLOCK_SEPARATOR.join(
        ["Alpha line one.\nAlpha line two.", "Beta block body."]
    )
    chunks = [
        _span_chunk(
            "Alpha line one. Alpha line two. Beta block body.", 0, 0, len(merged)
        )
    ]

    backfill_chunk_sidecars(chunks, blocks_path)

    assert _refs(chunks[0]) == ["b1", "b2"]


@pytest.mark.offline
def test_repeated_identical_blocks_map_via_distinct_spans(tmp_path: Path) -> None:
    # Identical content in two blocks; the span — not the text — disambiguates which
    # block each chunk belongs to.
    blocks_path = _write_blocks(
        tmp_path, [("b1", "Repeated text."), ("b2", "Repeated text.")]
    )
    merged = _BLOCK_SEPARATOR.join(["Repeated text.", "Repeated text."])
    b2_start = merged.index("Repeated text.", 1)
    chunks = [
        _span_chunk("Repeated text.", 0, 0, len("Repeated text.")),
        _span_chunk("Repeated text.", 1, b2_start, b2_start + len("Repeated text.")),
    ]

    backfill_chunk_sidecars(chunks, blocks_path)

    assert _refs(chunks[0]) == ["b1"]
    assert _refs(chunks[1]) == ["b2"]


@pytest.mark.offline
def test_source_span_disambiguates_cross_block_from_later_duplicate(
    tmp_path: Path,
) -> None:
    # Regression: "ab\n\ncd" and a later "abcd" block collapse to the same
    # whitespace-stripped text. The old text-matching fallback would prefer the later
    # single-block "abcd" and strand the real cross-block chunk; the span is
    # authoritative and resolves each chunk to its true block(s).
    blocks_path = _write_blocks(tmp_path, [("b1", "ab"), ("b2", "cd"), ("b3", "abcd")])
    chunks = [
        _span_chunk("ab\n\ncd", 0, 0, 6),
        _span_chunk("abcd", 1, 8, 12),
    ]

    backfill_chunk_sidecars(chunks, blocks_path)

    assert _refs(chunks[0]) == ["b1", "b2"]
    assert _refs(chunks[1]) == ["b3"]


@pytest.mark.offline
def test_missing_span_fails_document(tmp_path: Path) -> None:
    blocks_path = _write_blocks(tmp_path, [("b1", "Present content.")])
    chunks = [_chunk("Present content.", 4)]

    with pytest.raises(ChunkBlockMatchError) as exc:
        backfill_chunk_sidecars(chunks, blocks_path)

    assert exc.value.chunk_order_index == 4


@pytest.mark.offline
def test_out_of_range_span_fails_document(tmp_path: Path) -> None:
    blocks_path = _write_blocks(tmp_path, [("b1", "Present content.")])
    chunks = [_span_chunk("Present content.", 5, 0, 10_000)]

    with pytest.raises(ChunkBlockMatchError) as exc:
        backfill_chunk_sidecars(chunks, blocks_path)

    assert exc.value.chunk_order_index == 5


@pytest.mark.offline
def test_mismatched_span_text_fails_document(tmp_path: Path) -> None:
    # The span points at "Alpha." but the chunk content is "Beta." — neither a
    # byte-exact nor a whitespace-stripped match, so the span is treated as absent.
    blocks_path = _write_blocks(tmp_path, [("b1", "Alpha."), ("b2", "Beta.")])
    chunks = [_span_chunk("Beta.", 6, 0, 6)]

    with pytest.raises(ChunkBlockMatchError) as exc:
        backfill_chunk_sidecars(chunks, blocks_path)

    assert exc.value.chunk_order_index == 6


@pytest.mark.offline
def test_empty_blocks_path_is_noop(tmp_path: Path) -> None:
    chunks = [_span_chunk("anything", 0, 0, 8)]
    backfill_chunk_sidecars(chunks, "")
    assert "sidecar" not in chunks[0]


@pytest.mark.offline
def test_meta_only_file_is_noop(tmp_path: Path) -> None:
    blocks_path = _write_blocks(tmp_path, [])
    chunks = [_span_chunk("anything", 0, 0, 8)]
    backfill_chunk_sidecars(chunks, blocks_path)
    assert "sidecar" not in chunks[0]


@pytest.mark.offline
def test_unreadable_path_is_noop(tmp_path: Path) -> None:
    chunks = [_span_chunk("anything", 0, 0, 8)]
    backfill_chunk_sidecars(chunks, str(tmp_path / "does_not_exist.blocks.jsonl"))
    assert "sidecar" not in chunks[0]


@pytest.mark.offline
def test_existing_sidecar_is_preserved(tmp_path: Path) -> None:
    blocks_path = _write_blocks(tmp_path, [("b1", "Alpha."), ("b2", "Beta.")])
    # A P/mm-style chunk that already carries a valid sidecar must be left untouched,
    # even though it has no source span.
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
def test_multimodal_tag_span_matches_verbatim(tmp_path: Path) -> None:
    block = 'See: <table id="tb-abc-0001" format="json">[[1,2]]</table> done.'
    blocks_path = _write_blocks(tmp_path, [("b1", block)])
    chunks = [_span_chunk(block, 0, 0, len(block))]

    backfill_chunk_sidecars(chunks, blocks_path)

    assert _refs(chunks[0]) == ["b1"]


@pytest.mark.offline
def test_whitespace_only_block_excluded_from_refs(tmp_path: Path) -> None:
    # The middle row is whitespace-only -> not part of the merged text and never
    # contributes a ref. A chunk whose span covers b1..b3 lists only b1 and b3.
    blocks_path = _write_blocks(
        tmp_path, [("b1", "First."), ("bws", "   "), ("b3", "Third.")]
    )
    merged = _BLOCK_SEPARATOR.join(["First.", "Third."])
    chunks = [_span_chunk(merged, 0, 0, len(merged))]

    backfill_chunk_sidecars(chunks, blocks_path)

    assert _refs(chunks[0]) == ["b1", "b3"]


@pytest.mark.offline
def test_empty_chunk_skipped_then_following_matches(tmp_path: Path) -> None:
    blocks_path = _write_blocks(tmp_path, [("b1", "Real content.")])
    chunks = [_chunk("", 0), _span_chunk("Real content.", 1, 0, len("Real content."))]

    backfill_chunk_sidecars(chunks, blocks_path)

    assert "sidecar" not in chunks[0]
    assert _refs(chunks[1]) == ["b1"]


@pytest.mark.offline
def test_replacement_char_chunk_skipped_not_failed(tmp_path: Path) -> None:
    # A multi-byte UTF-8 char split at a token-window boundary decodes to U+FFFD in
    # both the chunk content and its span probe, so the chunk is unlocatable by any
    # means. It must be SKIPPED (no sidecar), not raise and FAIL the whole document;
    # the following clean chunk (with a valid span) still maps correctly.
    b1, b2 = "Status update 🎉 done.", "Clean tail block."
    blocks_path = _write_blocks(tmp_path, [("b1", b1), ("b2", b2)])
    merged = _BLOCK_SEPARATOR.join([b1, b2])
    b2_start = merged.index(b2)
    chunks = [
        # First chunk lost a byte at the emoji boundary -> U+FFFD, no usable span.
        _chunk("Status update � done.", 0),
        _span_chunk(b2, 1, b2_start, b2_start + len(b2)),
    ]

    backfill_chunk_sidecars(chunks, blocks_path)

    assert "sidecar" not in chunks[0]  # provenance degraded for the corrupt chunk
    assert _refs(chunks[1]) == ["b2"]  # clean chunk still resolves
