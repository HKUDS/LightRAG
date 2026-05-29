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
    # "common" appears inside b1; the next chunk's true home is b2. End-first
    # search must not re-match the occurrence inside the previous chunk.
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
def test_overlapping_chunks_match_via_start_fallback(tmp_path: Path) -> None:
    # Simulate F/R token overlap: chunk 2 begins inside chunk 1's span.
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
