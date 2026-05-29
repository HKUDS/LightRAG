"""Integration tests: real F chunker + hard-split + sidecar backfill + chunks dict.

Unlike ``tests/sidecar/test_backfill.py`` (which hand-builds chunk lists), these
drive the actual ``chunking_by_fixed_token`` chunker over the exact merged text a
parsed document would yield, then apply the real
``enforce_chunk_token_limit_before_embedding`` hard-split and
``build_chunks_dict_from_chunking_result`` persistence step — verifying that
backfilled sidecars are precise per final slice and survive into the chunks dict.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lightrag.chunker import chunking_by_fixed_token
from lightrag.sidecar import backfill_chunk_sidecars
from lightrag.utils import (
    Tokenizer,
    enforce_chunk_token_limit_before_embedding,
)
from lightrag.utils_pipeline import build_chunks_dict_from_chunking_result

_BLOCK_SEPARATOR = "\n\n"


class _CharTokenizerImpl:
    """Deterministic char-per-token tokenizer; decode(encode(x)) == x so F
    chunks are verbatim substrings and token sizes are character counts."""

    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


def _tokenizer() -> Tokenizer:
    return Tokenizer("char", _CharTokenizerImpl())


def _write_blocks(tmp_path: Path, blocks: list[tuple[str, str]]) -> tuple[str, str]:
    """Write blocks.jsonl; return (path, merged_text)."""
    path = tmp_path / "doc.blocks.jsonl"
    lines = [json.dumps({"type": "meta", "format": "lightrag", "version": "1.0"})]
    parts: list[str] = []
    for blockid, content in blocks:
        lines.append(
            json.dumps(
                {
                    "type": "content",
                    "blockid": blockid,
                    "content": content,
                    "heading": "",
                    "parent_headings": [],
                    "level": 1,
                }
            )
        )
        if content.strip():
            parts.append(content)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path), _BLOCK_SEPARATOR.join(parts)


@pytest.mark.offline
def test_real_fixed_token_chunks_get_full_coverage(tmp_path: Path) -> None:
    blocks = [(f"b{i}", f"Block number {i} body content here.") for i in range(6)]
    blocks_path, merged = _write_blocks(tmp_path, blocks)
    tok = _tokenizer()

    chunks = chunking_by_fixed_token(
        tok, merged, chunk_token_size=40, chunk_overlap_token_size=5
    )
    assert len(chunks) >= 2  # multiple blocks per chunk at this size

    backfill_chunk_sidecars(chunks, blocks_path)

    # Every chunk is located and carries block provenance.
    for ch in chunks:
        assert ch["sidecar"]["type"] == "block"
        assert ch["sidecar"]["refs"]
    # Union of all refs covers every block.
    seen = {r["id"] for ch in chunks for r in ch["sidecar"]["refs"]}
    assert seen == {b for b, _ in blocks}


@pytest.mark.offline
def test_hard_split_slices_get_precise_refs(tmp_path: Path) -> None:
    # One large block followed by a small one. A single fixed-token chunk covers
    # both; the hard-split then breaks the big-content chunk into slices that
    # each lie inside one block, so refs must NOT all inherit the full set.
    big = "A" * 120
    blocks_path, merged = _write_blocks(tmp_path, [("big", big), ("small", "tail.")])
    tok = _tokenizer()

    chunks = chunking_by_fixed_token(
        tok, merged, chunk_token_size=200, chunk_overlap_token_size=0
    )
    assert len(chunks) == 1  # everything in one chunk pre-split

    chunks = enforce_chunk_token_limit_before_embedding(chunks, tok, max_tokens=30)
    assert len(chunks) > 1  # hard-split fired

    backfill_chunk_sidecars(chunks, blocks_path)

    # The early slices live entirely inside "big" -> single ref, not both blocks.
    assert chunks[0]["sidecar"]["refs"] == [{"type": "block", "id": "big"}]
    # "small" is referenced only by the slice that actually reaches its content.
    small_refs = [
        ch
        for ch in chunks
        if any(r["id"] == "small" for r in ch["sidecar"]["refs"])
    ]
    assert len(small_refs) == 1


@pytest.mark.offline
def test_backfilled_sidecar_persists_into_chunks_dict(tmp_path: Path) -> None:
    blocks_path, merged = _write_blocks(
        tmp_path, [("b1", "Alpha body."), ("b2", "Beta body.")]
    )
    tok = _tokenizer()

    chunks = chunking_by_fixed_token(tok, merged, chunk_token_size=200)
    backfill_chunk_sidecars(chunks, blocks_path)

    chunks_dict = build_chunks_dict_from_chunking_result(
        chunks, doc_id="doc-xyz", file_path="doc.docx"
    )
    assert chunks_dict  # non-empty
    for record in chunks_dict.values():
        assert "sidecar" in record
        assert record["sidecar"]["type"] == "block"
        assert record["sidecar"]["refs"]
