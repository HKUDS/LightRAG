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
def test_real_overlap_tail_chunk_maps_to_next_block(tmp_path: Path) -> None:
    # End-to-end reproduction of the overlap-tail ambiguity with the real chunker.
    # Simple case: b1="aa", b2="a", chunk_size=3, overlap=1 -> ["aa", "a", "a"]. The
    # middle window [2:5] = "\n\na" strips to b2's "a" (the overlap landed on the
    # separator), so the tail chunks belong to b2 — not the earlier "a" inside b1.
    blocks_path, merged = _write_blocks(tmp_path, [("b1", "aa"), ("b2", "a")])
    tok = _tokenizer()

    chunks = chunking_by_fixed_token(
        tok,
        merged,
        chunk_token_size=3,
        chunk_overlap_token_size=1,
        _emit_source_span=True,
    )
    assert [c["content"] for c in chunks] == ["aa", "a", "a"]

    backfill_chunk_sidecars(chunks, blocks_path)

    assert [r["id"] for r in chunks[0]["sidecar"]["refs"]] == ["b1"]
    assert [r["id"] for r in chunks[1]["sidecar"]["refs"]] == ["b2"]
    assert [r["id"] for r in chunks[2]["sidecar"]["refs"]] == ["b2"]


@pytest.mark.offline
def test_real_overlap_on_stripped_separator_does_not_strand_chunks(
    tmp_path: Path,
) -> None:
    # Harder end-to-end case: b1="a", b2="abab", chunk_size=2, overlap=1 ->
    # ["a", "", "a", "ab", "ba", "ab", "b"]. The token overlap repeatedly lands on the
    # stripped separator, so consecutive non-empty chunks can share a normalized start
    # and only the end advances. The empty chunk is skipped; every other tail chunk
    # must resolve into b2 without raising ChunkBlockMatchError.
    blocks_path, merged = _write_blocks(tmp_path, [("b1", "a"), ("b2", "abab")])
    tok = _tokenizer()

    chunks = chunking_by_fixed_token(
        tok,
        merged,
        chunk_token_size=2,
        chunk_overlap_token_size=1,
        _emit_source_span=True,
    )
    assert [c["content"] for c in chunks] == ["a", "", "a", "ab", "ba", "ab", "b"]

    backfill_chunk_sidecars(chunks, blocks_path)

    assert [r["id"] for r in chunks[0]["sidecar"]["refs"]] == ["b1"]
    # The empty chunk is skipped (no sidecar); all remaining chunks belong to b2.
    assert "sidecar" not in chunks[1]
    for ch in chunks[2:]:
        assert [r["id"] for r in ch["sidecar"]["refs"]] == ["b2"]


@pytest.mark.offline
def test_real_identical_adjacent_blocks_no_cross_block_artifact(
    tmp_path: Path,
) -> None:
    # End-to-end: identical adjacent blocks b1="aa", b2="aa", chunk_size=2, overlap=0
    # -> ["aa", "", "aa"]. Stripping glues them to "aaaa", where "aa" also matches at
    # offset 1 across the (removed) separator. The final chunk must reference only b2,
    # not spuriously span both blocks.
    blocks_path, merged = _write_blocks(tmp_path, [("b1", "aa"), ("b2", "aa")])
    tok = _tokenizer()

    chunks = chunking_by_fixed_token(
        tok,
        merged,
        chunk_token_size=2,
        chunk_overlap_token_size=0,
        _emit_source_span=True,
    )
    assert [c["content"] for c in chunks] == ["aa", "", "aa"]

    backfill_chunk_sidecars(chunks, blocks_path)

    assert chunks[0]["_source_span"] == {"start": 0, "end": 2}
    assert chunks[2]["_source_span"] == {"start": 4, "end": 6}

    assert [r["id"] for r in chunks[0]["sidecar"]["refs"]] == ["b1"]
    assert "sidecar" not in chunks[1]
    assert [r["id"] for r in chunks[2]["sidecar"]["refs"]] == ["b2"]


@pytest.mark.offline
def test_real_fixed_token_chunks_get_full_coverage(tmp_path: Path) -> None:
    blocks = [(f"b{i}", f"Block number {i} body content here.") for i in range(6)]
    blocks_path, merged = _write_blocks(tmp_path, blocks)
    tok = _tokenizer()

    chunks = chunking_by_fixed_token(
        tok,
        merged,
        chunk_token_size=40,
        chunk_overlap_token_size=5,
        _emit_source_span=True,
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
        tok,
        merged,
        chunk_token_size=200,
        chunk_overlap_token_size=0,
        _emit_source_span=True,
    )
    assert len(chunks) == 1  # everything in one chunk pre-split
    assert chunks[0]["_source_span"] == {"start": 0, "end": len(merged)}

    chunks = enforce_chunk_token_limit_before_embedding(chunks, tok, max_tokens=30)
    assert len(chunks) > 1  # hard-split fired

    backfill_chunk_sidecars(chunks, blocks_path)

    # The early slices live entirely inside "big" -> single ref, not both blocks.
    assert chunks[0]["sidecar"]["refs"] == [{"type": "block", "id": "big"}]
    # "small" is referenced only by the slice that actually reaches its content.
    small_refs = [
        ch for ch in chunks if any(r["id"] == "small" for r in ch["sidecar"]["refs"])
    ]
    assert len(small_refs) == 1


@pytest.mark.offline
def test_hard_split_multi_sentence_rejoin_keeps_provenance(tmp_path: Path) -> None:
    # A single block whose sentences are separated by single spaces. The hard
    # split regroups whole sentence units and rejoins them with "\n\n", so the
    # resulting slice content is NOT byte-verbatim in the source. Span propagation
    # must fall back to whitespace-normalized matching instead of dropping the span
    # — otherwise span-first backfill would wrongly FAIL the document.
    block = "ab. cd. ef. gh. ij. kl."
    blocks_path, merged = _write_blocks(tmp_path, [("b1", block)])
    tok = _tokenizer()

    chunks = chunking_by_fixed_token(
        tok,
        merged,
        chunk_token_size=200,
        chunk_overlap_token_size=0,
        _emit_source_span=True,
    )
    assert len(chunks) == 1
    assert chunks[0]["_source_span"] == {"start": 0, "end": len(merged)}

    chunks = enforce_chunk_token_limit_before_embedding(chunks, tok, max_tokens=7)
    assert len(chunks) > 1  # hard split fired into multiple slices
    # At least one slice rejoined sentence units with "\n\n" (not byte-verbatim),
    # which is exactly the case the normalized span fallback must cover.
    assert any("\n\n" in ch["content"] for ch in chunks)

    # Must NOT raise: every slice keeps a span via the normalized fallback, and
    # each maps back to the single source block it came from.
    backfill_chunk_sidecars(chunks, blocks_path)

    for ch in chunks:
        assert [r["id"] for r in ch["sidecar"]["refs"]] == ["b1"]


@pytest.mark.offline
def test_real_tiktoken_token_windows_match_verbatim(tmp_path: Path) -> None:
    # The char tokenizer guarantees decode(encode(x)) == x; a real BPE tokenizer
    # does not split on character boundaries, so this exercises that tiktoken's
    # decoded token windows (after the chunker's .strip()) are still locatable in
    # the reconstructed merged text — including across the token-overlap fallback.
    tiktoken = pytest.importorskip("tiktoken")
    del tiktoken  # only needed to gate the test
    from lightrag.utils import TiktokenTokenizer

    tok = TiktokenTokenizer()
    blocks = [
        ("b1", "The quick brown fox jumps over the lazy dog repeatedly."),
        ("b2", "Pack my box with five dozen liquor jugs, said the printer."),
        ("b3", "How vexingly quick daft zebras jump across the meadow."),
    ]
    blocks_path, merged = _write_blocks(tmp_path, blocks)

    chunks = chunking_by_fixed_token(
        tok,
        merged,
        chunk_token_size=12,
        chunk_overlap_token_size=3,
        _emit_source_span=True,
    )
    assert len(chunks) >= 2  # small window + overlap -> multiple chunks

    backfill_chunk_sidecars(chunks, blocks_path)

    for ch in chunks:
        assert ch["sidecar"]["type"] == "block"
        assert ch["sidecar"]["refs"]
    # Union of all refs covers every block.
    seen = {r["id"] for ch in chunks for r in ch["sidecar"]["refs"]}
    assert seen == {b for b, _ in blocks}


@pytest.mark.offline
def test_real_tiktoken_multibyte_boundary_degrades_not_fails(tmp_path: Path) -> None:
    # Regression: tiktoken is byte-level, so a 4-byte UTF-8 char (emoji / rare CJK
    # extension) can have its bytes split across a token-window boundary. Decoding the
    # partial window yields U+FFFD in BOTH the chunk content and its span probe, so the
    # chunk is unlocatable by span or by text. Span-first backfill must skip provenance
    # for the corrupt chunks while still attributing the clean ones, not FAIL the whole
    # document.
    pytest.importorskip("tiktoken")
    from lightrag.utils import TiktokenTokenizer

    tok = TiktokenTokenizer()
    # Emoji are supplementary-plane (4-byte) chars that force byte-fallback tokens.
    block = "Status update 🎉🚀 progress 😀😁😂 and more text 🔥💡✅ keep going. " * 20
    blocks_path, merged = _write_blocks(tmp_path, [("b1", block)])

    chunks = chunking_by_fixed_token(
        tok,
        merged,
        chunk_token_size=18,
        chunk_overlap_token_size=4,
        _emit_source_span=True,
    )
    # The window splits at least one emoji -> some chunks carry U+FFFD and lack a span.
    assert any("�" in c["content"] for c in chunks)
    assert any("_source_span" not in c for c in chunks)

    # Must NOT raise: corrupt chunks are skipped, the rest are attributed.
    backfill_chunk_sidecars(chunks, blocks_path)

    for ch in chunks:
        if "�" in ch["content"]:
            assert "sidecar" not in ch  # provenance degraded, document not failed
        elif ch["content"].strip():  # empty tail chunks are skipped entirely
            assert ch["sidecar"]["refs"] == [{"type": "block", "id": "b1"}]
    # At least the clean chunks resolved into the single source block.
    assert any("sidecar" in ch for ch in chunks)


@pytest.mark.offline
def test_backfilled_sidecar_persists_into_chunks_dict(tmp_path: Path) -> None:
    blocks_path, merged = _write_blocks(
        tmp_path, [("b1", "Alpha body."), ("b2", "Beta body.")]
    )
    tok = _tokenizer()

    chunks = chunking_by_fixed_token(
        tok, merged, chunk_token_size=200, _emit_source_span=True
    )
    backfill_chunk_sidecars(chunks, blocks_path)

    chunks_dict = build_chunks_dict_from_chunking_result(
        chunks, doc_id="doc-xyz", file_path="doc.docx"
    )
    assert chunks_dict  # non-empty
    for record in chunks_dict.values():
        assert "sidecar" in record
        assert "_source_span" not in record
        assert record["sidecar"]["type"] == "block"
        assert record["sidecar"]["refs"]


@pytest.mark.offline
def test_span_first_disambiguates_repeated_cross_block_text(tmp_path: Path) -> None:
    # Latest pathological case: the real first chunk is b1 + separator + b2
    # ("ab\n\ncd"), but stripping whitespace makes it textually equal to b3
    # ("abcd"). Span-first backfill must map by source coverage, not by the
    # ambiguous stripped string.
    blocks_path, merged = _write_blocks(
        tmp_path, [("b1", "ab"), ("b2", "cd"), ("b3", "abcd")]
    )
    tok = _tokenizer()

    chunks = chunking_by_fixed_token(
        tok,
        merged,
        chunk_token_size=6,
        chunk_overlap_token_size=0,
        _emit_source_span=True,
    )
    assert [c["content"] for c in chunks] == ["ab\n\ncd", "abcd"]

    backfill_chunk_sidecars(chunks, blocks_path)

    assert [r["id"] for r in chunks[0]["sidecar"]["refs"]] == ["b1", "b2"]
    assert [r["id"] for r in chunks[1]["sidecar"]["refs"]] == ["b3"]


@pytest.mark.offline
def test_split_by_character_only_emits_exact_source_spans() -> None:
    tok = _tokenizer()
    content = "  alpha|beta|gamma  "

    chunks = chunking_by_fixed_token(
        tok,
        content,
        chunk_token_size=20,
        split_by_character="|",
        split_by_character_only=True,
        _emit_source_span=True,
    )

    assert [c["content"] for c in chunks] == ["alpha", "beta", "gamma"]
    assert [c["_source_span"] for c in chunks] == [
        {"start": 2, "end": 7},
        {"start": 8, "end": 12},
        {"start": 13, "end": 18},
    ]
