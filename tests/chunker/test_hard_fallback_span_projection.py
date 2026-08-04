"""Regression tests for ``enforce_chunk_token_limit_before_embedding``'s
two-stage ``_source_span`` projection (design: "Tokenizer safe splitting and
truncation refactor").

A hard-split child's span is normally computed by adding the parent's own
``_source_span.start`` to the child's local offset within the parent's
``content``. That arithmetic is only valid when the parent's ``content`` is a
byte-verbatim slice of the original document text (``parent_span.end -
parent_span.start == len(parent_content)``); the V chunking strategy rejoins
sentences with a single space, so it is not always true. These tests pin the
two-stage whitespace-normalized projection that replaces the naive add for
that case, and its explicit fail-fast/soft-drop rules.
"""

from __future__ import annotations

import pytest

from lightrag.exceptions import ChunkBlockMatchError
from lightrag.utils import (
    Tokenizer,
    TokenizerInterface,
    enforce_chunk_token_limit_before_embedding,
)

pytestmark = pytest.mark.offline


class _CharTokenizer(TokenizerInterface):
    def encode(self, content: str) -> list[int]:
        return [ord(ch) % 1000 for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


def _tok() -> Tokenizer:
    return Tokenizer("char", _CharTokenizer())


def test_projection_fixes_the_double_space_regression_example():
    """Pinned example from the design doc: the source document has TWO
    separate U+0020 spaces between "a" and "b" (length 4), but the parent
    chunk's own content was whitespace-normalized down to a single space
    (length 3). The naive ``parent_start + local_offset`` addition would
    point one character too early -- at the space, not "b".
    """
    document = "a" + " " + " " + "b"
    parent_content = "a" + " " + "b"
    assert len(document) == 4
    assert len(parent_content) == 3

    chunking_result = [
        {
            "content": parent_content,
            "tokens": len(_tok().encode(parent_content)),
            "chunk_order_index": 0,
            "chunk_id": "c0",
            "_source_span": {"start": 0, "end": len(document)},
        }
    ]

    out = enforce_chunk_token_limit_before_embedding(
        chunking_result,
        _tok(),
        max_tokens=1,
        overlap_tokens=0,
        source_content=document,
    )

    by_content = {dp["content"]: dp.get("_source_span") for dp in out}
    b_span = by_content.get("b")
    assert b_span is not None, f"expected a 'b' slice among {list(by_content)}"
    assert (b_span["start"], b_span["end"]) == (3, 4)
    assert document[b_span["start"] : b_span["end"]] == "b"


def test_missing_source_content_drops_child_spans_without_raising():
    """A caller with no provenance concept at all (e.g. ainsert_custom_chunks)
    can omit ``source_content``; a parent needing the two-stage projection
    then just loses its children's spans instead of raising."""
    parent_content = "a" + " " + "b"
    chunking_result = [
        {
            "content": parent_content,
            "tokens": 3,
            "chunk_order_index": 0,
            "chunk_id": "c0",
            "_source_span": {"start": 0, "end": 4},
        }
    ]

    out = enforce_chunk_token_limit_before_embedding(
        chunking_result, _tok(), max_tokens=1, overlap_tokens=0, source_content=None
    )

    assert len(out) > 1
    assert all("_source_span" not in dp for dp in out)


def test_genuinely_diverged_content_raises_chunk_block_match_error():
    """When source_content IS supplied but the parent content has diverged
    from it beyond whitespace (not just re-spaced), the projection cannot be
    trusted at all and must fail loudly rather than emit a wrong span."""
    parent_content = "a" + " " + "b"
    diverged_document = "WXYZ"  # shares no characters with parent_content
    chunking_result = [
        {
            "content": parent_content,
            "tokens": 3,
            "chunk_order_index": 0,
            "chunk_id": "c0",
            "_source_span": {"start": 0, "end": 4},
        }
    ]

    with pytest.raises(ChunkBlockMatchError):
        enforce_chunk_token_limit_before_embedding(
            chunking_result,
            _tok(),
            max_tokens=1,
            overlap_tokens=0,
            source_content=diverged_document,
        )


def test_direct_arithmetic_fast_path_when_lengths_match():
    """When parent_span length equals len(parent_content) (the common F/R
    case), no projection is needed and the direct-arithmetic offsets are
    exact -- this must keep working even without source_content."""
    document = "hello world, this is a test document body."
    chunking_result = [
        {
            "content": document,
            "tokens": len(_tok().encode(document)),
            "chunk_order_index": 0,
            "chunk_id": "c0",
            "_source_span": {"start": 0, "end": len(document)},
        }
    ]

    out = enforce_chunk_token_limit_before_embedding(
        chunking_result, _tok(), max_tokens=5, overlap_tokens=0, source_content=None
    )

    assert len(out) > 1
    for dp in out:
        span = dp["_source_span"]
        assert document[span["start"] : span["end"]] == dp["content"]


def test_projection_is_built_once_per_parent_and_reused_for_all_children():
    """The per-parent projection is O(parent length) to build; it must not be
    rebuilt once per child (which would make a single parent's hard split
    O(children x parent length))."""
    build_calls = 0
    import lightrag.utils as lr_utils

    original = lr_utils._parent_to_source_projection

    def _counting_projection(*args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        return original(*args, **kwargs)

    lr_utils._parent_to_source_projection = _counting_projection
    try:
        # Sentences separated by a single space in parent_content, but by a
        # double space in the document -- forces the projection path for
        # every one of this parent's several hard-split children.
        parent_content = " ".join(f"word{i:02d}" for i in range(20))
        document = "  ".join(f"word{i:02d}" for i in range(20))
        chunking_result = [
            {
                "content": parent_content,
                "tokens": len(_tok().encode(parent_content)),
                "chunk_order_index": 0,
                "chunk_id": "c0",
                "_source_span": {"start": 0, "end": len(document)},
            }
        ]
        out = enforce_chunk_token_limit_before_embedding(
            chunking_result,
            _tok(),
            max_tokens=6,
            overlap_tokens=0,
            source_content=document,
        )
        assert len(out) > 2  # multiple children actually produced
    finally:
        lr_utils._parent_to_source_projection = original

    assert build_calls == 1


def test_overlap_tokens_produce_real_overlap_and_still_pass_backfill():
    from lightrag.sidecar.backfill import backfill_chunk_sidecars
    import json
    from pathlib import Path

    def _write_blocks(tmp_path: Path, content: str) -> str:
        path = tmp_path / "doc.blocks.jsonl"
        lines = [json.dumps({"type": "meta", "format": "lightrag", "version": "1.0"})]
        lines.append(
            json.dumps(
                {
                    "type": "content",
                    "blockid": "b1",
                    "content": content,
                    "heading": "",
                    "parent_headings": [],
                    "level": 1,
                }
            )
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(path)

    import tempfile

    document = "abcdefghijklmnopqrstuvwxyz" * 3
    with tempfile.TemporaryDirectory() as tmp:
        blocks_path = _write_blocks(Path(tmp), document)
        chunking_result = [
            {
                "content": document,
                "tokens": len(_tok().encode(document)),
                "chunk_order_index": 0,
                "chunk_id": "c0",
                "_source_span": {"start": 0, "end": len(document)},
            }
        ]
        out = enforce_chunk_token_limit_before_embedding(
            chunking_result,
            _tok(),
            max_tokens=10,
            overlap_tokens=3,
            source_content=document,
        )
        assert len(out) > 1
        # Real overlap: consecutive spans share characters.
        spans = [dp["_source_span"] for dp in out]
        assert any(
            spans[i]["start"] < spans[i - 1]["end"] for i in range(1, len(spans))
        )
        backfill_chunk_sidecars(out, blocks_path)
        for dp in out:
            assert dp["sidecar"]["refs"] == [{"type": "block", "id": "b1"}]


def test_unsplittable_oversized_chunk_raises_token_budget_error():
    """A chunk that cannot be safely split at all (not even its first
    Unicode code point fits max_tokens) must fail the document loudly --
    swallowing TokenBudgetError here and re-emitting the original, still
    oversized chunk unchanged would silently defeat the whole safety
    contract this function exists to enforce."""
    from lightrag.utils import TokenBudgetError, TokenizerInterface

    class _HeavyTokenizer(TokenizerInterface):
        def encode(self, content: str) -> list[int]:
            return [0] * (2 * len(content))

        def decode(self, tokens: list[int]) -> str:
            return "x" * (len(tokens) // 2)

    tok = Tokenizer("heavy", _HeavyTokenizer())
    chunking_result = [
        {
            "content": "abc",
            "tokens": 6,
            "chunk_order_index": 0,
            "chunk_id": "c0",
        }
    ]

    with pytest.raises(TokenBudgetError):
        enforce_chunk_token_limit_before_embedding(
            chunking_result, tok, max_tokens=1, overlap_tokens=0
        )


def test_oversized_chunk_is_encoded_only_once_for_the_full_content():
    """The oversized-check and the split must not each independently encode
    the full chunk content -- split_by_token_limit's own fast-path check
    covers "is this within budget", so there is no separate pre-check encode
    call for content that turns out to fit, and exactly one full-content
    encode for content that does not."""

    class _CountingTokenizer(TokenizerInterface):
        def __init__(self):
            self.full_content_encode_calls = 0
            self._content_len = None

        def encode(self, content: str) -> list[int]:
            if self._content_len is not None and len(content) == self._content_len:
                self.full_content_encode_calls += 1
            return [ord(ch) % 1000 for ch in content]

        def decode(self, tokens: list[int]) -> str:
            return "".join(chr(t) for t in tokens)

    content = "abcdefghij" * 100  # 1000 chars
    underlying = _CountingTokenizer()
    underlying._content_len = len(content)
    tok = Tokenizer("counting", underlying)

    chunking_result = [
        {"content": content, "tokens": 1000, "chunk_order_index": 0, "chunk_id": "c0"}
    ]
    out = enforce_chunk_token_limit_before_embedding(
        chunking_result, tok, max_tokens=100, overlap_tokens=0
    )
    assert len(out) > 1
    # Exactly one call encoded the entire original content -- the single
    # fast-path check inside split_by_token_limit, not a separate pre-check.
    assert underlying.full_content_encode_calls == 1
