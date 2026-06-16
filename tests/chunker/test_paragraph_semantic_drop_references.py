"""Tests for the paragraph-semantic ``drop_references`` option (chunking=P).

A reference block is dropped only when it BOTH sits within the last
``references_tail_n`` content blocks AND its heading matches a reference prefix.
The switch is the only knob that flows through ``chunk_options``; the tail
window and heading prefixes are read live from env at chunk time (verified here
by mutating env between calls, proving they are NOT snapshotted).

Assertions check the *content* of the emitted chunks (each block carries a
distinctive marker) rather than heading names, because LevelMerge may merge
small blocks and rewrite the surviving heading.
"""

import json
import logging

import pytest

from lightrag.chunker.paragraph_semantic import (
    _format_dropped_headings,
    _is_reference_heading,
    chunking_by_paragraph_semantic,
)
from lightrag.constants import DEFAULT_P_REFERENCES_HEADINGS
from lightrag.utils import Tokenizer, TokenizerInterface, logger as _lightrag_logger


class _CharTokenizer(TokenizerInterface):
    """1:1 character-to-token mapping — keeps math obvious in assertions."""

    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


def _make_tokenizer() -> Tokenizer:
    return Tokenizer(model_name="char", tokenizer=_CharTokenizer())


def _row(heading: str, content: str, *, level: int = 1) -> dict:
    return {
        "type": "content",
        "blockid": heading or "blk",
        "format": "plain_text",
        "content": content,
        "heading": heading,
        "parent_headings": [],
        "level": level,
        "session_type": "body",
        "table_slice": "none",
        "positions": [],
    }


def _write_blocks_jsonl(tmp_path, rows: list[dict]) -> str:
    path = tmp_path / "doc.blocks.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
    return str(path)


def _all_content(chunks: list[dict]) -> str:
    return "\n".join(c["content"] for c in chunks)


# --------------------------------------------------------------------------- #
# _is_reference_heading (word boundary vs. plain CJK prefix)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "heading,expected",
    [
        ("References", True),
        ("references", True),  # case-insensitive
        ("REFERENCES", True),
        ("Reference", True),
        ("Bibliography", True),
        ("References [1-50]", True),  # word boundary: next char is non-alnum
        ("参考文献", True),
        ("参考文献列表", True),  # CJK: plain prefix, no word boundary
        ("Referenced work", False),  # ASCII word boundary excludes this
        ("Related Work", False),
        ("", False),
    ],
)
def test_is_reference_heading(heading, expected):
    assert _is_reference_heading(heading, DEFAULT_P_REFERENCES_HEADINGS) is expected


# --------------------------------------------------------------------------- #
# _format_dropped_headings (length-bounded log rendering)
# --------------------------------------------------------------------------- #


def test_format_dropped_headings_short_list():
    assert _format_dropped_headings(["References"]) == "'References'"
    assert (
        _format_dropped_headings(["References", "Bibliography"])
        == "'References', 'Bibliography'"
    )


def test_format_dropped_headings_truncates_long_heading():
    out = _format_dropped_headings(["X" * 200], max_each=60)
    # 60 kept chars + an ellipsis, all within one repr'd string.
    assert "X" * 60 + "…" in out
    assert "X" * 61 not in out


def test_format_dropped_headings_caps_item_count():
    out = _format_dropped_headings([f"H{i}" for i in range(12)], max_items=5)
    assert out.count("'H") == 5  # only 5 listed
    assert "(+7 more)" in out


# --------------------------------------------------------------------------- #
# Filtering behaviour (assert on content markers, robust to LevelMerge)
# --------------------------------------------------------------------------- #


@pytest.mark.offline
def test_drops_trailing_reference_block(tmp_path):
    tokenizer = _make_tokenizer()
    rows = [
        _row("Introduction", "INTRO_MARKER intro body"),
        _row("Method", "METHOD_MARKER method body"),
        _row("References", "REF_MARKER [1] Foo. [2] Bar."),
    ]
    blocks_path = _write_blocks_jsonl(tmp_path, rows)

    body = _all_content(
        chunking_by_paragraph_semantic(
            tokenizer, "", 2000, blocks_path=blocks_path, drop_references=True
        )
    )
    assert "REF_MARKER" not in body
    assert "INTRO_MARKER" in body and "METHOD_MARKER" in body


@pytest.mark.offline
def test_default_keeps_reference_block(tmp_path):
    tokenizer = _make_tokenizer()
    rows = [
        _row("Introduction", "INTRO_MARKER intro body"),
        _row("References", "REF_MARKER [1] Foo."),
    ]
    blocks_path = _write_blocks_jsonl(tmp_path, rows)

    # drop_references defaults to False — references survive.
    body = _all_content(
        chunking_by_paragraph_semantic(tokenizer, "", 2000, blocks_path=blocks_path)
    )
    assert "REF_MARKER" in body


@pytest.mark.offline
def test_mid_document_reference_section_outside_window_kept(tmp_path):
    tokenizer = _make_tokenizer()
    # A "References" heading mid-document, with two non-reference blocks after
    # it, so it falls outside the last-2 window and must NOT be dropped.
    rows = [
        _row("Introduction", "INTRO_MARKER intro body"),
        _row("References", "REF_MARKER discusses references"),
        _row("Method", "METHOD_MARKER method body"),
        _row("Results", "RESULTS_MARKER results body"),
    ]
    blocks_path = _write_blocks_jsonl(tmp_path, rows)

    body = _all_content(
        chunking_by_paragraph_semantic(
            tokenizer,
            "",
            2000,
            blocks_path=blocks_path,
            drop_references=True,
            references_tail_n=2,
        )
    )
    assert "REF_MARKER" in body


@pytest.mark.offline
def test_word_boundary_keeps_referenced(tmp_path):
    tokenizer = _make_tokenizer()
    rows = [
        _row("Introduction", "INTRO_MARKER intro body"),
        _row("参考文献列表", "CJKREF_MARKER [1] Foo."),
        _row("Referenced datasets", "DATASET_MARKER dataset details"),
    ]
    blocks_path = _write_blocks_jsonl(tmp_path, rows)

    body = _all_content(
        chunking_by_paragraph_semantic(
            tokenizer, "", 2000, blocks_path=blocks_path, drop_references=True
        )
    )
    # CJK prefix block dropped; "Referenced ..." kept (ASCII word boundary).
    assert "CJKREF_MARKER" not in body
    assert "DATASET_MARKER" in body


@pytest.mark.offline
def test_all_blocks_match_keeps_to_avoid_empty(tmp_path):
    tokenizer = _make_tokenizer()
    rows = [
        _row("References", "REF1_MARKER [1] Foo."),
        _row("Bibliography", "REF2_MARKER [2] Bar."),
    ]
    blocks_path = _write_blocks_jsonl(tmp_path, rows)

    # Both blocks match within the last-2 window; dropping all would leave no
    # content, so the chunker keeps them and warns instead.
    chunks = chunking_by_paragraph_semantic(
        tokenizer, "", 2000, blocks_path=blocks_path, drop_references=True
    )
    body = _all_content(chunks)
    assert chunks  # not an empty document
    assert "REF1_MARKER" in body and "REF2_MARKER" in body


# --------------------------------------------------------------------------- #
# Detection knobs are read LIVE from env (not snapshotted)
# --------------------------------------------------------------------------- #


@pytest.mark.offline
def test_tail_n_read_live_from_env(tmp_path, monkeypatch):
    tokenizer = _make_tokenizer()
    rows = [
        _row("Introduction", "INTRO_MARKER intro body"),
        _row("References", "REF_MARKER [1] Foo."),
        _row("Appendix", "APPENDIX_MARKER appendix body"),
    ]
    blocks_path = _write_blocks_jsonl(tmp_path, rows)

    # tail_n=1: only the last block ("Appendix") is in the window → References
    # is outside it and survives.
    monkeypatch.setenv("CHUNK_P_REFERENCES_TAIL_N", "1")
    body = _all_content(
        chunking_by_paragraph_semantic(
            tokenizer, "", 2000, blocks_path=blocks_path, drop_references=True
        )
    )
    assert "REF_MARKER" in body

    # Widen the window to 2 → References now falls inside and is dropped. Same
    # call, only the env changed: proves the knob is read live, not snapshotted.
    monkeypatch.setenv("CHUNK_P_REFERENCES_TAIL_N", "2")
    body = _all_content(
        chunking_by_paragraph_semantic(
            tokenizer, "", 2000, blocks_path=blocks_path, drop_references=True
        )
    )
    assert "REF_MARKER" not in body


@pytest.mark.offline
def test_drop_references_emits_info_log(tmp_path, caplog):
    tokenizer = _make_tokenizer()
    rows = [
        _row("Introduction", "INTRO_MARKER intro body"),
        _row("References", "REF_MARKER [1] Foo."),
    ]
    blocks_path = _write_blocks_jsonl(tmp_path, rows)

    # The lightrag logger sets propagate=False, so caplog can't see it by
    # default — enable propagation for the duration of the call.
    original_propagate = _lightrag_logger.propagate
    _lightrag_logger.propagate = True
    try:
        with caplog.at_level(logging.INFO, logger=_lightrag_logger.name):
            chunking_by_paragraph_semantic(
                tokenizer, "", 2000, blocks_path=blocks_path, drop_references=True
            )
    finally:
        _lightrag_logger.propagate = original_propagate

    info_msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert any("drop_references" in m and "References" in m for m in info_msgs), (
        info_msgs
    )


@pytest.mark.offline
def test_headings_read_live_from_env(tmp_path, monkeypatch):
    tokenizer = _make_tokenizer()
    rows = [
        _row("Introduction", "INTRO_MARKER intro body"),
        _row("Literature", "LIT_MARKER [1] Foo. [2] Bar."),
    ]
    blocks_path = _write_blocks_jsonl(tmp_path, rows)

    # Custom prefix list makes "Literature" a reference heading.
    monkeypatch.setenv("CHUNK_P_REFERENCES_HEADINGS", "Literature|参考文献")
    body = _all_content(
        chunking_by_paragraph_semantic(
            tokenizer, "", 2000, blocks_path=blocks_path, drop_references=True
        )
    )
    assert "LIT_MARKER" not in body
