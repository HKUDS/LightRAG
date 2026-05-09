"""Regression tests for paragraph-semantic Stage B oversized-table handling."""

import json

import pytest

from lightrag.chunker.paragraph_semantic import (
    _expand_block_with_table_splits,
    _split_rows_by_tokens,
)
from lightrag.utils import Tokenizer, TokenizerInterface


class _CharTokenizer(TokenizerInterface):
    """1:1 character-to-token mapping — keeps math obvious in assertions."""

    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


def _make_tokenizer() -> Tokenizer:
    return Tokenizer(model_name="char", tokenizer=_CharTokenizer())


@pytest.mark.offline
def test_split_rows_by_tokens_few_rows_huge_total_no_empty_slice():
    # Reproduces the bug where target_chunks > len(rows) made target_rows
    # < 1, so int((i+1)*target_rows) collapsed to start and the loop
    # appended empty slices (which would later serialise as <table>[]…).
    tokenizer = _make_tokenizer()
    # 3 rows that each individually exceed target_max — forces
    # math.ceil(total/target_ideal) and math.ceil(total/target_max) to
    # both be much greater than len(rows).
    rows = [
        [{"col": "x" * 800}],
        [{"col": "y" * 800}],
        [{"col": "z" * 800}],
    ]

    chunks = _split_rows_by_tokens(
        rows,
        tokenizer,
        target_max=200,
        target_ideal=150,
        last_min=64,
    )

    assert chunks, "expected at least one chunk"
    for chunk in chunks:
        assert chunk, "Stage B must never emit an empty row slice"
    # Concatenation preserves all rows in order.
    flat: list = []
    for chunk in chunks:
        flat.extend(chunk)
    assert flat == rows


@pytest.mark.offline
def test_split_rows_by_tokens_balanced_split_yields_one_row_per_chunk():
    # When target_chunks gets capped at len(rows), each chunk holds one
    # row — verifies the cap kicks in and forward progress is preserved.
    tokenizer = _make_tokenizer()
    rows = [[{"col": "a" * 300}] for _ in range(4)]

    chunks = _split_rows_by_tokens(
        rows,
        tokenizer,
        target_max=200,
        target_ideal=150,
        last_min=10,  # low enough that the tail-merge step doesn't fire
    )

    assert all(chunk for chunk in chunks)
    # Each row appears exactly once across the chunks.
    flat: list = []
    for chunk in chunks:
        flat.extend(chunk)
    assert flat == rows


def _build_oversized_table_text(num_rows: int, row_payload_size: int) -> str:
    rows = [[f"r{idx}-" + "x" * row_payload_size] for idx in range(num_rows)]
    return f'<table id="tb-1" format="json">{json.dumps(rows)}</table>'


@pytest.mark.offline
def test_expand_block_assigns_first_and_last_roles_to_glued_blocks():
    # An oversized table sandwiched between leading and trailing paragraphs
    # produces three slices: "first" (glued with leading paras),
    # "middle" (standalone), "last" (glued with trailing paras). Before
    # the fix, the first/last blocks defaulted to "none" and lost their
    # directional merge-protection.
    tokenizer = _make_tokenizer()
    table_text = _build_oversized_table_text(num_rows=6, row_payload_size=200)
    block = {
        "heading": "Section",
        "parent_headings": ["Doc"],
        "level": 2,
        "paragraphs": [
            {"text": "lead paragraph", "is_table": False},
            {"text": table_text, "is_table": True},
            {"text": "trailing paragraph", "is_table": False},
        ],
    }

    out = _expand_block_with_table_splits(
        block,
        tokenizer=tokenizer,
        table_max=400,
        table_ideal=300,
        table_min_last=128,
    )

    roles = [b["table_chunk_role"] for b in out]
    assert roles[0] == "first", f"expected leading block role=first, got {roles}"
    assert roles[-1] == "last", f"expected trailing block role=last, got {roles}"
    assert all(
        r == "middle" for r in roles[1:-1]
    ), f"expected middle slices between first/last, got {roles}"

    # Boundary glue still works: leading text sits inside the first block,
    # trailing text sits inside the last block.
    assert any(
        p["text"] == "lead paragraph" for p in out[0]["paragraphs"]
    ), "leading paragraph must glue with the first table slice"
    assert any(
        p["text"] == "trailing paragraph" for p in out[-1]["paragraphs"]
    ), "trailing paragraph must glue with the last table slice"


@pytest.mark.offline
def test_expand_block_two_oversized_tables_separates_last_and_first_roles():
    # Two oversized tables in the same heading block: the tail of the first
    # split must carry role="last" and not be silently merged into the
    # head of the second split (which must carry role="first").
    tokenizer = _make_tokenizer()
    block = {
        "heading": "Section",
        "parent_headings": [],
        "level": 2,
        "paragraphs": [
            {
                "text": _build_oversized_table_text(num_rows=4, row_payload_size=200),
                "is_table": True,
            },
            {"text": "between tables", "is_table": False},
            {
                "text": _build_oversized_table_text(num_rows=4, row_payload_size=200),
                "is_table": True,
            },
        ],
    }

    out = _expand_block_with_table_splits(
        block,
        tokenizer=tokenizer,
        table_max=400,
        table_ideal=300,
        table_min_last=128,
    )

    roles = [b["table_chunk_role"] for b in out]
    # We expect the role sequence to start with "first", end with "last",
    # and contain at least one "last" -> "first" transition (the boundary
    # between the two oversized tables) without any boundary block losing
    # its role.
    assert roles[0] == "first"
    assert roles[-1] == "last"
    assert "last" in roles
    # The transition: there must be a "last" immediately followed by a
    # "first" somewhere in the middle of the role sequence.
    transitions = list(zip(roles, roles[1:]))
    assert (
        ("last", "first") in transitions
    ), f"expected a last->first boundary between the two split tables, got {roles}"
