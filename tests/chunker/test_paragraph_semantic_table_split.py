"""Regression tests for paragraph-semantic TableRowSplit oversized-table handling."""

import json
import logging

import pytest

from lightrag.chunker.paragraph_semantic import (
    _detect_table_format,
    _expand_block_with_table_splits,
    _merge_small_blocks,
    _new_block,
    _split_html_rows,
    _split_long_block,
    _split_rows_by_tokens,
    _split_table_text,
    chunking_by_paragraph_semantic,
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


@pytest.fixture
def _propagate_lightrag_logger(monkeypatch):
    # The ``lightrag`` logger sets ``propagate=False`` (see lightrag.utils), so
    # caplog's root handler captures nothing by default. Flip it on for the
    # duration of a test that asserts on the degrade warning.
    monkeypatch.setattr(logging.getLogger("lightrag"), "propagate", True)


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
        assert chunk, "TableRowSplit must never emit an empty row slice"
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


def _write_blocks_jsonl(tmp_path, content: str) -> str:
    path = tmp_path / "doc.blocks.jsonl"
    row = {
        "type": "content",
        "heading": "Section",
        "parent_headings": [],
        "level": 2,
        "content": content,
    }
    path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    return str(path)


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
    assert all(
        "表格片段" not in b["heading"] for b in out
    ), "TableRowSplit should not expose legacy table-fragment heading suffixes"


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


@pytest.mark.offline
def test_expand_block_duplicates_short_text_between_oversized_tables():
    tokenizer = _make_tokenizer()
    bridge = "between tables"
    block = {
        "heading": "Section",
        "parent_headings": [],
        "level": 2,
        "paragraphs": [
            {
                "text": _build_oversized_table_text(num_rows=4, row_payload_size=200),
                "is_table": True,
            },
            {"text": bridge, "is_table": False},
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
        target_max=800,
        chunk_overlap_token_size=100,
    )

    roles = [b["table_chunk_role"] for b in out]
    boundary_idx = next(
        i
        for i, (left, right) in enumerate(zip(roles, roles[1:]))
        if (left, right) == ("last", "first")
    )
    assert bridge in out[boundary_idx]["content"]
    assert bridge in out[boundary_idx + 1]["content"]


@pytest.mark.offline
def test_expand_block_emits_middle_text_when_table_bridge_is_long():
    tokenizer = _make_tokenizer()
    bridge = ("A" * 45) + ("B" * 50) + ("C" * 45)
    block = {
        "heading": "Section",
        "parent_headings": [],
        "level": 2,
        "paragraphs": [
            {
                "text": _build_oversized_table_text(num_rows=6, row_payload_size=120),
                "is_table": True,
            },
            {"text": bridge, "is_table": False},
            {
                "text": _build_oversized_table_text(num_rows=6, row_payload_size=120),
                "is_table": True,
            },
        ],
    }

    out = _expand_block_with_table_splits(
        block,
        tokenizer=tokenizer,
        table_max=260,
        table_ideal=180,
        table_min_last=32,
        target_max=400,
        chunk_overlap_token_size=45,
    )

    # The standalone middle block carries R-style overlap with the text that
    # went left (prefix) and right (suffix): because each side's slice is
    # itself ≤ the overlap budget, the middle re-covers the whole bridge — the
    # bridge is never fragmented, only its head/tail are *also* duplicated into
    # the neighbouring table blocks.
    middle_idx = next(
        i
        for i, blk in enumerate(out)
        if blk["table_chunk_role"] == "none" and blk["content"] == bridge
    )
    assert out[middle_idx - 1]["table_chunk_role"] == "last"
    assert "A" * 45 in out[middle_idx - 1]["content"]
    assert "B" * 50 not in out[middle_idx - 1]["content"]
    assert out[middle_idx + 1]["table_chunk_role"] == "first"
    assert out[middle_idx + 1]["content"].startswith("C" * 45)
    assert "B" * 50 not in out[middle_idx + 1]["content"]
    # The overlap never drags table markup into the middle text block.
    assert "<table" not in out[middle_idx]["content"]
    assert all(b["tokens"] <= 400 for b in out), [b["tokens"] for b in out]


@pytest.mark.offline
def test_bridge_single_side_budget_capped_at_half_target_max():
    # §7 guarantee: even with a huge configured overlap, the bridge text
    # duplicated into each table boundary block is capped at target_max // 2
    # so it can never dominate the block. The bridge is a long run of a
    # character that never appears in the table payload, so counting it in
    # each boundary block measures exactly how much bridge text was copied in.
    tokenizer = _make_tokenizer()
    bridge = "Z" * 1000
    block = {
        "heading": "Section",
        "parent_headings": [],
        "level": 2,
        "paragraphs": [
            {
                "text": _build_oversized_table_text(num_rows=8, row_payload_size=30),
                "is_table": True,
            },
            {"text": bridge, "is_table": False},
            {
                "text": _build_oversized_table_text(num_rows=8, row_payload_size=30),
                "is_table": True,
            },
        ],
    }
    target_max = 400
    half = target_max // 2

    out = _expand_block_with_table_splits(
        block,
        tokenizer=tokenizer,
        table_max=200,
        table_ideal=140,
        table_min_last=48,
        target_max=target_max,
        chunk_overlap_token_size=10_000,  # huge → only the half-cap can bind
    )

    # The long bridge yields [... "last"+prefix, "none" full-bridge middle,
    # suffix+"first" ...]; locate the middle and read its neighbours.
    middle_idx = next(
        i
        for i, blk in enumerate(out)
        if blk["table_chunk_role"] == "none" and blk["content"] == bridge
    )
    left = out[middle_idx - 1]
    right = out[middle_idx + 1]
    assert left["table_chunk_role"] == "last"
    assert right["table_chunk_role"] == "first"
    left_z = left["content"].count("Z")
    right_z = right["content"].count("Z")
    # The huge overlap would have copied far more than `half` without the cap;
    # the bridge (1000) and the small table slices leave ample headroom, so the
    # half-cap is the binding constraint and each side gets exactly `half`.
    assert left_z == half, left_z
    assert right_z == half, right_z


@pytest.mark.offline
def test_public_chunking_adds_part_suffixes_to_all_table_split_fragments(tmp_path):
    tokenizer = _make_tokenizer()
    body = "\n".join(
        [
            "lead paragraph",
            _build_oversized_table_text(num_rows=6, row_payload_size=200),
            "trailing paragraph",
        ]
    )
    blocks_path = _write_blocks_jsonl(tmp_path, body)

    chunks = chunking_by_paragraph_semantic(
        tokenizer,
        body,
        chunk_token_size=800,
        blocks_path=blocks_path,
        chunk_overlap_token_size=0,
    )

    assert len(chunks) > 1
    assert [chunk["heading"]["heading"] for chunk in chunks] == [
        f"Section [part {idx}]" for idx in range(1, len(chunks) + 1)
    ]
    assert all("表格片段" not in chunk["heading"]["heading"] for chunk in chunks)


# ---------------------------------------------------------------------------
# Table-aware fallback tests (row-boundary first, character last).
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_detect_table_format_explicit_attr():
    assert _detect_table_format('id="t1" format="json"', "[]") == "json"
    assert _detect_table_format("format='html'", "<tr></tr>") == "html"
    # Unknown formats fall through (force the caller to use char fallback).
    assert _detect_table_format('format="markdown"', "...") is None


@pytest.mark.offline
def test_detect_table_format_sniff_when_attrs_silent():
    assert _detect_table_format("", '[{"a":1}]') == "json"
    assert _detect_table_format("", "<tr><td>x</td></tr>") == "html"
    # Body that doesn't look like JSON or HTML → unknown.
    assert _detect_table_format("", "plain text rows") is None


@pytest.mark.offline
def test_split_html_rows_extracts_tr_elements():
    body = (
        "<thead><tr><th>h</th></tr></thead>"
        "<tbody><tr><td>a</td></tr><tr><td>b</td></tr></tbody>"
    )
    rows = _split_html_rows(body)
    assert rows is not None
    assert len(rows) == 3
    # Each row carries its parent wrapper so the chunk serialiser can
    # rebuild <thead>/<tbody> instead of dropping them silently.
    assert [w for w, _ in rows] == ["thead", "tbody", "tbody"]
    assert all(tr.startswith("<tr") and tr.endswith("</tr>") for _, tr in rows)


@pytest.mark.offline
def test_split_html_rows_no_tr_returns_none():
    assert _split_html_rows("just text, no rows") is None
    assert _split_html_rows("") is None


@pytest.mark.offline
def test_split_table_text_single_row_oversized_falls_to_character_split():
    # A 1-row table whose single cell is huge cannot be reduced via row
    # boundary — the function must fall to character splitting and respect
    # target_max on every output piece.
    tokenizer = _make_tokenizer()
    rows = [[{"col": "x" * 2000}]]
    table_text = f'<table id="tb-1" format="json">{json.dumps(rows)}</table>'

    pieces = _split_table_text(
        table_text,
        tokenizer=tokenizer,
        target_max=500,
        target_ideal=350,
        last_min=128,
    )

    assert len(pieces) >= 2, "single-row oversized table must produce multiple pieces"
    # Every piece honors the cap (this is the contract violation the user
    # reported when the previous code emitted a single 2000-token table).
    assert all(_count_tokens(tokenizer, p) <= 500 for p in pieces)


@pytest.mark.offline
def test_split_table_text_multirow_one_huge_row_degrades_whole_table_to_recursive():
    # A multi-row table where most rows fit but one row is itself huge. A single
    # row that exceeds the cap can no longer be expressed at row boundaries, so
    # the WHOLE table degrades to a recursive character split (rather than
    # emitting a mix of <table> slices and orphaned character fragments).
    tokenizer = _make_tokenizer()
    small_row = [{"col": "ok"}]
    huge_row = [{"col": "z" * 2000}]
    rows = [small_row, huge_row, small_row]
    table_text = f'<table id="tb-1" format="json">{json.dumps(rows)}</table>'

    pieces = _split_table_text(
        table_text,
        tokenizer=tokenizer,
        target_max=500,
        target_ideal=350,
        last_min=64,
    )

    assert len(pieces) >= 2, "oversized row must force a multi-piece recursive split"
    assert all(_count_tokens(tokenizer, p) <= 500 for p in pieces)
    # The whole table degraded: it was shredded as one recursive split, so no
    # piece survives as a complete re-wrapped <table>...</table> slice (the old
    # mixed output kept legal per-row table slices alongside text fragments).
    assert not any(p.startswith("<table ") and p.endswith("</table>") for p in pieces)


@pytest.mark.offline
def test_split_table_text_html_table_split_by_tr():
    # HTML-format table: rows are <tr>...</tr>; each output fragment must
    # remain a legal <table {attrs}>{rows}</table> string.
    tokenizer = _make_tokenizer()
    body = "".join(f"<tr><td>{'r' * 200}</td></tr>" for _ in range(5))
    table_text = f'<table id="tb-h1" format="html">{body}</table>'

    pieces = _split_table_text(
        table_text,
        tokenizer=tokenizer,
        target_max=500,
        target_ideal=350,
        last_min=64,
    )

    assert len(pieces) >= 2
    # All pieces should be legal <table>...</table> fragments (none of the
    # rows individually exceeds target_max, so no character fallback).
    assert all(p.startswith("<table ") and p.endswith("</table>") for p in pieces)
    assert all(_count_tokens(tokenizer, p) <= 500 for p in pieces)


@pytest.mark.offline
def test_split_table_text_html_preserves_thead_tbody_wrappers():
    # When an HTML table mixes <thead> and <tbody>, the row splitter
    # used to drop the wrappers entirely — the chunked output came back
    # as bare <tr> sequences. The fix re-emits each wrapper around its
    # rows in every chunk so the table structure survives splitting.
    tokenizer = _make_tokenizer()
    head_row = "<tr><th>" + ("h" * 80) + "</th></tr>"
    body_rows = "".join(f"<tr><td>{'b' * 80}{i}</td></tr>" for i in range(4))
    body = f"<thead>{head_row}</thead><tbody>{body_rows}</tbody>"
    table_text = f'<table id="tb-mixed" format="html">{body}</table>'

    pieces = _split_table_text(
        table_text,
        tokenizer=tokenizer,
        target_max=400,
        target_ideal=280,
        last_min=64,
    )

    # Multiple chunks expected and every chunk must remain a legal
    # <table>-wrapped fragment.
    assert len(pieces) >= 2
    assert all(p.startswith("<table ") and p.endswith("</table>") for p in pieces)
    # Every chunk that contains the header row must still wrap it in
    # <thead>...</thead>; every chunk with body rows must wrap them in
    # <tbody>...</tbody>. Before the fix, both wrappers vanished.
    for piece in pieces:
        if "<th>" in piece:
            assert "<thead>" in piece and "</thead>" in piece, piece
        if "<td>" in piece:
            assert "<tbody>" in piece and "</tbody>" in piece, piece
    # Round-trip: concatenating just the row payloads from every chunk
    # recovers the original row sequence in order.
    extracted_rows: list[str] = []
    import re

    for piece in pieces:
        extracted_rows.extend(
            re.findall(r"<tr\b[^>]*>.*?</tr>", piece, re.DOTALL | re.IGNORECASE)
        )
    expected_rows = re.findall(r"<tr\b[^>]*>.*?</tr>", body, re.DOTALL | re.IGNORECASE)
    assert extracted_rows == expected_rows


@pytest.mark.offline
def test_split_table_text_html_injects_thead_into_nonfirst_slices():
    """A long HTML table with an HTML <thead> header_body: the first slice keeps
    its own in-body <thead> (not double-injected); every middle/last slice has
    the raw <thead> spliced in exactly once, with rowspan/colspan preserved;
    every slice stays within the cap."""
    tokenizer = _make_tokenizer()
    header = (
        '<thead><tr><th rowspan="2">Metric</th><th colspan="2">Group</th></tr></thead>'
    )
    head_row = '<tr><th rowspan="2">Metric</th><th colspan="2">Group</th></tr>'
    body_rows = "".join(f"<tr><td>{'d' * 120}{i}</td></tr>" for i in range(6))
    # Body carries its own <thead> (the source header) plus the data rows.
    body = f"<thead>{head_row}</thead><tbody>{body_rows}</tbody>"
    table_text = f'<table id="tb-h" format="html">{body}</table>'

    pieces = _split_table_text(
        table_text,
        tokenizer=tokenizer,
        target_max=500,
        target_ideal=350,
        last_min=64,
        header_body=header,
    )

    assert len(pieces) >= 3  # first / middle(s) / last
    assert all(p.startswith("<table ") and p.endswith("</table>") for p in pieces)
    assert all(_count_tokens(tokenizer, p) <= 500 for p in pieces)

    # First slice keeps its own header and is NOT injected with a second copy.
    assert pieces[0].count("<thead>") == 1
    # Every non-first slice gets the header spliced in exactly once, verbatim
    # (rowspan/colspan preserved), immediately after the opening <table …>.
    for piece in pieces[1:]:
        assert piece.count("<thead>") == 1, piece
        assert header in piece, piece
        assert 'rowspan="2"' in piece and 'colspan="2"' in piece, piece
        # The <thead> sits at the front of the body, ahead of the data rows.
        assert piece.index("<thead>") < piece.index("<tbody>"), piece


@pytest.mark.offline
def test_split_table_text_raises_on_html_table_with_json_header():
    """Cross-format guard: an HTML table fed a JSON-array header_body must raise
    (the sidecar header does not belong to this table)."""
    tokenizer = _make_tokenizer()
    body = "".join(f"<tr><td>{'r' * 200}</td></tr>" for _ in range(5))
    table_text = f'<table id="tb-h1" format="html">{body}</table>'
    with pytest.raises(ValueError):
        _split_table_text(
            table_text,
            tokenizer=tokenizer,
            target_max=500,
            target_ideal=350,
            last_min=64,
            header_body='[["H1", "H2"]]',
        )


@pytest.mark.offline
def test_split_table_text_unknown_format_falls_to_character():
    # No format attr, body that doesn't look like JSON/HTML → unknown.
    tokenizer = _make_tokenizer()
    table_text = '<table id="weird">' + ("plain row text " * 300) + "</table>"

    pieces = _split_table_text(
        table_text,
        tokenizer=tokenizer,
        target_max=500,
        target_ideal=350,
        last_min=64,
    )

    assert len(pieces) >= 2
    assert all(_count_tokens(tokenizer, p) <= 500 for p in pieces)


@pytest.mark.offline
def test_expand_block_single_row_table_no_longer_left_intact():
    # TableRowSplit integration: previously a single-row oversized table was
    # appended back to cur_paras unchanged, leading the block to reach
    # AnchorSplit with the table whole and the character fallback shredding
    # the <table> tag. After the fix, TableRowSplit itself produces multiple
    # pieces for such a table.
    tokenizer = _make_tokenizer()
    rows = [[{"col": "x" * 2000}]]  # single huge row
    table_text = f'<table id="tb-1" format="json">{json.dumps(rows)}</table>'
    block = {
        "heading": "Section",
        "parent_headings": [],
        "level": 2,
        "paragraphs": [
            {"text": "lead", "is_table": False},
            {"text": table_text, "is_table": True},
            {"text": "trail", "is_table": False},
        ],
    }

    out = _expand_block_with_table_splits(
        block,
        tokenizer=tokenizer,
        table_max=400,
        table_ideal=300,
        table_min_last=128,
    )

    # Multiple sub-blocks must be produced; the oversized table no longer
    # passes through whole.
    assert len(out) >= 2
    # First/last role protection still fires when the table was reduced.
    roles = [b["table_chunk_role"] for b in out]
    assert (
        "first" in roles or "last" in roles
    ), f"expected first/last role assignment after table split, got {roles}"


@pytest.mark.offline
def test_split_long_block_table_dominant_no_anchor_keeps_some_table_markup():
    # AnchorSplit integration: a block dominated by an oversized table with no
    # anchor candidates used to be character-split end-to-end, destroying
    # the <table> tag. After the fix, at least some output sub-blocks
    # retain legal <table>...</table> markup for the rows that fit.
    tokenizer = _make_tokenizer()
    # Many small rows -> row-boundary split produces multiple legal
    # <table> fragments, none of which individually exceed target_max.
    rows = [[{"col": f"r{i}-" + "v" * 200}] for i in range(8)]
    table_text = f'<table id="tb-1" format="json">{json.dumps(rows)}</table>'

    paragraphs = [
        {"text": "Sufficiently long lead paragraph " * 30, "is_table": False},
        {"text": table_text, "is_table": True},
    ]

    sub_blocks = _split_long_block(
        paragraphs,
        heading="Heading",
        parent_headings=[],
        level=2,
        table_chunk_role="none",
        tokenizer=tokenizer,
        target_max=600,
        target_ideal=450,
    )

    # Every sub-block respects the cap.
    assert all(b["tokens"] <= 600 for b in sub_blocks)
    # At least one sub-block keeps an unbroken <table> fragment somewhere
    # in its content (proof that row-boundary preservation kicked in).
    contents = [b["content"] for b in sub_blocks]
    assert any(
        ("<table " in c and "</table>" in c) for c in contents
    ), "expected at least one sub-block to retain a legal <table> fragment"


@pytest.mark.offline
def test_split_table_text_budgets_wrapper_overhead_for_target_max():
    # ``_split_rows_by_tokens`` measures only the body (json.dumps(rows));
    # the surrounding ``<table {attrs}></table>`` wrapper costs tokens too.
    # Without wrapper-aware budgeting, a chunk whose body just fits
    # target_max would overflow once wrapped and trigger character
    # fallback — shredding the row structure for no good reason.
    tokenizer = _make_tokenizer()
    # A long attrs string forces a non-trivial wrapper overhead so the
    # body-only budget previously chosen (==target_max) overflows when
    # the wrapper is added back in.
    attrs_padding = "x" * 80
    rows = [[{"col": "y" * 80}] for _ in range(4)]
    table_text = f'<table id="{attrs_padding}" format="json">{json.dumps(rows)}</table>'

    pieces = _split_table_text(
        table_text,
        tokenizer=tokenizer,
        target_max=250,
        target_ideal=180,
        last_min=64,
    )

    # Every output piece honors the cap.
    assert all(_count_tokens(tokenizer, p) <= 250 for p in pieces), [
        _count_tokens(tokenizer, p) for p in pieces
    ]
    # Row structure preserved — none of the pieces fell back to
    # character fragments because of accidental wrapper overflow.
    assert all(p.startswith("<table ") and p.endswith("</table>") for p in pieces)


def _count_tokens(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text))


# ---------------------------------------------------------------------------
# HeaderRecovery — the source table's repeating header is budgeted out of the
# per-slice cap and re-injected into the <table> body of every non-first slice
# AT SPLIT TIME (not backfilled at output), so a slice + its header always
# stays ≤ target_max, and split slices are frozen against LevelMerge so the
# header is never duplicated by a re-merge.
# ---------------------------------------------------------------------------


_HEADER_BODY = '[["H1", "H2"]]'
# After injection (or naturally on the first slice) a JSON slice's <table> body
# begins with the header row.
_INJECTED_PREFIX = '<table id="tb-1" format="json">[["H1", "H2"]'

# Sentinel distinguishing "write tables.json without a table_header" from
# "do not write tables.json at all".
_NO_SIDECAR = object()


def _write_tables_json(tmp_path, headers: dict) -> None:
    """Write a ``doc.tables.json`` beside ``doc.blocks.jsonl``.

    ``headers`` maps table id -> header string; a ``None`` value emits an entry
    WITHOUT the ``table_header`` field (a table that has no repeating header).
    """
    tables: dict = {}
    for tid, header in headers.items():
        entry = {"id": tid, "format": "json", "content": "[]", "caption": ""}
        if header is not None:
            entry["table_header"] = header
        tables[tid] = entry
    path = tmp_path / "doc.tables.json"
    path.write_text(
        json.dumps({"version": "1.0", "tables": tables}, ensure_ascii=False),
        encoding="utf-8",
    )


def _build_table_with_header(num_data_rows: int, payload_size: int) -> str:
    # First row is the real header; the rest are data rows. After row-splitting,
    # only the "first" slice keeps the header row; middle/last slices must have
    # it re-injected.
    rows = [["H1", "H2"]] + [
        [f"r{idx}-" + "x" * payload_size, "y"] for idx in range(num_data_rows)
    ]
    return f'<table id="tb-1" format="json">{json.dumps(rows)}</table>'


def _chunk_with_oversized_table(
    tmp_path,
    *,
    heading: str = "Section",
    sidecar=_HEADER_BODY,
    num_data_rows: int = 6,
    payload_size: int = 200,
    trailing: str = "trailing paragraph",
    chunk_token_size: int = 800,
):
    tokenizer = _make_tokenizer()
    body = "\n".join(
        [
            "lead paragraph",
            _build_table_with_header(
                num_data_rows=num_data_rows, payload_size=payload_size
            ),
            trailing,
        ]
    )
    if sidecar is _NO_SIDECAR:
        pass  # leave the directory without a tables.json
    elif sidecar is None:
        _write_tables_json(tmp_path, {"tb-1": None})  # entry without table_header
    else:
        _write_tables_json(tmp_path, {"tb-1": sidecar})
    path = tmp_path / "doc.blocks.jsonl"
    row = {
        "type": "content",
        "heading": heading,
        "parent_headings": [],
        "level": 2 if heading else 0,
        "content": body,
    }
    path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    chunks = chunking_by_paragraph_semantic(
        tokenizer,
        body,
        chunk_token_size=chunk_token_size,
        blocks_path=str(path),
        chunk_overlap_token_size=0,
    )
    return chunks


def _table_chunks(chunks):
    return [c for c in chunks if '<table id="tb-1"' in c["content"]]


def _slice_starts_with_header(chunk) -> bool:
    content = chunk["content"]
    tbl = content[content.index('<table id="tb-1"') :]
    return tbl.startswith(_INJECTED_PREFIX)


@pytest.mark.offline
def test_extract_table_id_variants():
    from lightrag.table_markup import extract_table_id

    assert extract_table_id('id="tb-1" format="json"') == "tb-1"
    assert extract_table_id("format='html' id='tb-h'") == "tb-h"
    assert extract_table_id('format="json"') is None
    assert extract_table_id("") is None
    # ``id`` that is only the tail of another attribute name must not match —
    # neither a hyphen-joined one (``data-id``) nor a word-char prefix
    # (``grid``); a standalone ``id`` alongside such an attribute still wins.
    assert extract_table_id('data-id="x"') is None
    assert extract_table_id('grid="y"') is None
    assert extract_table_id('data-id="x" id="tb-9"') == "tb-9"


@pytest.mark.offline
def test_inject_header_into_table_slice_json_and_html():
    from lightrag.chunker.paragraph_semantic import _inject_header_into_table_slice

    # JSON: header rows are prepended to the row array, attrs (incl. id) kept.
    json_slice = '<table id="tb-1" format="json">[["a", "b"]]</table>'
    assert (
        _inject_header_into_table_slice(json_slice, '[["H1", "H2"]]')
        == '<table id="tb-1" format="json">[["H1", "H2"], ["a", "b"]]</table>'
    )

    # HTML: the raw <thead> header is spliced verbatim ahead of the body,
    # preserving any rowspan/colspan markup it carries.
    html_slice = (
        '<table id="tb-h" format="html"><tbody><tr><td>a</td></tr></tbody></table>'
    )
    html_header = '<thead><tr><th colspan="2">H</th></tr></thead>'
    assert _inject_header_into_table_slice(html_slice, html_header) == (
        '<table id="tb-h" format="html">'
        '<thead><tr><th colspan="2">H</th></tr></thead>'
        "<tbody><tr><td>a</td></tr></tbody></table>"
    )

    # Unparseable header → no injection.
    assert _inject_header_into_table_slice(json_slice, "not json") is None
    # Non-<table> fragment → no injection.
    assert _inject_header_into_table_slice("just text", '[["H1", "H2"]]') is None


@pytest.mark.offline
def test_inject_header_raises_on_cross_format():
    """A header whose format disagrees with the slice's content format means the
    sidecar header does not belong to this table — inject must raise, not
    silently emit a malformed slice."""
    from lightrag.chunker.paragraph_semantic import _inject_header_into_table_slice

    json_slice = '<table id="tb-1" format="json">[["a", "b"]]</table>'
    html_slice = (
        '<table id="tb-h" format="html"><tbody><tr><td>a</td></tr></tbody></table>'
    )
    html_header = "<thead><tr><th>H</th></tr></thead>"

    # HTML header into a JSON slice → mismatch.
    with pytest.raises(ValueError):
        _inject_header_into_table_slice(json_slice, html_header)
    # JSON header into an HTML slice → mismatch.
    with pytest.raises(ValueError):
        _inject_header_into_table_slice(html_slice, '[["H1", "H2"]]')


@pytest.mark.offline
def test_inject_header_skips_html_slice_that_already_has_thead():
    from lightrag.chunker.paragraph_semantic import _inject_header_into_table_slice

    # HTML: a slice that already carries a <thead> is left untouched rather
    # than gaining a second one.
    slice_with_thead = (
        '<table id="tb-h" format="html">'
        "<thead><tr><th>H2a</th><th>H2b</th></tr></thead>"
        "<tbody><tr><td>d</td></tr></tbody></table>"
    )
    html_header = "<thead><tr><th>H1a</th><th>H1b</th></tr></thead>"
    assert _inject_header_into_table_slice(slice_with_thead, html_header) is None


@pytest.mark.offline
def test_full_pipeline_injects_html_thead_into_split_html_table(tmp_path):
    """End-to-end through chunking_by_paragraph_semantic: an oversized HTML table
    whose tables.json carries a raw <thead> header has that header re-injected
    (verbatim, spans preserved) into every non-first table chunk."""
    html_header = (
        '<thead><tr><th rowspan="2">Metric</th>'
        '<th colspan="2">Group</th></tr></thead>'
    )
    head_row = '<tr><th rowspan="2">Metric</th><th colspan="2">Group</th></tr>'
    data_rows = "".join(f"<tr><td>{'d' * 160}{i}</td></tr>" for i in range(6))
    table = (
        '<table id="tb-h" format="html">'
        f"<thead>{head_row}</thead><tbody>{data_rows}</tbody></table>"
    )
    body = "\n".join(["lead paragraph", table, "trailing paragraph"])

    # tables.json sidecar with an HTML-format header.
    (tmp_path / "doc.tables.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "tables": {
                    "tb-h": {
                        "id": "tb-h",
                        "format": "html",
                        "content": table,
                        "caption": "",
                        "table_header": html_header,
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    blocks_path = tmp_path / "doc.blocks.jsonl"
    blocks_path.write_text(
        json.dumps(
            {
                "type": "content",
                "heading": "Section",
                "parent_headings": [],
                "level": 2,
                "content": body,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    tokenizer = _make_tokenizer()
    chunks = chunking_by_paragraph_semantic(
        tokenizer,
        body,
        chunk_token_size=900,
        blocks_path=str(blocks_path),
        chunk_overlap_token_size=0,
    )

    table_chunks = [c for c in chunks if '<table id="tb-h"' in c["content"]]
    assert len(table_chunks) >= 2  # the table was split
    # Every table chunk carries exactly one <thead> with spans intact; the
    # first keeps its own, later ones got the sidecar header spliced in.
    for chunk in table_chunks:
        assert chunk["content"].count("<thead>") == 1, chunk["content"]
        assert 'rowspan="2"' in chunk["content"]
        assert 'colspan="2"' in chunk["content"]
    # At least one non-first chunk literally contains the injected header.
    assert any(html_header in c["content"] for c in table_chunks[1:])


@pytest.mark.offline
def test_split_table_text_multi_row_header_pinned_not_duplicated():
    # A multi-row header is pinned out of the split body, so every slice gets
    # the full header exactly once — never a cut remnant duplicated into a
    # later slice (e.g. [H1, H2, H2, ...]).
    from lightrag.table_markup import TABLE_TAG_RE

    tokenizer = _make_tokenizer()
    h1 = ["H1", "h1"]
    h2 = ["H2", "h2"]
    header_rows = [h1, h2]
    header_body = json.dumps(header_rows)
    rows = [h1, h2] + [[f"D{i}", "x" * 30] for i in range(6)]
    table_text = f'<table id="tb-1" format="json">{json.dumps(rows)}</table>'

    pieces = _split_table_text(
        table_text,
        tokenizer=tokenizer,
        target_max=160,
        target_ideal=120,
        last_min=10,
        header_body=header_body,
    )

    assert len(pieces) >= 2, "expected the data rows to be split across slices"
    for piece in pieces:
        body_rows = json.loads(TABLE_TAG_RE.match(piece).group("body"))
        # Full header present exactly once at the top of every slice.
        assert body_rows[:2] == header_rows, body_rows
        assert body_rows[2:].count(h1) == 0 and body_rows[2:].count(h2) == 0, body_rows
        assert _count_tokens(tokenizer, piece) <= 160


@pytest.mark.offline
def test_split_table_text_preserves_data_row_equal_to_header():
    # A genuine data row whose values equal the (single-row) header must be
    # preserved, not mistaken for a header remnant and dropped. Pinning the
    # header out of the body guarantees this: the body is split as pure data
    # and the header is prepended verbatim.
    from lightrag.table_markup import TABLE_TAG_RE

    tokenizer = _make_tokenizer()
    header_row = ["Label", "Value"]
    header_body = json.dumps([header_row])
    # The 3rd data row is identical to the header (e.g. a repeated label row).
    data_rows = [
        ["a", "x" * 40],
        ["b", "y" * 40],
        ["Label", "Value"],
        ["c", "z" * 40],
    ]
    rows = [header_row] + data_rows
    table_text = f'<table id="tb-1" format="json">{json.dumps(rows)}</table>'

    pieces = _split_table_text(
        table_text,
        tokenizer=tokenizer,
        target_max=130,
        target_ideal=90,
        last_min=10,
        header_body=header_body,
    )

    assert len(pieces) >= 2, "expected the data to be split across slices"
    # Reconstruct the data by stripping the one prepended header row from each
    # slice; every original data row (including the header-lookalike) survives.
    recovered: list = []
    for piece in pieces:
        body_rows = json.loads(TABLE_TAG_RE.match(piece).group("body"))
        assert body_rows[0] == header_row, body_rows
        recovered.extend(body_rows[1:])
    assert recovered == data_rows, recovered


@pytest.mark.offline
def test_split_table_text_resplits_reducible_slice_to_keep_header():
    # A reducible multi-row non-first slice whose wrapped table fits target_max
    # but whose table+header would exceed it must be split further at row
    # boundaries so each sub-slice still carries the header — rather than being
    # accepted whole and then silently left header-less. Here the splitter
    # emits two 2-row chunks at 119 tokens (target_max=120, effective cap=106),
    # which must be re-split so every non-first slice keeps its header.
    from lightrag.table_markup import TABLE_TAG_RE

    tokenizer = _make_tokenizer()
    header_row = ["HH", "HH"]
    header_body = json.dumps([header_row])
    rows = [header_row] + [[f"D{i}", "x" * 28] for i in range(4)]
    table_text = f'<table id="tb-1" format="json">{json.dumps(rows)}</table>'

    pieces = _split_table_text(
        table_text,
        tokenizer=tokenizer,
        target_max=120,
        target_ideal=120,
        last_min=10,
        header_body=header_body,
    )

    assert len(pieces) >= 3
    # Every piece honors the hard cap.
    assert all(_count_tokens(tokenizer, p) <= 120 for p in pieces), [
        _count_tokens(tokenizer, p) for p in pieces
    ]
    # No non-first slice is left header-less: each carries the recovered header
    # (every data row here is small enough that header + row fits target_max).
    for piece in pieces[1:]:
        body_rows = json.loads(TABLE_TAG_RE.match(piece).group("body"))
        assert body_rows[0] == header_row, body_rows


@pytest.mark.offline
def test_split_table_text_budgets_header_before_splitting():
    # With a header supplied, every emitted slice must stay ≤ target_max even
    # though the header is prepended into the non-first slices — proving the
    # header tokens were budgeted out BEFORE the split, not backfilled after.
    tokenizer = _make_tokenizer()
    header_body = '[["' + "H" * 60 + '", "' + "K" * 60 + '"]]'  # non-trivial header
    rows = [["H" * 60, "K" * 60]] + [[f"r{i}-" + "y" * 60, "z"] for i in range(8)]
    table_text = f'<table id="tb-1" format="json">{json.dumps(rows)}</table>'

    pieces = _split_table_text(
        table_text,
        tokenizer=tokenizer,
        target_max=300,
        target_ideal=220,
        last_min=64,
        header_body=header_body,
    )

    assert len(pieces) >= 3, "expected the oversized table to split into 3+ slices"
    # Every slice honors the cap including its (injected) header.
    assert all(_count_tokens(tokenizer, p) <= 300 for p in pieces), [
        _count_tokens(tokenizer, p) for p in pieces
    ]
    # Non-first slices carry the recovered header row.
    assert all(
        p.startswith('<table id="tb-1" format="json">[["' + "H" * 60)
        for p in pieces[1:]
    )


@pytest.mark.offline
def test_split_table_text_degrades_when_header_would_exceed_cap(
    caplog, _propagate_lightrag_logger
):
    # Cap edge: a single-row slice whose wrapped <table> sits just under
    # target_max yet leaves no room for the header it would need. Rather than
    # silently keeping it header-less (the old size-guard behaviour), the whole
    # table degrades to a recursive split so the header is preserved as text,
    # the hard cap still holds, and a warning is logged.
    tokenizer = _make_tokenizer()
    target_max = 200
    header_body = '[["HHHHHHHHHH", "KKKKKKKKKK"]]'  # ~30 tokens of header
    # row0 is the real header (pinned out of the body); row1 is small; row2 is a
    # single big row whose wrapped <table> size sits just under target_max, so
    # it fits the bare cap yet leaves no room for the ~30-token header.
    rows = [["HHHHHHHHHH", "KKKKKKKKKK"], ["s", "y"], ["x" * 140, "y"]]
    table_text = f'<table id="tb-1" format="json">{json.dumps(rows)}</table>'

    with caplog.at_level(logging.WARNING, logger="lightrag"):
        pieces = _split_table_text(
            table_text,
            tokenizer=tokenizer,
            target_max=target_max,
            target_ideal=150,
            last_min=64,
            header_body=header_body,
        )

    assert len(pieces) >= 2
    # The cap holds for every piece, and the whole table degraded: no piece
    # survives as a complete re-wrapped <table>...</table> slice.
    assert all(_count_tokens(tokenizer, p) <= target_max for p in pieces), [
        _count_tokens(tokenizer, p) for p in pieces
    ]
    assert not any(p.startswith("<table ") and p.endswith("</table>") for p in pieces)
    # The header content survives in the recursive fallback (not dropped).
    assert "HHHHHHHHHH" in "".join(pieces)
    assert any("degrading the whole table" in rec.message for rec in caplog.records), [
        rec.message for rec in caplog.records
    ]


@pytest.mark.offline
def test_split_table_text_pinned_header_oversized_row_preserves_header(
    caplog, _propagate_lightrag_logger
):
    # Codex P2 regression: a pinned-header JSON table whose data portion holds a
    # single row larger than target_max. The old code character-split the
    # header-less ``wrapped`` body, so the later injection pass (which can't
    # repair a non-<table> fragment) dropped the header entirely. The whole
    # table must now degrade to a recursive split that keeps the header text.
    tokenizer = _make_tokenizer()
    target_max = 300
    header_body = '[["HEADERCOL_A", "HEADERCOL_B"]]'
    # row0 repeats the header (pinned out of the body); row1 is a single huge
    # data row whose content alone blows past target_max.
    rows = [["HEADERCOL_A", "HEADERCOL_B"], ["x" * 2000, "y"]]
    table_text = f'<table id="tb-7" format="json">{json.dumps(rows)}</table>'

    with caplog.at_level(logging.WARNING, logger="lightrag"):
        pieces = _split_table_text(
            table_text,
            tokenizer=tokenizer,
            target_max=target_max,
            target_ideal=220,
            last_min=64,
            header_body=header_body,
        )

    assert len(pieces) >= 2
    assert all(_count_tokens(tokenizer, p) <= target_max for p in pieces), [
        _count_tokens(tokenizer, p) for p in pieces
    ]
    # The header column names survive somewhere in the emitted fragments.
    assert "HEADERCOL_A" in "".join(pieces)
    assert any(
        "tb-7" in rec.message and "degrading the whole table" in rec.message
        for rec in caplog.records
    ), [rec.message for rec in caplog.records]


@pytest.mark.offline
def test_header_injected_into_every_table_slice(tmp_path):
    chunks = _chunk_with_oversized_table(tmp_path)
    table_chunks = _table_chunks(chunks)

    assert len(table_chunks) >= 2, "expected the oversized table to be split"
    # Every slice's <table> now begins with the header row: the "first" slice
    # naturally (it kept the real header row), middle/last via re-injection.
    assert all(_slice_starts_with_header(c) for c in table_chunks), [
        c["content"][:90] for c in table_chunks
    ]


@pytest.mark.offline
def test_every_chunk_respects_cap_after_header_injection(tmp_path):
    # The Codex P2 finding: a near-cap slice + a recovered header must not push
    # the chunk past the hard cap. With split-time budgeting this holds for a
    # table that has substantial trailing prose glued to its last slice too.
    chunk_token_size = 800
    long_trailing = " ".join(f"word{i}" for i in range(120))  # ~ hundreds of tokens
    chunks = _chunk_with_oversized_table(
        tmp_path,
        num_data_rows=10,
        payload_size=120,
        trailing=long_trailing,
        chunk_token_size=chunk_token_size,
    )
    tokenizer = _make_tokenizer()
    assert _table_chunks(chunks), "expected a split header-bearing table"
    for c in chunks:
        assert _count_tokens(tokenizer, c["content"]) <= chunk_token_size, c["content"][
            :120
        ]
        assert c["tokens"] <= chunk_token_size


@pytest.mark.offline
def test_split_table_slices_never_remerge_no_duplicate_header(tmp_path):
    # Split slices are frozen against LevelMerge. Two slices of one table are
    # never recombined into a single chunk (which, post-injection, would carry
    # the header twice). The header appears at most once per chunk.
    chunks = _chunk_with_oversized_table(
        tmp_path, num_data_rows=8, payload_size=120, trailing="", chunk_token_size=800
    )
    table_chunks = _table_chunks(chunks)

    assert len(table_chunks) >= 2, "table must remain split, not reassembled"
    for c in table_chunks:
        # The header signature must not be duplicated inside any single chunk.
        assert c["content"].count('"H1", "H2"') <= 1, c["content"]


@pytest.mark.offline
def test_levelmerge_freezes_first_and_last_slices():
    # Deterministic freeze guard at the LevelMerge layer: a small "first" block
    # immediately followed by a small "last" block of the SAME 2-slice table —
    # both under target_ideal and same parent path — would have been reassembled
    # into one whole table before the freeze. Now neither merge direction fires,
    # so the two slices stay as two separate chunks (no duplicated header).
    tokenizer = _make_tokenizer()
    first = _new_block(
        heading="Section",
        parent_headings=["Doc"],
        level=2,
        paragraphs=[
            {
                "text": '<table id="tb-1" format="json">[["H1","H2"],["a","b"]]</table>',
                "is_table": True,
            }
        ],
        table_chunk_role="first",
        tokenizer=tokenizer,
    )
    last = _new_block(
        heading="Section",
        parent_headings=["Doc"],
        level=2,
        paragraphs=[
            {
                "text": '<table id="tb-1" format="json">[["H1","H2"],["c","d"]]</table>',
                "is_table": True,
            }
        ],
        table_chunk_role="last",
        tokenizer=tokenizer,
    )

    merged = _merge_small_blocks(
        [first, last],
        tokenizer=tokenizer,
        target_max=10_000,  # generous cap so size never blocks a merge
        target_ideal=10_000,  # both blocks count as below-ideal
        small_tail_threshold=10_000,
    )

    assert len(merged) == 2, "frozen split slices must not be reassembled"
    assert all(b["content"].count('"H1","H2"') == 1 for b in merged)


@pytest.mark.offline
def test_no_injection_when_source_table_has_no_header(tmp_path):
    # tables.json lists the table but WITHOUT a table_header field → nothing is
    # fabricated; only the first slice (which kept the real header) starts with it.
    chunks = _chunk_with_oversized_table(tmp_path, sidecar=None)
    table_chunks = _table_chunks(chunks)

    assert len(table_chunks) >= 2
    starts = [_slice_starts_with_header(c) for c in table_chunks]
    assert starts[0] is True, "first slice keeps the table's own header row"
    assert any(s is False for s in starts[1:]), "later slices must not be injected"


@pytest.mark.offline
def test_no_injection_when_tables_json_missing(tmp_path):
    # No tables.json at all → silent degrade: no error, no injection.
    chunks = _chunk_with_oversized_table(tmp_path, sidecar=_NO_SIDECAR)
    table_chunks = _table_chunks(chunks)

    assert len(table_chunks) >= 2
    starts = [_slice_starts_with_header(c) for c in table_chunks]
    assert starts[0] is True
    assert any(s is False for s in starts[1:])


@pytest.mark.offline
def test_injection_works_when_source_heading_empty(tmp_path):
    # A preamble block (empty source heading) still gets the header injected.
    chunks = _chunk_with_oversized_table(tmp_path, heading="")
    table_chunks = _table_chunks(chunks)

    assert len(table_chunks) >= 2
    assert all(_slice_starts_with_header(c) for c in table_chunks)
