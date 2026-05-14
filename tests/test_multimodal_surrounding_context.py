"""Unit tests for the native multimodal surrounding-context extractor.

See ``docs/NativeMultimodalSurroundingContextPlan-zh.md``.

These tests use a 1:1 character-token mapping so the expected token
budgets in each scenario stay obvious without coupling to tiktoken's
BPE.  The helper functions exercised here are pure (no async, no
network), so the suite runs offline.
"""

import json

import pytest

from lightrag.multimodal_context import (
    build_surrounding,
    enrich_sidecars_with_surrounding,
    find_target_span,
    load_chunk_separators,
)
from lightrag.utils import Tokenizer, TokenizerInterface


class _CharTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


def _tokenizer() -> Tokenizer:
    return Tokenizer(model_name="char", tokenizer=_CharTokenizer())


# ---------------------------------------------------------------------------
# Target-tag locator
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_find_target_span_drawing_in_mixed_content():
    content = (
        "leading text. "
        '<drawing id="dr-abcd-0001" format="png" path="img.png" src="img" /> '
        "trailing text."
    )
    span = find_target_span("drawings", "dr-abcd-0001", content)
    assert span is not None
    start, end = span
    assert content[start:end].startswith('<drawing id="dr-abcd-0001"')
    assert content[start:end].endswith("/>")


@pytest.mark.offline
def test_find_target_span_table_with_id_anywhere_in_attrs():
    # id is not first attribute — locator must still find it.
    content = (
        'before <table format="json" id="tb-abcd-0007">[[1,2],[3,4]]</table> after'
    )
    span = find_target_span("tables", "tb-abcd-0007", content)
    assert span is not None
    snippet = content[span[0] : span[1]]
    assert snippet.endswith("</table>")
    assert 'id="tb-abcd-0007"' in snippet


@pytest.mark.offline
def test_find_target_span_table_cite_marker():
    content = 'before <cite type="table" refid="tb-abcd-0007">表1</cite> after'
    span = find_target_span("tables", "tb-abcd-0007", content)
    assert span is not None
    assert content[span[0] : span[1]].startswith("<cite")


@pytest.mark.offline
def test_find_target_span_equation():
    content = 'A <equation id="eq-abcd-0002" format="latex">x^2</equation> B'
    span = find_target_span("equations", "eq-abcd-0002", content)
    assert span is not None
    assert content[span[0] : span[1]].endswith("</equation>")


@pytest.mark.offline
def test_find_target_span_unknown_id_returns_none():
    content = '<drawing id="dr-1" />'
    assert find_target_span("drawings", "dr-other", content) is None


# ---------------------------------------------------------------------------
# Drawings & equations surrounding
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_drawing_surrounding_kept_within_block_only():
    tok = _tokenizer()
    block = (
        "paragraph one ends. paragraph two. "
        '<drawing id="dr-1" path="a.png" src="a" /> '
        "paragraph three. paragraph four."
    )
    span = find_target_span("drawings", "dr-1", block)
    surr = build_surrounding(
        kind="drawings",
        block_content=block,
        span=span,
        tokenizer=tok,
        leading_max_tokens=2000,
        trailing_max_tokens=2000,
        separators=load_chunk_separators(),
    )
    assert surr["leading"].endswith("paragraph two. ")
    assert surr["trailing"].startswith(" paragraph three.")


@pytest.mark.offline
def test_equation_surrounding_protects_drawing_atom():
    tok = _tokenizer()
    block = (
        '<drawing id="dr-prev" path="a.png" src="a" caption="Fig 1" />'
        " intro text. "
        '<equation id="eq-1" format="latex">a+b=c</equation>'
        " conclusion text."
    )
    span = find_target_span("equations", "eq-1", block)
    surr = build_surrounding(
        kind="equations",
        block_content=block,
        span=span,
        tokenizer=tok,
        leading_max_tokens=2000,
        trailing_max_tokens=2000,
        separators=load_chunk_separators(),
    )
    # Parser-internal id/path/src are stripped, but caption survives and
    # the drawing tag stays atomic (not cut in half).
    assert '<drawing caption="Fig 1" />' in surr["leading"]
    assert "/>" in surr["leading"]
    # No half-open drawing/equation tags
    assert surr["leading"].count("<drawing") == surr["leading"].count("/>")


# ---------------------------------------------------------------------------
# Tables surrounding: other tables must be stripped before token counting.
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_table_surrounding_strips_other_tables_before_counting():
    tok = _tokenizer()
    block = (
        '<table id="tb-other" format="json">[["a","b"],["c","d"]]</table> '
        "narrative text describing the report. "
        '<table id="tb-target" format="json">[["x","y"]]</table>'
        " concluding remarks."
    )
    span = find_target_span("tables", "tb-target", block)
    surr = build_surrounding(
        kind="tables",
        block_content=block,
        span=span,
        tokenizer=tok,
        leading_max_tokens=2000,
        trailing_max_tokens=2000,
        separators=load_chunk_separators(),
    )
    # Sibling table must NOT appear in surrounding.
    assert "<table" not in surr["leading"]
    assert "</table>" not in surr["leading"]
    assert "<table" not in surr["trailing"]
    assert "narrative text" in surr["leading"]
    assert "concluding remarks" in surr["trailing"]


@pytest.mark.offline
def test_table_surrounding_supports_cite_marker_and_strips_sibling_cites():
    tok = _tokenizer()
    block = (
        'prefix <cite type="table" refid="tb-other">表0</cite> '
        'narrative <cite type="table" refid="tb-target">表1</cite> suffix'
    )
    span = find_target_span("tables", "tb-target", block)
    surr = build_surrounding(
        kind="tables",
        block_content=block,
        span=span,
        tokenizer=tok,
        leading_max_tokens=2000,
        trailing_max_tokens=2000,
        separators=load_chunk_separators(),
    )
    assert "tb-other" not in surr["leading"]
    assert "表0" not in surr["leading"]
    assert "narrative " in surr["leading"]
    assert surr["trailing"] == " suffix"


# ---------------------------------------------------------------------------
# Custom CHUNK_R_SEPARATORS via env
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_chunk_r_separators_env_drives_segment_boundary(monkeypatch):
    # Only the pipe character is a separator: text must split at '|'.
    monkeypatch.setenv("CHUNK_R_SEPARATORS", json.dumps(["|"]))
    seps = load_chunk_separators()
    assert seps == ["|"]
    tok = _tokenizer()
    # 3 segments separated by '|'; budget = 12 chars/tokens; each seg is
    # 10 chars including the trailing '|', so 1 whole segment fits, 2 do not.
    block = 'aaaaaaaaa|bbbbbbbbb|<drawing id="d" />|ccccccccc|ddddddddd'
    span = find_target_span("drawings", "d", block)
    surr = build_surrounding(
        kind="drawings",
        block_content=block,
        span=span,
        tokenizer=tok,
        leading_max_tokens=12,
        trailing_max_tokens=12,
        separators=seps,
    )
    # Leading should end at a '|' boundary (one whole segment), not be
    # char-truncated.
    assert surr["leading"].endswith("|")
    # And contain whole segment closest to target.
    assert "bbbbbbbbb|" in surr["leading"]


# ---------------------------------------------------------------------------
# Char fallback when the closest segment alone exceeds the budget.
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_oversized_closest_segment_char_truncated():
    tok = _tokenizer()
    # Single huge "segment" (no separator) right before the target.
    big = "X" * 5000
    block = big + '<drawing id="d" />'
    span = find_target_span("drawings", "d", block)
    surr = build_surrounding(
        kind="drawings",
        block_content=block,
        span=span,
        tokenizer=tok,
        leading_max_tokens=200,
        trailing_max_tokens=200,
        separators=load_chunk_separators(),
    )
    assert len(tok.encode(surr["leading"])) <= 200
    assert surr["trailing"] == ""
    # The suffix should be a tail of the X-run.
    assert surr["leading"].endswith("X")


@pytest.mark.offline
def test_oversized_trailing_char_truncated_at_head():
    tok = _tokenizer()
    big = "Y" * 5000
    block = '<drawing id="d" />' + big
    span = find_target_span("drawings", "d", block)
    surr = build_surrounding(
        kind="drawings",
        block_content=block,
        span=span,
        tokenizer=tok,
        leading_max_tokens=200,
        trailing_max_tokens=200,
        separators=load_chunk_separators(),
    )
    assert len(tok.encode(surr["trailing"])) <= 200
    assert surr["trailing"].startswith("Y")


# ---------------------------------------------------------------------------
# Drawings/equations surrounding: JSON / HTML table row trimming.
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_drawing_surrounding_row_trims_oversized_json_table():
    tok = _tokenizer()
    # 10 rows of repeating cells; whole table is ~> budget.
    rows = [[f"r{i}c0", f"r{i}c1"] for i in range(10)]
    big_table = '<table id="tb-big" format="json">' + json.dumps(rows) + "</table>"
    block = big_table + ' <drawing id="d" />'
    span = find_target_span("drawings", "d", block)
    # Budget chosen so only a few rows of the JSON table fit.
    surr = build_surrounding(
        kind="drawings",
        block_content=block,
        span=span,
        tokenizer=tok,
        leading_max_tokens=80,
        trailing_max_tokens=80,
        separators=load_chunk_separators(),
    )
    # Result must be a complete (smaller) <table>...</table>, contain
    # closing tag, and fit within budget.
    leading = surr["leading"]
    assert "<table " in leading
    assert (
        leading.rstrip().endswith("</table>")
        or leading.rstrip().endswith("</table> ")
        or "</table>" in leading
    )
    assert len(tok.encode(leading)) <= 80
    # Should keep tail rows (closest to target — last rows by index)
    assert "r9c0" in leading
    # Should not include rows from the far side.
    assert "r0c0" not in leading


@pytest.mark.offline
def test_drawing_surrounding_row_trims_oversized_html_table():
    tok = _tokenizer()
    rows_html = "".join(f"<tr><td>r{i}c0</td><td>r{i}c1</td></tr>" for i in range(10))
    body = f"<tbody>{rows_html}</tbody>"
    big_table = f'<table id="tb-h" format="html">{body}</table>'
    block = f'<drawing id="d" /> {big_table}'
    span = find_target_span("drawings", "d", block)
    surr = build_surrounding(
        kind="drawings",
        block_content=block,
        span=span,
        tokenizer=tok,
        leading_max_tokens=120,
        trailing_max_tokens=120,
        separators=load_chunk_separators(),
    )
    trailing = surr["trailing"]
    assert "<table " in trailing
    assert "</table>" in trailing
    assert "<tbody>" in trailing
    assert "</tbody>" in trailing
    assert len(tok.encode(trailing)) <= 120
    # For trailing we keep head rows.
    assert "r0c0" in trailing
    assert "r9c0" not in trailing


@pytest.mark.offline
def test_drawing_surrounding_char_trims_oversized_single_json_row():
    tok = _tokenizer()
    row_text = "A" * 200 + "TAIL"
    big_table = (
        '<table id="tb-big" format="json">'
        + json.dumps([[row_text]], ensure_ascii=False)
        + "</table>"
    )
    block = big_table + '<drawing id="d" />'
    span = find_target_span("drawings", "d", block)
    surr = build_surrounding(
        kind="drawings",
        block_content=block,
        span=span,
        tokenizer=tok,
        leading_max_tokens=90,
        trailing_max_tokens=90,
        separators=load_chunk_separators(),
    )

    leading = surr["leading"]
    assert leading.startswith("<table ")
    assert leading.endswith("</table>")
    assert "TAIL" in leading
    assert len(tok.encode(leading)) <= 90

    body = leading[leading.index(">") + 1 : -len("</table>")]
    parsed = json.loads(body)
    assert isinstance(parsed, list)


@pytest.mark.offline
def test_drawing_surrounding_char_trims_oversized_single_html_row():
    tok = _tokenizer()
    row_text = "HEAD" + "B" * 200
    big_table = (
        '<table id="tb-h" format="html">'
        f"<tbody><tr><td>{row_text}</td></tr></tbody>"
        "</table>"
    )
    block = f'<drawing id="d" />{big_table}'
    span = find_target_span("drawings", "d", block)
    surr = build_surrounding(
        kind="drawings",
        block_content=block,
        span=span,
        tokenizer=tok,
        leading_max_tokens=100,
        trailing_max_tokens=100,
        separators=load_chunk_separators(),
    )

    trailing = surr["trailing"]
    assert trailing.startswith("<table ")
    assert trailing.endswith("</table>")
    assert "<tr><td>" in trailing
    assert "HEAD" in trailing
    assert len(tok.encode(trailing)) <= 100


# ---------------------------------------------------------------------------
# enrich_sidecars_with_surrounding: idempotency + modality gating.
# ---------------------------------------------------------------------------


def _write_blocks(tmp_path, base, blocks):
    blocks_path = tmp_path / f"{base}.blocks.jsonl"
    lines = [json.dumps({"type": "meta", "format": "lightrag"})]
    for b in blocks:
        lines.append(json.dumps(b, ensure_ascii=False))
    blocks_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return blocks_path


def _write_sidecar(path, root_key, items):
    path.write_text(
        json.dumps(
            {"version": "1.0", root_key: items},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


@pytest.mark.offline
def test_enrich_only_updates_enabled_modalities(tmp_path):
    base = "doc"
    blockid = "b1"
    content = (
        "intro. "
        '<drawing id="dr-1" path="a.png" src="a" />'
        " middle "
        '<table id="tb-1" format="json">[["a"]]</table>'
        " tail "
        '<equation id="eq-1" format="latex">e</equation>'
        " end."
    )
    _write_blocks(
        tmp_path,
        base,
        [
            {
                "type": "content",
                "blockid": blockid,
                "format": "plain_text",
                "content": content,
                "heading": "h",
                "parent_headings": [],
                "level": 1,
            }
        ],
    )
    drawings_path = tmp_path / f"{base}.drawings.json"
    tables_path = tmp_path / f"{base}.tables.json"
    equations_path = tmp_path / f"{base}.equations.json"
    _write_sidecar(
        drawings_path,
        "drawings",
        {"dr-1": {"id": "dr-1", "blockid": blockid, "heading": "h"}},
    )
    _write_sidecar(
        tables_path,
        "tables",
        {"tb-1": {"id": "tb-1", "blockid": blockid, "heading": "h"}},
    )
    _write_sidecar(
        equations_path,
        "equations",
        {"eq-1": {"id": "eq-1", "blockid": blockid, "heading": "h"}},
    )

    counts = enrich_sidecars_with_surrounding(
        blocks_path=str(tmp_path / f"{base}.blocks.jsonl"),
        enabled_modalities={"drawings"},
        tokenizer=_tokenizer(),
        leading_max_tokens=2000,
        trailing_max_tokens=2000,
    )
    assert counts["drawings"] == 1
    assert counts["tables"] == 0
    assert counts["equations"] == 0

    drawings = json.loads(drawings_path.read_text(encoding="utf-8"))
    tables = json.loads(tables_path.read_text(encoding="utf-8"))
    equations = json.loads(equations_path.read_text(encoding="utf-8"))
    assert "surrounding" in drawings["drawings"]["dr-1"]
    assert drawings["drawings"]["dr-1"]["surrounding"]["leading"].startswith("intro.")
    assert "surrounding" not in tables["tables"]["tb-1"]
    assert "surrounding" not in equations["equations"]["eq-1"]


@pytest.mark.offline
def test_enrich_runs_even_when_llm_analyze_result_present(tmp_path):
    """Idempotency: existing ``llm_analyze_result`` does not block
    surrounding backfill — we treat the two fields as independent."""
    base = "doc"
    blockid = "b1"
    content = 'prefix. <drawing id="dr-1" path="a.png" src="a" /> suffix.'
    _write_blocks(
        tmp_path,
        base,
        [
            {
                "type": "content",
                "blockid": blockid,
                "format": "plain_text",
                "content": content,
                "heading": "h",
                "parent_headings": [],
                "level": 1,
            }
        ],
    )
    drawings_path = tmp_path / f"{base}.drawings.json"
    _write_sidecar(
        drawings_path,
        "drawings",
        {
            "dr-1": {
                "id": "dr-1",
                "blockid": blockid,
                "heading": "h",
                "llm_analyze_result": {
                    "name": "x",
                    "summary": "",
                    "detail_description": "",
                },
            }
        },
    )

    counts = enrich_sidecars_with_surrounding(
        blocks_path=str(tmp_path / f"{base}.blocks.jsonl"),
        enabled_modalities={"drawings"},
        tokenizer=_tokenizer(),
        leading_max_tokens=2000,
        trailing_max_tokens=2000,
    )
    assert counts["drawings"] == 1
    payload = json.loads(drawings_path.read_text(encoding="utf-8"))
    item = payload["drawings"]["dr-1"]
    assert item["llm_analyze_result"]["name"] == "x"  # untouched
    assert item["surrounding"]["leading"].startswith("prefix.")
    assert item["surrounding"]["trailing"].startswith(" suffix.")


@pytest.mark.offline
def test_enrich_does_not_cross_block_boundaries(tmp_path):
    base = "doc"
    block_a = "earlier block content."
    block_b = 'later block. <drawing id="dr-1" path="a.png" src="a" /> tail.'
    _write_blocks(
        tmp_path,
        base,
        [
            {
                "type": "content",
                "blockid": "bA",
                "format": "plain_text",
                "content": block_a,
                "heading": "h1",
                "parent_headings": [],
                "level": 1,
            },
            {
                "type": "content",
                "blockid": "bB",
                "format": "plain_text",
                "content": block_b,
                "heading": "h2",
                "parent_headings": [],
                "level": 1,
            },
        ],
    )
    drawings_path = tmp_path / f"{base}.drawings.json"
    _write_sidecar(
        drawings_path,
        "drawings",
        {"dr-1": {"id": "dr-1", "blockid": "bB", "heading": "h2"}},
    )

    enrich_sidecars_with_surrounding(
        blocks_path=str(tmp_path / f"{base}.blocks.jsonl"),
        enabled_modalities={"drawings"},
        tokenizer=_tokenizer(),
        leading_max_tokens=2000,
        trailing_max_tokens=2000,
    )
    payload = json.loads(drawings_path.read_text(encoding="utf-8"))
    surr = payload["drawings"]["dr-1"]["surrounding"]
    # Must come from block B only — content of block A absent.
    assert "earlier block content" not in surr["leading"]
    assert surr["leading"].startswith("later block.")


# ---------------------------------------------------------------------------
# Per-half token budgets via SURROUNDING_LEADING/TRAILING_MAX_TOKENS env vars.
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_env_var_leading_and_trailing_budgets_apply_independently(
    tmp_path, monkeypatch
):
    # Asymmetric budgets must produce asymmetric leading / trailing sizes.
    monkeypatch.setenv("SURROUNDING_LEADING_MAX_TOKENS", "5")
    monkeypatch.setenv("SURROUNDING_TRAILING_MAX_TOKENS", "20")

    base = "doc"
    blockid = "b1"
    content = "X" * 200 + '<drawing id="dr-1" path="a.png" src="a" />' + "Y" * 200
    _write_blocks(
        tmp_path,
        base,
        [
            {
                "type": "content",
                "blockid": blockid,
                "format": "plain_text",
                "content": content,
                "heading": "h",
                "parent_headings": [],
                "level": 1,
            }
        ],
    )
    drawings_path = tmp_path / f"{base}.drawings.json"
    _write_sidecar(
        drawings_path,
        "drawings",
        {"dr-1": {"id": "dr-1", "blockid": blockid, "heading": "h"}},
    )

    tok = _tokenizer()
    enrich_sidecars_with_surrounding(
        blocks_path=str(tmp_path / f"{base}.blocks.jsonl"),
        enabled_modalities={"drawings"},
        tokenizer=tok,
    )

    surr = json.loads(drawings_path.read_text(encoding="utf-8"))["drawings"]["dr-1"][
        "surrounding"
    ]
    assert len(tok.encode(surr["leading"])) <= 5
    assert len(tok.encode(surr["trailing"])) <= 20
    # Trailing is allowed to use its larger budget, so it must be strictly
    # longer than leading here.
    assert len(surr["trailing"]) > len(surr["leading"])


# ---------------------------------------------------------------------------
# Parser-internal markup stripping inside surrounding (mirrors what
# ``strip_internal_multimodal_markup_for_extraction`` does for chunk
# content before entity extraction).  The cleaning happens *before*
# token-budgeted truncation, so the saved budget reflects what the
# LLM actually receives and a truncation point can never land inside
# an unprocessed ``id="…"`` attribute.
# ---------------------------------------------------------------------------


@pytest.mark.offline
def test_surrounding_strips_drawing_id_path_src():
    tok = _tokenizer()
    block = (
        "leading prose. "
        '<drawing id="dr-x" path="figs/a.png" src="raw/a.png" caption="Fig 1" />'
        " between. "
        '<equation id="eq-target" format="latex">x=1</equation>'
        " trailing prose."
    )
    span = find_target_span("equations", "eq-target", block)
    surr = build_surrounding(
        kind="equations",
        block_content=block,
        span=span,
        tokenizer=tok,
        leading_max_tokens=2000,
        trailing_max_tokens=2000,
        separators=load_chunk_separators(),
    )
    leading = surr["leading"]
    assert '<drawing caption="Fig 1" />' in leading
    assert 'id="dr-x"' not in leading
    assert "path=" not in leading
    assert "src=" not in leading


@pytest.mark.offline
def test_surrounding_strips_table_internal_id():
    tok = _tokenizer()
    block = (
        "prefix. "
        '<table id="tb-x" format="json" caption="Sales">[[1,2],[3,4]]</table>'
        " between. "
        '<drawing id="dr-target" caption="Fig 2" />'
        " suffix."
    )
    span = find_target_span("drawings", "dr-target", block)
    surr = build_surrounding(
        kind="drawings",
        block_content=block,
        span=span,
        tokenizer=tok,
        leading_max_tokens=2000,
        trailing_max_tokens=2000,
        separators=load_chunk_separators(),
    )
    leading = surr["leading"]
    assert '<table format="json" caption="Sales">[[1,2],[3,4]]</table>' in leading
    assert 'id="tb-x"' not in leading


@pytest.mark.offline
def test_surrounding_strips_cite_refid_keeping_visible_text():
    tok = _tokenizer()
    block = (
        "see "
        '<cite type="table" refid="tb-x">Table 1</cite>'
        " for details. "
        '<drawing id="dr-target" caption="Fig 3" />'
        " end."
    )
    span = find_target_span("drawings", "dr-target", block)
    surr = build_surrounding(
        kind="drawings",
        block_content=block,
        span=span,
        tokenizer=tok,
        leading_max_tokens=2000,
        trailing_max_tokens=2000,
        separators=load_chunk_separators(),
    )
    leading = surr["leading"]
    # Surrounding path uses keep_cite_tag=True: the cite wrapper survives
    # (so the VLM/LLM can tell "Table 1" is a reference to an external
    # table, not inline prose) but the parser-internal refid is gone.
    assert '<cite type="table">Table 1</cite>' in leading
    assert "refid=" not in leading
    assert "tb-x" not in leading


@pytest.mark.offline
def test_surrounding_keeps_equation_cite_tag_and_strips_refid():
    """In production, equations without LaTeX content emit as
    ``<cite type="equation" refid="eq-…">公式 N</cite>`` rather than a
    full ``<equation>`` tag.  Surrounding must keep the wrapper so the
    multimodal analyzer can recognize the visible label as an external
    referent, not inline prose."""
    tok = _tokenizer()
    block = (
        "see "
        '<cite type="equation" refid="eq-y">公式 2</cite>'
        " above. "
        '<drawing id="dr-target" caption="Fig 4" />'
        " end."
    )
    span = find_target_span("drawings", "dr-target", block)
    surr = build_surrounding(
        kind="drawings",
        block_content=block,
        span=span,
        tokenizer=tok,
        leading_max_tokens=2000,
        trailing_max_tokens=2000,
        separators=load_chunk_separators(),
    )
    leading = surr["leading"]
    assert '<cite type="equation">公式 2</cite>' in leading
    assert "refid=" not in leading
    assert "eq-y" not in leading


@pytest.mark.offline
def test_strip_happens_before_budget_truncation():
    """Regression guard for the strip-before-truncate ordering.

    Constructs a leading source whose raw form (with id/path/src) exceeds
    the budget while its stripped form fits.  If strip ran *after*
    truncation, the budget would be measured against the bloated raw
    string and the saved surrounding would be cut early (possibly mid-
    attribute, leaving ``id="…`` residue).
    """
    tok = _tokenizer()
    # Raw drawing tag including attrs (~67 chars), stripped form is
    # just '<drawing caption="C" />' (~24 chars).  Budget at 30 sits
    # between the two — raw is too big, stripped fits.
    block = (
        '<drawing id="dr-prev" path="some/long/path.png" src="raw/long/path.png"'
        ' caption="C" />'
        '<equation id="eq-1" format="latex">y</equation>'
        " tail."
    )
    span = find_target_span("equations", "eq-1", block)
    surr = build_surrounding(
        kind="equations",
        block_content=block,
        span=span,
        tokenizer=tok,
        leading_max_tokens=30,
        trailing_max_tokens=2000,
        separators=load_chunk_separators(),
    )
    leading = surr["leading"]
    # Whole stripped tag must be present — proves strip ran before
    # the budget gate.
    assert leading == '<drawing caption="C" />'
    # And no parser-internal markers leaked through.
    assert "id=" not in leading
    assert "path=" not in leading
    assert "src=" not in leading


@pytest.mark.offline
def test_enrich_overwrites_surrounding_when_budget_changes(tmp_path):
    """Idempotency: rerunning with a smaller budget overwrites the prior
    surrounding, demonstrating that ``SURROUNDING_LEADING_MAX_TOKENS``
    changes propagate without needing to clear sidecars first."""
    base = "doc"
    blockid = "b1"
    content = "L" * 500 + '<drawing id="dr-1" caption="C" />' + "T" * 500
    _write_blocks(
        tmp_path,
        base,
        [
            {
                "type": "content",
                "blockid": blockid,
                "format": "plain_text",
                "content": content,
                "heading": "h",
                "parent_headings": [],
                "level": 1,
            }
        ],
    )
    drawings_path = tmp_path / f"{base}.drawings.json"
    _write_sidecar(
        drawings_path,
        "drawings",
        {"dr-1": {"id": "dr-1", "blockid": blockid, "heading": "h"}},
    )

    tok = _tokenizer()
    enrich_sidecars_with_surrounding(
        blocks_path=str(tmp_path / f"{base}.blocks.jsonl"),
        enabled_modalities={"drawings"},
        tokenizer=tok,
        leading_max_tokens=300,
        trailing_max_tokens=300,
    )
    first = json.loads(drawings_path.read_text(encoding="utf-8"))["drawings"]["dr-1"][
        "surrounding"
    ]
    first_leading_len = len(first["leading"])
    first_trailing_len = len(first["trailing"])

    enrich_sidecars_with_surrounding(
        blocks_path=str(tmp_path / f"{base}.blocks.jsonl"),
        enabled_modalities={"drawings"},
        tokenizer=tok,
        leading_max_tokens=50,
        trailing_max_tokens=50,
    )
    second = json.loads(drawings_path.read_text(encoding="utf-8"))["drawings"]["dr-1"][
        "surrounding"
    ]
    # New budget is smaller, so saved surrounding must shrink — proving
    # the previous value was overwritten, not preserved.
    assert len(second["leading"]) < first_leading_len
    assert len(second["trailing"]) < first_trailing_len
    assert len(tok.encode(second["leading"])) <= 50
    assert len(tok.encode(second["trailing"])) <= 50


@pytest.mark.offline
def test_env_var_invalid_value_falls_back_to_default(monkeypatch):
    # An unparseable env value must not crash; it falls back to 2000.
    monkeypatch.setenv("SURROUNDING_LEADING_MAX_TOKENS", "not-a-number")
    monkeypatch.setenv("SURROUNDING_TRAILING_MAX_TOKENS", "not-a-number")
    from lightrag.multimodal_context import (
        DEFAULT_SURROUNDING_MAX_TOKENS,
        _resolve_surrounding_budget,
    )

    leading, trailing = _resolve_surrounding_budget(None, None)
    assert leading == DEFAULT_SURROUNDING_MAX_TOKENS
    assert trailing == DEFAULT_SURROUNDING_MAX_TOKENS
