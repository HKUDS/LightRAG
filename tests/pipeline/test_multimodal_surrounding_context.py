"""Unit tests for the native multimodal surrounding-context extractor.

The ``surrounding`` field is specified in ``docs/LightRAGSidecarFormat.md``
(section 7).

These tests use a 1:1 character-token mapping so the expected token
budgets in each scenario stay obvious without coupling to tiktoken's
BPE.  The helper functions exercised here are pure (no async, no
network), so the suite runs offline.
"""

import json

import pytest

from lightrag.multimodal_context import (
    DEFAULT_SURROUNDING_MAX_TOKENS,
    _resolve_surrounding_budget,
    build_surrounding,
    enrich_sidecars_with_surrounding,
    find_target_span,
    load_chunk_separators,
)
from lightrag.utils import Tokenizer, TokenizerInterface

pytestmark = pytest.mark.offline


class _CharTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


def _tokenizer() -> Tokenizer:
    return Tokenizer(model_name="char", tokenizer=_CharTokenizer())


def _surround(
    kind: str,
    item_id: str,
    block: str,
    *,
    leading: int = 2000,
    trailing: int = 2000,
    separators: list[str] | None = None,
) -> dict[str, str]:
    """Locate ``item_id`` in ``block`` and build its surrounding."""
    span = find_target_span(kind, item_id, block)
    assert span is not None, f"{kind}/{item_id} not found in block"
    return build_surrounding(
        kind=kind,
        block_content=block,
        span=span,
        tokenizer=_tokenizer(),
        leading_max_tokens=leading,
        trailing_max_tokens=trailing,
        separators=load_chunk_separators() if separators is None else separators,
    )


def _ntok(text: str) -> int:
    return len(_tokenizer().encode(text))


# ---------------------------------------------------------------------------
# Target-tag locator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "item_id", "content", "expected"),
    [
        pytest.param(
            "tables",
            "tb-abcd-0007",
            'before <table format="json" id="tb-abcd-0007">[[1,2]]</table> after',
            '<table format="json" id="tb-abcd-0007">[[1,2]]</table>',
            id="table-id-not-first-attribute",
        ),
        pytest.param(
            "drawings",
            "im-other",
            '<drawing id="im-1" />',
            None,
            id="unknown-id",
        ),
    ],
)
def test_find_target_span(kind, item_id, content, expected):
    span = find_target_span(kind, item_id, content)
    got = None if span is None else content[span[0] : span[1]]
    assert got == expected


# ---------------------------------------------------------------------------
# Tables surrounding: the target is the table, other tables are stripped
# before token counting.
# ---------------------------------------------------------------------------


def test_table_surrounding_strips_other_tables_before_counting():
    surr = _surround(
        "tables",
        "tb-target",
        '<table id="tb-other" format="json">[["a","b"],["c","d"]]</table> '
        "narrative text describing the report. "
        '<table id="tb-target" format="json">[["x","y"]]</table>'
        " concluding remarks.",
    )
    assert surr == {
        "leading": " narrative text describing the report. ",
        "trailing": " concluding remarks.",
    }


def test_table_surrounding_supports_cite_marker_and_strips_sibling_cites():
    surr = _surround(
        "tables",
        "tb-target",
        'prefix <cite type="table" refid="tb-other">表0</cite> '
        'narrative <cite type="table" refid="tb-target">表1</cite> suffix',
    )
    assert surr == {"leading": "prefix  narrative ", "trailing": " suffix"}


def test_table_surrounding_anchors_on_the_tag_not_a_preceding_cite():
    """A cite to the same id before the table must not become the anchor:
    the text between the two belongs to ``leading``, not ``trailing``."""
    surr = _surround(
        "tables",
        "tb-1",
        "as shown in Table 1, "
        '<cite type="table" refid="tb-1">Table 1</cite>: '
        '<table id="tb-1" format="json">[[1,2]]</table>'
        " concluding remarks.",
    )
    assert surr == {
        "leading": "as shown in Table 1, : ",
        "trailing": " concluding remarks.",
    }


# ---------------------------------------------------------------------------
# Segmentation and char fallback
# ---------------------------------------------------------------------------


def test_chunk_r_separators_env_drives_segment_boundary(monkeypatch):
    # Only the pipe character is a separator: text must split at '|'.
    monkeypatch.setenv("CHUNK_R_SEPARATORS", json.dumps(["|"]))
    seps = load_chunk_separators()
    assert seps == ["|"]
    # Budget = 12 chars/tokens; each segment is 10 chars including the
    # trailing '|', so exactly one whole segment fits on each side.
    surr = _surround(
        "drawings",
        "d",
        'aaaaaaaaa|bbbbbbbbb|<drawing id="d" />|ccccccccc|ddddddddd',
        leading=12,
        trailing=12,
        separators=seps,
    )
    assert surr["leading"] == "bbbbbbbbb|"


@pytest.mark.parametrize(
    ("block", "side", "kept_char"),
    [
        pytest.param("X" * 5000 + '<drawing id="d" />', "leading", "X", id="leading"),
        pytest.param('<drawing id="d" />' + "Y" * 5000, "trailing", "Y", id="trailing"),
    ],
)
def test_oversized_closest_segment_char_truncated(block, side, kept_char):
    """A single separator-less run larger than the budget is cut by
    characters, keeping the part closest to the target."""
    surr = _surround("drawings", "d", block, leading=200, trailing=200)
    assert surr[side] == kept_char * 200


# ---------------------------------------------------------------------------
# Drawings/equations surrounding: JSON / HTML table row trimming.
# ---------------------------------------------------------------------------


def test_drawing_surrounding_row_trims_oversized_json_table():
    rows = [[f"r{i}c0", f"r{i}c1"] for i in range(10)]
    block = (
        '<table id="tb-big" format="json">'
        + json.dumps(rows)
        + '</table> <drawing id="d" />'
    )
    surr = _surround("drawings", "d", block, leading=80, trailing=80)
    # Whole rows closest to the target survive, as a complete table.
    assert surr["leading"] == (
        '<table format="json">[["r8c0", "r8c1"], ["r9c0", "r9c1"]]</table> '
    )


def test_drawing_surrounding_row_trims_oversized_html_table():
    rows_html = "".join(f"<tr><td>r{i}c0</td><td>r{i}c1</td></tr>" for i in range(10))
    block = (
        '<drawing id="d" /> '
        f'<table id="tb-h" format="html"><tbody>{rows_html}</tbody></table>'
    )
    trailing = _surround("drawings", "d", block, leading=120, trailing=120)["trailing"]
    assert trailing.startswith(" <table ")
    assert trailing.endswith("</tbody></table>")
    assert _ntok(trailing) <= 120
    # For trailing we keep head rows.
    assert "r0c0" in trailing
    assert "r9c0" not in trailing


def test_drawing_surrounding_char_trims_oversized_single_json_row():
    big_table = (
        '<table id="tb-big" format="json">'
        + json.dumps([["A" * 200 + "TAIL"]])
        + "</table>"
    )
    leading = _surround(
        "drawings", "d", big_table + '<drawing id="d" />', leading=90, trailing=90
    )["leading"]
    assert leading.startswith("<table ")
    assert leading.endswith("</table>")
    assert "TAIL" in leading
    assert _ntok(leading) <= 90
    body = leading[leading.index(">") + 1 : -len("</table>")]
    assert isinstance(json.loads(body), list)


def test_drawing_surrounding_char_trims_oversized_single_html_row():
    big_table = (
        '<table id="tb-h" format="html">'
        f"<tbody><tr><td>HEAD{'B' * 200}</td></tr></tbody>"
        "</table>"
    )
    trailing = _surround(
        "drawings", "d", f'<drawing id="d" />{big_table}', leading=100, trailing=100
    )["trailing"]
    assert trailing.startswith("<table ")
    assert trailing.endswith("</table>")
    assert "<tr><td>HEAD" in trailing
    assert _ntok(trailing) <= 100


# ---------------------------------------------------------------------------
# Parser-internal markup stripping inside surrounding.  The per-tag rules
# are pinned by the ``strip_internal_multimodal_markup_for_extraction``
# tests in ``tests/pipeline/test_pipeline_release_closure.py``; here we pin
# only what the surrounding path adds: the cite wrapper is kept, and the
# cleaning happens *before* token-budgeted truncation.
# ---------------------------------------------------------------------------


def test_surrounding_strips_cite_refid_keeping_the_wrapper():
    surr = _surround(
        "drawings",
        "im-target",
        'see <cite type="table" refid="tb-x">Table 1</cite> for details. '
        '<drawing id="im-target" caption="Fig 3" /> end.',
    )
    # The wrapper survives (so the VLM/LLM can tell "Table 1" refers to an
    # external table, not inline prose) but the parser-internal refid is gone.
    assert surr["leading"] == 'see <cite type="table">Table 1</cite> for details. '


def test_strip_happens_before_budget_truncation():
    """The raw drawing tag (with id/path/src) exceeds the budget while its
    stripped form fits.  If stripping ran after truncation, the budget would
    be measured against the raw string and the tag cut, possibly mid-
    attribute."""
    surr = _surround(
        "equations",
        "eq-1",
        '<drawing id="im-prev" path="some/long/path.png" src="raw/long/path.png"'
        ' caption="C" />'
        '<equation id="eq-1" format="latex">y</equation>'
        " tail.",
        leading=30,
    )
    assert surr["leading"] == '<drawing caption="C" />'


# ---------------------------------------------------------------------------
# enrich_sidecars_with_surrounding
# ---------------------------------------------------------------------------


def _write_blocks(tmp_path, base, blocks):
    """Write ``<base>.blocks.jsonl`` with a meta header and ``(blockid,
    content)`` content blocks; return its path as a string."""
    blocks_path = tmp_path / f"{base}.blocks.jsonl"
    lines = [json.dumps({"type": "meta", "format": "lightrag"})]
    for blockid, content in blocks:
        lines.append(
            json.dumps(
                {
                    "type": "content",
                    "blockid": blockid,
                    "format": "plain_text",
                    "content": content,
                    "heading": "h",
                    "parent_headings": [],
                    "level": 1,
                },
                ensure_ascii=False,
            )
        )
    blocks_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(blocks_path)


def _write_sidecar(tmp_path, base, root_key, items):
    path = tmp_path / f"{base}.{root_key}.json"
    path.write_text(
        json.dumps({"version": "1.0", root_key: items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _item(item_id, blockid="b1", **extra):
    return {item_id: {"id": item_id, "blockid": blockid, "heading": "h", **extra}}


def _read_item(path, root_key, item_id):
    return json.loads(path.read_text(encoding="utf-8"))[root_key][item_id]


def test_enrich_only_updates_enabled_modalities(tmp_path):
    blocks_path = _write_blocks(
        tmp_path,
        "doc",
        [
            (
                "b1",
                'intro. <drawing id="im-1" path="a.png" src="a" />'
                ' middle <table id="tb-1" format="json">[["a"]]</table>'
                ' tail <equation id="eq-1" format="latex">e</equation> end.',
            )
        ],
    )
    drawings = _write_sidecar(tmp_path, "doc", "drawings", _item("im-1"))
    tables = _write_sidecar(tmp_path, "doc", "tables", _item("tb-1"))
    equations = _write_sidecar(tmp_path, "doc", "equations", _item("eq-1"))

    counts = enrich_sidecars_with_surrounding(
        blocks_path=blocks_path,
        enabled_modalities={"drawings"},
        tokenizer=_tokenizer(),
        leading_max_tokens=2000,
        trailing_max_tokens=2000,
    )
    assert counts == {"drawings": 1, "tables": 0, "equations": 0}
    assert _read_item(drawings, "drawings", "im-1")["surrounding"][
        "leading"
    ].startswith("intro.")
    assert "surrounding" not in _read_item(tables, "tables", "tb-1")
    assert "surrounding" not in _read_item(equations, "equations", "eq-1")


def test_enrich_does_not_cross_block_boundaries(tmp_path):
    blocks_path = _write_blocks(
        tmp_path,
        "doc",
        [
            ("bA", "earlier block content."),
            ("bB", 'later block. <drawing id="im-1" path="a.png" src="a" /> tail.'),
        ],
    )
    drawings = _write_sidecar(tmp_path, "doc", "drawings", _item("im-1", "bB"))

    enrich_sidecars_with_surrounding(
        blocks_path=blocks_path,
        enabled_modalities={"drawings"},
        tokenizer=_tokenizer(),
        leading_max_tokens=2000,
        trailing_max_tokens=2000,
    )
    assert _read_item(drawings, "drawings", "im-1")["surrounding"] == {
        "leading": "later block. ",
        "trailing": " tail.",
    }


def test_enrich_rewrite_overwrites_surrounding_and_keeps_other_fields(
    tmp_path, emulate_windows_newlines
):
    """Idempotent rewrite of an already-enriched sidecar.

    - A rerun with a smaller budget overwrites the prior ``surrounding``, so
      a budget change propagates without clearing sidecars first.
    - An existing ``llm_analyze_result`` neither blocks the backfill nor is
      touched by it.
    - The whole sidecar is re-serialized with ``indent=2``; it must keep LF
      line endings on every platform (``emulate_windows_newlines`` makes an
      unpinned ``newline`` write CRLF on Linux too).
    """
    blocks_path = _write_blocks(
        tmp_path,
        "doc",
        [("b1", "L" * 500 + '<drawing id="im-1" caption="C" />' + "T" * 500)],
    )
    analysis = {"name": "x", "summary": "", "detail_description": ""}
    drawings = _write_sidecar(
        tmp_path, "doc", "drawings", _item("im-1", llm_analyze_result=analysis)
    )

    def run(budget):
        counts = enrich_sidecars_with_surrounding(
            blocks_path=blocks_path,
            enabled_modalities={"drawings"},
            tokenizer=_tokenizer(),
            leading_max_tokens=budget,
            trailing_max_tokens=budget,
        )
        assert counts["drawings"] == 1
        return _read_item(drawings, "drawings", "im-1")

    first = run(300)
    assert first["surrounding"] == {"leading": "L" * 300, "trailing": "T" * 300}
    assert first["llm_analyze_result"] == analysis

    second = run(50)
    assert second["surrounding"] == {"leading": "L" * 50, "trailing": "T" * 50}
    assert second["llm_analyze_result"] == analysis
    assert b"\r\n" not in drawings.read_bytes(), "the rewrite wrote CRLF"


# ---------------------------------------------------------------------------
# Per-half token budgets via SURROUNDING_LEADING/TRAILING_MAX_TOKENS env vars.
# ---------------------------------------------------------------------------


def test_env_var_leading_and_trailing_budgets_apply_independently(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("SURROUNDING_LEADING_MAX_TOKENS", "5")
    monkeypatch.setenv("SURROUNDING_TRAILING_MAX_TOKENS", "20")
    blocks_path = _write_blocks(
        tmp_path,
        "doc",
        [("b1", "X" * 200 + '<drawing id="im-1" path="a.png" src="a" />' + "Y" * 200)],
    )
    drawings = _write_sidecar(tmp_path, "doc", "drawings", _item("im-1"))

    enrich_sidecars_with_surrounding(
        blocks_path=blocks_path,
        enabled_modalities={"drawings"},
        tokenizer=_tokenizer(),
    )
    assert _read_item(drawings, "drawings", "im-1")["surrounding"] == {
        "leading": "X" * 5,
        "trailing": "Y" * 20,
    }


def test_env_var_invalid_value_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("SURROUNDING_LEADING_MAX_TOKENS", "not-a-number")
    monkeypatch.setenv("SURROUNDING_TRAILING_MAX_TOKENS", "not-a-number")
    assert _resolve_surrounding_budget(None, None) == (
        DEFAULT_SURROUNDING_MAX_TOKENS,
        DEFAULT_SURROUNDING_MAX_TOKENS,
    )
