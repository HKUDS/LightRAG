"""Unit tests for ``lightrag.parser.markdown.extract`` (pure extraction).

Locks in the supported-subset contract: ATX heading splitting, fenced-code
suppression, pipe / HTML table recognition with headers, block-level ``$$``
math (and inline ``$`` left alone), and inline image resolution via a stubbed
resolver.
"""

from __future__ import annotations

from lightrag.parser.markdown.extract import (
    PREFACE_HEADING,
    ResolvedImage,
    extract_markdown,
)


class _StubResolver:
    """Resolves ``http(s)`` → external link, everything else → local bytes."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def resolve(self, src: str) -> ResolvedImage:
        self.calls.append(src)
        if src.startswith(("http://", "https://")):
            return ResolvedImage(kind="external", url=src, fmt="png")
        return ResolvedImage(
            kind="local",
            asset_ref="sha:" + src,
            data=b"BYTES:" + src.encode(),
            suggested_name="img.png",
            fmt="png",
        )


def _extract(md: str):
    return extract_markdown(md, image_resolver=_StubResolver())


def test_headings_split_blocks_and_track_parents():
    md = "# A\nintro\n## B\nbody\n### C\ndeep\n# D\nlast"
    ex = _extract(md)
    summary = [(b["heading"], b["level"], b["parent_headings"]) for b in ex.blocks]
    assert summary == [
        ("A", 1, []),
        ("B", 2, ["A"]),
        ("C", 3, ["A", "B"]),
        ("D", 1, []),
    ]
    # Heading line is rendered into the block body (markdown-style).
    assert ex.blocks[0]["content"].startswith("# A")


def test_content_before_first_heading_is_preface():
    ex = _extract("loose intro line\n\n# Real")
    assert ex.blocks[0]["heading"] == PREFACE_HEADING
    assert ex.blocks[0]["level"] == 0
    assert "loose intro line" in ex.blocks[0]["content"]


def test_fenced_code_suppresses_all_detection():
    md = "# H\n```python\n# not a heading\n$$ not eq $$\n| not | table |\n![x](no.png)\n```\n"
    ex = _extract(md)
    # Only the single real heading produced a block; fence content stays verbatim.
    assert len(ex.blocks) == 1
    assert not ex.tables and not ex.equations and not ex.drawings
    assert "# not a heading" in ex.blocks[0]["content"]
    assert "$$ not eq $$" in ex.blocks[0]["content"]


def test_tilde_fence_supported():
    md = "# H\n~~~\n## still code\n~~~\nafter"
    ex = _extract(md)
    assert len(ex.blocks) == 1
    assert "## still code" in ex.blocks[0]["content"]


def test_pipe_table_with_header():
    md = "| Name | Age |\n|------|-----|\n| Bob | 30 |\n| Sue | 25 |\n"
    ex = _extract(md)
    assert len(ex.tables) == 1
    (table,) = ex.tables.values()
    assert table["kind"] == "pipe"
    assert table["rows"] == [["Bob", "30"], ["Sue", "25"]]
    assert table["header"] == [["Name", "Age"]]


def test_pipe_table_requires_delimiter_row():
    # A pipe-containing line with no delimiter row underneath is plain text.
    md = "| just | text |\nnot a delimiter\n"
    ex = _extract(md)
    assert not ex.tables


def test_pipe_line_over_thematic_break_is_not_a_table():
    # A pipe-containing paragraph followed by a bare ``---`` (thematic break /
    # setext underline) has mismatched column counts (2 vs 1) and so is NOT a
    # GFM table — it must stay plain text rather than be misrecognised.
    md = "foo | bar\n---\nnext paragraph\n"
    ex = _extract(md)
    assert not ex.tables
    assert "foo | bar" in ex.blocks[0]["content"]


def test_pipe_table_column_count_must_match_header():
    # Delimiter row column count differs from the header → not a table.
    md = "| a | b | c |\n| --- | --- |\n| 1 | 2 | 3 |\n"
    ex = _extract(md)
    assert not ex.tables


def test_html_table_captured_verbatim_spanning_lines():
    md = (
        "<table>\n"
        "<thead><tr><th>K</th></tr></thead>\n"
        "<tbody><tr><td>a</td></tr></tbody>\n"
        "</table>\n"
    )
    ex = _extract(md)
    assert len(ex.tables) == 1
    (table,) = ex.tables.values()
    assert table["kind"] == "html"
    assert "<thead>" in table["html"] and "</table>" in table["html"]


def test_block_equation_single_and_multiline():
    md = "$$ E = mc^2 $$\n\ntext\n\n$$\n\\sum x\n$$\n"
    ex = _extract(md)
    latexes = list(ex.equations.values())
    assert "E = mc^2" in latexes
    assert "\\sum x" in latexes


def test_inline_dollar_not_recognized_as_equation():
    ex = _extract("cost is $5 and $10 today")
    assert not ex.equations
    assert "$5" in ex.blocks[0]["content"]


def test_unclosed_block_equation_is_plain_text():
    ex = _extract("$$\nE = mc^2\nno closing")
    assert not ex.equations
    assert "$$" in ex.blocks[0]["content"]


def test_inline_image_local_and_external_and_dedup():
    md = "![a](one.png) and ![b](one.png) and ![c](http://x/y.png)"
    resolver = _StubResolver()
    ex = extract_markdown(md, image_resolver=resolver)
    # Three occurrences → three drawings; same local src deduped to one asset.
    assert len(ex.drawings) == 3
    assert len(ex.assets) == 1
    kinds = sorted(d["kind"] for d in ex.drawings.values())
    assert kinds == ["external", "local", "local"]
    # Asset dedup is by ``asset_ref`` in extract (the resolver itself owns any
    # per-src call caching), so the stub sees one call per occurrence.
    assert resolver.calls.count("one.png") == 2


def test_base64_src_not_echoed_into_drawing_src():
    md = "![x](data:image/png;base64,QUJD)"

    class _B64:
        def resolve(self, src):
            return ResolvedImage(
                kind="local",
                asset_ref="sha:b64",
                data=b"ABC",
                suggested_name="image.png",
                fmt="png",
            )

    ex = extract_markdown(md, image_resolver=_B64())
    (drawing,) = ex.drawings.values()
    assert drawing["src"] == ""


def test_reference_style_image_not_recognized():
    # ``![alt][id]`` reference-style is out of the supported subset.
    ex = _extract("![alt][id]\n\n[id]: real.png")
    assert not ex.drawings
