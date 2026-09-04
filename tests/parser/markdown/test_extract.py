"""Unit tests for ``lightrag.parser.markdown.extract`` (pure extraction).

Locks in the supported-subset contract: ATX heading splitting, fenced-code
suppression, pipe / HTML table recognition with headers, block-level ``$$``
math (and inline ``$`` left alone), and inline image resolution via a stubbed
resolver.
"""

from __future__ import annotations

import itertools
import re
import time

import pytest

from lightrag.parser.markdown.extract import (
    PREFACE_HEADING,
    ResolvedImage,
    _replace_inline_images,
    extract_markdown,
)


_LEGACY_INLINE_IMAGE_RE = re.compile(
    r'!\[(?P<alt>[^\]]*)\]\(\s*(?P<src><[^>]*>|[^)\s]+)(?:\s+"[^"]*")?\s*\)'
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


def test_skipped_heading_levels_keep_same_level_headings_as_siblings():
    ex = _extract("# A\n### B\n### C")
    summary = [(b["heading"], b["parent_headings"]) for b in ex.blocks]
    assert summary == [
        ("A", []),
        ("B", ["A"]),
        ("C", ["A"]),
    ]


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


def test_html_table_preserves_same_line_trailing_text():
    ex = _extract("<table><tr><td>a</td></tr></table> trailing prose")

    (table,) = ex.tables.values()
    assert table["html"] == "<table><tr><td>a</td></tr></table>"
    assert ex.blocks[0]["content"].endswith(" trailing prose")


def test_html_table_preserves_trailing_text_after_multiline_close():
    ex = _extract("<table>\n<tr><td>a</td></tr>\n</table> trailing prose")

    (table,) = ex.tables.values()
    assert table["html"].endswith("</table>")
    assert ex.blocks[0]["content"].endswith(" trailing prose")


def test_html_table_preserves_trailing_text_after_unicode_content():
    ex = _extract("<table><tr><td>İ</td></tr></table>tail")

    (table,) = ex.tables.values()
    assert table["html"] == "<table><tr><td>İ</td></tr></table>"
    assert ex.blocks[0]["content"].endswith("tail")


def test_html_table_ignores_closing_tag_text_inside_attribute():
    ex = _extract('<table data-note="</table>"><tr><td>a</td></tr></table> tail')

    (table,) = ex.tables.values()
    assert table["html"] == ('<table data-note="</table>"><tr><td>a</td></tr></table>')
    assert ex.blocks[0]["content"].endswith(" tail")


def test_html_table_ignores_quotes_inside_comments():
    ex = _extract("<table><!-- don't remove --><tr><td>a</td></tr></table> tail")

    (table,) = ex.tables.values()
    assert table["html"] == ("<table><!-- don't remove --><tr><td>a</td></tr></table>")
    assert ex.blocks[0]["content"].endswith(" tail")


def test_html_table_processes_inline_image_in_trailing_text():
    resolver = _StubResolver()
    ex = extract_markdown(
        "<table><tr><td>a</td></tr></table> ![plot](plot.png)",
        image_resolver=resolver,
    )

    assert len(ex.drawings) == 1
    assert resolver.calls == ["plot.png"]
    assert "![plot](plot.png)" not in ex.blocks[0]["content"]


def test_html_table_ignores_unmatched_quote_inside_textarea():
    """Codex finding: an apostrophe inside RCDATA content (textarea/title)
    must not be tracked as an attribute quote -- it previously left the
    parser's quote-tracking state open indefinitely, swallowing the real
    </table> that followed."""
    ex = _extract(
        "<table>\n<tr><td><textarea>x<y's value</textarea></td></tr>\n</table> tail"
    )

    (table,) = ex.tables.values()
    assert table["html"].endswith("</table>")
    assert "<y's value</textarea>" in table["html"]
    assert ex.blocks[0]["content"].endswith(" tail")


def test_html_table_raw_mode_waits_for_opening_tag_to_close():
    """A decoy closing sequence inside the textarea's own opening-tag
    attributes (still ordinary markup, quote-tracked as normal) must not be
    mistaken for the real content boundary -- raw mode must not begin until
    that opening tag's own unquoted ">" is reached."""
    ex = _extract(
        '<table><textarea data-note="</textarea>">x<y\'s value</textarea></table> tail'
    )

    (table,) = ex.tables.values()
    assert table["html"].endswith("</table>")
    assert 'data-note="</textarea>">x<y\'s value</textarea>' in table["html"]
    assert ex.blocks[0]["content"].endswith(" tail")


def test_html_table_raw_mode_closing_tag_split_across_lines():
    """A raw-text closing tag broken by a line break, e.g. </textarea then a
    bare > on the next line, is still valid HTML and must still be found --
    a per-line search would miss it and lose the rest of the document."""
    ex = _extract("<table><textarea>x</textarea\n>\n</table> tail")

    (table,) = ex.tables.values()
    assert table["html"].endswith("</table>")
    assert ex.blocks[0]["content"].endswith(" tail")


def test_html_table_raw_mode_tag_name_split_across_lines():
    """Same hazard, but the element name itself is split by the line
    break rather than the trailing whitespace before ">"."""
    ex = _extract("<table><textarea>x</texta\nrea></table> tail")

    (table,) = ex.tables.values()
    assert table["html"].endswith("</table>")
    assert ex.blocks[0]["content"].endswith(" tail")


def test_html_table_ignores_tag_like_text_inside_script():
    """RCDATA/raw-text elements (script here) must not have their content
    scanned for tags either -- a literal </table>-shaped string inside
    <script> content must not be mistaken for the real closing tag."""
    ex = _extract(
        '<table><tr><td><script>if (a < b) { x.close("</table>"); }</script>'
        "</td></tr></table> tail"
    )

    (table,) = ex.tables.values()
    assert table["html"].endswith("</table>")
    assert 'x.close("</table>")' in table["html"]
    assert ex.blocks[0]["content"].endswith(" tail")


def test_html_table_ignores_quote_inside_style():
    """Same unmatched-quote hazard as textarea, for the other raw-text
    element (style)."""
    ex = _extract(
        '<table><tr><td><style>content: "it\'s";</style></td></tr></table> tail'
    )

    (table,) = ex.tables.values()
    assert table["html"].endswith("</table>")
    assert 'content: "it\'s";' in table["html"]
    assert ex.blocks[0]["content"].endswith(" tail")


def test_html_table_ignores_comment_marker_inside_title():
    """RCDATA content must not be scanned for HTML comments either -- a
    literal <!-- inside <title> text must stay literal, not open a
    (never-closing) comment state."""
    ex = _extract(
        "<table><tr><td><title>a <!-- not a comment</title></td></tr></table> tail"
    )

    (table,) = ex.tables.values()
    assert table["html"].endswith("</table>")
    assert "<!-- not a comment</title>" in table["html"]
    assert ex.blocks[0]["content"].endswith(" tail")


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


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("before ![alt](image.png) after", "before <image.png> after"),
        ('![alt](<a b.png> "caption")', "<<a b.png>>"),
        ("![](<a>suffix)", "<<a>suffix>"),
        ("![broken ![nested](ok.png)", "<ok.png>"),
    ],
)
def test_inline_image_scanner_preserves_supported_image_grammar(line, expected):
    assert _replace_inline_images(line, lambda src: f"<{src}>") == expected


def test_inline_image_scanner_matches_legacy_regex_for_short_title_inputs():
    def legacy_replace(line: str) -> str:
        return _LEGACY_INLINE_IMAGE_RE.sub(
            lambda match: f"<{match.group('src')}>", line
        )

    # The first case previously exposed a cache query that fell back to an
    # earlier start after a title search. The generated short corpus protects
    # the title and alternate-source grammar without a timing dependency.
    lines = (
        'srcb"bb![a]( ![a]( " "title")<![a](>',
        '![a](src "title") ![b](other "caption")',
        '![a](<a b> "title") ![b](<c>d)',
    )
    pieces = ("![a](", "![", "]", "(", ")", "<", ">", " ", '"', "title", "src")
    corpus = itertools.chain(
        lines,
        ("".join(parts) for parts in itertools.product(pieces, repeat=4)),
    )
    for line in corpus:
        assert _replace_inline_images(line, lambda src: f"<{src}>") == legacy_replace(
            line
        )


def test_inline_image_scanner_scales_linearly_for_malformed_candidates():
    def elapsed(repetitions: int) -> float:
        line = "![](" * repetitions
        started = time.perf_counter()
        for _ in range(5):
            assert _replace_inline_images(line, lambda src: src) == line
        return time.perf_counter() - started

    small = elapsed(1_024)
    large = elapsed(8_192)
    # Eight times more input must not become the old quadratic 64x workload.
    assert large / max(small, 1e-9) < 16
