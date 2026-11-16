"""Literal image examples must not trigger image resolution."""

import pytest

from lightrag.parser.markdown.extract import (
    ResolvedImage,
    _replace_inline_images,
    extract_markdown,
)

pytestmark = pytest.mark.offline


def _resolved(line: str) -> tuple[list[str], str]:
    calls: list[str] = []
    out = _replace_inline_images(line, lambda src: calls.append(src) or "<img>")
    return calls, out


@pytest.mark.parametrize(
    "literal",
    [
        r"\![example](example.png)",
        "`![example](example.png)`",
        "``a ` tick ![example](example.png)``",
        "``![example](example.png)` still code``",
        # A backslash escapes one backtick; the rest of the run still opens.
        r"\``![example](example.png)`",
        # The code span opened in the alt text swallows its ``]``.
        "![a `b](example.png)`",
        # An escaped ``]`` never ends the alt text.
        r"![a\](example.png)",
    ],
)
def test_literal_images_are_preserved_without_resolving(literal):
    calls = []

    class Resolver:
        def resolve(self, src):
            calls.append(src)
            return ResolvedImage(kind="external", url=src)

    result = extract_markdown(
        literal + " and ![real](real.png)", image_resolver=Resolver()
    )
    assert calls == ["real.png"]
    assert len(result.drawings) == 1
    assert literal in result.blocks[0]["content"]


@pytest.mark.parametrize(
    "prefix",
    [
        r"\\",  # an escaped backslash escapes nothing after it
        "`unclosed ",  # an unclosed run protects nothing
        r"\`",  # an escaped single backtick opens nothing
        r"`a\`",  # a backslash does not escape a closing run
        "``a`` ",  # a closed span ends before the image
    ],
)
def test_escaped_backslash_or_unclosed_code_does_not_hide_image(prefix):
    calls = []

    class Resolver:
        def resolve(self, src):
            calls.append(src)
            return ResolvedImage(kind="external", url=src)

    result = extract_markdown(prefix + "![real](real.png)", image_resolver=Resolver())
    assert calls == ["real.png"]
    assert len(result.drawings) == 1


@pytest.mark.parametrize(
    "line",
    [
        "![a `b` c](real.png)",  # a span closed inside the alt text
        r"![a\\](real.png)",  # an escaped backslash before ``]``
        '![a ``b`` c](real.png "title")',
    ],
)
def test_code_and_escapes_closed_inside_alt_text_keep_the_image(line):
    calls, out = _resolved(line)
    assert calls == ["real.png"]
    assert out == "<img>"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        # After the unclosed ``x`` run scans the whole line, the closed `a`
        # span must not make the later `b ...` span look unclosed.
        ("``x `a` `b ![i](i.png)` `", []),
        ("``x `a` `in ``` t` ![real](real.png) ```", ["real.png"]),
    ],
)
def test_a_closed_span_does_not_hide_a_later_closer(line, expected):
    calls, _ = _resolved(line)
    assert calls == expected


def test_skipped_code_spans_are_copied_verbatim():
    line = "`a` ``b ` c`` ![x](x.png) `![y](y.png)` tail"
    calls, out = _resolved(line)
    assert calls == ["x.png"]
    assert out == "`a` ``b ` c`` <img> `![y](y.png)` tail"
