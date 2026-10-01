"""Literal image examples must not trigger image resolution."""

import pytest

from lightrag.parser.markdown.extract import ResolvedImage, extract_markdown

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    "literal",
    [
        r"\![example](example.png)",
        "`![example](example.png)`",
        "``a ` tick ![example](example.png)``",
        "``![example](example.png)` still code``",
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


@pytest.mark.parametrize("prefix", [r"\\", "`unclosed ", r"\`"])
def test_escaped_backslash_or_unclosed_code_does_not_hide_image(prefix):
    calls = []

    class Resolver:
        def resolve(self, src):
            calls.append(src)
            return ResolvedImage(kind="external", url=src)

    result = extract_markdown(prefix + "![real](real.png)", image_resolver=Resolver())
    assert calls == ["real.png"]
    assert len(result.drawings) == 1
