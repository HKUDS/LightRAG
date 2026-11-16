"""The README translations stay line-for-line aligned with README.md.

Edits that reached ``README.md`` alone let the translations drift: by the time
it was noticed they were 18-19 lines off and were missing a banner, a
documentation table row, renamed env vars and, in one of them, whole
paragraphs. AGENTS.md ("README translations") therefore requires every README
change to land in all four files in the same commit, with the same line count
and the same line structure. This pins that rule, because it is a property of
the tree, not of any one change: a pull request that edits ``README.md`` alone
now fails here instead of in a reviewer's attention.

"Same structure" means line N has the same *kind* in every file -- blank,
heading of a given level, list item at a given indent, table row with a given
column count, HTML tag, quote, code fence, code line, or prose. The prose
itself is translated and is not compared.
"""

from __future__ import annotations

import pathlib
import re

import pytest

pytestmark = pytest.mark.offline

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_ENGLISH = _REPO_ROOT / "README.md"
_TRANSLATIONS = sorted(_REPO_ROOT.glob("README-*.md"))

_HEADING = re.compile(r"(#{1,6}) ")
_BULLET = re.compile(r"[-*+] ")
_NUMBERED = re.compile(r"\d+\. ")
_HTML_TAG = re.compile(r"</?[A-Za-z]*")
_UNESCAPED_PIPE = re.compile(r"(?<!\\)\|")


def _line_kinds(lines: list[str]) -> list[str]:
    """Classify every line by its Markdown role, ignoring its wording."""
    kinds = []
    in_fence = False
    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if stripped.startswith("```"):
            kinds.append("code fence")
            in_fence = not in_fence
        elif not stripped:
            kinds.append("blank")
        elif in_fence:
            # Code is copied, not translated, except for its comments; a
            # translated comment may legitimately start differently.
            kinds.append("code line")
        elif match := _HEADING.match(line):
            kinds.append(f"h{len(match[1])} heading")
        elif stripped.startswith("|"):
            cells = len(_UNESCAPED_PIPE.findall(stripped)) - 1
            kinds.append(f"table row ({cells} columns)")
        elif stripped.startswith("<"):
            kinds.append(f"html {_HTML_TAG.match(stripped)[0]}")
        elif stripped.startswith(">"):
            kinds.append("quote")
        elif _BULLET.match(stripped):
            kinds.append(f"bullet (indent {indent})")
        elif _NUMBERED.match(stripped):
            kinds.append(f"numbered item (indent {indent})")
        else:
            kinds.append("text")
    return kinds


def _first_divergence(english: list[str], translated: list[str]) -> int | None:
    """Index of the first line whose kind differs, or None when all match."""
    for index, (a, b) in enumerate(zip(english, translated)):
        if a != b:
            return index
    if len(english) != len(translated):
        return min(len(english), len(translated))
    return None


def _read_lines(path: pathlib.Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def test_translations_exist():
    """Guards the glob: an empty match would make every check below vacuous."""
    assert _ENGLISH.is_file()
    assert [p.name for p in _TRANSLATIONS] == [
        "README-id.md",
        "README-ja.md",
        "README-zh.md",
    ], "a README translation was added or removed: update this list and AGENTS.md"


@pytest.mark.parametrize("translation", _TRANSLATIONS, ids=lambda p: p.name)
def test_translation_has_the_english_line_count(translation):
    english, translated = _read_lines(_ENGLISH), _read_lines(translation)

    assert len(translated) == len(english), (
        f"{translation.name} has {len(translated)} lines, README.md has "
        f"{len(english)}. Apply the README change to every language in the "
        "same commit (AGENTS.md, 'README translations')."
    )


@pytest.mark.parametrize("translation", _TRANSLATIONS, ids=lambda p: p.name)
def test_translation_has_the_english_line_structure(translation):
    english, translated = _read_lines(_ENGLISH), _read_lines(translation)
    english_kinds, translated_kinds = _line_kinds(english), _line_kinds(translated)

    index = _first_divergence(english_kinds, translated_kinds)

    if index is not None:

        def describe(lines, kinds):
            if index >= len(lines):
                return "<end of file>"
            return f"{kinds[index]}: {lines[index][:100]!r}"

        pytest.fail(
            f"{translation.name} diverges from README.md at line {index + 1}.\n"
            f"  README.md:  {describe(english, english_kinds)}\n"
            f"  {translation.name}: {describe(translated, translated_kinds)}\n"
            "Keep the English line breaks and Markdown structure "
            "(AGENTS.md, 'README translations')."
        )


def test_line_kinds_tell_the_structures_apart():
    """The classifier must separate what a translation could get wrong."""
    kinds = _line_kinds(
        [
            "# Title",
            "## Section",
            "",
            "- item",
            "  - nested",
            "1. numbered",
            "| a | b |",
            "| a | b | c |",
            "<div>",
            "> note",
            "Prose.",
            "```bash",
            "# a comment, not a heading",
            "```",
        ]
    )

    assert kinds == [
        "h1 heading",
        "h2 heading",
        "blank",
        "bullet (indent 0)",
        "bullet (indent 2)",
        "numbered item (indent 0)",
        "table row (2 columns)",
        "table row (3 columns)",
        "html <div",
        "quote",
        "text",
        "code fence",
        "code line",
        "code fence",
    ]


def test_first_divergence_reports_the_first_mismatch_or_the_shorter_end():
    assert _first_divergence(["a", "b"], ["a", "b"]) is None
    assert _first_divergence(["a", "b", "c"], ["a", "x", "c"]) == 1
    assert _first_divergence(["a", "b"], ["a", "b", "c"]) == 2
    assert _first_divergence(["a", "b", "c"], ["a", "b"]) == 2
