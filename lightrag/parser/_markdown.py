"""Shared markdown rendering helpers for parser engines.

Used by the native docx parser and the external (mineru / docling) IR
builders so heading-line rendering stays identical across engines. This is
a leaf module with no heavy imports — ``lightrag/parser/__init__.py`` only
carries a docstring — so all three engines can import it without risking a
circular dependency.
"""

from __future__ import annotations

import re

# Markdown caps heading levels at 6 (``######``); deeper outline levels are
# clamped to 6 rather than emitting an illegal 7+ run of ``#``.
MAX_HEADING_LEVEL = 6

# A heading line that is ALREADY markdown: 1-6 ``#`` followed by a space.
# Used to avoid double-prefixing text that an upstream engine emitted with
# its own markdown heading marker (e.g. mineru/docling extracting ``# Foo``).
_MD_HEADING_RE = re.compile(r"^#{1,6} ")


def strip_heading_markdown_prefix(text: str) -> str:
    """Return heading metadata without an existing markdown heading prefix.

    The content renderer may keep a source line such as ``"# Foo"`` verbatim
    to avoid double-prefixing, but structured metadata (``heading``,
    ``parent_headings``, doc title) must stay clean.
    """
    return _MD_HEADING_RE.sub("", text, count=1)


def render_heading_line(level: int, text: str) -> str:
    """Render a heading as a markdown-prefixed content line.

    Args:
        level: 1-based heading level (1 = H1). Values < 1 are treated as 1;
            values > :data:`MAX_HEADING_LEVEL` are clamped so a level >= 7
            heading still gets ``######``.
        text: The heading text.

    Returns:
        ``text`` unchanged when it already starts with a markdown heading
        prefix (``^#{1,6} ``); otherwise ``"#" * clamped_level + " " + text``.
    """
    if _MD_HEADING_RE.match(text):
        return text
    hashes = "#" * min(max(level, 1), MAX_HEADING_LEVEL)
    return f"{hashes} {text}"


__all__ = [
    "MAX_HEADING_LEVEL",
    "render_heading_line",
    "strip_heading_markdown_prefix",
]
