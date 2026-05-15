"""Placeholder token rendering for spec-shaped multimodal tags.

Adapters populate :attr:`IRBlock.content_template` with ``{{TBL:k}}``,
``{{IMG:k}}``, ``{{EQ:k}}`` and ``{{EQI:k}}`` tokens. The writer assigns
``tb-`` / ``im-`` / ``eq-`` ids, then calls :func:`render_template` to
substitute the spec-shaped XML-style tags described in
``LightRAGSidecarFormat-zh.md`` §3.3.
"""

from __future__ import annotations

import json
import re
from typing import Callable

_TOKEN_RE = re.compile(r"\{\{(TBL|IMG|EQ|EQI):([A-Za-z0-9_\-]+)\}\}")


def xml_attr_escape(value: str) -> str:
    """Escape an attribute value for an XML-style tag attribute."""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def caption_attr(caption: str) -> str:
    """Render a leading-space ``caption="..."`` attribute; empty when absent.

    Matches the existing native_docx adapter convention exactly so consumers
    that grep for ``caption="``-prefixed substrings keep working.
    """
    return f' caption="{xml_attr_escape(caption)}"' if caption else ""


def render_table_tag(table_id: str, fmt: str, body: str) -> str:
    """``<table id="tb-..." format="json|html">body</table>`` per spec §3.3.

    ``body`` is the table content; for ``json`` it is the JSON array, for
    ``html`` it is the raw ``<table>...</table>`` HTML inside (the outer
    wrapper is added here).
    """
    return (
        f'<table id="{xml_attr_escape(table_id)}" '
        f'format="{xml_attr_escape(fmt)}">{body}</table>'
    )


def render_drawing_tag(
    drawing_id: str,
    fmt: str,
    caption: str,
    path: str,
    src: str,
) -> str:
    """``<drawing id="im-..." format="..." caption="..." path="..." src="..." />``."""
    return (
        f'<drawing id="{xml_attr_escape(drawing_id)}" '
        f'format="{xml_attr_escape(fmt)}"'
        f"{caption_attr(caption)} "
        f'path="{xml_attr_escape(path)}" '
        f'src="{xml_attr_escape(src)}" />'
    )


def render_equation_tag(
    eq_id: str | None,
    latex: str,
    caption: str = "",
) -> str:
    """Block equation: ``<equation id="eq-..." format="latex" caption="...">latex</equation>``.

    Inline equation (``eq_id is None``): ``<equation format="latex">latex</equation>``
    — no id, never written to ``equations.json``. Caption is preserved for
    both forms (spec §3.3 allows ``caption`` on ``<equation>``).
    """
    if eq_id is None:
        return f'<equation format="latex"{caption_attr(caption)}>{latex}</equation>'
    return (
        f'<equation id="{xml_attr_escape(eq_id)}" '
        f'format="latex"{caption_attr(caption)}>{latex}</equation>'
    )


def render_template(
    template: str,
    *,
    table_renderer: Callable[[str], str],
    drawing_renderer: Callable[[str], str],
    equation_renderer: Callable[[str], str],
    inline_equation_renderer: Callable[[str], str],
) -> str:
    """Replace ``{{TBL:k}}`` / ``{{IMG:k}}`` / ``{{EQ:k}}`` / ``{{EQI:k}}``.

    Each renderer takes the placeholder *key* (the ``k`` portion) and returns
    the rendered XML-style tag.
    """

    def _replace(match: "re.Match[str]") -> str:
        kind, key = match.group(1), match.group(2)
        if kind == "TBL":
            return table_renderer(key)
        if kind == "IMG":
            return drawing_renderer(key)
        if kind == "EQ":
            return equation_renderer(key)
        return inline_equation_renderer(key)

    return _TOKEN_RE.sub(_replace, template)


def table_body_for_rows(rows: list[list[str]]) -> str:
    """Encode rows as the JSON body that lives inside ``<table format="json">``."""
    return json.dumps(rows, ensure_ascii=False)
