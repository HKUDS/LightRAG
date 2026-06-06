"""Shared helpers for parsing and re-emitting ``<table>`` markup.

These primitives are used by the paragraph-semantic chunker (TableRowSplit
oversized-table re-split) and by the native multimodal surrounding-context
extractor.  Both call sites need to:

* recognise a post-rewrite ``<table id="…" format="…">…</table>`` tag,
* decide whether the body is JSON or HTML,
* enumerate row-level units (JSON list items or HTML ``<tr>`` rows along
  with their ``<thead>`` / ``<tbody>`` / ``<tfoot>`` wrappers), and
* re-serialise a subset of rows while preserving the structural wrappers.

Keeping the regexes and helpers in one place avoids subtle drift when
either consumer evolves.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Strict regex for a post-rewrite table tag emitted by the sidecar
# writer (``lightrag.sidecar.writer``):
#   <table id="tb-…" format="json"[ caption="…"]>{rows_json}</table>
# blocks.jsonl invariants guarantee the tag has no embedded newlines.
TABLE_TAG_RE = re.compile(
    r"<table\s+(?P<attrs>[^>]*)>(?P<body>.*?)</table>",
    re.DOTALL,
)

# Format detection regex inside the attrs string, e.g. format="json".
_TABLE_FORMAT_RE = re.compile(r"""format\s*=\s*["'](?P<fmt>[^"']+)["']""")

# HTML <tr>...</tr> row extractor.  Standard HTML disallows nested <tr>,
# so a non-greedy match is sufficient for well-formed input.
HTML_TR_RE = re.compile(r"<tr\b[^>]*>.*?</tr>", re.DOTALL | re.IGNORECASE)

# Combined scanner for row-grouping wrappers and rows themselves.  Used
# to attribute each <tr> to its surrounding <thead>/<tbody>/<tfoot> so
# the wrapper can be reconstructed around chunk boundaries instead of
# being silently dropped during row-level table splitting.
HTML_ROW_PARTS_RE = re.compile(
    r"(?P<wrap></?(?:thead|tbody|tfoot)\b[^>]*>)" r"|(?P<tr><tr\b[^>]*>.*?</tr>)",
    re.DOTALL | re.IGNORECASE,
)
HTML_WRAPPER_TAG_RE = re.compile(
    r"<(?P<slash>/?)(?P<name>thead|tbody|tfoot)\b", re.IGNORECASE
)


def detect_table_format(attrs: str, body: str) -> str | None:
    """Return ``"json"``, ``"html"`` or ``None`` for a parsed ``<table>`` tag.

    Prefers an explicit ``format="…"`` attribute.  When silent, sniffs
    the body: a leading ``[`` / ``{`` (after whitespace) implies JSON;
    the presence of any ``<tr`` tag implies HTML.  Anything else is
    unknown and the caller should fall back to character splitting.
    """
    match = _TABLE_FORMAT_RE.search(attrs or "")
    if match:
        fmt = match.group("fmt").strip().lower()
        if fmt in {"json", "html"}:
            return fmt
        return None
    stripped = (body or "").lstrip()
    if stripped.startswith(("[", "{")):
        return "json"
    if "<tr" in stripped.lower():
        return "html"
    return None


def parse_table_tag(text: str) -> tuple[str, list[Any]] | None:
    """Parse a JSON ``<table …>{rows_json}</table>``.

    Returns ``(attrs_str, rows)`` or ``None`` if the tag is malformed
    (does not match ``TABLE_TAG_RE``, body is not JSON, or body decodes
    to something other than a list).
    """
    match = TABLE_TAG_RE.match((text or "").strip())
    if not match:
        return None
    body = match.group("body")
    try:
        rows = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(rows, list):
        return None
    return match.group("attrs"), rows


def split_html_rows(body: str) -> list[tuple[str, str]] | None:
    """Extract ``<tr>...</tr>`` rows tagged with their wrapper context.

    Returns a list of ``(wrapper_name, tr_str)`` tuples where
    ``wrapper_name`` is ``"thead"`` / ``"tbody"`` / ``"tfoot"`` (lower-
    cased) for rows that sit inside the corresponding wrapper, or ``""``
    for rows outside any of those wrappers.  ``None`` signals "no row
    found" so the caller falls through to character splitting.

    Whitespace, captions, comments, ``<colgroup>`` and any other text
    outside the recognised row-wrappers is dropped — this is a regex
    extractor, not a full DOM parser.  Wrapper attributes (e.g.
    ``<thead class="…">``) are also dropped on re-emission; chunked
    output uses bare wrapper tags.
    """
    rows: list[tuple[str, str]] = []
    current_wrapper = ""
    for match in HTML_ROW_PARTS_RE.finditer(body or ""):
        wrap = match.group("wrap")
        tr = match.group("tr")
        if wrap is not None:
            tag = HTML_WRAPPER_TAG_RE.match(wrap)
            if tag:
                slash = tag.group("slash")
                name = tag.group("name").lower()
                if slash == "/":
                    if current_wrapper == name:
                        current_wrapper = ""
                else:
                    current_wrapper = name
        elif tr is not None:
            rows.append((current_wrapper, tr))
    if not rows:
        return None
    return rows


def serialize_html_rows(rows: list[tuple[str, str]]) -> str:
    """Re-emit ``(wrapper, tr)`` rows grouped under their original
    ``<thead>`` / ``<tbody>`` / ``<tfoot>`` wrappers.

    Consecutive rows sharing the same wrapper name collapse into a
    single wrapper block; transitions emit a closing tag for the
    previous wrapper and an opening tag for the next.  Rows tagged with
    ``""`` (no wrapper) emit bare ``<tr>...</tr>``.
    """
    parts: list[str] = []
    current_wrapper = ""
    for wrapper, tr in rows:
        if wrapper != current_wrapper:
            if current_wrapper:
                parts.append(f"</{current_wrapper}>")
            if wrapper:
                parts.append(f"<{wrapper}>")
            current_wrapper = wrapper
        parts.append(tr)
    if current_wrapper:
        parts.append(f"</{current_wrapper}>")
    return "".join(parts)
