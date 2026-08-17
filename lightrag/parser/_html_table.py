"""Shared HTML-table parsing helpers for parser engines.

Pure functions that recover structural facts from an HTML ``<table>`` string
without a heavy dependency: row/column counts (colspan-aware), the verbatim
``<thead>`` substring, table-payload detection, and ``<html>/<body>`` wrapper
stripping. Originally private to the mineru IR builder; lifted into a leaf
module so the native markdown IR builder can reuse the exact same logic
(merged-cell semantics must survive identically across engines).

Leaf module with no parser-layer imports — safe for any engine to import.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser

from lightrag.utils import logger


@dataclass
class HTMLTableInfo:
    num_rows: int = 0
    num_cols: int = 0


class _HTMLTableInfoParser(HTMLParser):
    """Count ``<tr>`` rows and their (colspan-aware) column widths.

    Used only to recover ``num_rows`` / ``num_cols`` when the engine did not
    supply them; the ``<thead>`` header itself is preserved verbatim by
    :func:`extract_thead_html`, not reconstructed here.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        # ``col_count`` (sum of colspans) for each completed top-level ``<tr>``.
        self.row_col_counts: list[int] = []
        self._tr_depth = 0
        self._cell_depth = 0
        self._row_col_count = 0
        self._cell_colspan = 1

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            if self._tr_depth == 0:
                self._row_col_count = 0
            self._tr_depth += 1
            return
        if tag in {"td", "th"} and self._tr_depth > 0:
            if self._cell_depth == 0:
                self._cell_colspan = _cell_span(attrs, "colspan")
            self._cell_depth += 1

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._cell_depth > 0:
            self._cell_depth -= 1
            if self._cell_depth == 0:
                self._row_col_count += self._cell_colspan
                self._cell_colspan = 1
            return
        if tag == "tr" and self._tr_depth > 0:
            self._tr_depth -= 1
            # ``col_count > 0`` ⇔ the row held at least one cell (colspan ≥ 1),
            # so an empty ``<tr></tr>`` is skipped exactly as before.
            if self._tr_depth == 0 and self._row_col_count > 0:
                self.row_col_counts.append(self._row_col_count)
                self._row_col_count = 0


def _cell_span(attrs: list[tuple[str, str | None]], name: str) -> int:
    """Read a ``colspan``/``rowspan`` attribute as an int ``>= 1`` (default 1)."""
    for key, value in attrs:
        if key.lower() != name:
            continue
        try:
            return max(int(value or "1"), 1)
        except ValueError:
            return 1
    return 1


def extract_html_table_info(html: str) -> HTMLTableInfo:
    parser = _HTMLTableInfoParser()
    try:
        parser.feed(html or "")
        parser.close()
    except Exception as exc:  # pragma: no cover - HTMLParser is forgiving.
        logger.debug("[html_table] failed to parse table HTML: %s", exc)
        return HTMLTableInfo()
    return HTMLTableInfo(
        num_rows=len(parser.row_col_counts),
        num_cols=max(parser.row_col_counts, default=0),
    )


def extract_thead_html(html: str) -> str | None:
    """Return the first top-level ``<thead …>…</thead>`` substring verbatim.

    The raw markup is kept so merged-cell semantics (``rowspan`` / ``colspan``)
    survive into ``tables.json`` and, later, into every repeated header chunk
    of a split table. Returns ``None`` when the table has no ``<thead>`` or the
    ``<thead>`` carries no visible text (a blank spacer row, which would
    otherwise emit empty ``<th>`` headers).
    """
    stripped = (html or "").strip()
    lower = stripped.lower()
    start = find_html_tag(lower, "thead")
    if start < 0:
        return None
    close = lower.find("</thead>", start)
    if close < 0:
        return None
    thead = stripped[start : close + len("</thead>")]
    # Blank check: drop a header whose cells hold no non-whitespace text.
    if not re.sub(r"<[^>]+>", "", thead).strip():
        return None
    return thead


def looks_like_html_table_payload(body: str) -> bool:
    lower = (body or "").lstrip().lower()
    return any(
        starts_with_html_tag(lower, tag)
        for tag in ("table", "thead", "tbody", "tfoot", "tr", "html", "body")
    )


def unwrap_html_table(payload: str) -> str:
    """Strip a ``<html>/<body>`` document wrapper that a table model
    sometimes emits, returning the outermost ``<table…>…</table>`` span. Keeps
    a single clean ``<table>`` so the writer does not nest tables and the
    non-greedy ``TABLE_TAG_RE`` is not truncated at an inner ``</table>``.
    Falls back to the stripped payload when no ``<table>`` element exists."""
    stripped = (payload or "").strip()
    lower = stripped.lower()
    start = _find_table_open(lower)
    if start < 0:
        return stripped
    close = lower.rfind("</table>")
    if close < start:
        return stripped
    return stripped[start : close + len("</table>")]


def _find_table_open(lower: str) -> int:
    """First index of a real ``<table`` start tag (not e.g. ``<tablefoo``).
    Returns -1 when none is present."""
    return find_html_tag(lower, "table")


def find_html_tag(lower: str, tag: str) -> int:
    """First index of a real ``<tag`` start tag (not e.g. ``<tablefoo`` for
    ``tag="table"``). ``lower`` must already be lower-cased. Returns -1 when
    none is present."""
    needle = f"<{tag}"
    idx = 0
    while True:
        idx = lower.find(needle, idx)
        if idx < 0:
            return -1
        nxt = idx + len(needle)
        if nxt >= len(lower) or lower[nxt] in {" ", "\t", "\r", "\n", ">", "/"}:
            return idx
        idx = nxt


def starts_with_html_tag(lower: str, tag: str) -> bool:
    prefix = f"<{tag}"
    if not lower.startswith(prefix):
        return False
    if len(lower) == len(prefix):
        return True
    return lower[len(prefix)] in {" ", "\t", "\r", "\n", ">", "/"}


def html_table_inner_body(html: str) -> str:
    stripped = (html or "").strip()
    lower = stripped.lower()
    if not starts_with_html_tag(lower, "table"):
        return stripped
    open_end = _open_tag_end(stripped)
    close_start = lower.rfind("</table>")
    if open_end < 0 or close_start <= open_end:
        return stripped
    return stripped[open_end + 1 : close_start].strip()


def _open_tag_end(html: str) -> int:
    """Index of the ``>`` closing the leading tag, skipping quoted attribute
    values so a ``>`` inside an attribute (e.g. ``<table data-x="a>b">``) does
    not terminate the tag early. Returns -1 when no closing ``>`` is found."""
    quote: str | None = None
    for idx, ch in enumerate(html):
        if quote is not None:
            if ch == quote:
                quote = None
        elif ch in {'"', "'"}:
            quote = ch
        elif ch == ">":
            return idx
    return -1


__all__ = [
    "HTMLTableInfo",
    "extract_html_table_info",
    "extract_thead_html",
    "looks_like_html_table_payload",
    "unwrap_html_table",
    "find_html_tag",
    "starts_with_html_tag",
    "html_table_inner_body",
]
