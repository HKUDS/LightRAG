"""Keep ``rowspan`` cells whole when an HTML table is cut between rows.

Row splitting cuts a table between two ``<tr>`` rows, but a cell with
``rowspan="N"`` covers the next ``N - 1`` rows too. Cut as is, the cell claims
rows its slice does not hold, and the first rows of the next slice have no cell
for that column, so their values shift one column left under the header.

:func:`build_rowspan_slicer` returns a slicer that makes every slice a table on
its own, or ``None`` when the slices must be emitted exactly as cut.

Rules for callers and for changes to this module:

* **All or nothing per table.** A table is repaired only when its whole body
  matches the strict grammar below. Anything else is never partly interpreted:
  the builder returns ``None`` and the caller slices the rows as they are,
  which is what happened before this module existed. Never add a per-row or
  per-cell fallback: a layout built from a table that was only partly
  understood can move a cell into the wrong column.
* **The grammar is closed.** New HTML forms are supported by extending the
  grammar and its tests, not by tolerating them in the scanner.
* **Work is bounded.** The layout stops at :data:`LAYOUT_BUDGET` steps, so a
  hostile table cannot make it quadratic in time or memory.
* A repeated cell costs tokens the row splitter never counted. The caller must
  re-check every serialized slice against its cap.

Supported grammar, the limits and the reasons are in
``docs/ParagraphSemanticChunking.md`` (section 3.2.1).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from string import ascii_lowercase, ascii_uppercase

from lightrag.table_markup import HTML_ROW_PARTS_RE, HTML_WRAPPER_TAG_RE
from lightrag.utils import logger

# The largest span values HTML defines. A larger value is not clamped here:
# the table is left unrepaired.
_SPAN_LIMITS = {"rowspan": 65534, "colspan": 1000}

# Upper bound on the layout's work for one table: spans carried into each row
# plus the columns they cover, summed over the rows. Tables with a few label
# columns stay far below it.
LAYOUT_BUDGET = 200_000

# HTML's ASCII whitespace. Python's ``\s`` also matches characters such as
# U+00A0, which HTML reads as part of an attribute name, so it is never used.
_WS = " \t\n\f\r"
_W = r"[ \t\n\f\r]"
_NAME = r"""[^ \t\n\f\r"'<>/=]+"""
_ATTR_VALUE = r"""(?:"(?P<dq>[^"]*)"|'(?P<sq>[^']*)'|(?P<bare>[^ \t\n\f\r"'=<>`]+))"""
_ATTR_RE = re.compile(rf"""{_W}+(?P<name>{_NAME})(?:{_W}*={_W}*{_ATTR_VALUE})?""")
# A start tag's attributes: quoted values are consumed whole, so a ``>`` inside
# one does not end the tag.
_ATTRS = (
    rf"""(?:{_W}+{_NAME}(?:{_W}*={_W}*"""
    rf"""(?:"[^"]*"|'[^']*'|[^ \t\n\f\r"'=<>`]+))?)*{_W}*"""
)
_FLAGS = re.DOTALL | re.IGNORECASE | re.ASCII
_ROW_RE = re.compile(rf"<tr(?:{_ATTRS})>(?P<inner>.*)</tr>", _FLAGS)
_CELL_RE = re.compile(
    rf"{_W}*<(?P<tag>td|th)(?P<attrs>{_ATTRS})>(?P<inner>.*?)</(?P=tag){_W}*>",
    _FLAGS,
)
# A complete tag inside a cell. Requiring every tag of a cell's content to be
# complete proves that the ``</td>`` which ended the cell was not inside a
# quoted attribute value of a nested element.
_INNER_TAG_RE = re.compile(
    rf"""</?(?P<name>[A-Za-z][^ \t\n\f\r<>/"'=]*){_ATTRS}/?>""", re.ASCII
)
# Tags that end a cell when HTML meets them inside one: a nested table, or a
# cell whose end tag was omitted so that the next start tag landed in it.
_CELL_ENDING_TAGS = frozenset(
    "table thead tbody tfoot tr td th caption col colgroup".split()
)
# Elements whose content HTML does not read as markup (so a ``</td>`` in it is
# text), and those that change how the content after them is parsed. The first
# line is the complete list of elements that switch the HTML tokenizer's state.
_OPAQUE_TAGS = frozenset(
    "script style textarea title xmp iframe noembed noframes noscript plaintext "
    "svg math template select".split()
)
_ASCII_LOWER = str.maketrans(ascii_uppercase, ascii_lowercase)
# Wrapper/row sequence of a supported body: an optional ``<thead>`` block, then
# the body rows either bare or inside a single ``<tbody>``.
_BODY_SHAPE_RE = re.compile(r"(?:Hr*h)?(?:r+|Br+b)")
_SHAPE_CODE = {"thead": "H", "tbody": "B", "tfoot": "F"}


class _Unsupported(Exception):
    """The table is outside the supported grammar; the message says why."""


@dataclass(frozen=True)
class _Cell:
    html: str  # the cell exactly as written
    tag: str
    attrs: str  # attributes without ``rowspan``
    inner: str
    rowspan: int
    colspan: int

    def with_rowspan(self, rowspan: int) -> str:
        if rowspan == self.rowspan:
            return self.html
        span = f' rowspan="{rowspan}"' if rowspan > 1 else ""
        return f"<{self.tag}{self.attrs}{span}>{self.inner}</{self.tag}>"


def _span_value(name: str, match: re.Match[str]) -> int:
    raw = match.group("dq") or match.group("sq") or match.group("bare") or ""
    raw = raw.strip(_WS)
    if not (raw.isascii() and raw.isdecimal()):
        raise _Unsupported(f"{name} is not a positive integer")
    # Compare the digit count first: ``int`` refuses a very long digit string.
    limit = _SPAN_LIMITS[name]
    value = int(raw) if len(raw) <= len(str(limit)) else limit + 1
    if not 1 <= value <= limit:
        raise _Unsupported(f"{name} is outside 1..{_SPAN_LIMITS[name]}")
    return value


def _check_cell_content(inner: str) -> None:
    pos = inner.find("<")
    while pos != -1:
        following = inner[pos + 1 : pos + 2]
        if following.isascii() and (following.isalpha() or following == "/"):
            tag = _INNER_TAG_RE.match(inner, pos)
            if tag is None:
                raise _Unsupported("a tag inside a cell that cannot be read")
            name = tag.group("name").translate(_ASCII_LOWER)
            if name in _CELL_ENDING_TAGS:
                raise _Unsupported("table markup inside a cell")
            if name in _OPAQUE_TAGS:
                raise _Unsupported(f"a <{name}> element inside a cell")
            pos = inner.find("<", tag.end())
        elif following in ("!", "?"):
            raise _Unsupported("a declaration inside a cell")
        else:
            # A ``<`` that starts no tag is text.
            pos = inner.find("<", pos + 1)


def _parse_cell(match: re.Match[str]) -> _Cell:
    inner = match.group("inner")
    _check_cell_content(inner)
    attrs = match.group("attrs")
    spans: dict[str, int] = {}
    kept: list[str] = []
    # ``attrs`` matched the grammar, so the attributes tile it from the start
    # and only whitespace can follow them; scanning that tail is quadratic.
    for attr in _ATTR_RE.finditer(attrs.rstrip(_WS)):
        name = attr.group("name").translate(_ASCII_LOWER)
        if name not in _SPAN_LIMITS:
            kept.append(attr.group(0))
            continue
        if name in spans:
            raise _Unsupported(f"a cell repeats {name}")
        spans[name] = _span_value(name, attr)
        if name == "colspan":
            kept.append(attr.group(0))
    return _Cell(
        html=match.group(0).lstrip(_WS),
        tag=match.group("tag"),
        attrs="".join(kept),
        inner=inner,
        rowspan=spans.get("rowspan", 1),
        colspan=spans.get("colspan", 1),
    )


def _parse_row(tr: str) -> tuple[str, list[_Cell], str]:
    """Split a ``<tr>`` into its start tag, its cells and its end tag."""
    row = _ROW_RE.fullmatch(tr)
    if row is None:
        raise _Unsupported("a row tag that cannot be read")
    inner = row.group("inner")
    cells: list[_Cell] = []
    pos = 0
    while (match := _CELL_RE.match(inner, pos)) is not None:
        cells.append(_parse_cell(match))
        pos = match.end()
    if inner[pos:].strip(_WS):
        raise _Unsupported("row content that is not a complete cell")
    return tr[: row.start("inner")], cells, tr[row.end("inner") :]


def _header_row_count(body: str, row_count: int) -> int:
    """Check the body's shape; return how many leading rows are ``<thead>`` rows."""
    if "<!--" in body:
        raise _Unsupported("an HTML comment")
    if HTML_ROW_PARTS_RE.sub("", body).strip(_WS):
        raise _Unsupported("content outside the rows")
    shape: list[str] = []
    for part in HTML_ROW_PARTS_RE.finditer(body):
        if part.group("tr") is not None:
            shape.append("r")
            continue
        tag = HTML_WRAPPER_TAG_RE.match(part.group("wrap"))
        code = _SHAPE_CODE[tag.group("name").lower()]
        shape.append(code.lower() if tag.group("slash") else code)
    signature = "".join(shape)
    if signature.count("r") != row_count or not _BODY_SHAPE_RE.fullmatch(signature):
        raise _Unsupported("row groups other than one <thead> and one body")
    return signature.index("h") - 1 if "h" in signature else 0


class RowspanSlicer:
    """Cut row ranges of one table with every ``rowspan`` cell made local.

    Built by :func:`build_rowspan_slicer`. :meth:`slice` returns
    ``rows[start:end]`` where

    * a cell that starts before ``start`` and still covers it is repeated in
      the first row, at its column, spanning the rows it covers in the slice;
    * a cell whose ``rowspan`` runs past ``end`` is cut to the rows left.

    ``<thead>`` rows and rows that need no change are returned unchanged.
    """

    def __init__(
        self,
        rows: list[tuple[str, str]],
        header_rows: int,
        parsed: list[tuple[str, list[tuple[int, _Cell]], str]],
        carried: list[list[tuple[int, int, _Cell]]],
    ) -> None:
        self._rows = rows
        self._header_rows = header_rows
        # Per body row: start tag, (column, cell) pairs, end tag.
        self._parsed = parsed
        # Per body row: (column, last body row covered, cell) of each span
        # that an earlier row started and that still covers this one.
        self._carried = carried

    def slice(self, start: int, end: int) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for index in range(start, end):
            wrapper, tr = self._rows[index]
            row = index - self._header_rows
            if row < 0:
                out.append((wrapper, tr))
                continue
            open_tag, placed, close_tag = self._parsed[row]
            # Rows of the slice that remain from this one on.
            left = end - index
            pieces = [
                (col, cell.with_rowspan(min(cell.rowspan, left)))
                for col, cell in placed
            ]
            changed = any(cell.rowspan > left for _col, cell in placed)
            if index == start:
                for col, last, cell in self._carried[row]:
                    pieces.append((col, cell.with_rowspan(min(last - row + 1, left))))
                    changed = True
            if changed:
                pieces.sort(key=lambda piece: piece[0])
                tr = open_tag + "".join(html for _col, html in pieces) + close_tag
            out.append((wrapper, tr))
        return out


def _layout(
    body_rows: list[str],
) -> tuple[
    list[tuple[str, list[tuple[int, _Cell]], str]],
    list[list[tuple[int, int, _Cell]]],
    bool,
]:
    parsed: list[tuple[str, list[tuple[int, _Cell]], str]] = []
    carried: list[list[tuple[int, int, _Cell]]] = []
    active: dict[int, tuple[int, int, _Cell]] = {}
    has_spans = False
    work = 0
    last_row = len(body_rows) - 1
    for row, tr in enumerate(body_rows):
        for col in [c for c, span in active.items() if span[1] < row]:
            del active[col]
        occupied: set[int] = set()
        for col, _last, cell in active.values():
            work += 1 + cell.colspan
            if work > LAYOUT_BUDGET:
                raise _Unsupported("too many overlapping spans")
            occupied.update(range(col, col + cell.colspan))
        carried.append(list(active.values()))
        open_tag, cells, close_tag = _parse_row(tr)
        placed: list[tuple[int, _Cell]] = []
        col = 0
        for cell in cells:
            while col in occupied:
                col += 1
            if cell.colspan > 1 and occupied:
                work += cell.colspan
                if work > LAYOUT_BUDGET:
                    raise _Unsupported("too many overlapping spans")
                if not occupied.isdisjoint(range(col + 1, col + cell.colspan)):
                    raise _Unsupported("a cell overlaps a span from an earlier row")
            placed.append((col, cell))
            if cell.rowspan > 1:
                has_spans = True
                active[col] = (col, min(row + cell.rowspan - 1, last_row), cell)
            col += cell.colspan
        # Cells fill the free columns from the left, so the row has a hole
        # exactly when a carried span lies beyond the last of its own cells.
        # A cell repeated into such a row would land in the hole's column.
        if occupied and max(occupied) >= col:
            covered = len(occupied) + sum(cell.colspan for _col, cell in placed)
            if covered <= max(occupied):
                raise _Unsupported("an empty column left of a spanned cell")
        parsed.append((open_tag, placed, close_tag))
    return parsed, carried, has_spans


def build_rowspan_slicer(
    body: str, rows: list[tuple[str, str]], *, table_label: str
) -> RowspanSlicer | None:
    """Return a slicer for ``rows``, or ``None`` to slice them exactly as cut.

    ``rows`` must be ``split_html_rows(body)``. ``None`` means either that no
    body cell spans rows, or that the table is outside the supported grammar;
    the second case logs one warning naming ``table_label`` and the reason.
    """
    if not any(w != "thead" and "rowspan" in tr.lower() for w, tr in rows):
        return None
    try:
        header_rows = _header_row_count(body, len(rows))
        parsed, carried, has_spans = _layout([tr for _w, tr in rows[header_rows:]])
    except _Unsupported as exc:
        logger.warning(
            "Table %s: rowspan cells are not repaired across row slices (%s); "
            "a slice that starts inside a span may misalign its columns",
            table_label,
            exc,
        )
        return None
    if not has_spans:
        return None
    return RowspanSlicer(rows, header_rows, parsed, carried)
