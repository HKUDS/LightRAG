"""rowspan cells across the row-split boundaries of an oversized HTML table."""

from html.parser import HTMLParser

import pytest

from lightrag.chunker import _html_rowspan
from lightrag.chunker._html_rowspan import build_rowspan_slicer
from lightrag.chunker.paragraph_semantic import _split_table_text
from lightrag.table_markup import split_html_rows
from lightrag.utils import Tokenizer, TokenizerInterface

pytestmark = pytest.mark.offline


class _CharTokenizer(TokenizerInterface):
    """One token per character, so a table's size is its length."""

    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


_SEGMENT_HEADER = (
    "<thead><tr><th>Segment</th><th>Quarter</th><th>Revenue</th></tr></thead>"
)


def _segment_table(label_attrs: str = "", wrap: str = "{}") -> str:
    """Cloud/Hardware labels each span their four quarter rows."""
    segments = {
        "Cloud": [("Q1", "1,200"), ("Q2", "1,350"), ("Q3", "1,410"), ("Q4", "1,520")],
        "Hardware": [("Q1", "800"), ("Q2", "760"), ("Q3", "790"), ("Q4", "810")],
    }
    body = []
    for segment, quarters in segments.items():
        for i, (quarter, revenue) in enumerate(quarters):
            label = (
                f'<td rowspan="{len(quarters)}"{label_attrs}>'
                f"{wrap.format(segment)}</td>"
                if i == 0
                else ""
            )
            body.append(f"<tr>{label}<td>{quarter}</td><td>{revenue}</td></tr>")
    return (
        f'<table id="tb-seg" format="html">{_SEGMENT_HEADER}'
        f"<tbody>{''.join(body)}</tbody></table>"
    )


def _colspan_table() -> str:
    """Per group: a 4-row label, a 2x2 block, and a 2-row cell before the last column."""
    pad = "x" * 30
    body = []
    for g in range(3):
        body.append(
            f'<tr><td rowspan="4">L{g}</td><td rowspan="2" colspan="2">B{g}</td>'
            f'<td>{g}-1 {pad}</td><td rowspan="2">R{g}</td><td>{g}-t1</td></tr>'
            f"<tr><td>{g}-2 {pad}</td><td>{g}-t2</td></tr>"
            f"<tr><td>{g}-a</td><td>{g}-b</td><td>{g}-3 {pad}</td><td>{g}-r</td>"
            f"<td>{g}-t3</td></tr>"
            f'<tr><td colspan="2">{g}-c</td><td>{g}-4 {pad}</td><td>{g}-s</td>'
            f"<td>{g}-t4</td></tr>"
        )
    return f'<table id="tb-col" format="html"><tbody>{"".join(body)}</tbody></table>'


def _tall_label_table() -> str:
    """One wide label spanning 12 short rows: its copy dominates every slice."""
    rows = "".join(
        f'<tr><td rowspan="12">{"L" * 100}</td><td>v{i}</td></tr>'
        if i == 0
        else f"<tr><td>v{i}</td></tr>"
        for i in range(12)
    )
    return f'<table id="tb-tall" format="html"><tbody>{rows}</tbody></table>'


class _BodyRows(HTMLParser):
    """(text, rowspan, colspan) of every ``<td>``, row by row."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[list[list]] = []
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.rows.append([])
        elif tag == "td":
            found: dict = {}
            for name, value in attrs:
                found.setdefault(name, value)
            spans = [int(found.get(n) or 1) for n in ("rowspan", "colspan")]
            self.cell = ["", *spans]
            self.rows[-1].append(self.cell)

    def handle_endtag(self, tag):
        if tag == "td":
            self.cell = None

    def handle_data(self, data):
        if self.cell is not None:
            self.cell[0] += data


def _body_grid(piece: str) -> list[list[str]]:
    """Expand a piece's body rows to a text grid, spans filled in.

    Read with ``html.parser`` so the oracle shares nothing with the code under
    test. Fails when a span runs past the piece's last row or a row leaves a
    column uncovered, so the grid alone shows a dropped or misplaced cell.
    """
    parser = _BodyRows()
    parser.feed(piece.split("<tbody>", 1)[1])
    pending: dict[tuple[int, int], str] = {}
    grid = []
    for r, cells in enumerate(parser.rows):
        row = {c: t for (rr, c), t in pending.items() if rr == r}
        col = 0
        for text, rowspan, colspan in cells:
            while col in row:
                col += 1
            assert r + rowspan <= len(parser.rows), ("rowspan past its slice", piece)
            for dr in range(rowspan):
                for dc in range(colspan):
                    if dr:
                        pending[(r + dr, col + dc)] = text
                    else:
                        row[col + dc] = text
            col += colspan
        assert sorted(row) == list(range(len(row))), ("uncovered column", piece)
        grid.append([row[c] for c in sorted(row)])
    return grid


@pytest.mark.parametrize(
    ("table", "header_body", "target_max", "opens_on"),
    [
        # Cut after Cloud's Q3: the next slice opens on Cloud's Q4 row.
        pytest.param(_segment_table, _SEGMENT_HEADER, 420, ["Cloud", "Q4"], id="label"),
        # A quoted ``>`` is part of its tag, in the label's own attributes and
        # in an element inside the label; a ``<`` that starts no tag is text.
        pytest.param(
            lambda: _segment_table(
                ' title="a > b"', '<span title="1 > 0">{}</span> < 9<br/>'
            ),
            _SEGMENT_HEADER,
            500,
            ["Cloud < 9", "Q4"],
            id="quoted-gt",
        ),
        # Cut after Cloud's Q2: the label is repeated with rowspan="2".
        pytest.param(
            _segment_table, _SEGMENT_HEADER, 300, ["Cloud", "Q3"], id="label-2"
        ),
        # One-row slices inside every span: repeated cells keep colspan and column.
        pytest.param(_colspan_table, None, 200, ["L0", "B0", "B0"], id="colspan"),
        # Only the repeated label overflows: re-splitting must still cut rows
        # instead of degrading the table to a character split.
        pytest.param(_tall_label_table, None, 200, ["L" * 100, "v6"], id="tall-label"),
    ],
)
def test_rowspan_cells_are_local_to_each_slice(
    table, header_body, target_max, opens_on
):
    source = table()
    pieces = _split_table_text(
        source,
        tokenizer=Tokenizer(model_name="char", tokenizer=_CharTokenizer()),
        target_max=target_max,
        target_ideal=int(target_max * 0.8),
        last_min=20,
        header_body=header_body,
    )
    assert len(pieces) >= 2
    assert all(piece.startswith("<table") for piece in pieces)
    assert all(len(piece) <= target_max for piece in pieces)
    # Rejoined, the slices' grids are the source table's grid row for row.
    grids = [_body_grid(piece) for piece in pieces]
    assert [row for grid in grids for row in grid] == _body_grid(source)
    # The case really cuts through the span it names: a later slice opens on
    # a continuation row, which only a repeated spanned cell can complete.
    assert opens_on in [grid[0][: len(opens_on)] for grid in grids[1:]], pieces


def _rows(*cells_per_row: str) -> str:
    return "".join(f"<tr>{cells}</tr>" for cells in cells_per_row)


_SPAN_ROWS = _rows('<td rowspan="2">A</td><td>a1</td>', "<td>a2</td>")


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        pytest.param(
            f"<tbody>{_SPAN_ROWS}<!-- note --></tbody>", "comment", id="comment"
        ),
        pytest.param(
            f"<caption>C</caption><tbody>{_SPAN_ROWS}</tbody>",
            "outside the rows",
            id="caption",
        ),
        pytest.param(
            f"<tbody>{_SPAN_ROWS}</tbody><tbody>{_rows('<td>B</td><td>b</td>')}</tbody>",
            "row groups",
            id="two-tbody",
        ),
        pytest.param(
            f"<tbody>{_SPAN_ROWS}</tbody><tfoot>{_rows('<td>F</td><td>f</td>')}</tfoot>",
            "row groups",
            id="tfoot",
        ),
        pytest.param(
            _rows('<td rowspan="2">A<td>a1</td>', "<td>a2</td>"),
            "inside a cell",
            id="omitted-end-tag",
        ),
        pytest.param(
            _rows('<td rowspan="2">A</td>stray<td>a1</td>', "<td>a2</td>"),
            "not a complete cell",
            id="text-between-cells",
        ),
        pytest.param(
            _rows('<td rowspan="0">A</td><td>a1</td>', "<td>a2</td>"),
            "rowspan is outside",
            id="rowspan-zero",
        ),
        pytest.param(
            _rows('<td rowspan="2x">A</td><td>a1</td>', "<td>a2</td>"),
            "rowspan is not a positive integer",
            id="rowspan-not-a-number",
        ),
        pytest.param(
            _rows('<td rowspan="2" colspan="1001">A</td><td>a1</td>', "<td>a2</td>"),
            "colspan is outside",
            id="colspan-over-limit",
        ),
        pytest.param(
            # The first ``</td>`` sits inside a quoted attribute of the span.
            _rows(
                '<td rowspan="2"><span title="</td><td>">A</span></td><td>a1</td>',
                "<td>a2</td>",
            ),
            "cannot be read",
            id="delimiter-in-nested-attribute",
        ),
        pytest.param(
            # The ``</td><td>`` inside a textarea is text, not two cell tags.
            _rows(
                '<td rowspan="2"><textarea></td><td></textarea>A</td><td>a1</td>',
                "<td>a2</td>",
            ),
            "<textarea> element inside a cell",
            id="raw-text-element",
        ),
        pytest.param(
            # An end tag that closes the cell and its row group in HTML.
            _rows('<td rowspan="2">A</tbody></td><td>a1</td>', "<td>a2</td>"),
            "table markup inside a cell",
            id="group-end-tag-in-cell",
        ),
        pytest.param(
            # U+00A0 between the cells is text, not whitespace.
            _rows('<td rowspan="2">A</td>\u00a0<td>a1</td>', "<td>a2</td>"),
            "not a complete cell",
            id="nbsp-between-cells",
        ),
        pytest.param(
            _rows('<td rowspan="2">A<![CDATA[x]]></td><td>a1</td>', "<td>a2</td>"),
            "declaration inside a cell",
            id="declaration-in-cell",
        ),
        pytest.param(
            # Longer than ``int`` accepts: still a refusal, not a crash.
            _rows(f'<td rowspan="{"9" * 5000}">A</td><td>a1</td>', "<td>a2</td>"),
            "rowspan is outside",
            id="rowspan-huge",
        ),
        pytest.param(
            # Row 2 holds one cell: A, repeated there, would move to column 2.
            _rows('<td>X</td><td>Y</td><td rowspan="2">A</td>', "<td>B</td>"),
            "empty column",
            id="hole-before-span",
        ),
        pytest.param(
            # B's colspan runs into the column A still covers.
            _rows(
                '<td>X</td><td rowspan="2">A</td><td>Z</td>',
                '<td colspan="2">B</td>',
            ),
            "overlaps a span",
            id="cell-overlaps-span",
        ),
        pytest.param(
            _rows('<td rowspan="2" rowspan="3">A</td><td>a1</td>', "<td>a2</td>"),
            "repeats rowspan",
            id="repeated-attribute",
        ),
    ],
)
def test_table_outside_the_grammar_is_left_as_cut(body, reason, lightrag_log_records):
    rows = split_html_rows(body)

    assert build_rowspan_slicer(body, rows, table_label="tb-x") is None
    # One warning names the table and the reason; nothing is repaired in part.
    messages = [record.getMessage() for record in lightrag_log_records]
    assert len(messages) == 1
    assert "tb-x" in messages[0] and reason in messages[0]


def test_layout_stops_at_its_work_budget(monkeypatch, lightrag_log_records):
    # Three spans carried into each of three rows, two steps each: 18 steps.
    spans = "".join(f'<td rowspan="4">c{i}</td>' for i in range(3))
    body = _rows(spans, "<td>x</td>", "<td>y</td>", "<td>z</td>")
    rows = split_html_rows(body)
    monkeypatch.setattr(_html_rowspan, "LAYOUT_BUDGET", 18)
    assert build_rowspan_slicer(body, rows, table_label="tb-x") is not None

    monkeypatch.setattr(_html_rowspan, "LAYOUT_BUDGET", 17)
    assert build_rowspan_slicer(body, rows, table_label="tb-x") is None
    assert [
        "too many overlapping spans" in r.getMessage() for r in lightrag_log_records
    ] == [True]


@pytest.mark.parametrize(
    "body",
    [
        # ``data-rowspan`` is not a span attribute.
        pytest.param(
            _rows('<td data-rowspan="2">A</td>', "<td>B</td>"), id="data-rowspan"
        ),
        # U+00A0 is not whitespace in HTML: one attribute named "data-x\u00a0rowspan".
        pytest.param(
            _rows('<td data-x\u00a0rowspan="2">A</td>', "<td>B</td>"), id="nbsp-in-name"
        ),
        # A quoted value that merely mentions rowspan.
        pytest.param(
            _rows('<td title="rowspan=2">A</td>', "<td>B</td>"), id="quoted-mention"
        ),
    ],
)
def test_table_without_a_body_rowspan_needs_no_slicer(body, lightrag_log_records):
    assert build_rowspan_slicer(body, split_html_rows(body), table_label="t") is None
    assert lightrag_log_records == []
