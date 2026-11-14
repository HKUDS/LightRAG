"""Document-wide PDF extraction budget (lightrag/parser/legacy/pdf_budget.py).

pypdf bounds each stream, not the document: pages sharing one content stream,
or one form XObject invoked many times, let a tiny PDF make pypdf parse and
return without bound. These tests build such PDFs at a small scale and lower
the budget through the env knobs, which are read at parse time.
"""

from __future__ import annotations

import io

import pytest
from pypdf import PageObject, PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject,
    DecodedStreamObject,
    DictionaryObject,
    FloatObject,
    NameObject,
)

from lightrag.parser.legacy import LegacyExtractionError, extract_text
from lightrag.parser.legacy import pdf_budget
from lightrag.parser.legacy.pdf_budget import (
    PdfExtractionBudget,
    PdfExtractionBudgetError,
)

pytestmark = pytest.mark.offline

_ENV = ("PDF_MAX_CONTENT_BYTES", "PDF_MAX_CONTENT_RATIO", "PDF_MAX_EXTRACTED_CHARS")


@pytest.fixture(autouse=True)
def _default_budget(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)


def _text_ops(line: bytes, repeat: int) -> bytes:
    return b"BT /F1 12 Tf\n" + (b"(" + line + b") Tj\n") * repeat + b"ET\n"


class _Builder:
    def __init__(self) -> None:
        self.writer = PdfWriter()
        font = self.writer._add_object(
            DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                }
            )
        )
        self.fonts = DictionaryObject({NameObject("/F1"): font})

    def resources(self, xobjects: dict[str, object] | None = None):
        res = DictionaryObject({NameObject("/Font"): self.fonts})
        if xobjects:
            res[NameObject("/XObject")] = DictionaryObject(
                {NameObject(k): v for k, v in xobjects.items()}
            )
        return res

    def stream(self, data: bytes):
        s = DecodedStreamObject()
        s.set_data(data)
        return self.writer._add_object(s.flate_encode())

    def form(self, data: bytes, xobjects: dict[str, object] | None = None):
        s = DecodedStreamObject()
        s.set_data(data)
        s.update(
            {
                NameObject("/Type"): NameObject("/XObject"),
                NameObject("/Subtype"): NameObject("/Form"),
                NameObject("/BBox"): ArrayObject(
                    [FloatObject(0), FloatObject(0), FloatObject(612), FloatObject(792)]
                ),
                NameObject("/Resources"): self.resources(xobjects),
            }
        )
        return self.writer._add_object(s.flate_encode())

    def page(self, contents, xobjects: dict[str, object] | None = None) -> None:
        page = self.writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = self.resources(xobjects)
        page[NameObject("/Contents")] = contents

    def bytes(self) -> bytes:
        buf = io.BytesIO()
        self.writer.write(buf)
        return buf.getvalue()


def _shared_stream_pdf(pages: int, data: bytes) -> bytes:
    b = _Builder()
    shared = b.stream(data)
    for _ in range(pages):
        b.page(shared)
    return b.bytes()


def _count_page_extractions(monkeypatch) -> list[int]:
    calls: list[int] = []
    original = PageObject.extract_text

    def spy(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PageObject, "extract_text", spy)
    return calls


def test_ordinary_pdf_text_is_unchanged():
    b = _Builder()
    for word in (b"Alpha", b"Beta", b"Gamma"):
        b.page(b.stream(_text_ops(word, 1)))
    data = b.bytes()

    expected = "".join(
        page.extract_text() + "\n" for page in PdfReader(io.BytesIO(data)).pages
    )
    assert extract_text(data, "pdf") == expected
    assert "Beta" in expected


def test_shared_content_stream_is_refused_before_any_page_is_parsed(monkeypatch):
    # Each page is far below pypdf's per-stream ceiling; together they are not.
    data = _shared_stream_pdf(40, _text_ops(b"A" * 1000, 30))
    monkeypatch.setenv("PDF_MAX_CONTENT_BYTES", str(400_000))
    calls = _count_page_extractions(monkeypatch)

    with pytest.raises(PdfExtractionBudgetError, match="PDF_MAX_CONTENT_BYTES"):
        extract_text(data, "pdf", file_path="bomb.pdf")
    assert calls == []

    monkeypatch.setenv("PDF_MAX_CONTENT_BYTES", str(4_000_000))
    assert extract_text(data, "pdf").count("A" * 1000) == 40 * 30


def test_budget_error_is_a_legacy_extraction_error():
    assert issubclass(PdfExtractionBudgetError, LegacyExtractionError)
    assert issubclass(PdfExtractionBudgetError, ValueError)


def test_ratio_gate_bounds_content_relative_to_file_size(monkeypatch):
    data = _shared_stream_pdf(20, _text_ops(b"B" * 1000, 30))
    monkeypatch.setattr(pdf_budget, "DEFAULT_PDF_CONTENT_RATIO_FLOOR_BYTES", 1024)
    monkeypatch.setenv("PDF_MAX_CONTENT_BYTES", "0")
    content = 20 * len(_text_ops(b"B" * 1000, 30))
    assert content / len(data) > 50

    monkeypatch.setenv("PDF_MAX_CONTENT_RATIO", "10")
    with pytest.raises(PdfExtractionBudgetError, match="PDF_MAX_CONTENT_RATIO"):
        extract_text(data, "pdf")

    monkeypatch.setenv("PDF_MAX_CONTENT_RATIO", "0")
    assert "B" * 1000 in extract_text(data, "pdf")


def test_text_free_content_is_charged(monkeypatch):
    # No text at all, so only the content budget can stop it.
    data = _shared_stream_pdf(30, b"0 0 m 1 1 l S\n" * 5000)
    monkeypatch.setenv("PDF_MAX_EXTRACTED_CHARS", "0")
    monkeypatch.setenv("PDF_MAX_CONTENT_BYTES", str(1_000_000))
    with pytest.raises(PdfExtractionBudgetError, match="PDF_MAX_CONTENT_BYTES"):
        extract_text(data, "pdf")


def test_form_xobject_is_charged_at_every_invocation(monkeypatch):
    b = _Builder()
    form_data = _text_ops(b"F" * 100, 50)
    form = b.form(form_data)
    page_data = b"q /X0 Do Q\n" * 200
    b.page(b.stream(page_data), {"/X0": form})
    data = b.bytes()

    monkeypatch.setenv("PDF_MAX_CONTENT_BYTES", str(100 * len(form_data)))
    with pytest.raises(PdfExtractionBudgetError, match="form XObject /X0"):
        extract_text(data, "pdf")

    monkeypatch.setenv("PDF_MAX_CONTENT_BYTES", str(300 * len(form_data)))
    assert extract_text(data, "pdf").count("F" * 100) == 200 * 50


def test_nested_form_xobjects_are_charged_through_their_own_resources():
    b = _Builder()
    inner_data = _text_ops(b"I" * 10, 5)
    inner = b.form(inner_data)
    outer_data = b"q /In Do Q\n" * 10
    outer = b.form(outer_data, {"/In": inner})
    page_data = b"q /Out Do Q\n" * 10
    b.page(b.stream(page_data), {"/Out": outer})
    reader = PdfReader(io.BytesIO(b.bytes()))
    page = reader.pages[0]

    budget = PdfExtractionBudget(1, "nested.pdf")
    budget.check_page_contents([page])
    text = page.extract_text(**budget.visitors(page))
    budget.finish_page(text)

    assert text.count("I" * 10) == 100 * 5
    assert budget.content_bytes == (
        len(page_data) + 10 * len(outer_data) + 100 * len(inner_data)
    )


def test_text_budget_trips_inside_a_form_xobject_pypdf_would_swallow(monkeypatch):
    # pypdf catches Exception around a form XObject; the tripped budget must
    # still surface instead of being logged and dropped.
    b = _Builder()
    form = b.form(_text_ops(b"T" * 100, 100))
    b.page(b.stream(b"q /X0 Do Q\n"), {"/X0": form})
    data = b.bytes()

    monkeypatch.setenv("PDF_MAX_EXTRACTED_CHARS", "1000")
    with pytest.raises(PdfExtractionBudgetError, match="PDF_MAX_EXTRACTED_CHARS"):
        extract_text(data, "pdf")


def test_text_budget_is_document_wide(monkeypatch):
    b = _Builder()
    for _ in range(5):
        b.page(b.stream(_text_ops(b"C" * 100, 10)))
    data = b.bytes()

    monkeypatch.setenv("PDF_MAX_EXTRACTED_CHARS", "3000")
    calls = _count_page_extractions(monkeypatch)
    with pytest.raises(PdfExtractionBudgetError, match="PDF_MAX_EXTRACTED_CHARS"):
        extract_text(data, "pdf")
    # Pages are each 1000 chars: the fourth page trips while it is read.
    assert len(calls) == 4

    monkeypatch.setenv("PDF_MAX_EXTRACTED_CHARS", "6000")
    assert extract_text(data, "pdf").count("C" * 100) == 50
