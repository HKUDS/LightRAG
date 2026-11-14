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
    EncodedStreamObject,
    FloatObject,
    NameObject,
    NullObject,
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


# --- Streams pypdf cannot decode -------------------------------------------
# pypdf caches a successful decode but not a failed one, so a budget that
# decodes on its own must not make a failing stream cost once per page or per
# invocation. An unknown filter makes the failure cheap and deterministic.


def _bogus(b: _Builder, form_resources=None):
    s = EncodedStreamObject()
    s._data = b"x" * 64
    s[NameObject("/Filter")] = NameObject("/BogusDecode")
    if form_resources is not None:
        s.update(
            {
                NameObject("/Type"): NameObject("/XObject"),
                NameObject("/Subtype"): NameObject("/Form"),
                NameObject("/Resources"): form_resources,
            }
        )
    return b.writer._add_object(s)


def _count_decodes(monkeypatch) -> list[int]:
    calls: list[int] = []
    original = EncodedStreamObject.get_data

    def spy(self, *args, **kwargs):
        if self.get("/Filter") == "/BogusDecode":
            calls.append(id(self))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(EncodedStreamObject, "get_data", spy)
    return calls


def test_failing_shared_page_stream_fails_once_like_pypdf(monkeypatch):
    b = _Builder()
    bogus = _bogus(b)
    for _ in range(30):
        b.page(bogus)
    data = b.bytes()
    with pytest.raises(Exception) as unbudgeted:
        PdfReader(io.BytesIO(data)).pages[0].extract_text()

    calls = _count_decodes(monkeypatch)
    with pytest.raises(type(unbudgeted.value)):
        extract_text(data, "pdf")
    assert len(calls) == 1


@pytest.mark.parametrize("resources", ["missing", "empty"])
def test_failing_contents_of_pages_pypdf_skips_do_not_fail(monkeypatch, resources):
    # Without /Resources no pypdf release parses the page; with empty ones
    # recent pypdf does not either. The budget may decode such a stream once.
    b = _Builder()
    bogus = _bogus(b)
    for _ in range(30):
        page = b.writer.add_blank_page(width=612, height=792)
        page[NameObject("/Contents")] = bogus
        if resources == "missing":
            del page["/Resources"]
        else:
            page[NameObject("/Resources")] = DictionaryObject()
    data = b.bytes()
    try:
        expected = "".join(
            page.extract_text() + "\n" for page in PdfReader(io.BytesIO(data)).pages
        )
    except Exception as exc:  # pypdf 6.1.0 parses a page with empty resources
        expected = exc

    calls = _count_decodes(monkeypatch)
    if isinstance(expected, Exception):
        with pytest.raises(type(expected)):
            extract_text(data, "pdf")
        assert len(calls) <= 2
    else:
        assert extract_text(data, "pdf") == expected == "\n" * 30
        assert len(calls) == (0 if resources == "missing" else 1)


def _form_page(b: _Builder, form, invocations: int) -> None:
    b.page(
        b.stream(_text_ops(b"Body", 1) + b"/X0 Do\n" * invocations),
        {"/X0": form},
    )


def test_failing_form_invoked_again_is_refused(monkeypatch):
    b = _Builder()
    _form_page(b, _bogus(b, form_resources=b.resources()), 20)
    calls = _count_decodes(monkeypatch)
    with pytest.raises(PdfExtractionBudgetError, match="cannot be decoded"):
        extract_text(b.bytes(), "pdf")
    # Ours once, pypdf's swallowed attempt once, then the refusal.
    assert len(calls) <= 2


def test_failing_form_invoked_once_still_extracts(monkeypatch):
    b = _Builder()
    _form_page(b, _bogus(b, form_resources=b.resources()), 1)
    assert "Body" in extract_text(b.bytes(), "pdf")


def test_failing_form_with_empty_resources_is_not_refused(monkeypatch):
    # Recent pypdf never decodes a form whose /Resources is empty.
    b = _Builder()
    _form_page(b, _bogus(b, form_resources=DictionaryObject()), 20)
    calls = _count_decodes(monkeypatch)
    assert "Body" in extract_text(b.bytes(), "pdf")
    assert len(set(calls)) <= 1 and len(calls) <= 21


def test_form_without_resources_is_not_charged():
    b = _Builder()
    s = DecodedStreamObject()
    s.set_data(_text_ops(b"N" * 100, 100))
    s.update(
        {
            NameObject("/Type"): NameObject("/XObject"),
            NameObject("/Subtype"): NameObject("/Form"),
        }
    )
    form = b.writer._add_object(s.flate_encode())
    page_data = b"/X0 Do\n" * 50
    b.page(b.stream(page_data), {"/X0": form})
    page = PdfReader(io.BytesIO(b.bytes())).pages[0]

    budget = PdfExtractionBudget(1, "noresources.pdf")
    budget.check_page_contents([page])
    budget.finish_page(page.extract_text(**budget.visitors(page)))
    assert budget.content_bytes == len(page_data)


@pytest.mark.parametrize(
    "entry",
    [NullObject(), NameObject("/NotAForm"), DictionaryObject()],
    ids=["null", "name", "plain-dict"],
)
def test_xobject_entries_that_are_not_streams_are_skipped_like_pypdf(entry):
    b = _Builder()
    ref = b.writer._add_object(entry)
    b.page(b.stream(_text_ops(b"Keep", 1) + b"/X0 Do\n"), {"/X0": ref})
    data = b.bytes()
    expected = "".join(
        page.extract_text() + "\n" for page in PdfReader(io.BytesIO(data)).pages
    )
    assert "Keep" in expected
    assert extract_text(data, "pdf") == expected


def _traceback_frames(exc: BaseException) -> int:
    frames, seen, node = 0, set(), exc
    while node is not None and id(node) not in seen:
        seen.add(id(node))
        tb = node.__traceback__
        while tb is not None:
            frames, tb = frames + 1, tb.tb_next
        node = node.__cause__ or node.__context__
    return frames


def test_memoized_decode_failure_does_not_accumulate_tracebacks():
    # Every invocation of a tolerated failing form re-raises the memoized
    # error; the frames must not pile up on that one instance.
    b = _Builder()
    form = _bogus(b, form_resources=DictionaryObject())
    b.page(b.stream(b"/X0 Do\n" * 2000), {"/X0": form})
    page = PdfReader(io.BytesIO(b.bytes())).pages[0]

    budget = PdfExtractionBudget(1, "tb.pdf")
    budget.check_page_contents([page])
    budget.finish_page(page.extract_text(**budget.visitors(page)))

    failures = [r for _, r in budget._decoded.values() if isinstance(r, Exception)]
    assert len(failures) == 1
    assert _traceback_frames(failures[0]) < 10


# --- Objects that fail to resolve ------------------------------------------
# The budget resolves objects inside pypdf's visitor callbacks, where an error
# escapes pypdf's own catch. Whatever pypdf does with a corrupt object, the
# budgeted extraction must do the same.


def _corrupt_object(data: bytes, marker: bytes) -> bytes:
    """Overwrite the body of the indirect object holding ``marker``."""
    out = bytearray(data)
    at = out.index(marker)
    start = out.rindex(b" obj", 0, at) + len(b" obj")
    end = at + len(marker)
    out[start:end] = b")" + b"\0" * (end - start - 1)
    return bytes(out)


def _unbudgeted(data: bytes):
    try:
        return "".join(
            page.extract_text() + "\n" for page in PdfReader(io.BytesIO(data)).pages
        )
    except Exception as exc:
        return type(exc)


def _assert_same_as_pypdf(data: bytes) -> None:
    expected = _unbudgeted(data)
    if isinstance(expected, type):
        # Same outcome, not necessarily the same type: once one resolution of
        # a corrupt object has failed, pypdf's reader reports a retry as a
        # self-reference loop instead.
        with pytest.raises(Exception) as raised:
            extract_text(data, "pdf")
        assert not isinstance(raised.value, PdfExtractionBudgetError)
    else:
        assert extract_text(data, "pdf") == expected


def _marked_resources(b: _Builder):
    res = b.resources()
    res[NameObject("/Marker")] = NameObject("/CorruptMe")
    return b.writer._add_object(res)


def test_form_with_corrupt_indirect_resources_is_skipped_like_pypdf():
    b = _Builder()
    s = DecodedStreamObject()
    s.set_data(_text_ops(b"Form", 1))
    s.update(
        {
            NameObject("/Type"): NameObject("/XObject"),
            NameObject("/Subtype"): NameObject("/Form"),
            NameObject("/Resources"): _marked_resources(b),
        }
    )
    form = b.writer._add_object(s)
    b.page(b.stream(_text_ops(b"Keep", 1) + b"/X0 Do\n"), {"/X0": form})
    data = _corrupt_object(b.bytes(), b"/Marker /CorruptMe")

    assert "Keep" in _unbudgeted(data)
    _assert_same_as_pypdf(data)


def test_page_with_corrupt_indirect_resources_matches_pypdf():
    b = _Builder()
    b.page(b.stream(_text_ops(b"Other", 1)))
    page = b.writer.add_blank_page(width=612, height=792)
    page[NameObject("/Resources")] = _marked_resources(b)
    page[NameObject("/Contents")] = b.stream(_text_ops(b"Gone", 1))
    _assert_same_as_pypdf(_corrupt_object(b.bytes(), b"/Marker /CorruptMe"))


def test_corrupt_contents_of_page_with_empty_resources_matches_pypdf():
    b = _Builder()
    b.page(b.stream(_text_ops(b"Other", 1)))
    page = b.writer.add_blank_page(width=612, height=792)
    page[NameObject("/Resources")] = DictionaryObject()
    marked = b.writer._add_object(
        DictionaryObject({NameObject("/Marker"): NameObject("/CorruptMe")})
    )
    page[NameObject("/Contents")] = marked
    _assert_same_as_pypdf(_corrupt_object(b.bytes(), b"/Marker /CorruptMe"))


def test_form_inheriting_resources_through_parent_is_charged():
    # pypdf resolves a form's /Resources through /Parent, as for a page, and
    # parses it; the budget must charge it rather than treat it as skipped.
    b = _Builder()
    form_data = _text_ops(b"P" * 10, 5)
    s = DecodedStreamObject()
    s.set_data(form_data)
    s.update(
        {
            NameObject("/Type"): NameObject("/XObject"),
            NameObject("/Subtype"): NameObject("/Form"),
            NameObject("/Parent"): b.writer._add_object(
                DictionaryObject({NameObject("/Resources"): b.resources()})
            ),
        }
    )
    form = b.writer._add_object(s.flate_encode())
    page_data = b"/X0 Do\n" * 20
    b.page(b.stream(page_data), {"/X0": form})
    data = b.bytes()
    page = PdfReader(io.BytesIO(data)).pages[0]

    budget = PdfExtractionBudget(1, "parent.pdf")
    budget.check_page_contents([page])
    text = page.extract_text(**budget.visitors(page))
    budget.finish_page(text)
    assert budget.content_bytes == len(page_data) + 20 * len(form_data)
    # Recent pypdf extracts the form's text; 6.1.0 returns none for it.
    assert text + "\n" == _unbudgeted(data)


def test_stream_without_subtype_is_skipped_like_pypdf():
    # pypdf reads /Subtype as a required key inside its catch, so a stream
    # without one is skipped; it must not be decoded, charged or refused.
    b = _Builder()
    s = EncodedStreamObject()
    s._data = b"x" * 64
    s[NameObject("/Filter")] = NameObject("/BogusDecode")
    s[NameObject("/Resources")] = b.resources()
    _form_page(b, b.writer._add_object(s), 20)
    data = b.bytes()
    assert "Body" in _unbudgeted(data)
    _assert_same_as_pypdf(data)
