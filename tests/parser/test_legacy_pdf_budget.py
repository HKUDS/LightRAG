"""Resource-limited PDF extraction (lightrag/parser/legacy/pdf_budget.py).

pypdf runs in a child process with CPU and memory limits, because what it does
with a hostile PDF cannot be predicted from outside. These tests cover the
limits, the two cheap gates that run first, the failure paths of the child
process, and that a PDF pypdf can extract comes back exactly as pypdf returns
it.
"""

from __future__ import annotations

import ast
import io
import random
import subprocess
import sys
import time
from pathlib import Path

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
from lightrag.parser.legacy import pdf_budget, pdf_worker
from lightrag.parser.legacy.pdf_budget import PdfExtractionBudgetError

pytestmark = pytest.mark.offline

_ENV = (
    "PDF_MAX_CONTENT_BYTES",
    "PDF_MAX_CONTENT_RATIO",
    "PDF_MAX_EXTRACTED_CHARS",
    "PDF_MAX_CPU_SECONDS",
    "PDF_MAX_MEMORY_MB",
    "PDF_EXTRACT_IN_SUBPROCESS",
)


@pytest.fixture(autouse=True)
def _default_limits(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(params=["subprocess", "in-process"])
def mode(request, monkeypatch):
    if request.param == "in-process":
        monkeypatch.setenv("PDF_EXTRACT_IN_SUBPROCESS", "false")
    return request.param


@pytest.fixture
def in_process(monkeypatch):
    monkeypatch.setenv("PDF_EXTRACT_IN_SUBPROCESS", "false")


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


def _bogus(b: _Builder, form_resources=None):
    """A stream pypdf cannot decode: cheap, deterministic failure."""
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


def _corrupt_object(data: bytes, marker: bytes) -> bytes:
    """Overwrite the body of the indirect object holding ``marker``."""
    out = bytearray(data)
    at = out.index(marker)
    start = out.rindex(b" obj", 0, at) + len(b" obj")
    end = at + len(marker)
    out[start:end] = b")" + b"\0" * (end - start - 1)
    return bytes(out)


def _unbudgeted(data: bytes):
    """What plain pypdf returns for ``data``: the text, or the exception type."""
    try:
        return "".join(
            page.extract_text() + "\n" for page in PdfReader(io.BytesIO(data)).pages
        )
    except Exception as exc:
        return type(exc)


def _assert_same_as_pypdf(data: bytes) -> None:
    expected = _unbudgeted(data)
    if isinstance(expected, type):
        with pytest.raises(LegacyExtractionError) as raised:
            extract_text(data, "pdf")
        assert not isinstance(raised.value, PdfExtractionBudgetError)
    else:
        assert extract_text(data, "pdf") == expected


def _cmap_font_pdf(pages: int, entries: int = 20_000) -> bytes:
    """Pages sharing one font whose ToUnicode CMap pypdf re-parses per page."""
    b = _Builder()
    lines = [b"begincmap", b"1 begincodespacerange <0000> <FFFF> endcodespacerange"]
    for start in range(0, entries, 100):
        lines.append(b"100 beginbfchar")
        lines += [b"<%04X> <0041>" % code for code in range(start, start + 100)]
        lines.append(b"endbfchar")
    lines.append(b"endcmap")
    font = b.writer._add_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
                NameObject("/ToUnicode"): b.stream(b"\n".join(lines)),
            }
        )
    )
    b.fonts = DictionaryObject({NameObject("/F1"): font})
    resources = b.writer._add_object(b.resources())
    contents = b.stream(_text_ops(b"Hi", 1))
    for _ in range(pages):
        page = b.writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = resources
        page[NameObject("/Contents")] = contents
    return b.bytes()


# --- Output is pypdf's output -------------------------------------------------


def test_ordinary_pdf_text_is_unchanged(mode):
    b = _Builder()
    for word in (b"Alpha", b"Beta", b"Gamma"):
        b.page(b.stream(_text_ops(word, 1)))
    data = b.bytes()

    expected = _unbudgeted(data)
    assert "Beta" in expected
    assert extract_text(data, "pdf") == expected


def test_pypdf_does_not_run_in_this_process(monkeypatch):
    b = _Builder()
    b.page(b.stream(_text_ops(b"Isolated", 1)))
    data = b.bytes()

    def boom(self, *args, **kwargs):
        raise AssertionError("pypdf extraction ran in the caller's process")

    monkeypatch.setattr(PageObject, "extract_text", boom)
    assert "Isolated" in extract_text(data, "pdf")

    monkeypatch.setenv("PDF_EXTRACT_IN_SUBPROCESS", "false")
    with pytest.raises(LegacyExtractionError, match="caller's process"):
        extract_text(data, "pdf")


def test_in_process_mode_starts_no_process(in_process, monkeypatch):
    b = _Builder()
    b.page(b.stream(_text_ops(b"Inline", 1)))

    def no_popen(*args, **kwargs):
        raise AssertionError("a child process was started")

    monkeypatch.setattr(subprocess, "Popen", no_popen)
    assert "Inline" in extract_text(b.bytes(), "pdf")


def test_worker_module_never_imports_lightrag():
    # The child runs this file by path; importing the package there would cost
    # seconds and hundreds of megabytes for every PDF.
    tree = ast.parse(Path(pdf_worker.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert "lightrag" not in imported
    top_level = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
    }
    assert top_level <= set(sys.stdlib_module_names) | {"__future__"}


# --- The two deterministic gates ---------------------------------------------


def _set(monkeypatch, **env):
    for name, value in env.items():
        monkeypatch.setenv(name, str(value))


def test_shared_content_stream_is_refused(mode, monkeypatch):
    # Each page is far below pypdf's per-stream ceiling; together they are not.
    data = _shared_stream_pdf(40, _text_ops(b"A" * 1000, 30))
    _set(monkeypatch, PDF_MAX_CONTENT_BYTES=400_000)
    with pytest.raises(PdfExtractionBudgetError, match="PDF_MAX_CONTENT_BYTES"):
        extract_text(data, "pdf", file_path="bomb.pdf")

    _set(monkeypatch, PDF_MAX_CONTENT_BYTES=4_000_000)
    assert extract_text(data, "pdf").count("A" * 1000) == 40 * 30


def test_content_gate_refuses_before_any_page_is_parsed(in_process, monkeypatch):
    data = _shared_stream_pdf(40, _text_ops(b"A" * 1000, 30))
    _set(monkeypatch, PDF_MAX_CONTENT_BYTES=400_000)
    calls: list[int] = []
    monkeypatch.setattr(
        PageObject, "extract_text", lambda self, *a, **k: calls.append(1) or ""
    )
    with pytest.raises(PdfExtractionBudgetError, match="bomb.pdf"):
        extract_text(data, "pdf", file_path="bomb.pdf")
    assert calls == []


def test_budget_error_is_a_legacy_extraction_error():
    assert issubclass(PdfExtractionBudgetError, LegacyExtractionError)
    assert issubclass(PdfExtractionBudgetError, ValueError)


def test_ratio_gate_bounds_content_relative_to_file_size(mode, monkeypatch):
    data = _shared_stream_pdf(20, _text_ops(b"B" * 1000, 30))
    monkeypatch.setattr(pdf_budget, "DEFAULT_PDF_CONTENT_RATIO_FLOOR_BYTES", 1024)
    _set(monkeypatch, PDF_MAX_CONTENT_BYTES=0)
    assert 20 * len(_text_ops(b"B" * 1000, 30)) / len(data) > 50

    _set(monkeypatch, PDF_MAX_CONTENT_RATIO=10)
    with pytest.raises(PdfExtractionBudgetError, match="PDF_MAX_CONTENT_RATIO"):
        extract_text(data, "pdf")

    _set(monkeypatch, PDF_MAX_CONTENT_RATIO=0)
    assert "B" * 1000 in extract_text(data, "pdf")


def test_text_budget_is_document_wide(mode, monkeypatch):
    b = _Builder()
    for _ in range(5):
        b.page(b.stream(_text_ops(b"C" * 100, 10)))
    data = b.bytes()

    _set(monkeypatch, PDF_MAX_EXTRACTED_CHARS=3000)
    with pytest.raises(PdfExtractionBudgetError, match="PDF_MAX_EXTRACTED_CHARS"):
        extract_text(data, "pdf")

    _set(monkeypatch, PDF_MAX_EXTRACTED_CHARS=6000)
    assert extract_text(data, "pdf").count("C" * 100) == 50


def test_failing_shared_page_stream_is_decoded_once_by_the_gate(
    in_process, monkeypatch
):
    # pypdf caches a successful decode but not a failed one; the gate must not
    # decode a failing shared stream again for every page.
    b = _Builder()
    bogus = _bogus(b)
    for _ in range(30):
        b.page(bogus)
    calls: list[int] = []
    original = EncodedStreamObject.get_data

    def spy(self, *args, **kwargs):
        if self.get("/Filter") == "/BogusDecode":
            calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(EncodedStreamObject, "get_data", spy)
    with pytest.raises(LegacyExtractionError, match="BogusDecode"):
        extract_text(b.bytes(), "pdf")
    # Once by the gate, once by pypdf at the first page.
    assert len(calls) == 2


def test_shared_contents_array_is_traversed_once(in_process, monkeypatch):
    b = _Builder()
    empty = b.stream(b"")
    array = b.writer._add_object(ArrayObject([empty] * 500))
    for _ in range(200):
        page = b.writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject()
        page[NameObject("/Contents")] = array
    data = b.bytes()
    calls: list[int] = []
    original = pdf_worker._resolve

    def spy(obj):
        calls.append(1)
        return original(obj)

    monkeypatch.setattr(pdf_worker, "_resolve", spy)
    assert extract_text(data, "pdf") == _unbudgeted(data)
    # 200 pages + 500 entries, not 200 x 500.
    assert len(calls) <= 200 + 500 + 10


# --- Process limits ------------------------------------------------------------


def test_cpu_limit_stops_work_the_gates_cannot_see(monkeypatch):
    # No content stream is large here: the cost is the font's CMap, which
    # pypdf parses again for every page. Only the CPU limit bounds it.
    data = _cmap_font_pdf(pages=400)
    _set(monkeypatch, PDF_MAX_CPU_SECONDS=1)
    started = time.monotonic()
    with pytest.raises(PdfExtractionBudgetError, match="PDF_MAX_CPU_SECONDS"):
        extract_text(data, "pdf", file_path="fonts.pdf")
    assert time.monotonic() - started < 30


def test_cpu_limit_stops_a_repeated_form(monkeypatch):
    b = _Builder()
    form = b.form(b"0 0 m 1 1 l S\n" * 5000)
    b.page(b.stream(b"/X0 Do\n" * 4000), {"/X0": form})
    _set(monkeypatch, PDF_MAX_CPU_SECONDS=1)
    with pytest.raises(PdfExtractionBudgetError, match="PDF_MAX_CPU_SECONDS"):
        extract_text(b.bytes(), "pdf")


@pytest.mark.skipif(sys.platform == "win32", reason="no memory limit on Windows")
def test_memory_limit_stops_the_process(monkeypatch):
    b = _Builder()
    for index in range(6):
        b.page(b.stream(b"%" + bytes([65 + index]) * 50_000_000 + b"\n"))
    _set(
        monkeypatch,
        PDF_MAX_MEMORY_MB=150,
        PDF_MAX_CONTENT_BYTES=0,
        PDF_MAX_CONTENT_RATIO=0,
    )
    with pytest.raises(PdfExtractionBudgetError, match="PDF_MAX_MEMORY_MB"):
        extract_text(b.bytes(), "pdf")


def test_memory_limit_ignores_the_size_of_the_calling_process(monkeypatch):
    # On Linux a child inherits its parent's peak resident size. A server
    # holding more than the limit must not make every PDF look over it.
    # Slow enough for the watchdog to look several times.
    data = _cmap_font_pdf(pages=15)
    ballast = b"\x01" * (300 * 1024 * 1024)
    _set(monkeypatch, PDF_MAX_MEMORY_MB=150)
    assert extract_text(data, "pdf") == "AA\n" * 15  # the CMap maps both codes to A
    assert len(ballast) == 300 * 1024 * 1024


def test_wall_clock_timeout_is_the_last_resort(monkeypatch):
    data = _cmap_font_pdf(pages=400)
    # CPU limit far away, so only the parent's timeout can stop the child.
    _set(monkeypatch, PDF_MAX_CPU_SECONDS=1000)
    monkeypatch.setattr(pdf_budget, "_WALL_CLOCK_FACTOR", 0)
    monkeypatch.setattr(pdf_budget, "_WALL_CLOCK_SLACK_SECONDS", 1)
    with pytest.raises(PdfExtractionBudgetError, match="wall-clock"):
        extract_text(data, "pdf")


# --- Failure paths of the child process ------------------------------------------


def _fake_worker(tmp_path, monkeypatch, source: str) -> None:
    script = tmp_path / "fake_worker.py"
    script.write_text(source, encoding="utf-8")
    monkeypatch.setattr(pdf_budget, "_WORKER_PATH", str(script))


def test_worker_exiting_without_status_is_an_error(tmp_path, monkeypatch):
    _fake_worker(
        tmp_path, monkeypatch, "import sys\nsys.stderr.write('boom\\n')\nsys.exit(7)\n"
    )
    with pytest.raises(LegacyExtractionError, match="exit code 7.*boom") as raised:
        extract_text(b"%PDF-1.7", "pdf")
    assert not isinstance(raised.value, PdfExtractionBudgetError)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_worker_killed_by_a_signal_is_a_budget_refusal(tmp_path, monkeypatch):
    _fake_worker(
        tmp_path,
        monkeypatch,
        "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n",
    )
    with pytest.raises(PdfExtractionBudgetError, match="signal 9"):
        extract_text(b"%PDF-1.7", "pdf")


def test_text_with_lone_surrogates_round_trips(tmp_path, monkeypatch):
    _fake_worker(
        tmp_path,
        monkeypatch,
        "import os, sys\n"
        "sys.stdin.buffer.read()\n"
        "os.write(1, 'a\\ud800b'.encode('utf-8', 'surrogatepass'))\n"
        'os.write(2, b\'{"kind": "ok"}\\n\')\n',
    )
    assert extract_text(b"%PDF-1.7", "pdf") == "a\ud800b"


def test_process_that_cannot_start_names_the_opt_out(monkeypatch):
    monkeypatch.setattr(sys, "executable", "/nonexistent/python")
    with pytest.raises(LegacyExtractionError, match="PDF_EXTRACT_IN_SUBPROCESS"):
        extract_text(b"%PDF-1.7", "pdf")


def test_pypdf_error_keeps_its_type_name_and_message(mode):
    with pytest.raises(LegacyExtractionError, match="PdfReadError|PdfStreamError"):
        extract_text(b"this is not a pdf", "pdf")


def test_pypdf_warnings_reach_the_lightrag_log(lightrag_log_records):
    b = _Builder()
    b.page(
        b.stream(_text_ops(b"Body", 1) + b"/X0 Do\n"),
        {"/X0": _bogus(b, form_resources=b.resources())},
    )
    assert "Body" in extract_text(b.bytes(), "pdf", file_path="warn.pdf")
    messages = [record.getMessage() for record in lightrag_log_records]
    assert any("warn.pdf" in m and "XFormObject" in m for m in messages)


# --- PDFs pypdf extracts must still extract -------------------------------------
# The first, in-process design mirrored pypdf's resource and form handling and
# failed each of these documents at some point. They now only have to match.


def _odd_pdfs() -> dict[str, bytes]:
    pdfs: dict[str, bytes] = {}

    b = _Builder()
    b.page(b.stream(_text_ops(b"Keep", 1)))
    page = b.writer.add_blank_page(width=612, height=792)
    page[NameObject("/Resources")] = NullObject()
    page[NameObject("/Contents")] = _bogus(b)
    pdfs["page with null resources and undecodable contents"] = b.bytes()

    b = _Builder()
    broken = _bogus(b, form_resources=b.resources())
    for _ in range(3):
        b.page(b.stream(_text_ops(b"Body", 1) + b"/X0 Do\n"), {"/X0": broken})
    pdfs["undecodable form used on every page"] = b.bytes()

    for label, entry in (
        ("null", NullObject()),
        ("name", NameObject("/NotAForm")),
        ("plain dictionary", DictionaryObject()),
    ):
        b = _Builder()
        ref = b.writer._add_object(entry)
        b.page(b.stream(_text_ops(b"Keep", 1) + b"/X0 Do\n"), {"/X0": ref})
        pdfs[f"XObject entry that is a {label}"] = b.bytes()

    b = _Builder()
    marked = b.resources()
    marked[NameObject("/Marker")] = NameObject("/CorruptMe")
    s = DecodedStreamObject()
    s.set_data(_text_ops(b"Form", 1))
    s.update(
        {
            NameObject("/Type"): NameObject("/XObject"),
            NameObject("/Subtype"): NameObject("/Form"),
            NameObject("/Resources"): b.writer._add_object(marked),
        }
    )
    b.page(
        b.stream(_text_ops(b"Keep", 1) + b"/X0 Do\n"),
        {"/X0": b.writer._add_object(s)},
    )
    pdfs["form with corrupt indirect resources"] = _corrupt_object(
        b.bytes(), b"/Marker /CorruptMe"
    )

    b = _Builder()
    s = EncodedStreamObject()
    s._data = b"x" * 64
    s[NameObject("/Filter")] = NameObject("/BogusDecode")
    s[NameObject("/Resources")] = b.resources()
    b.page(
        b.stream(_text_ops(b"Body", 1) + b"/X0 Do\n" * 20),
        {"/X0": b.writer._add_object(s)},
    )
    pdfs["stream without subtype invoked repeatedly"] = b.bytes()

    b = _Builder()
    b.page(b.stream(_text_ops(b"Body", 1) + b"Do\n/X0 Do\n"), {"/X0": b.form(b"")})
    pdfs["Do without an operand"] = b.bytes()

    b = _Builder()
    bogus = _bogus(b)
    for _ in range(5):
        b.page(bogus)
    pdfs["undecodable shared page stream"] = b.bytes()
    return pdfs


_ODD_PDFS = _odd_pdfs()


@pytest.mark.parametrize("label", sorted(_ODD_PDFS))
def test_odd_pdf_matches_pypdf(label):
    _assert_same_as_pypdf(_ODD_PDFS[label])


def test_odd_pdfs_cover_both_outcomes():
    outcomes = {label: _unbudgeted(data) for label, data in _ODD_PDFS.items()}
    assert isinstance(outcomes["undecodable shared page stream"], type)
    extracted = [o for o in outcomes.values() if not isinstance(o, type)]
    assert len(extracted) == len(outcomes) - 1
    assert all("Keep" in text or "Body" in text for text in extracted)


def _random_pdf(rng: random.Random) -> bytes:
    """A small PDF with randomly broken resources, contents and XObjects."""
    b = _Builder()
    add = b.writer._add_object
    name = NameObject

    def resources_for(kind: str, xobjects: dict[str, object]) -> dict:
        res = b.resources(xobjects)
        if kind == "own":
            return {name("/Resources"): res}
        if kind == "indirect":
            return {name("/Resources"): add(res)}
        if kind == "empty":
            return {name("/Resources"): DictionaryObject()}
        if kind == "null":
            return {name("/Resources"): NullObject()}
        if kind == "name":
            return {name("/Resources"): name("/Oops")}
        if kind == "parent":
            node = add(DictionaryObject({name("/Resources"): res}))
            for _ in range(rng.choice([0, 2, 120])):
                node = add(DictionaryObject({name("/Parent"): node}))
            return {name("/Parent"): node}
        return {}

    kinds = ["own"] * 4 + ["indirect", "empty", "null", "name", "parent", "missing"]
    forms: list[object] = []

    def entry(depth: int) -> object:
        roll = rng.random()
        if roll < 0.55 or not forms:
            ops = bytearray(_text_ops(b"F", rng.randint(0, 2)))
            xobjects: dict[str, object] = {}
            if depth < 2:
                for index in range(rng.randint(0, 2)):
                    xobjects[f"/X{index}"] = entry(depth + 1)
                    ops += b"/X%d Do\n" % index * rng.choice([1, 2])
            if rng.random() < 0.15:
                ops += b"Do\n/Nope Do\n"
            if rng.random() < 0.12:
                s = EncodedStreamObject()
                s._data = b"x" * 40
                s[name("/Filter")] = name("/BogusDecode")
            else:
                s = DecodedStreamObject()
                s.set_data(bytes(ops))
                s = s.flate_encode()
            if rng.random() > 0.08:
                s[name("/Subtype")] = name("/Form")
            s.update(resources_for(rng.choice(kinds), xobjects))
            forms.append(add(s))
            return forms[-1]
        if roll < 0.7:
            return rng.choice(forms)
        return add(rng.choice([NullObject(), name("/No"), DictionaryObject()]))

    shared = b.stream(_text_ops(b"S", 1))
    for number in range(rng.randint(1, 3)):
        page = b.writer.add_blank_page(width=612, height=792)
        del page["/Resources"]
        ops = bytearray(_text_ops(b"P%d" % number, 1))
        xobjects = {}
        for index in range(rng.randint(0, 3)):
            xobjects[f"/X{index}"] = entry(0)
            ops += b"q /X%d Do Q\n" % index * rng.choice([1, 3])
        page.update(resources_for(rng.choice(kinds[:-2] + ["missing"]), xobjects))
        own = b.stream(bytes(ops))
        page[name("/Contents")] = rng.choice(
            [
                own,
                own,
                ArrayObject([own, shared, add(NullObject()), add(DictionaryObject())]),
                ArrayObject([own, _bogus(b)]),
                _bogus(b),
                NullObject(),
            ]
        )
    return b.bytes()


def test_random_broken_pdfs_match_pypdf(in_process):
    # Differential check against plain pypdf: whatever pypdf does with a
    # malformed structure, the limited extraction must do the same.
    for seed in range(150):
        data = _random_pdf(random.Random(seed))
        expected = _unbudgeted(data)
        try:
            got: object = extract_text(data, "pdf")
        except PdfExtractionBudgetError:
            raise
        except LegacyExtractionError:
            got = None
        if isinstance(expected, type):
            assert got is None, f"seed {seed}: pypdf fails, extraction did not"
        else:
            assert got == expected, f"seed {seed}"
