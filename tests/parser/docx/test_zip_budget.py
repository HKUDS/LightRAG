"""Decompression budget for the native DOCX engine (GHSA-2wpj-ffvv-2pq8).

A ``.docx`` is a ZIP. Before this guard, ``validate_source`` checked only
existence / is-file / suffix, so a 449 KiB archive declaring 200 MiB of
``word/document.xml`` was admitted and parsed: measured peak RSS 1180 MB on a
449 KiB upload, with ``MAX_UPLOAD_SIZE`` (100 MB, compressed) and
``MAX_REQUEST_BODY_BYTES`` (1 MiB, request body) both active and neither
applying, because both bound the compressed form.

The fix-proof test here is
``test_high_ratio_docx_is_refused_before_any_member_is_decompressed``: it fails
on the pre-fix parser by *returning normally* (no exception), which is the
behaviour that let the bomb through — not by an AttributeError about a missing
symbol.
"""

from __future__ import annotations

import io
import struct
import zipfile
from pathlib import Path

import pytest

from lightrag.parser.docx.parser import NativeDocxParser
from lightrag.parser.docx.zip_budget import (
    DocxDecompressionBudgetError,
    enforce_docx_decompression_budget,
)

_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
    "</Types>"
)

# Present so the benign fixtures are openable by python-docx, not just by
# zipfile: the legacy-engine test parses one for real.
_ROOT_RELS = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
    "</Relationships>"
)


def _write_docx(path: Path, *, body_xml: str, extra_members: int = 0) -> Path:
    """Write a minimal .docx whose document.xml holds ``body_xml``."""
    doc = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body_xml}</w:body></w:document>"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("[Content_Types].xml", _CONTENT_TYPES)
        zf.writestr("_rels/.rels", _ROOT_RELS)
        zf.writestr("word/document.xml", doc)
        for i in range(extra_members):
            zf.writestr(f"word/media/image{i}.png", b"x")
    return path


def _bomb(path: Path, *, uncompressed_mib: int = 200) -> Path:
    """A high-ratio .docx: many small text nodes summing to ~N MiB.

    Deliberately NOT one giant ``<w:t>``: lxml caps a single text node at
    10 MB, so the single-node shape reports a parse error instead of the
    unbounded expansion this guard exists to stop. Many ordinary-sized nodes
    is both the realistic worst case and the one that actually expands.
    """
    para = "<w:p><w:r><w:t>" + ("A" * 900) + "</w:t></w:r></w:p>"
    return _write_docx(
        path, body_xml=para * ((uncompressed_mib * 1024 * 1024) // len(para))
    )


def _benign(path: Path) -> Path:
    return _write_docx(path, body_xml="<w:p><w:r><w:t>hello world</w:t></w:r></w:p>")


# --- fix proof -------------------------------------------------------------


def test_high_ratio_docx_is_refused_before_any_member_is_decompressed(
    tmp_path, monkeypatch
):
    """The bomb is refused, and refused without decompressing anything.

    Two assertions in one test on purpose: "raises" alone would also be
    satisfied by a guard that reads every member to measure it, which would
    reintroduce the cost being defended against. The budget must come from
    the central directory (``infolist()``), so no member read may happen.
    """
    src = _bomb(tmp_path / "bomb.docx")
    # A 200 MiB expansion from well under a megabyte on disk.
    assert src.stat().st_size < 1024 * 1024

    def _explode(*args, **kwargs):  # pragma: no cover - only on regression
        raise AssertionError("budget check decompressed a member")

    monkeypatch.setattr(zipfile.ZipFile, "read", _explode)
    monkeypatch.setattr(zipfile.ZipFile, "open", _explode)

    with pytest.raises(DocxDecompressionBudgetError) as exc:
        NativeDocxParser().validate_source_blocking(src, "bomb.docx")

    message = str(exc.value)
    assert "uncompressed bytes" in message
    assert "DOCX_MAX_COMPRESSION_RATIO" in message


def test_budget_error_is_an_exception_the_pipeline_can_catch():
    # The parse worker catches ``except Exception``; a BaseException-only
    # error type would tear down the worker instead of failing one document.
    assert issubclass(DocxDecompressionBudgetError, Exception)
    # ValueError specifically: validate_source's existing rejection is a
    # ValueError, and callers discriminate on nothing finer.
    assert issubclass(DocxDecompressionBudgetError, ValueError)


# --- the gates, one at a time ----------------------------------------------


def test_absolute_ceiling_refuses_a_low_ratio_archive(tmp_path, monkeypatch):
    """Ratio alone is not the guard: a low-ratio archive over the hard cap
    must still be refused, or an incompressible 2 GB .docx sails through."""
    monkeypatch.setenv("DOCX_MAX_UNCOMPRESSED_BYTES", str(4 * 1024 * 1024))
    monkeypatch.setenv("DOCX_MAX_COMPRESSION_RATIO", "0")  # ratio gate off
    src = _bomb(tmp_path / "big.docx", uncompressed_mib=8)

    with pytest.raises(DocxDecompressionBudgetError) as exc:
        enforce_docx_decompression_budget(src, "big.docx")
    assert "DOCX_MAX_UNCOMPRESSED_BYTES" in str(exc.value)


def test_ratio_gate_refuses_below_the_absolute_ceiling(tmp_path, monkeypatch):
    """The ratio gate is what closes the amplification: 512 MiB alone would
    still let a ~1 MB upload expand 500x."""
    monkeypatch.setenv("DOCX_MAX_UNCOMPRESSED_BYTES", str(512 * 1024 * 1024))
    monkeypatch.setenv("DOCX_MAX_COMPRESSION_RATIO", "100")
    monkeypatch.setenv("DOCX_RATIO_FLOOR_BYTES", str(16 * 1024 * 1024))
    src = _bomb(tmp_path / "ratio.docx", uncompressed_mib=200)

    total = sum(i.file_size for i in zipfile.ZipFile(src).infolist())
    assert total < 512 * 1024 * 1024  # under the hard cap...
    with pytest.raises(DocxDecompressionBudgetError):  # ...refused anyway
        enforce_docx_decompression_budget(src, "ratio.docx")


def test_ratio_floor_exempts_small_documents(tmp_path, monkeypatch):
    """Small documents legitimately compress far better than large ones; a
    floor-less ratio gate would refuse ordinary files."""
    monkeypatch.setenv("DOCX_MAX_COMPRESSION_RATIO", "2")  # absurdly strict
    monkeypatch.setenv("DOCX_RATIO_FLOOR_BYTES", str(16 * 1024 * 1024))
    src = _write_docx(
        tmp_path / "small.docx",
        body_xml="<w:p><w:r><w:t>" + ("A" * 200_000) + "</w:t></w:r></w:p>",
    )
    total = sum(i.file_size for i in zipfile.ZipFile(src).infolist())
    assert total / src.stat().st_size > 2  # would trip the ratio...
    assert total < 16 * 1024 * 1024  # ...but sits under the floor
    enforce_docx_decompression_budget(src, "small.docx")


def test_entry_count_ceiling(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCX_MAX_ENTRIES", "5")
    src = _write_docx(
        tmp_path / "many.docx",
        body_xml="<w:p/>",
        extra_members=20,
    )
    with pytest.raises(DocxDecompressionBudgetError) as exc:
        enforce_docx_decompression_budget(src, "many.docx")
    assert "DOCX_MAX_ENTRIES" in str(exc.value)


def test_limits_are_read_live_and_can_be_disabled(tmp_path, monkeypatch):
    """An operator whose legitimate document is refused must be able to raise
    the limit without a code change — the reason these are env knobs."""
    src = _bomb(tmp_path / "bomb.docx")
    with pytest.raises(DocxDecompressionBudgetError):
        enforce_docx_decompression_budget(src, "bomb.docx")

    monkeypatch.setenv("DOCX_MAX_UNCOMPRESSED_BYTES", "0")  # 0 = gate off
    monkeypatch.setenv("DOCX_MAX_COMPRESSION_RATIO", "0")
    monkeypatch.setenv("DOCX_MAX_ENTRIES", "0")
    enforce_docx_decompression_budget(src, "bomb.docx")


# --- no false refusals ------------------------------------------------------


def test_ordinary_docx_passes_untouched(tmp_path):
    src = _benign(tmp_path / "ok.docx")
    parser = NativeDocxParser()
    parser.validate_source(src, "ok.docx")
    parser.validate_source_blocking(src, "ok.docx")


def test_non_zip_named_docx_is_passed_through_not_rejected_here(tmp_path):
    """The budget weighs declared expansion; it does not adjudicate format.

    An unreadable archive has no declared expansion, and python-docx already
    reports "not a .docx" downstream, more precisely. Rejecting it here would
    change the error existing callers see for an unrelated reason — and it
    does: 25 tests across tests/parser/ stand up a placeholder .docx with the
    real extract mocked out, and a format verdict in validate_source fails
    every one of them.
    """
    src = tmp_path / "fake.docx"
    src.write_bytes(b"this is not a zip archive")
    NativeDocxParser().validate_source(src, "fake.docx")


def test_suffix_rejection_still_precedes_the_budget(tmp_path):
    """The pre-existing verdict must not be reordered behind a zip open —
    a .txt is refused for what it is, not for being an unreadable archive."""
    src = tmp_path / "note.txt"
    src.write_text("hello")
    with pytest.raises(ValueError) as exc:
        NativeDocxParser().validate_source(src, "note.txt")
    assert "does not support pending file" in str(exc.value)


# --- the assumption the pre-check rests on ---------------------------------


def test_a_lying_central_directory_is_self_limiting_not_a_bypass():
    """Pins the library semantics that make a metadata-only check sufficient.

    The budget is read from the central directory, which an attacker writes.
    Understating a member does not buy unbounded expansion: CPython's zipfile
    stops decompressing at the declared ``file_size`` and then fails the CRC,
    so a liar gets at most the size it declared — which is the quantity the
    budget bounds. If a future zipfile (or a switch to another zip reader)
    ever decompresses past the declared size, this test fails and the guard
    needs a counting reader, not just the pre-check.
    """
    payload = b"A" * (8 * 1024 * 1024)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("big.txt", payload)
    raw = bytearray(buf.getvalue())

    central = raw.find(b"PK\x01\x02")
    local = raw.find(b"PK\x03\x04")
    assert central > 0 and local >= 0
    assert struct.unpack_from("<I", raw, central + 24)[0] == len(payload)
    struct.pack_into("<I", raw, central + 24, 1024)  # lie: claim 1 KiB
    struct.pack_into("<I", raw, local + 22, 1024)

    zf = zipfile.ZipFile(io.BytesIO(bytes(raw)))
    assert sum(i.file_size for i in zf.infolist()) == 1024  # what we'd gate on

    with pytest.raises(zipfile.BadZipFile):
        zf.read("big.txt")  # truncated at the lie, then CRC-failed


# --- the legacy engine parses .docx too ------------------------------------


def test_legacy_engine_is_guarded_too(tmp_path, monkeypatch):
    """Guarding only the native path leaves a routing bypass.

    The legacy extractor also hands a .docx to python-docx, and the default
    routing (``*:native-teP,*:legacy-R``) reaches it both as a fallback and
    from a ``bomb.[legacy].docx`` filename hint. It takes bytes rather than a
    path, which is why the budget accepts either.
    """
    from lightrag.parser.legacy.extractors import _extract_docx

    payload = _bomb(tmp_path / "bomb.docx").read_bytes()
    with pytest.raises(DocxDecompressionBudgetError):
        _extract_docx(payload)

    # And an ordinary document still parses through the same entry point.
    monkeypatch.delenv("DOCX_MAX_COMPRESSION_RATIO", raising=False)
    assert isinstance(_extract_docx(_benign(tmp_path / "ok.docx").read_bytes()), str)


def test_budget_accepts_bytes_and_path_equivalently(tmp_path):
    src = _bomb(tmp_path / "bomb.docx")
    with pytest.raises(DocxDecompressionBudgetError) as from_path:
        enforce_docx_decompression_budget(src, "bomb.docx")
    with pytest.raises(DocxDecompressionBudgetError) as from_bytes:
        enforce_docx_decompression_budget(src.read_bytes(), "bomb.docx")
    assert str(from_path.value) == str(from_bytes.value)


# --- the budget must not run on the event loop -----------------------------
#
# validate_source is called inline in NativeParserBase.parse, on the event
# loop; only _extract_sync runs in the parser executor. Opening the archive
# of a .docx with a huge central directory takes seconds (measured: 5.2s for
# 100k members, 24.3s for 500k), so a budget check placed in validate_source
# stalls every unrelated request — the exact symptom the budget exists to
# prevent, re-created by the guard itself.


def test_validate_source_does_not_open_the_archive(tmp_path, monkeypatch):
    src = _benign(tmp_path / "ok.docx")

    def _explode(*args, **kwargs):  # pragma: no cover - only on regression
        raise AssertionError("validate_source opened the zip on the event loop")

    monkeypatch.setattr(zipfile, "ZipFile", _explode)
    NativeDocxParser().validate_source(src, "ok.docx")


def test_budget_runs_in_the_executor_before_artifact_cleanup():
    """The base must call the blocking hook inside _extract_sync, ahead of the
    rmtree — a refusal must not have destroyed the previous attempt's output.
    """
    import inspect

    from lightrag.parser.native_base import NativeParserBase

    body = inspect.getsource(NativeParserBase.parse)
    extract_sync = body.index("def _extract_sync()")
    call = body.index("validate_source_blocking", extract_sync)
    rmtree = body.index("shutil.rmtree", extract_sync)
    assert call < rmtree, "blocking validation must precede parsed_dir cleanup"
    # And it must NOT be called inline in parse() before that.
    assert "validate_source_blocking" not in body[:extract_sync]


# --- an entry-count bomb is refused without building every ZipInfo ---------


def test_entry_count_bomb_is_refused_without_opening_the_archive(tmp_path, monkeypatch):
    """ZipFile() materializes one ZipInfo per member before infolist() can be
    counted, so checking the count there means paying the whole cost to then
    refuse. The EOCD record carries the total; read that instead."""
    src = tmp_path / "many.docx"
    with zipfile.ZipFile(src, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("[Content_Types].xml", _CONTENT_TYPES)
        for i in range(300):
            zf.writestr(f"m/{i}", b"")
    monkeypatch.setenv("DOCX_MAX_ENTRIES", "50")

    def _explode(*args, **kwargs):  # pragma: no cover - only on regression
        raise AssertionError("entry-count refusal opened the central directory")

    monkeypatch.setattr(zipfile, "ZipFile", _explode)
    with pytest.raises(DocxDecompressionBudgetError) as exc:
        enforce_docx_decompression_budget(src, "many.docx")
    assert "DOCX_MAX_ENTRIES" in str(exc.value)


def test_eocd_prefilter_is_fail_open(tmp_path, monkeypatch):
    """A tail that carries no readable EOCD must fall through to the
    authoritative infolist() count, never refuse on its own — a parsing bug
    here must not be able to reject a legitimate document."""
    from lightrag.parser.docx.zip_budget import _declared_entry_count

    junk = tmp_path / "junk.docx"
    junk.write_bytes(b"not a zip at all")
    assert _declared_entry_count(junk) is None
    # ...and the caller treats that as "no verdict": a non-zip is passed
    # through to python-docx, exactly as before the prefilter existed.
    enforce_docx_decompression_budget(junk, "junk.docx")


def test_eocd_count_matches_reality_for_an_ordinary_archive(tmp_path):
    from lightrag.parser.docx.zip_budget import _declared_entry_count

    src = _write_docx(tmp_path / "ok.docx", body_xml="<w:p/>", extra_members=7)
    with zipfile.ZipFile(src) as zf:
        real = len(zf.infolist())
    assert _declared_entry_count(src) == real


def test_eocd_prefilter_survives_a_zip_comment(tmp_path, monkeypatch):
    """The EOCD is not necessarily the last 22 bytes — a trailing comment
    pushes it back, and the record must still be found."""
    from lightrag.parser.docx.zip_budget import _declared_entry_count

    src = tmp_path / "commented.docx"
    with zipfile.ZipFile(src, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("[Content_Types].xml", _CONTENT_TYPES)
        for i in range(20):
            zf.writestr(f"m/{i}", b"")
        zf.comment = b"x" * 5000
    assert _declared_entry_count(src) == 21
