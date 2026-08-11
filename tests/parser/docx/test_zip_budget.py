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
import os
import struct
import zipfile
from functools import lru_cache
from pathlib import Path

import pytest

from lightrag.parser.docx.parser import NativeDocxParser
from lightrag.parser.docx.zip_budget import (
    DocxDecompressionBudgetError,
    enforce_docx_decompression_budget,
)

# CI runs only ``-m offline``; these use tmp files only, no live services.
pytestmark = pytest.mark.offline

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


@lru_cache(maxsize=8)
def _bomb_bytes(uncompressed_mib: int) -> bytes:
    """The bomb archive bytes for a given uncompressed size, built once.

    Deliberately NOT one giant ``<w:t>``: lxml caps a single text node at
    10 MB, so the single-node shape reports a parse error instead of the
    unbounded expansion this guard exists to stop. Many ordinary-sized nodes
    is both the realistic worst case and the one that actually expands.

    Cached because deflating a ~200 MiB body at level 9 costs ~1s and six
    tests build the byte-identical fixture — the cache turns six builds into
    one with no loss of coverage.
    """
    para = "<w:p><w:r><w:t>" + ("A" * 900) + "</w:t></w:r></w:p>"
    body_xml = para * ((uncompressed_mib * 1024 * 1024) // len(para))
    buf = io.BytesIO()
    _write_docx(buf, body_xml=body_xml)
    return buf.getvalue()


def _bomb(path: Path, *, uncompressed_mib: int = 200) -> Path:
    """A high-ratio .docx written to ``path`` from the cached bomb bytes."""
    path.write_bytes(_bomb_bytes(uncompressed_mib))
    return path


def _benign(path: Path) -> Path:
    return _write_docx(path, body_xml="<w:p><w:r><w:t>hello world</w:t></w:r></w:p>")


def _write_padded_high_ratio_zip(
    path: Path,
    *,
    member_sizes: list[int],
    padding_size: int,
) -> Path:
    """Write high-ratio members plus unrelated incompressible stored padding."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for index, size in enumerate(member_sizes):
            zf.writestr(f"payload-{index}.bin", b"\0" * size)
        zf.writestr(
            "padding.bin",
            os.urandom(padding_size),
            compress_type=zipfile.ZIP_STORED,
        )
    return path


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


def test_member_ratio_rejects_stored_padding_before_decompression(
    tmp_path, monkeypatch
):
    """Unrelated stored bytes must not dilute a 1000x member below the cap."""
    monkeypatch.setenv("DOCX_MAX_UNCOMPRESSED_BYTES", str(8 * 1024 * 1024))
    monkeypatch.setenv("DOCX_MAX_COMPRESSION_RATIO", "100")
    monkeypatch.setenv("DOCX_RATIO_FLOOR_BYTES", str(1024 * 1024))
    src = _write_padded_high_ratio_zip(
        tmp_path / "padded.docx",
        member_sizes=[4 * 1024 * 1024],
        padding_size=64 * 1024,
    )

    with zipfile.ZipFile(src) as zf:
        infos = zf.infolist()
        total = sum(info.file_size for info in infos)
        assert total / src.stat().st_size < 100
        assert infos[0].file_size / infos[0].compress_size > 1000

    def _explode(*args, **kwargs):  # pragma: no cover - only on regression
        raise AssertionError("member-ratio guard decompressed a member")

    monkeypatch.setattr(zipfile.ZipFile, "read", _explode)
    monkeypatch.setattr(zipfile.ZipFile, "open", _explode)

    with pytest.raises(DocxDecompressionBudgetError) as exc:
        NativeDocxParser().validate_source_blocking(src, "padded.docx")

    message = str(exc.value)
    assert "over-ratio members" in message
    assert "payload-0.bin" not in message


def test_real_stored_content_supports_repetitive_xml_one_for_one(tmp_path, monkeypatch):
    """Real media may offset excess XML expansion, but only byte for byte.

    The former absolute member-floor rule rejected this mixed archive even
    though its total ratio is low. The stored bytes here model JPEG/media
    payloads; unlike a ratio multiplier, their allowance equals their cost.
    """
    mib = 1024 * 1024
    monkeypatch.setenv("DOCX_MAX_UNCOMPRESSED_BYTES", str(16 * mib))
    monkeypatch.setenv("DOCX_MAX_COMPRESSION_RATIO", "100")
    monkeypatch.setenv("DOCX_RATIO_FLOOR_BYTES", str(mib))
    src = _write_padded_high_ratio_zip(
        tmp_path / "mixed.docx",
        member_sizes=[6 * mib],
        padding_size=5 * mib,
    )

    with zipfile.ZipFile(src) as zf:
        infos = zf.infolist()
        assert infos[0].file_size / infos[0].compress_size > 100
        assert sum(info.file_size for info in infos) / src.stat().st_size < 100

    enforce_docx_decompression_budget(src, src.name)


@pytest.mark.parametrize("suffix", ["docx", "pptx", "xlsx"])
def test_padded_member_is_rejected_by_every_legacy_ooxml_entrypoint(
    tmp_path, monkeypatch, suffix
):
    from lightrag.parser.legacy.extractors import extract_text

    monkeypatch.setenv("DOCX_MAX_UNCOMPRESSED_BYTES", str(8 * 1024 * 1024))
    monkeypatch.setenv("DOCX_MAX_COMPRESSION_RATIO", "100")
    monkeypatch.setenv("DOCX_RATIO_FLOOR_BYTES", str(1024 * 1024))
    payload = _write_padded_high_ratio_zip(
        tmp_path / f"padded.{suffix}",
        member_sizes=[4 * 1024 * 1024],
        padding_size=64 * 1024,
    ).read_bytes()

    with pytest.raises(DocxDecompressionBudgetError):
        extract_text(payload, suffix, file_path=f"padded.{suffix}")


def test_ratio_floor_applies_to_cumulative_high_ratio_member_bytes(
    tmp_path, monkeypatch
):
    """Splitting a bomb into individually-sub-floor members must not bypass."""
    floor = 1024 * 1024
    monkeypatch.setenv("DOCX_MAX_UNCOMPRESSED_BYTES", str(8 * 1024 * 1024))
    monkeypatch.setenv("DOCX_MAX_COMPRESSION_RATIO", "100")
    monkeypatch.setenv("DOCX_RATIO_FLOOR_BYTES", str(floor))
    src = _write_padded_high_ratio_zip(
        tmp_path / "split.docx",
        member_sizes=[floor // 2, floor // 2, floor // 2],
        padding_size=64 * 1024,
    )

    with zipfile.ZipFile(src) as zf:
        infos = zf.infolist()
        assert all(info.file_size <= floor for info in infos)
        assert sum(info.file_size for info in infos) / src.stat().st_size < 100

    with pytest.raises(DocxDecompressionBudgetError, match="over-ratio members"):
        enforce_docx_decompression_budget(src, "split.docx")


@pytest.mark.parametrize("high_ratio_bytes", [512 * 1024, 1024 * 1024])
def test_cumulative_member_ratio_floor_keeps_small_content_exempt(
    tmp_path, monkeypatch, high_ratio_bytes
):
    """The exemption includes its exact boundary and ignores stored padding."""
    floor = 1024 * 1024
    monkeypatch.setenv("DOCX_MAX_UNCOMPRESSED_BYTES", str(8 * 1024 * 1024))
    monkeypatch.setenv("DOCX_MAX_COMPRESSION_RATIO", "100")
    monkeypatch.setenv("DOCX_RATIO_FLOOR_BYTES", str(floor))
    src = _write_padded_high_ratio_zip(
        tmp_path / f"small-{high_ratio_bytes}.docx",
        member_sizes=[high_ratio_bytes],
        padding_size=64 * 1024,
    )

    enforce_docx_decompression_budget(src, src.name)


def test_negative_ratio_floor_rejects_only_archives_with_over_ratio_members(
    tmp_path, monkeypatch
):
    """A negative floor removes the exemption; it does not reject ratio-safe ZIPs."""
    monkeypatch.setenv("DOCX_MAX_UNCOMPRESSED_BYTES", str(8 * 1024 * 1024))
    monkeypatch.setenv("DOCX_MAX_COMPRESSION_RATIO", "100")
    monkeypatch.setenv("DOCX_RATIO_FLOOR_BYTES", "-1")

    safe = tmp_path / "stored.docx"
    with zipfile.ZipFile(safe, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("stored.bin", b"A" * 1024)
    enforce_docx_decompression_budget(safe, safe.name)

    high_ratio = _write_padded_high_ratio_zip(
        tmp_path / "strict.docx",
        member_sizes=[512 * 1024],
        padding_size=64 * 1024,
    )
    with pytest.raises(DocxDecompressionBudgetError, match="over-ratio members"):
        enforce_docx_decompression_budget(high_ratio, high_ratio.name)


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
    from a ``bomb.[legacy].docx`` filename hint. The budget is enforced in the
    ``extract_text`` dispatcher, before the extractor opens the archive.
    """
    from lightrag.parser.legacy.extractors import extract_text

    payload = _bomb(tmp_path / "bomb.docx").read_bytes()
    with pytest.raises(DocxDecompressionBudgetError):
        extract_text(payload, "docx", file_path="bomb.docx")

    # And an ordinary document still parses through the same entry point.
    monkeypatch.delenv("DOCX_MAX_COMPRESSION_RATIO", raising=False)
    assert isinstance(
        extract_text(_benign(tmp_path / "ok.docx").read_bytes(), "docx"), str
    )


@pytest.mark.parametrize("suffix", ["pptx", "xlsx"])
def test_legacy_pptx_xlsx_share_the_budget(tmp_path, suffix):
    """.pptx / .xlsx are the same OPC/ZIP bomb class as .docx.

    They register in the legacy suffix dispatch and route to legacy by default
    (native handles only docx/md/textpack). The budget runs in the dispatcher
    BEFORE python-pptx/openpyxl open the archive, so this fires without those
    libraries — and _extract_xlsx would otherwise double the blowup by loading
    the workbook twice.
    """
    from lightrag.parser.legacy.extractors import extract_text

    payload = _bomb(tmp_path / f"bomb.{suffix}").read_bytes()
    with pytest.raises(DocxDecompressionBudgetError):
        extract_text(payload, suffix, file_path=f"bomb.{suffix}")


def test_budget_error_names_the_file_not_the_extension(tmp_path):
    """The refusal must name the source, not the literal 'docx'.

    LegacyParser.parse threads ctx.file_path through extract_text; the native
    path already names the real file for the identical refusal.
    """
    from lightrag.parser.legacy.extractors import extract_text

    payload = _bomb(tmp_path / "quarterly-report.docx").read_bytes()
    with pytest.raises(DocxDecompressionBudgetError) as exc:
        extract_text(payload, "docx", file_path="quarterly-report.docx")
    assert "quarterly-report.docx" in str(exc.value)


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


# --- a refusal must not destroy a previous successful parse ----------------


def _parse_via_real_pipeline(source_path, tmp_path, monkeypatch, doc_id):
    """Drive the real NativeParserBase.parse (executor, rollback path, all)."""
    import asyncio
    from unittest import mock

    from lightrag.constants import FULL_DOCS_FORMAT_PENDING_PARSE
    from lightrag.parser.base import ParseContext
    from lightrag.parser.debug import build_debug_rag
    from lightrag.parser.registry import get_parser

    def _stub_blocks(file_path, **_kwargs):
        return [
            {
                "uuid": "p1",
                "heading": "H",
                "content": "# H\nbody",
                "type": "text",
                "parent_headings": [],
                "level": 1,
            }
        ]

    with mock.patch(
        "lightrag.parser.docx.parse_document.extract_docx_blocks", _stub_blocks
    ):
        return asyncio.run(
            get_parser("native").parse(
                ParseContext(
                    build_debug_rag(),
                    doc_id,
                    str(source_path),
                    {"parse_format": FULL_DOCS_FORMAT_PENDING_PARSE, "content": ""},
                )
            )
        )


def test_refusal_preserves_a_previous_successful_parse(tmp_path, monkeypatch):
    """End-to-end: an over-budget re-parse must not delete the prior sidecar.

    The rollback in ``NativeParserBase.parse`` catches BaseException and
    removes ``parsed_dir``. A budget refusal raises from inside
    ``_extract_sync``, so without the cleanup_started guard it lands in that
    rollback and deletes a COMPLETE sidecar from an earlier successful parse —
    one a persisted ``sidecar_location`` still points at.
    """
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    source_path = input_dir / "doc.docx"

    # 1. A first parse that succeeds and leaves artifacts behind.
    _benign(source_path)
    result = _parse_via_real_pipeline(source_path, tmp_path, monkeypatch, "doc-keep")
    parsed_dir = Path(result.blocks_path).parent
    assert parsed_dir.is_dir()
    survivors = sorted(p.name for p in parsed_dir.iterdir())
    assert survivors, "first parse should have written artifacts"

    # 2. The same document is re-parsed after being swapped for a bomb.
    _bomb(source_path)
    with pytest.raises(DocxDecompressionBudgetError):
        _parse_via_real_pipeline(source_path, tmp_path, monkeypatch, "doc-keep")

    # 3. The earlier sidecar is still there.
    assert parsed_dir.is_dir(), "refusal deleted the previous attempt's sidecar"
    assert sorted(p.name for p in parsed_dir.iterdir()) == survivors


def test_a_failure_after_cleanup_began_still_rolls_back(tmp_path, monkeypatch):
    """The guard must not disable the rollback it narrows: once extract has
    started replacing parsed_dir, a failure still has to clean up."""
    from unittest import mock

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    source_path = input_dir / "doc.docx"
    _benign(source_path)

    result = _parse_via_real_pipeline(source_path, tmp_path, monkeypatch, "doc-roll")
    parsed_dir = Path(result.blocks_path).parent
    assert parsed_dir.is_dir()

    def _boom(*args, **kwargs):
        raise RuntimeError("extract blew up after the pre-clean")

    # A successful parse consumes the source out of INPUT_DIR, so re-create it
    # for the second attempt — the same thing the preservation test does by
    # swapping in the bomb.
    _benign(source_path)
    with (
        mock.patch("lightrag.parser.docx.parser.NativeDocxParser.extract", _boom),
        pytest.raises(RuntimeError),
    ):
        _parse_via_real_pipeline(source_path, tmp_path, monkeypatch, "doc-roll")

    assert not parsed_dir.exists(), "rollback must still remove a partial parse"


def test_cancel_during_validation_does_not_destroy_a_prior_sidecar(
    tmp_path, monkeypatch
):
    """Cancelling run_in_executor cancels the future, not the worker thread.

    Without a checkpoint after validation, the abandoned worker walks on into
    the rmtree: it deletes a COMPLETE sidecar from an earlier parse and leaves
    a partial one, after the coroutine has already raised CancelledError. The
    source has been consumed out of INPUT_DIR by then, so nothing can rebuild
    it, and the persisted sidecar_location points at a file that is gone.
    """
    import asyncio
    import threading
    from unittest import mock

    from lightrag.constants import FULL_DOCS_FORMAT_PENDING_PARSE
    from lightrag.parser.base import ParseContext
    from lightrag.parser.debug import build_debug_rag
    from lightrag.parser.registry import get_parser

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    source_path = input_dir / "doc.docx"
    _benign(source_path)

    result = _parse_via_real_pipeline(source_path, tmp_path, monkeypatch, "doc-cancel")
    parsed_dir = Path(result.blocks_path).parent
    survivors = sorted(p.name for p in parsed_dir.iterdir())
    assert "doc.blocks.jsonl" in survivors

    entered = threading.Event()
    release = threading.Event()
    reached_extract = threading.Event()

    def _slow_validate(self, source, file_path):
        # A large-but-legal archive: the scan is slow, not refused.
        entered.set()
        release.wait(10)

    def _spy_extract(self, source, **kwargs):
        reached_extract.set()
        return [], {}, {}

    async def _run():
        _benign(source_path)  # the first parse consumed the source
        with (
            mock.patch.object(
                NativeDocxParser, "validate_source_blocking", _slow_validate
            ),
            mock.patch.object(NativeDocxParser, "extract", _spy_extract),
        ):
            task = asyncio.create_task(
                get_parser("native").parse(
                    ParseContext(
                        build_debug_rag(),
                        "doc-cancel",
                        str(source_path),
                        {
                            "parse_format": FULL_DOCS_FORMAT_PENDING_PARSE,
                            "content": "",
                        },
                    )
                )
            )
            await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            # Let the abandoned worker run on past validation.
            release.set()
            # Pre-fix it proceeds into extract; post-fix it stops at the
            # checkpoint and this simply times out.
            await asyncio.to_thread(reached_extract.wait, 2)

    asyncio.run(_run())

    assert not reached_extract.is_set(), "abandoned worker continued into extraction"
    assert parsed_dir.is_dir()
    assert sorted(p.name for p in parsed_dir.iterdir()) == survivors


def test_pipeline_cancel_event_raises_the_catchable_cancel_type(tmp_path, monkeypatch):
    """The checkpoint must raise LLMBridgePipelineCancelled, not CancelledError.

    The production cancel path (_watch_pipeline_cancellation) only SETS
    ctx.pipeline_cancel_event; it never cancels the worker task. So the future
    stays live and the coroutine re-raises whatever the worker raised. A
    CancelledError there is a BaseException the parse worker's ``except
    (PipelineCancelledException, LLMBridgePipelineCancelled)`` / ``except
    Exception`` clauses both miss — the doc strands in PARSING and the worker
    task dies. This is the branch the shipped test does not cover: it uses
    ``task.cancel()``, where the future is already cancelled so awaiting it
    raises CancelledError regardless of the inner exception, hiding the type.

    Fix-proof: on the pre-fix code the worker raises asyncio.CancelledError and
    this ``pytest.raises(LLMBridgePipelineCancelled)`` fails behaviorally.
    """
    import asyncio
    import threading
    import time
    from unittest import mock

    from lightrag.constants import FULL_DOCS_FORMAT_PENDING_PARSE
    from lightrag.parser.base import ParseContext
    from lightrag.parser.debug import build_debug_rag
    from lightrag.parser.llm_bridge import LLMBridgePipelineCancelled
    from lightrag.parser.registry import get_parser

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    source_path = input_dir / "doc.docx"
    _benign(source_path)

    pipeline_cancel_event = threading.Event()
    entered = threading.Event()

    def _slow_validate(self, source, file_path):
        entered.set()
        # Wait until the coroutine has set the pipeline cancel event, so the
        # checkpoint below sees it — WITHOUT the task itself being cancelled.
        for _ in range(1000):
            if pipeline_cancel_event.is_set():
                return
            time.sleep(0.005)

    async def _run():
        with mock.patch.object(
            NativeDocxParser, "validate_source_blocking", _slow_validate
        ):
            ctx = ParseContext(
                build_debug_rag(),
                "doc-cancel-event",
                str(source_path),
                {"parse_format": FULL_DOCS_FORMAT_PENDING_PARSE, "content": ""},
            )
            ctx.pipeline_cancel_event = pipeline_cancel_event
            task = asyncio.create_task(get_parser("native").parse(ctx))
            await asyncio.to_thread(entered.wait, 5)
            pipeline_cancel_event.set()  # production cancel: set, do NOT cancel
            with pytest.raises(LLMBridgePipelineCancelled):
                await task

    asyncio.run(_run())


def test_parser_shutdown_stops_the_worker_before_cleanup(tmp_path, monkeypatch):
    """finalize_storages() must also stop a worker still inside validation.

    _shutdown_parser_executor() sets the shutdown event and then calls
    executor.shutdown(wait=False) — it does NOT wait for a running extract; its
    documented contract is that in-flight work exits via the event. A worker
    still inside validate_source_blocking would otherwise walk on to rmtree
    parsed_dir and extract into storages being torn down, destroying a prior
    complete sidecar exactly as the cancel case does.

    The event is captured once at parse start on purpose: the shutdown helper
    replaces the attribute with a fresh, unset Event after setting the old one,
    so re-reading it would observe the replacement and miss the shutdown.

    Fix-proof: pre-fix the checkpoint tested only the per-parse and pipeline
    events, so the worker proceeded into extract and no exception was raised.
    """
    import asyncio
    import threading
    import time
    from unittest import mock

    from lightrag.constants import FULL_DOCS_FORMAT_PENDING_PARSE
    from lightrag.parser.base import ParseContext
    from lightrag.parser.debug import build_debug_rag
    from lightrag.parser.llm_bridge import LLMBridgeShutdown
    from lightrag.parser.registry import get_parser

    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    source_path = input_dir / "doc.docx"
    _benign(source_path)

    # A first parse that succeeds and leaves a complete sidecar behind.
    result = _parse_via_real_pipeline(
        source_path, tmp_path, monkeypatch, "doc-shutdown"
    )
    parsed_dir = Path(result.blocks_path).parent
    survivors = sorted(p.name for p in parsed_dir.iterdir())
    assert "doc.blocks.jsonl" in survivors

    shutdown_event = threading.Event()
    entered = threading.Event()
    reached_extract = threading.Event()

    def _slow_validate(self, source, file_path):
        entered.set()
        for _ in range(1000):
            if shutdown_event.is_set():
                return
            time.sleep(0.005)

    def _spy_extract(self, source, **kwargs):
        reached_extract.set()
        return [], {}, {}

    async def _run():
        _benign(source_path)  # the first parse consumed the source
        with (
            mock.patch.object(
                NativeDocxParser, "validate_source_blocking", _slow_validate
            ),
            mock.patch.object(NativeDocxParser, "extract", _spy_extract),
        ):
            rag = build_debug_rag()
            rag._parser_shutdown_event = shutdown_event
            ctx = ParseContext(
                rag,
                "doc-shutdown",
                str(source_path),
                {"parse_format": FULL_DOCS_FORMAT_PENDING_PARSE, "content": ""},
            )
            task = asyncio.create_task(get_parser("native").parse(ctx))
            await asyncio.to_thread(entered.wait, 5)
            # What _shutdown_parser_executor does: set the live event, then
            # swap in a fresh one without waiting for the running thread.
            shutdown_event.set()
            rag._parser_shutdown_event = threading.Event()
            with pytest.raises(LLMBridgeShutdown):
                await task

    asyncio.run(_run())

    assert not reached_extract.is_set(), "worker continued into extraction"
    assert parsed_dir.is_dir(), "shutdown destroyed the previous sidecar"
    assert sorted(p.name for p in parsed_dir.iterdir()) == survivors
