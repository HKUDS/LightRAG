"""Unit tests for drawing placeholder emission in ``drawing_image_extractor``.

Focus: the ``path``/``src`` contract. ``path`` is local-only (a file the
extractor materialized under ``<base>.blocks.assets/``); linked/external
image references travel through ``src`` and never through ``path``.
"""

from __future__ import annotations

import tempfile
import zipfile
from errno import EMFILE, ENOSPC
from pathlib import Path
from unittest import mock
from xml.etree import ElementTree as ET

import pytest

from lightrag.parser.docx.drawing_image_extractor import (
    IMAGE_REL_TYPE,
    DrawingExtractionContext,
    DrawingRelationship,
    extract_drawing_placeholder_from_element,
    extract_vml_image_placeholder_from_element,
    parse_drawing_attributes,
)

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_WP = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
_A = "http://schemas.openxmlformats.org/drawingml/2006/main"
_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_V = "urn:schemas-microsoft-com:vml"


def _drawing_element(blip_attr: str, rel_id: str) -> ET.Element:
    return ET.fromstring(
        f'<w:drawing xmlns:w="{_W}" xmlns:wp="{_WP}" xmlns:a="{_A}" xmlns:r="{_R}">'
        "<wp:inline>"
        '<wp:docPr id="7" name="fig"/>'
        f'<a:blip r:{blip_attr}="{rel_id}"/>'
        "</wp:inline>"
        "</w:drawing>"
    )


def _pict_element(rel_id: str) -> ET.Element:
    return ET.fromstring(
        f'<w:pict xmlns:w="{_W}" xmlns:v="{_V}" xmlns:r="{_R}">'
        '<v:shape id="s1" alt="vml fig">'
        f'<v:imagedata r:id="{rel_id}"/>'
        "</v:shape>"
        "</w:pict>"
    )


def _context_with_relationship(rel: DrawingRelationship) -> DrawingExtractionContext:
    ctx = DrawingExtractionContext(docx_path=Path("unused.docx"))
    ctx.relationships[rel.rel_id] = rel
    return ctx


def _external_rel(target: str, image_format: str) -> DrawingRelationship:
    return DrawingRelationship(
        rel_id="rId9",
        target=target,
        target_mode="External",
        rel_type=IMAGE_REL_TYPE,
        image_format=image_format,
    )


def _embedded_rel(rel_id: str, filename: str) -> DrawingRelationship:
    return DrawingRelationship(
        rel_id=rel_id,
        target=f"media/{filename}",
        target_mode="",
        rel_type=IMAGE_REL_TYPE,
        part_name=f"/word/media/{filename}",
        content_type="image/png",
        image_format="png",
    )


@pytest.mark.offline
@pytest.mark.parametrize(
    ("target", "image_format"),
    [
        ("https://example.com/diagrams/architecture.png", "png"),
        ("../images/legacy.gif", "gif"),
    ],
)
def test_linked_blip_target_lands_in_src_not_path(
    target: str, image_format: str
) -> None:
    ctx = _context_with_relationship(_external_rel(target, image_format))
    placeholder = extract_drawing_placeholder_from_element(
        _drawing_element("link", "rId9"), ctx
    )

    attrs = parse_drawing_attributes(placeholder)
    assert attrs["src"] == target
    assert attrs["format"] == image_format
    assert "path" not in attrs


@pytest.mark.offline
def test_vml_external_imagedata_target_lands_in_src_not_path() -> None:
    ctx = _context_with_relationship(
        _external_rel("https://example.com/shape.png", "png")
    )
    placeholder = extract_vml_image_placeholder_from_element(_pict_element("rId9"), ctx)

    attrs = parse_drawing_attributes(placeholder)
    assert attrs["src"] == "https://example.com/shape.png"
    assert attrs["format"] == "png"
    assert "path" not in attrs


@pytest.mark.offline
def test_embedded_blip_streams_to_local_path_in_bounded_reads(
    tmp_path: Path, monkeypatch
) -> None:
    docx_path = tmp_path / "doc.docx"
    with zipfile.ZipFile(docx_path, "w") as zf:
        zf.writestr("word/media/image1.png", b"PNGDATA")

    read_sizes = []
    original_read = zipfile.ZipExtFile.read

    def _record_read_size(self, size=-1):
        read_sizes.append(size)
        return original_read(self, size)

    monkeypatch.setattr(zipfile.ZipExtFile, "read", _record_read_size)

    export_dir = tmp_path / "doc.blocks.assets"
    export_dir.mkdir()
    ctx = DrawingExtractionContext(
        docx_path=docx_path,
        export_dir_name="doc.blocks.assets",
        export_dir_path=export_dir,
    )
    ctx.relationships["rId1"] = _embedded_rel("rId1", "image1.png")

    placeholder = extract_drawing_placeholder_from_element(
        _drawing_element("embed", "rId1"), ctx
    )

    attrs = parse_drawing_attributes(placeholder)
    assert attrs["path"] == "doc.blocks.assets/image1.png"
    assert attrs["format"] == "png"
    assert "src" not in attrs
    assert (export_dir / "image1.png").read_bytes() == b"PNGDATA"
    assert read_sizes
    assert all(0 < size <= 1024 * 1024 for size in read_sizes)


@pytest.mark.offline
def test_single_image_budget_skips_before_read_and_records_warning(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("NATIVE_DOCX_IMAGE_MAX_BYTES", "8")
    monkeypatch.setenv("NATIVE_DOCX_IMAGE_MAX_TOTAL_BYTES", "64")
    docx_path = tmp_path / "doc.docx"
    with zipfile.ZipFile(docx_path, "w") as zf:
        zf.writestr("word/media/image.png", b"123456789")

    def _explode(*args, **kwargs):  # pragma: no cover - only on regression
        raise AssertionError("oversized image was opened")

    monkeypatch.setattr(zipfile.ZipFile, "open", _explode)
    warnings = {}
    export_dir = tmp_path / "assets"
    export_dir.mkdir()
    ctx = DrawingExtractionContext(
        docx_path=docx_path,
        export_dir_name="assets",
        export_dir_path=export_dir,
        parse_warnings=warnings,
    )
    rel = _embedded_rel("rId1", "image.png")
    ctx.relationships["rId1"] = rel

    with mock.patch(
        "lightrag.parser.docx.drawing_image_extractor.logger.warning"
    ) as log_warning:
        placeholder = extract_drawing_placeholder_from_element(
            _drawing_element("embed", "rId1"), ctx
        )
        for _ in range(499):
            assert ctx.export_embedded_image(rel) is None

    attrs = parse_drawing_attributes(placeholder)
    assert "path" not in attrs
    assert attrs["format"] == "png"
    assert warnings == {
        "docx_image_budget_skipped_count": 1,
        "docx_image_budget_skipped_bytes": 9,
    }
    log_warning.assert_called_once()
    assert "Skipping over-budget DOCX image" in log_warning.call_args.args[0]
    assert log_warning.call_args.args[1] == "/word/media/image.png"
    assert list(export_dir.iterdir()) == []


@pytest.mark.offline
def test_total_image_budget_counts_successes_once_and_skips_the_rest(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("NATIVE_DOCX_IMAGE_MAX_BYTES", "8")
    monkeypatch.setenv("NATIVE_DOCX_IMAGE_MAX_TOTAL_BYTES", "10")
    docx_path = tmp_path / "doc.docx"
    with zipfile.ZipFile(docx_path, "w") as zf:
        zf.writestr("word/media/first.png", b"123456")
        zf.writestr("word/media/second.png", b"abcdef")

    warnings = {}
    export_dir = tmp_path / "assets"
    export_dir.mkdir()
    ctx = DrawingExtractionContext(
        docx_path=docx_path,
        export_dir_name="assets",
        export_dir_path=export_dir,
        parse_warnings=warnings,
    )
    first = _embedded_rel("rId1", "first.png")
    second = _embedded_rel("rId2", "second.png")

    first_path = ctx.export_embedded_image(first)
    assert first_path == "assets/first.png"
    assert ctx.export_embedded_image(first) == first_path
    assert ctx.export_embedded_image(second) is None

    assert (export_dir / "first.png").read_bytes() == b"123456"
    assert not (export_dir / "second.png").exists()
    assert warnings == {
        "docx_image_budget_skipped_count": 1,
        "docx_image_budget_skipped_bytes": 6,
    }


@pytest.mark.offline
def test_exact_image_and_total_budget_boundary_is_allowed(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("NATIVE_DOCX_IMAGE_MAX_BYTES", "8")
    monkeypatch.setenv("NATIVE_DOCX_IMAGE_MAX_TOTAL_BYTES", "8")
    docx_path = tmp_path / "doc.docx"
    with zipfile.ZipFile(docx_path, "w") as zf:
        zf.writestr("word/media/image.png", b"12345678")

    export_dir = tmp_path / "assets"
    export_dir.mkdir()
    ctx = DrawingExtractionContext(
        docx_path=docx_path,
        export_dir_name="assets",
        export_dir_path=export_dir,
    )

    assert ctx.export_embedded_image(_embedded_rel("rId1", "image.png"))
    assert (export_dir / "image.png").read_bytes() == b"12345678"


@pytest.mark.offline
def test_temporary_file_cannot_delete_an_existing_dot_part_asset(
    tmp_path: Path,
) -> None:
    docx_path = tmp_path / "doc.docx"
    with zipfile.ZipFile(docx_path, "w") as zf:
        zf.writestr("word/media/.foo.part", b"FIRST")
        zf.writestr("word/media/foo", b"SECOND")

    export_dir = tmp_path / "assets"
    export_dir.mkdir()
    ctx = DrawingExtractionContext(
        docx_path=docx_path,
        export_dir_name="assets",
        export_dir_path=export_dir,
    )

    first = _embedded_rel("rId1", ".foo.part")
    second = _embedded_rel("rId2", "foo")
    assert ctx.export_embedded_image(first) == "assets/.foo.part"
    assert ctx.export_embedded_image(second) == "assets/foo"

    assert (export_dir / ".foo.part").read_bytes() == b"FIRST"
    assert (export_dir / "foo").read_bytes() == b"SECOND"
    assert sorted(path.name for path in export_dir.iterdir()) == [".foo.part", "foo"]


@pytest.mark.offline
def test_transient_stream_failure_retries_and_removes_partial_file(
    tmp_path: Path, monkeypatch
) -> None:
    data = b"A" * (1024 * 1024 + 1)
    docx_path = tmp_path / "doc.docx"
    with zipfile.ZipFile(docx_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("word/media/image.png", data)

    original_read = zipfile.ZipExtFile.read
    calls = 0

    def _fail_second_read(self, size=-1):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated stream failure")
        return original_read(self, size)

    monkeypatch.setattr(zipfile.ZipExtFile, "read", _fail_second_read)
    export_dir = tmp_path / "assets"
    export_dir.mkdir()
    ctx = DrawingExtractionContext(
        docx_path=docx_path,
        export_dir_name="assets",
        export_dir_path=export_dir,
        parse_warnings={},
    )
    rel = _embedded_rel("rId1", "image.png")

    with mock.patch(
        "lightrag.parser.docx.drawing_image_extractor.logger.warning"
    ) as log_warning:
        assert ctx.export_embedded_image(rel) is None
        assert calls == 2
        assert list(export_dir.iterdir()) == []

        assert ctx.export_embedded_image(rel) == "assets/image.png"

    assert (export_dir / "image.png").read_bytes() == data
    assert rel.part_name not in ctx._skipped_parts
    assert ctx.parse_warnings == {"docx_image_read_failed_count": 1}
    log_warning.assert_called_once()
    assert log_warning.call_args.args[4] == "later references will retry"


@pytest.mark.offline
def test_transient_zip_open_failure_retries_and_records_warning(
    tmp_path: Path, monkeypatch
) -> None:
    docx_path = tmp_path / "doc.docx"
    with zipfile.ZipFile(docx_path, "w") as zf:
        zf.writestr("word/media/image.png", b"PNGDATA")

    original_open = zipfile.ZipFile.open
    open_calls = 0

    def _fail_first_open(self, *args, **kwargs):
        nonlocal open_calls
        open_calls += 1
        if open_calls == 1:
            raise OSError(EMFILE, "simulated descriptor exhaustion")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", _fail_first_open)
    export_dir = tmp_path / "assets"
    export_dir.mkdir()
    ctx = DrawingExtractionContext(
        docx_path=docx_path,
        export_dir_name="assets",
        export_dir_path=export_dir,
        parse_warnings={},
    )
    rel = _embedded_rel("rId1", "image.png")

    with mock.patch(
        "lightrag.parser.docx.drawing_image_extractor.logger.warning"
    ) as log_warning:
        assert ctx.export_embedded_image(rel) is None
        assert ctx.export_embedded_image(rel) == "assets/image.png"

    assert open_calls == 2
    assert (export_dir / "image.png").read_bytes() == b"PNGDATA"
    assert rel.part_name not in ctx._skipped_parts
    assert ctx.parse_warnings == {"docx_image_read_failed_count": 1}
    log_warning.assert_called_once()
    assert "OSError" in log_warning.call_args.args[3]
    assert log_warning.call_args.args[4] == "later references will retry"


@pytest.mark.offline
def test_permanent_zip_failure_is_reported_once_and_memoized(
    tmp_path: Path, monkeypatch
) -> None:
    docx_path = tmp_path / "doc.docx"
    with zipfile.ZipFile(docx_path, "w") as zf:
        zf.writestr("word/media/image.png", b"PNGDATA")

    open_calls = 0

    def _fail_open(*args, **kwargs):
        nonlocal open_calls
        open_calls += 1
        raise zipfile.BadZipFile("simulated corrupt member")

    monkeypatch.setattr(zipfile.ZipFile, "open", _fail_open)
    export_dir = tmp_path / "assets"
    export_dir.mkdir()
    ctx = DrawingExtractionContext(
        docx_path=docx_path,
        export_dir_name="assets",
        export_dir_path=export_dir,
        parse_warnings={},
    )
    rel = _embedded_rel("rId1", "image.png")

    with mock.patch(
        "lightrag.parser.docx.drawing_image_extractor.logger.warning"
    ) as log_warning:
        assert ctx.export_embedded_image(rel) is None
        assert ctx.export_embedded_image(rel) is None

    assert open_calls == 1
    assert rel.part_name in ctx._skipped_parts
    assert ctx.parse_warnings == {"docx_image_read_failed_count": 1}
    log_warning.assert_called_once()
    assert log_warning.call_args.args[4] == (
        "the damaged or missing part will be skipped"
    )


@pytest.mark.offline
def test_output_write_failure_propagates_and_removes_partial_file(
    tmp_path: Path, monkeypatch
) -> None:
    docx_path = tmp_path / "doc.docx"
    with zipfile.ZipFile(docx_path, "w") as zf:
        zf.writestr("word/media/image.png", b"PNGDATA")

    export_dir = tmp_path / "assets"
    export_dir.mkdir()
    ctx = DrawingExtractionContext(
        docx_path=docx_path,
        export_dir_name="assets",
        export_dir_path=export_dir,
        parse_warnings={},
    )
    original_named_temporary_file = tempfile.NamedTemporaryFile

    class _FailingWriter:
        def __init__(self, output):
            self._output = output
            self.name = output.name

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self._output.__exit__(*args)

        def write(self, data: bytes):
            self._output.write(data[:1])
            raise OSError(ENOSPC, "simulated full disk")

    def _fail_partial_write(*args, **kwargs):
        return _FailingWriter(original_named_temporary_file(*args, **kwargs))

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", _fail_partial_write)

    with pytest.raises(OSError) as exc:
        ctx.export_embedded_image(_embedded_rel("rId1", "image.png"))

    assert exc.value.errno == ENOSPC
    assert list(export_dir.iterdir()) == []
    assert ctx.parse_warnings == {}
    assert ctx._used_filenames == {}
    assert ctx._skipped_parts == set()


@pytest.mark.offline
def test_failed_source_read_does_not_reserve_filename(tmp_path: Path, monkeypatch):
    docx_path = tmp_path / "doc.docx"
    with zipfile.ZipFile(docx_path, "w") as zf:
        zf.writestr("word/media/image.png", b"BROKEN")
        zf.writestr("word/alternate/image.png", b"GOOD")

    original_read = zipfile.ZipExtFile.read
    fail_once = True

    def _fail_first_read(self, size=-1):
        nonlocal fail_once
        if fail_once:
            fail_once = False
            raise OSError("simulated source failure")
        return original_read(self, size)

    monkeypatch.setattr(zipfile.ZipExtFile, "read", _fail_first_read)
    export_dir = tmp_path / "assets"
    export_dir.mkdir()
    ctx = DrawingExtractionContext(
        docx_path=docx_path,
        export_dir_name="assets",
        export_dir_path=export_dir,
    )
    failed = _embedded_rel("rId1", "image.png")
    successful = DrawingRelationship(
        rel_id="rId2",
        target="alternate/image.png",
        target_mode="",
        rel_type=IMAGE_REL_TYPE,
        part_name="/word/alternate/image.png",
        content_type="image/png",
        image_format="png",
    )

    assert ctx.export_embedded_image(failed) is None
    assert failed.part_name not in ctx._skipped_parts
    assert ctx.export_embedded_image(successful) == "assets/image.png"
    assert (export_dir / "image.png").read_bytes() == b"GOOD"
    assert not (export_dir / "image_2.png").exists()


@pytest.mark.offline
def test_non_positive_image_env_values_restore_safe_defaults(
    tmp_path: Path, monkeypatch
) -> None:
    from lightrag.constants import (
        DEFAULT_NATIVE_DOCX_IMAGE_MAX_BYTES,
        DEFAULT_NATIVE_DOCX_IMAGE_MAX_TOTAL_BYTES,
    )

    monkeypatch.setenv("NATIVE_DOCX_IMAGE_MAX_BYTES", "0")
    monkeypatch.setenv("NATIVE_DOCX_IMAGE_MAX_TOTAL_BYTES", "-1")
    ctx = DrawingExtractionContext(docx_path=tmp_path / "doc.docx")

    assert ctx._max_image_bytes == DEFAULT_NATIVE_DOCX_IMAGE_MAX_BYTES
    assert ctx._max_total_image_bytes == DEFAULT_NATIVE_DOCX_IMAGE_MAX_TOTAL_BYTES


@pytest.mark.offline
def test_native_parser_threads_image_budget_warnings(
    tmp_path: Path, monkeypatch
) -> None:
    from lightrag.parser.docx.parser import NativeDocxParser

    monkeypatch.setenv("NATIVE_DOCX_IMAGE_MAX_BYTES", "4")
    monkeypatch.setenv("NATIVE_DOCX_IMAGE_MAX_TOTAL_BYTES", "8")
    source = tmp_path / "doc.docx"
    with zipfile.ZipFile(source, "w") as zf:
        zf.writestr("word/media/image.png", b"12345")
    parsed_dir = tmp_path / "parsed"
    asset_dir = parsed_dir / "doc.blocks.assets"
    asset_dir.mkdir(parents=True)

    def _extract_blocks(
        file_path,
        drawing_context=None,
        parse_warnings=None,
        parse_metadata=None,
        **kwargs,
    ):
        assert drawing_context.parse_warnings is parse_warnings
        assert (
            drawing_context.export_embedded_image(_embedded_rel("rId1", "image.png"))
            is None
        )
        return [{"content": "body"}]

    with (
        mock.patch("lightrag.parser.docx.drawing_image_extractor.load_relationships"),
        mock.patch(
            "lightrag.parser.docx.parse_document.extract_docx_blocks",
            _extract_blocks,
        ),
    ):
        blocks, warnings, metadata = NativeDocxParser().extract(
            source,
            parsed_dir=parsed_dir,
            asset_dir=asset_dir,
            base_name="doc",
        )

    assert blocks == [{"content": "body"}]
    assert warnings == {
        "docx_image_budget_skipped_count": 1,
        "docx_image_budget_skipped_bytes": 5,
    }
    assert metadata == {}
