"""Cache validation tests for ``*.paddleocr_vl_raw`` bundles."""

from __future__ import annotations

from pathlib import Path

import pytest

from lightrag.parser.external import Manifest, ManifestFile, write_manifest
from lightrag.parser.external._common import compute_size_and_hash
import lightrag.parser.external.paddleocr_vl.cache as cache_mod
from lightrag.parser.external.paddleocr_vl.cache import (
    PaddleOCRVLParserOptions,
    is_bundle_valid,
)

# Access internal test helpers via module object (not in __all__)
current_endpoint_signature = cache_mod.current_endpoint_signature
current_options_signature = cache_mod.current_options_signature
from lightrag.parser.external.paddleocr_vl import (
    PADDLEOCR_VL_RAW_DIR_SUFFIX,
    clear_dir_contents,
    raw_dir_for_parsed_dir,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "PADDLEOCR_VL_API_MODE",
        "PADDLEOCR_VL_ENDPOINT",
        "PADDLEOCR_VL_OFFICIAL_ENDPOINT",
        "PADDLEOCR_VL_LOCAL_ENDPOINT",
        "PADDLEOCR_VL_MODEL",
        "PADDLEOCR_VL_PAGE_RANGES",
        "PADDLEOCR_VL_BATCH_ID",
        "PADDLEOCR_VL_USE_DOC_ORIENTATION_CLASSIFY",
        "PADDLEOCR_VL_USE_DOC_UNWARPING",
        "PADDLEOCR_VL_USE_LAYOUT_DETECTION",
        "PADDLEOCR_VL_USE_CHART_RECOGNITION",
        "PADDLEOCR_VL_LAYOUT_THRESHOLD",
        "PADDLEOCR_VL_LAYOUT_NMS",
        "PADDLEOCR_VL_LAYOUT_UNCLIP_RATIO",
        "PADDLEOCR_VL_LAYOUT_MERGE_BBOXES_MODE",
        "PADDLEOCR_VL_LAYOUT_SHAPE_MODE",
        "PADDLEOCR_VL_PROMPT_LABEL",
        "PADDLEOCR_VL_REPETITION_PENALTY",
        "PADDLEOCR_VL_TEMPERATURE",
        "PADDLEOCR_VL_TOP_P",
        "PADDLEOCR_VL_MIN_PIXELS",
        "PADDLEOCR_VL_MAX_PIXELS",
        "PADDLEOCR_VL_SHOW_FORMULA_NUMBER",
        "PADDLEOCR_VL_RESTRUCTURE_PAGES",
        "PADDLEOCR_VL_MERGE_TABLES",
        "PADDLEOCR_VL_RELEVEL_TITLES",
        "PADDLEOCR_VL_PRETTIFY_MARKDOWN",
        "PADDLEOCR_VL_VISUALIZE",
        "PADDLEOCR_VL_ENGINE_VERSION",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PADDLEOCR_VL_ENDPOINT", "http://paddle.test/api/v2/ocr/jobs")


def _bundle(tmp_path: Path, source: Path, *, options_signature: str | None = None) -> Path:
    raw_dir = tmp_path / "src.paddleocr_vl_raw"
    raw_dir.mkdir()
    result = raw_dir / "result.json"
    result.write_text('[{"ok": true}]', encoding="utf-8")
    asset = raw_dir / "imgs" / "fig.jpg"
    asset.parent.mkdir()
    asset.write_bytes(b"fakejpg")

    source_size, source_hash = compute_size_and_hash(source)
    result_size, result_hash = compute_size_and_hash(result)
    manifest = Manifest(
        engine="paddleocr_vl",
        source_content_hash=source_hash,
        source_size_bytes=source_size,
        source_filename_at_parse=source.name,
        critical_file=ManifestFile("result.json", result_size, result_hash),
        files=[ManifestFile("imgs/fig.jpg", asset.stat().st_size)],
        total_size_bytes=result_size + asset.stat().st_size,
        task_id="job-1",
        endpoint_signature=current_endpoint_signature(),
        engine_version="",
        options_signature=(
            options_signature
            if options_signature is not None
            else current_options_signature()
        ),
    )
    write_manifest(raw_dir, manifest)
    return raw_dir


@pytest.fixture
def source_file(tmp_path: Path) -> Path:
    source = tmp_path / "src.pdf"
    source.write_bytes(b"%PDF fake payload" * 32)
    return source


def test_valid_bundle_hits_cache(tmp_path: Path, source_file: Path) -> None:
    raw = _bundle(tmp_path, source_file)
    assert is_bundle_valid(raw, source_file) is True


def test_parser_options_include_official_documented_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PADDLEOCR_VL_PAGE_RANGES", "2,4-6")
    monkeypatch.setenv("PADDLEOCR_VL_BATCH_ID", "batch-1")
    monkeypatch.setenv("PADDLEOCR_VL_USE_DOC_ORIENTATION_CLASSIFY", "true")
    monkeypatch.setenv("PADDLEOCR_VL_USE_DOC_UNWARPING", "true")
    monkeypatch.setenv("PADDLEOCR_VL_USE_LAYOUT_DETECTION", "false")
    monkeypatch.setenv("PADDLEOCR_VL_USE_CHART_RECOGNITION", "true")
    monkeypatch.setenv("PADDLEOCR_VL_LAYOUT_THRESHOLD", "0.42")
    monkeypatch.setenv("PADDLEOCR_VL_LAYOUT_NMS", "true")
    monkeypatch.setenv("PADDLEOCR_VL_LAYOUT_UNCLIP_RATIO", "[1.0, 1.2]")
    monkeypatch.setenv("PADDLEOCR_VL_LAYOUT_MERGE_BBOXES_MODE", "small")
    monkeypatch.setenv("PADDLEOCR_VL_LAYOUT_SHAPE_MODE", "quad")
    monkeypatch.setenv("PADDLEOCR_VL_PROMPT_LABEL", "table")
    monkeypatch.setenv("PADDLEOCR_VL_REPETITION_PENALTY", "1.1")
    monkeypatch.setenv("PADDLEOCR_VL_TEMPERATURE", "0.2")
    monkeypatch.setenv("PADDLEOCR_VL_TOP_P", "0.9")
    monkeypatch.setenv("PADDLEOCR_VL_MIN_PIXELS", "1024")
    monkeypatch.setenv("PADDLEOCR_VL_MAX_PIXELS", "4096")
    monkeypatch.setenv("PADDLEOCR_VL_SHOW_FORMULA_NUMBER", "true")
    monkeypatch.setenv("PADDLEOCR_VL_RESTRUCTURE_PAGES", "true")
    monkeypatch.setenv("PADDLEOCR_VL_MERGE_TABLES", "false")
    monkeypatch.setenv("PADDLEOCR_VL_RELEVEL_TITLES", "false")
    monkeypatch.setenv("PADDLEOCR_VL_PRETTIFY_MARKDOWN", "true")
    monkeypatch.setenv("PADDLEOCR_VL_VISUALIZE", "false")

    options = PaddleOCRVLParserOptions.from_env()
    payload = options.request_payload()

    assert options.model == "PaddleOCR-VL-1.6"
    assert options.page_ranges == "2,4-6"
    assert options.batch_id == "batch-1"
    assert options.optional_payload.use_chart_recognition is True
    assert not hasattr(options, "doc_parsing_options")
    assert payload["model"] == "PaddleOCR-VL-1.6"
    assert payload["pageRanges"] == "2,4-6"
    assert payload["batchId"] == "batch-1"
    assert "apiMode" not in payload
    payload = payload["optionalPayload"]
    # Environment-set values
    assert payload["useDocOrientationClassify"] is True
    assert payload["useDocUnwarping"] is True
    assert payload["useLayoutDetection"] is False
    assert payload["useChartRecognition"] is True
    assert payload["layoutThreshold"] == 0.42
    assert payload["layoutNms"] is True
    assert payload["layoutUnclipRatio"] == [1.0, 1.2]
    assert payload["layoutMergeBboxesMode"] == "small"
    assert payload["layoutShapeMode"] == "quad"
    assert payload["promptLabel"] == "table"
    assert payload["repetitionPenalty"] == 1.1
    assert payload["temperature"] == 0.2
    assert payload["topP"] == 0.9
    assert payload["minPixels"] == 1024
    assert payload["maxPixels"] == 4096
    assert payload["showFormulaNumber"] is True
    assert payload["restructurePages"] is True
    assert payload["mergeTables"] is False
    assert payload["relevelTitles"] is False
    assert payload["prettifyMarkdown"] is True
    assert payload["visualize"] is False
    # Default values also present
    assert payload["useSealRecognition"] is True
    assert payload["useOcrForImageBlock"] is False
    assert payload["markdownIgnoreLabels"] == [
        "header",
        "header_image",
        "footer",
        "footer_image",
        "number",
        "footnote",
        "aside_text",
    ]


def test_parser_options_accept_per_file_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PADDLEOCR_VL_USE_CHART_RECOGNITION", "false")

    options = PaddleOCRVLParserOptions.from_env(
        overrides={
            "page_range": "1-3",
            "batch_id": "batch-file",
            "use_chart_recognition": True,
            "layout_threshold": 0.7,
            "layout_unclip_ratio": [1.0, 1.2],
            "prettify_markdown": True,
        }
    )

    assert options.page_ranges == "1-3"
    assert options.batch_id == "batch-file"
    payload = options.request_payload()
    assert payload["model"] == "PaddleOCR-VL-1.6"
    assert payload["pageRanges"] == "1-3"
    assert payload["batchId"] == "batch-file"
    assert "apiMode" not in payload
    payload = payload["optionalPayload"]
    # Overridden values
    assert payload["useChartRecognition"] is True
    assert payload["layoutThreshold"] == 0.7
    assert payload["layoutUnclipRatio"] == [1.0, 1.2]
    assert payload["prettifyMarkdown"] is True
    # Default values are also present
    assert payload["useDocOrientationClassify"] is False
    assert payload["useDocUnwarping"] is False
    assert payload["useLayoutDetection"] is True
    assert payload["useSealRecognition"] is True
    assert payload["useOcrForImageBlock"] is False
    assert payload["mergeTables"] is True
    assert payload["relevelTitles"] is True
    assert payload["layoutShapeMode"] == "auto"
    assert payload["promptLabel"] == "ocr"
    assert payload["repetitionPenalty"] == 1
    assert payload["temperature"] == 0
    assert payload["topP"] == 1
    assert payload["minPixels"] == 147384
    assert payload["maxPixels"] == 2822400
    assert payload["layoutNms"] is True
    assert payload["restructurePages"] is True
    assert payload["markdownIgnoreLabels"] == [
        "header",
        "header_image",
        "footer",
        "footer_image",
        "number",
        "footnote",
        "aside_text",
    ]


def test_parser_options_reject_none_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PADDLEOCR_VL_USE_CHART_RECOGNITION", "false")

    with pytest.raises(
        ValueError,
        match="PaddleOCR-VL option use_chart_recognition cannot be None",
    ):
        PaddleOCRVLParserOptions.from_env(
            overrides={"use_chart_recognition": None}
        )


def test_parser_options_reject_none_page_range_alias() -> None:
    with pytest.raises(
        ValueError,
        match="PaddleOCR-VL option page_range cannot be None",
    ):
        PaddleOCRVLParserOptions.from_env(overrides={"page_range": None})


def test_parser_option_coercion_uses_target_type_parameter() -> None:
    assert cache_mod._coerce_value("true", bool, default=False) is True
    assert cache_mod._coerce_value("7", int) == 7
    assert cache_mod._coerce_value("0.7", float) == 0.7
    assert cache_mod._coerce_value("  ocr  ", str) == "ocr"
    assert cache_mod._coerce_value('["header"]', list) == ["header"]
    assert cache_mod._coerce_value('{"temperature": 0.2}', dict) == {
        "temperature": 0.2
    }


def test_per_file_overrides_participate_in_cache_signature(
    tmp_path: Path, source_file: Path
) -> None:
    # Use an option that defaults to False, so setting to True changes the signature
    overrides = {"use_doc_orientation_classify": True}
    raw = _bundle(
        tmp_path,
        source_file,
        options_signature=PaddleOCRVLParserOptions.from_env(
            overrides=overrides
        ).signature(),
    )

    assert is_bundle_valid(raw, source_file, overrides=overrides) is True
    assert is_bundle_valid(raw, source_file) is False


def test_cache_keeps_mode_specific_endpoint_helpers_private() -> None:
    assert "current_official_endpoint_signature" not in cache_mod.__all__
    assert "current_local_endpoint_signature" not in cache_mod.__all__


def test_facade_exposes_raw_dir_helpers(tmp_path: Path) -> None:
    parsed_dir = tmp_path / "demo.pdf.parsed"
    raw_dir = raw_dir_for_parsed_dir(parsed_dir)
    assert raw_dir == tmp_path / f"demo.pdf{PADDLEOCR_VL_RAW_DIR_SUFFIX}"

    nested = raw_dir / "imgs"
    nested.mkdir(parents=True)
    (nested / "fig.jpg").write_bytes(b"fake")
    clear_dir_contents(raw_dir)
    assert list(raw_dir.iterdir()) == []


def test_endpoint_change_invalidates_cache(
    tmp_path: Path, source_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _bundle(tmp_path, source_file)
    monkeypatch.setenv("PADDLEOCR_VL_ENDPOINT", "http://other/api/v2/ocr/jobs")
    assert is_bundle_valid(raw, source_file) is False


def test_api_mode_change_invalidates_cache(
    tmp_path: Path, source_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _bundle(tmp_path, source_file)
    monkeypatch.setenv("PADDLEOCR_VL_API_MODE", "local")
    monkeypatch.setenv("PADDLEOCR_VL_LOCAL_ENDPOINT", "http://local-paddle")
    assert is_bundle_valid(raw, source_file) is False


def test_option_change_invalidates_cache(
    tmp_path: Path, source_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _bundle(tmp_path, source_file)
    # Use an option where the default value is False, so setting to True changes it
    monkeypatch.setenv("PADDLEOCR_VL_USE_DOC_ORIENTATION_CLASSIFY", "true")
    assert is_bundle_valid(raw, source_file) is False


def test_result_corruption_invalidates_cache(
    tmp_path: Path, source_file: Path
) -> None:
    raw = _bundle(tmp_path, source_file)
    result = raw / "result.json"
    result.write_bytes(b"X" * result.stat().st_size)
    assert is_bundle_valid(raw, source_file) is False
