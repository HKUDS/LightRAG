"""Cache validation for ``*.paddleocr_vl_raw/`` bundles."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from lightrag.parser.external._common import compute_size_and_hash
from lightrag.parser.external._manifest import load_manifest

MANIFEST_ENGINE = "paddleocr_vl"
CONTENT_LIST_FILENAME = "content_list.json"
VALID_PADDLEOCR_VL_API_MODES = {"official", "local"}
DEFAULT_PADDLEOCR_VL_API_MODE = "official"
DEFAULT_PADDLEOCR_VL_OFFICIAL_ENDPOINT = (
    "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
)
DEFAULT_PADDLEOCR_VL_MODEL = "PaddleOCR-VL-1.6"
DEFAULT_PADDLEOCR_VL_ENGINE_VERSION = "PaddleOCR-VL-1.6"

# Parser option defaults
DEFAULT_PADDLEOCR_VL_USE_DOC_ORIENTATION_CLASSIFY = False
DEFAULT_PADDLEOCR_VL_USE_DOC_UNWARPING = False
DEFAULT_PADDLEOCR_VL_USE_LAYOUT_DETECTION = True
DEFAULT_PADDLEOCR_VL_USE_CHART_RECOGNITION = True
DEFAULT_PADDLEOCR_VL_USE_SEAL_RECOGNITION = True
DEFAULT_PADDLEOCR_VL_VISUALIZE = False
DEFAULT_PADDLEOCR_VL_USE_OCR_FOR_IMAGE_BLOCK = False
DEFAULT_PADDLEOCR_VL_MERGE_TABLES = True
DEFAULT_PADDLEOCR_VL_MERGE_LAYOUT_BLOCKS = True
DEFAULT_PADDLEOCR_VL_RELEVEL_TITLES = True
DEFAULT_PADDLEOCR_VL_PRETTIFY_MARKDOWN = True
DEFAULT_PADDLEOCR_VL_LAYOUT_SHAPE_MODE = "auto"
DEFAULT_PADDLEOCR_VL_PROMPT_LABEL = "ocr"
DEFAULT_PADDLEOCR_VL_FORMAT_BLOCK_CONTENT = False
DEFAULT_PADDLEOCR_VL_REPETITION_PENALTY = 1
DEFAULT_PADDLEOCR_VL_TEMPERATURE = 0
DEFAULT_PADDLEOCR_VL_TOP_P = 1
DEFAULT_PADDLEOCR_VL_MIN_PIXELS = 147384
DEFAULT_PADDLEOCR_VL_MAX_PIXELS = 2822400
DEFAULT_PADDLEOCR_VL_SHOW_FORMULA_NUMBER = False
DEFAULT_PADDLEOCR_VL_LAYOUT_NMS = True
DEFAULT_PADDLEOCR_VL_RESTRUCTURE_PAGES = True
DEFAULT_PADDLEOCR_VL_MARKDOWN_IGNORE_LABELS: tuple[str, ...] = (
    "header",
    "header_image",
    "footer",
    "footer_image",
    "number",
    "footnote",
    "aside_text",
)


def _current_api_mode() -> str:
    mode = str(os.getenv("PADDLEOCR_VL_API_MODE")).strip().lower()
    return (
        mode if mode in VALID_PADDLEOCR_VL_API_MODES else DEFAULT_PADDLEOCR_VL_API_MODE
    )


def current_endpoint_signature() -> str:
    mode = _current_api_mode()
    endpoint = ""
    if mode == "official":
        endpoint = os.getenv(
            "PADDLEOCR_VL_OFFICIAL_ENDPOINT",
            os.getenv("PADDLEOCR_VL_ENDPOINT", DEFAULT_PADDLEOCR_VL_OFFICIAL_ENDPOINT),
        )
    if mode == "local":
        endpoint = os.getenv("PADDLEOCR_VL_LOCAL_ENDPOINT", "")

    return endpoint.strip().rstrip("/")


def _coerce_bool(value: Any, *, default: bool | None = None) -> bool | None:
    """Coerce a value to bool; return default if empty/unparseable."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    low = str(value).strip().lower()
    if low in {"1", "true", "yes", "y", "on"}:
        return True
    if low in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _coerce_number(
    value: Any, *, default: int | float | None = None
) -> int | float | None:
    """Coerce a value to int/float; return default if empty/unparseable."""
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        number = float(value)
        return int(number) if number.is_integer() else number
    raw = str(value).strip()
    if not raw:
        return default
    try:
        number = float(raw)
    except ValueError:
        return default
    return int(number) if number.is_integer() else number


def _coerce_string(value: Any, *, default: str | None = None) -> str | None:
    """Coerce a value to stripped string; return default if empty."""
    if value is None:
        return default
    raw = str(value).strip()
    return raw or default


def _coerce_list(value: Any, *, default: list[Any] | None = None) -> list[Any] | None:
    """Coerce a value to list; return default if empty/invalid."""
    if value is None:
        return default
    if isinstance(value, list):
        return value
    raw = str(value).strip()
    if not raw:
        return default
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else default
    except json.JSONDecodeError:
        return default


def _coerce_dict(
    value: Any, *, default: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Coerce a value to dict; return default if empty/invalid."""
    if value is None:
        return default
    if isinstance(value, dict):
        return value
    raw = str(value).strip()
    if not raw:
        return default
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else default
    except json.JSONDecodeError:
        return default


def _snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    if not name:
        return name
    components = name.split("_")
    if len(components) == 1:
        return name
    return components[0] + "".join(
        component.capitalize() for component in components[1:]
    )


def _build_payload(options: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field in fields(options):
        if field.name == "api_mode":
            continue
        value = getattr(options, field.name)
        if value is None:
            continue
        if hasattr(value, "request_payload"):
            value = value.request_payload()
        payload[_snake_to_camel(field.name)] = value
    return payload


@dataclass(frozen=True)
class DocParsingOptions:
    """Document parsing options for PaddleOCR-VL in `optionalPayload`."""

    use_doc_orientation_classify: bool | None
    use_doc_unwarping: bool | None
    use_layout_detection: bool | None
    use_chart_recognition: bool | None
    use_seal_recognition: bool | None
    use_ocr_for_image_block: bool | None
    layout_threshold: float | None
    layout_nms: bool | None
    layout_unclip_ratio: list[float] | None
    layout_merge_bboxes_mode: str | None
    layout_shape_mode: str | None
    prompt_label: str | None
    format_block_content: bool | None
    repetition_penalty: int | float | None
    temperature: int | float | None
    top_p: int | float | None
    min_pixels: int | float | None
    max_pixels: int | float | None
    merge_layout_blocks: bool | None
    markdown_ignore_labels: list[str] | None
    vlm_extra_args: dict[str, Any] | None
    prettify_markdown: bool | None
    show_formula_number: bool | None
    restructure_pages: bool | None
    merge_tables: bool | None
    relevel_titles: bool | None
    visualize: bool | None

    @classmethod
    def from_env(
        cls,
        *,
        overrides: "Mapping[str, Any] | None" = None,
    ) -> "DocParsingOptions":
        overrides = overrides or {}
        return cls(
            use_doc_orientation_classify=_coerce_bool(
                overrides.get(
                    "use_doc_orientation_classify",
                    os.getenv(
                        "PADDLEOCR_VL_USE_DOC_ORIENTATION_CLASSIFY",
                    ),
                ),
                default=DEFAULT_PADDLEOCR_VL_USE_DOC_ORIENTATION_CLASSIFY,
            ),
            use_doc_unwarping=_coerce_bool(
                overrides.get(
                    "use_doc_unwarping",
                    os.getenv(
                        "PADDLEOCR_VL_USE_DOC_UNWARPING",
                    ),
                ),
                default=DEFAULT_PADDLEOCR_VL_USE_DOC_UNWARPING,
            ),
            use_layout_detection=_coerce_bool(
                overrides.get(
                    "use_layout_detection",
                    os.getenv("PADDLEOCR_VL_USE_LAYOUT_DETECTION"),
                ),
                default=DEFAULT_PADDLEOCR_VL_USE_LAYOUT_DETECTION,
            ),
            use_chart_recognition=_coerce_bool(
                overrides.get(
                    "use_chart_recognition",
                    os.getenv("PADDLEOCR_VL_USE_CHART_RECOGNITION"),
                ),
                default=DEFAULT_PADDLEOCR_VL_USE_CHART_RECOGNITION,
            ),
            use_seal_recognition=_coerce_bool(
                overrides.get(
                    "use_seal_recognition",
                    os.getenv("PADDLEOCR_VL_USE_SEAL_RECOGNITION"),
                ),
                default=DEFAULT_PADDLEOCR_VL_USE_SEAL_RECOGNITION,
            ),
            use_ocr_for_image_block=_coerce_bool(
                overrides.get(
                    "use_ocr_for_image_block",
                    os.getenv("PADDLEOCR_VL_USE_OCR_FOR_IMAGE_BLOCK"),
                ),
                default=DEFAULT_PADDLEOCR_VL_USE_OCR_FOR_IMAGE_BLOCK,
            ),
            layout_threshold=_coerce_number(
                overrides.get(
                    "layout_threshold", os.getenv("PADDLEOCR_VL_LAYOUT_THRESHOLD")
                )
            ),
            layout_nms=_coerce_bool(
                overrides.get("layout_nms", os.getenv("PADDLEOCR_VL_LAYOUT_NMS")),
                default=DEFAULT_PADDLEOCR_VL_LAYOUT_NMS,
            ),
            layout_unclip_ratio=_coerce_list(
                overrides.get(
                    "layout_unclip_ratio",
                    os.getenv("PADDLEOCR_VL_LAYOUT_UNCLIP_RATIO"),
                )
            ),
            layout_merge_bboxes_mode=_coerce_string(
                overrides.get(
                    "layout_merge_bboxes_mode",
                    os.getenv("PADDLEOCR_VL_LAYOUT_MERGE_BBOXES_MODE"),
                )
            ),  # large(default), small, union
            layout_shape_mode=_coerce_string(
                overrides.get(
                    "layout_shape_mode",
                    os.getenv("PADDLEOCR_VL_LAYOUT_SHAPE_MODE"),
                ),
                default=DEFAULT_PADDLEOCR_VL_LAYOUT_SHAPE_MODE,
            ),  # rect, quad, poly, auto(default)
            prompt_label=_coerce_string(
                overrides.get("prompt_label", os.getenv("PADDLEOCR_VL_PROMPT_LABEL")),
                default=DEFAULT_PADDLEOCR_VL_PROMPT_LABEL,
            ),  # ocr(default), formula, table, chart
            format_block_content=_coerce_bool(
                overrides.get(
                    "format_block_content",
                    os.getenv("PADDLEOCR_VL_FORMAT_BLOCK_CONTENT"),
                ),
                default=DEFAULT_PADDLEOCR_VL_FORMAT_BLOCK_CONTENT,
            ),
            repetition_penalty=_coerce_number(
                overrides.get(
                    "repetition_penalty",
                    os.getenv("PADDLEOCR_VL_REPETITION_PENALTY"),
                ),
                default=DEFAULT_PADDLEOCR_VL_REPETITION_PENALTY,
            ),
            temperature=_coerce_number(
                overrides.get("temperature", os.getenv("PADDLEOCR_VL_TEMPERATURE")),
                default=DEFAULT_PADDLEOCR_VL_TEMPERATURE,
            ),
            top_p=_coerce_number(
                overrides.get("top_p", os.getenv("PADDLEOCR_VL_TOP_P")),
                default=DEFAULT_PADDLEOCR_VL_TOP_P,
            ),
            min_pixels=_coerce_number(
                overrides.get("min_pixels", os.getenv("PADDLEOCR_VL_MIN_PIXELS")),
                default=DEFAULT_PADDLEOCR_VL_MIN_PIXELS,
            ),
            max_pixels=_coerce_number(
                overrides.get("max_pixels", os.getenv("PADDLEOCR_VL_MAX_PIXELS")),
                default=DEFAULT_PADDLEOCR_VL_MAX_PIXELS,
            ),
            merge_layout_blocks=_coerce_bool(
                overrides.get(
                    "merge_layout_blocks",
                    os.getenv("PADDLEOCR_VL_MERGE_LAYOUT_BLOCKS"),
                ),
                default=DEFAULT_PADDLEOCR_VL_MERGE_LAYOUT_BLOCKS,
            ),
            markdown_ignore_labels=_coerce_list(
                overrides.get(
                    "markdown_ignore_labels",
                    os.getenv("PADDLEOCR_VL_MARKDOWN_IGNORE_LABELS"),
                ),
                default=list(DEFAULT_PADDLEOCR_VL_MARKDOWN_IGNORE_LABELS),
            ),
            vlm_extra_args=_coerce_dict(overrides.get("vlm_extra_args")),
            prettify_markdown=_coerce_bool(
                overrides.get(
                    "prettify_markdown",
                    os.getenv("PADDLEOCR_VL_PRETTIFY_MARKDOWN"),
                ),
                default=DEFAULT_PADDLEOCR_VL_PRETTIFY_MARKDOWN,
            ),
            show_formula_number=_coerce_bool(
                overrides.get(
                    "show_formula_number",
                    os.getenv("PADDLEOCR_VL_SHOW_FORMULA_NUMBER"),
                ),
                default=DEFAULT_PADDLEOCR_VL_SHOW_FORMULA_NUMBER,
            ),
            restructure_pages=_coerce_bool(
                overrides.get(
                    "restructure_pages",
                    os.getenv("PADDLEOCR_VL_RESTRUCTURE_PAGES"),
                ),
                default=DEFAULT_PADDLEOCR_VL_RESTRUCTURE_PAGES,
            ),
            merge_tables=_coerce_bool(
                overrides.get("merge_tables", os.getenv("PADDLEOCR_VL_MERGE_TABLES")),
                default=DEFAULT_PADDLEOCR_VL_MERGE_TABLES,
            ),
            relevel_titles=_coerce_bool(
                overrides.get(
                    "relevel_titles", os.getenv("PADDLEOCR_VL_RELEVEL_TITLES")
                ),
                default=DEFAULT_PADDLEOCR_VL_RELEVEL_TITLES,
            ),
            visualize=_coerce_bool(
                overrides.get("visualize", os.getenv("PADDLEOCR_VL_VISUALIZE")),
                default=DEFAULT_PADDLEOCR_VL_VISUALIZE,
            ),
        )

    def request_payload(self) -> dict[str, Any]:
        return _build_payload(self)


@dataclass(frozen=True)
class PaddleOCRVLParserOptions:
    """Effective PaddleOCR-VL request options.

    This mirrors MinerU's parser-options object: the live client uses it to
    build the service request, and the cache validator uses the same object to
    compute the options signature.
    """

    api_mode: str
    model: str
    page_ranges: str | None
    batch_id: str | None
    optional_payload: DocParsingOptions

    @classmethod
    def from_env(
        cls,
        *,
        api_mode: str | None = None,
        overrides: "Mapping[str, Any] | None" = None,
    ) -> "PaddleOCRVLParserOptions":
        mode = api_mode if api_mode is not None else _current_api_mode()
        overrides = overrides or {}
        optional_payload = DocParsingOptions.from_env(overrides=overrides)

        return cls(
            api_mode=mode,
            model=str(
                overrides.get(
                    "model", os.getenv("PADDLEOCR_VL_MODEL", DEFAULT_PADDLEOCR_VL_MODEL)
                )
            ),
            page_ranges=_coerce_string(
                overrides.get(
                    "page_ranges",
                    overrides.get("page_range", os.getenv("PADDLEOCR_VL_PAGE_RANGES")),
                )  # page_range → page_ranges alias (like MinerU)
            ),
            batch_id=_coerce_string(
                overrides.get("batch_id", os.getenv("PADDLEOCR_VL_BATCH_ID"))
            ),
            optional_payload=optional_payload,
        )

    def request_payload(self) -> dict[str, Any]:
        return _build_payload(self)

    def signature(self) -> str:
        payload = asdict(self)
        payload["signature_version"] = 1
        raw = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def snapshot_tunable_env(
    overrides: "Mapping[str, Any] | None" = None,
) -> dict[str, str]:
    """Return effective request options that change parser output bytes."""
    return {
        key: json.dumps(value, ensure_ascii=False, sort_keys=True)
        for key, value in asdict(
            PaddleOCRVLParserOptions.from_env(overrides=overrides)
        ).items()
    }


def current_options_signature(
    overrides: "Mapping[str, Any] | None" = None,
) -> str:
    return PaddleOCRVLParserOptions.from_env(overrides=overrides).signature()


def current_engine_version() -> str:
    return (
        os.getenv(
            "PADDLEOCR_VL_ENGINE_VERSION", DEFAULT_PADDLEOCR_VL_ENGINE_VERSION
        ).strip()
        or DEFAULT_PADDLEOCR_VL_ENGINE_VERSION
    )


def is_bundle_valid(
    raw_dir: Path,
    source_file: Path,
    *,
    overrides: "Mapping[str, Any] | None" = None,
) -> bool:
    """Return True iff the raw bundle matches the current source and options."""
    if not raw_dir.is_dir():
        return False

    manifest = load_manifest(raw_dir, expected_engine=MANIFEST_ENGINE)
    if manifest is None:
        return False

    try:
        if source_file.stat().st_size != int(manifest.source_size_bytes):
            return False
    except OSError:
        return False
    _, cur_hash = compute_size_and_hash(source_file)
    if cur_hash != manifest.source_content_hash:
        return False

    cur_mode = _current_api_mode()
    if manifest.api_mode and manifest.api_mode != cur_mode:
        return False

    cur_endpoint = current_endpoint_signature()
    if (
        cur_endpoint
        and manifest.endpoint_signature
        and cur_endpoint != manifest.endpoint_signature
    ):
        return False

    if not manifest.options_signature:
        return False
    if current_options_signature(overrides) != manifest.options_signature:
        return False

    cur_version = os.getenv("PADDLEOCR_VL_ENGINE_VERSION", "").strip()
    if (
        cur_version
        and manifest.engine_version
        and cur_version != manifest.engine_version
    ):
        return False

    crit = manifest.critical_file
    crit_path = raw_dir / crit.path
    try:
        if crit_path.stat().st_size != int(crit.size):
            return False
    except OSError:
        return False
    if crit.sha256:
        _, crit_actual = compute_size_and_hash(crit_path)
        if crit_actual != crit.sha256:
            return False

    for entry in manifest.files:
        path = raw_dir / entry.path
        try:
            if path.stat().st_size != int(entry.size):
                return False
        except OSError:
            return False

    return True


__all__ = [
    "CONTENT_LIST_FILENAME",
    "DEFAULT_PADDLEOCR_VL_API_MODE",
    "DEFAULT_PADDLEOCR_VL_ENGINE_VERSION",
    "DEFAULT_PADDLEOCR_VL_LAYOUT_NMS",
    "DEFAULT_PADDLEOCR_VL_LAYOUT_SHAPE_MODE",
    "DEFAULT_PADDLEOCR_VL_MARKDOWN_IGNORE_LABELS",
    "DEFAULT_PADDLEOCR_VL_MAX_PIXELS",
    "DEFAULT_PADDLEOCR_VL_MERGE_TABLES",
    "DEFAULT_PADDLEOCR_VL_MIN_PIXELS",
    "DEFAULT_PADDLEOCR_VL_MODEL",
    "DEFAULT_PADDLEOCR_VL_OFFICIAL_ENDPOINT",
    "DEFAULT_PADDLEOCR_VL_PROMPT_LABEL",
    "DEFAULT_PADDLEOCR_VL_RELEVEL_TITLES",
    "DEFAULT_PADDLEOCR_VL_REPETITION_PENALTY",
    "DEFAULT_PADDLEOCR_VL_RESTRUCTURE_PAGES",
    "DEFAULT_PADDLEOCR_VL_TEMPERATURE",
    "DEFAULT_PADDLEOCR_VL_TOP_P",
    "DEFAULT_PADDLEOCR_VL_USE_CHART_RECOGNITION",
    "DEFAULT_PADDLEOCR_VL_USE_DOC_ORIENTATION_CLASSIFY",
    "DEFAULT_PADDLEOCR_VL_USE_DOC_UNWARPING",
    "DEFAULT_PADDLEOCR_VL_USE_LAYOUT_DETECTION",
    "DEFAULT_PADDLEOCR_VL_USE_OCR_FOR_IMAGE_BLOCK",
    "DEFAULT_PADDLEOCR_VL_USE_SEAL_RECOGNITION",
    "MANIFEST_ENGINE",
    "PaddleOCRVLParserOptions",
    "VALID_PADDLEOCR_VL_API_MODES",
    "current_endpoint_signature",
    "current_engine_version",
    "current_options_signature",
    "is_bundle_valid",
    "snapshot_tunable_env",
]
