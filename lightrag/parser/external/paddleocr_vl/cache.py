"""Cache validation for ``*.paddleocr_vl_raw/`` bundles."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar, cast

from lightrag.parser.external._common import compute_size_and_hash
from lightrag.parser.external._manifest import load_manifest

MANIFEST_ENGINE = "paddleocr_vl"
CONTENT_LIST_FILENAME = "content_list.json"
VALID_PADDLEOCR_VL_API_MODES = {"official", "local"}
DEFAULT_PADDLEOCR_VL_API_MODE = "official"
DEFAULT_PADDLEOCR_VL_OFFICIAL_ENDPOINT = (
    "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
)
# PaddleOCR-VL-1.6 is the most recommended
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
DEFAULT_PADDLEOCR_VL_MAX_NEW_TOKENS = None
DEFAULT_PADDLEOCR_VL_SHOW_FORMULA_NUMBER = False
DEFAULT_PADDLEOCR_VL_RETURN_MARKDOWN_IMAGES = True
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
_UNSET = object()
_ValueT = TypeVar("_ValueT")
_LayoutThreshold = float | dict[int, float] | None
_LayoutUnclipRatio = (
    float | tuple[float, float] | dict[int, float | tuple[float, float]] | None
)
_LayoutMergeBboxesMode = str | dict[int, str] | None


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


def _coerce_value(
    value: Any,
    value_type: type[_ValueT],
    *,
    default: _ValueT | None = None,
) -> _ValueT | None:
    """Coerce a value to ``value_type``; return default if empty/unparseable."""
    if value is None:
        return default

    if value_type is bool:
        if isinstance(value, bool):
            return cast(_ValueT, value)
        low = str(value).strip().lower()
        if low in {"1", "true", "yes", "y", "on"}:
            return cast(_ValueT, True)
        if low in {"0", "false", "no", "n", "off"}:
            return cast(_ValueT, False)
        return default

    if value_type is int:
        if isinstance(value, bool):
            return default
        if isinstance(value, int):
            return cast(_ValueT, value)
        raw = str(value).strip()
        if not raw:
            return default
        try:
            return cast(_ValueT, int(raw))
        except ValueError:
            return default

    if value_type is float:
        if isinstance(value, bool):
            return default
        if isinstance(value, (int, float)):
            return cast(_ValueT, float(value))
        raw = str(value).strip()
        if not raw:
            return default
        try:
            return cast(_ValueT, float(raw))
        except ValueError:
            return default

    if value_type is str:
        raw = str(value).strip()
        return cast(_ValueT, raw) if raw else default

    if value_type is list:
        if isinstance(value, list):
            return cast(_ValueT, value)
        raw = str(value).strip()
        if not raw:
            return default
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return default
        return cast(_ValueT, parsed) if isinstance(parsed, list) else default

    if value_type is dict:
        if isinstance(value, dict):
            return cast(_ValueT, value)
        raw = str(value).strip()
        if not raw:
            return default
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return default
        return cast(_ValueT, parsed) if isinstance(parsed, dict) else default

    raise TypeError(f"Unsupported PaddleOCR-VL option type: {value_type!r}")


def _coerce_json_or_scalar(value: Any, *, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list, int, float, str)) and not isinstance(value, bool):
        if not isinstance(value, str):
            return value
        raw = value.strip()
        if not raw:
            return default
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return default


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _coerce_int_key_dict(
    value: Any,
    value_coercer: Callable[[Any], Any],
) -> dict[int, Any] | None:
    if not isinstance(value, dict):
        return None
    result: dict[int, Any] = {}
    for key, item in value.items():
        if isinstance(key, bool):
            return None
        try:
            category_id = int(key)
        except (TypeError, ValueError):
            return None
        coerced = value_coercer(item)
        if coerced is None:
            return None
        result[category_id] = coerced
    return result


def _coerce_unclip_value(value: Any) -> float | tuple[float, float] | None:
    scalar = _coerce_float(value)
    if scalar is not None:
        return scalar
    if isinstance(value, (list, tuple)) and len(value) == 2:
        first = _coerce_float(value[0])
        second = _coerce_float(value[1])
        if first is not None and second is not None:
            return (first, second)
    return None


def _coerce_threshold(value: Any) -> _LayoutThreshold:
    value = _coerce_json_or_scalar(value)
    scalar = _coerce_float(value)
    if scalar is not None:
        return scalar
    return _coerce_int_key_dict(value, _coerce_float)


def _coerce_unclip_ratio(value: Any) -> _LayoutUnclipRatio:
    value = _coerce_json_or_scalar(value)
    scalar_or_tuple = _coerce_unclip_value(value)
    if scalar_or_tuple is not None:
        return scalar_or_tuple
    return _coerce_int_key_dict(value, _coerce_unclip_value)


def _coerce_merge_bboxes_mode(value: Any) -> _LayoutMergeBboxesMode:
    value = _coerce_json_or_scalar(value)
    if isinstance(value, str):
        return value
    return _coerce_int_key_dict(value, _coerce_str)


def _coerce_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _override_env_custom(
    overrides: Mapping[str, Any],
    *keys: str,
    env_name: str,
    value_coercer: Callable[[Any], Any],
    default: Any = None,
) -> Any:
    value = _UNSET
    for key in keys:
        if key in overrides:
            value = overrides[key]
            break
    if value is _UNSET:
        value = os.getenv(env_name, _UNSET)
    if value is _UNSET:
        return default
    coerced = value_coercer(value)
    return coerced if coerced is not None else default


def _override_env(
    value_type: type[_ValueT],
    overrides: Mapping[str, Any],
    *keys: str,
    env_name: str,
    default: _ValueT | None = None,
) -> _ValueT | None:
    value = _UNSET
    for key in keys:
        if key in overrides:
            value = overrides[key]
            break
    if value is _UNSET:
        value = os.getenv(env_name, _UNSET)
    if value is _UNSET:
        return default
    return _coerce_value(value, value_type, default=default)


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
    layout_threshold: _LayoutThreshold
    layout_nms: bool | None
    layout_unclip_ratio: _LayoutUnclipRatio
    layout_merge_bboxes_mode: _LayoutMergeBboxesMode
    layout_shape_mode: str | None
    prompt_label: str | None
    format_block_content: bool | None
    repetition_penalty: float | None
    temperature: float | None
    top_p: float | None
    min_pixels: int | None
    max_pixels: int | None
    max_new_tokens: int | None
    merge_layout_blocks: bool | None
    markdown_ignore_labels: list[str] | None
    vlm_extra_args: dict[str, Any] | None
    prettify_markdown: bool | None
    show_formula_number: bool | None
    return_markdown_images: bool | None
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
            use_doc_orientation_classify=_override_env(
                bool,
                overrides,
                "use_doc_orientation_classify",
                env_name="PADDLEOCR_VL_USE_DOC_ORIENTATION_CLASSIFY",
                default=DEFAULT_PADDLEOCR_VL_USE_DOC_ORIENTATION_CLASSIFY,
            ),
            use_doc_unwarping=_override_env(
                bool,
                overrides,
                "use_doc_unwarping",
                env_name="PADDLEOCR_VL_USE_DOC_UNWARPING",
                default=DEFAULT_PADDLEOCR_VL_USE_DOC_UNWARPING,
            ),
            use_layout_detection=_override_env(
                bool,
                overrides,
                "use_layout_detection",
                env_name="PADDLEOCR_VL_USE_LAYOUT_DETECTION",
                default=DEFAULT_PADDLEOCR_VL_USE_LAYOUT_DETECTION,
            ),
            use_chart_recognition=_override_env(
                bool,
                overrides,
                "use_chart_recognition",
                env_name="PADDLEOCR_VL_USE_CHART_RECOGNITION",
                default=DEFAULT_PADDLEOCR_VL_USE_CHART_RECOGNITION,
            ),
            use_seal_recognition=_override_env(
                bool,
                overrides,
                "use_seal_recognition",
                env_name="PADDLEOCR_VL_USE_SEAL_RECOGNITION",
                default=DEFAULT_PADDLEOCR_VL_USE_SEAL_RECOGNITION,
            ),
            use_ocr_for_image_block=_override_env(
                bool,
                overrides,
                "use_ocr_for_image_block",
                env_name="PADDLEOCR_VL_USE_OCR_FOR_IMAGE_BLOCK",
                default=DEFAULT_PADDLEOCR_VL_USE_OCR_FOR_IMAGE_BLOCK,
            ),
            layout_threshold=_override_env_custom(
                overrides,
                "layout_threshold",
                env_name="PADDLEOCR_VL_LAYOUT_THRESHOLD",
                value_coercer=_coerce_threshold,
            ),
            layout_nms=_override_env(
                bool,
                overrides,
                "layout_nms",
                env_name="PADDLEOCR_VL_LAYOUT_NMS",
                default=DEFAULT_PADDLEOCR_VL_LAYOUT_NMS,
            ),
            layout_unclip_ratio=_override_env_custom(
                overrides,
                "layout_unclip_ratio",
                env_name="PADDLEOCR_VL_LAYOUT_UNCLIP_RATIO",
                value_coercer=_coerce_unclip_ratio,
            ),
            layout_merge_bboxes_mode=_override_env_custom(
                overrides,
                "layout_merge_bboxes_mode",
                env_name="PADDLEOCR_VL_LAYOUT_MERGE_BBOXES_MODE",
                value_coercer=_coerce_merge_bboxes_mode,
            ),
            layout_shape_mode=_override_env(
                str,
                overrides,
                "layout_shape_mode",
                env_name="PADDLEOCR_VL_LAYOUT_SHAPE_MODE",
                default=DEFAULT_PADDLEOCR_VL_LAYOUT_SHAPE_MODE,
            ),
            prompt_label=_override_env(
                str,
                overrides,
                "prompt_label",
                env_name="PADDLEOCR_VL_PROMPT_LABEL",
                default=DEFAULT_PADDLEOCR_VL_PROMPT_LABEL,
            ),
            format_block_content=_override_env(
                bool,
                overrides,
                "format_block_content",
                env_name="PADDLEOCR_VL_FORMAT_BLOCK_CONTENT",
                default=DEFAULT_PADDLEOCR_VL_FORMAT_BLOCK_CONTENT,
            ),
            repetition_penalty=_override_env(
                float,
                overrides,
                "repetition_penalty",
                env_name="PADDLEOCR_VL_REPETITION_PENALTY",
                default=DEFAULT_PADDLEOCR_VL_REPETITION_PENALTY,
            ),
            temperature=_override_env(
                float,
                overrides,
                "temperature",
                env_name="PADDLEOCR_VL_TEMPERATURE",
                default=DEFAULT_PADDLEOCR_VL_TEMPERATURE,
            ),
            top_p=_override_env(
                float,
                overrides,
                "top_p",
                env_name="PADDLEOCR_VL_TOP_P",
                default=DEFAULT_PADDLEOCR_VL_TOP_P,
            ),
            min_pixels=_override_env(
                int,
                overrides,
                "min_pixels",
                env_name="PADDLEOCR_VL_MIN_PIXELS",
                default=DEFAULT_PADDLEOCR_VL_MIN_PIXELS,
            ),
            max_pixels=_override_env(
                int,
                overrides,
                "max_pixels",
                env_name="PADDLEOCR_VL_MAX_PIXELS",
                default=DEFAULT_PADDLEOCR_VL_MAX_PIXELS,
            ),
            max_new_tokens=_override_env(
                int,
                overrides,
                "max_new_tokens",
                env_name="PADDLEOCR_VL_MAX_NEW_TOKENS",
                default=DEFAULT_PADDLEOCR_VL_MAX_NEW_TOKENS,
            ),
            merge_layout_blocks=_override_env(
                bool,
                overrides,
                "merge_layout_blocks",
                env_name="PADDLEOCR_VL_MERGE_LAYOUT_BLOCKS",
                default=DEFAULT_PADDLEOCR_VL_MERGE_LAYOUT_BLOCKS,
            ),
            markdown_ignore_labels=_override_env(
                list,
                overrides,
                "markdown_ignore_labels",
                env_name="PADDLEOCR_VL_MARKDOWN_IGNORE_LABELS",
                default=list(DEFAULT_PADDLEOCR_VL_MARKDOWN_IGNORE_LABELS),
            ),
            vlm_extra_args=_override_env(
                dict,
                overrides,
                "vlm_extra_args",
                env_name="PADDLEOCR_VL_VLM_EXTRA_ARGS",
            ),
            prettify_markdown=_override_env(
                bool,
                overrides,
                "prettify_markdown",
                env_name="PADDLEOCR_VL_PRETTIFY_MARKDOWN",
                default=DEFAULT_PADDLEOCR_VL_PRETTIFY_MARKDOWN,
            ),
            show_formula_number=_override_env(
                bool,
                overrides,
                "show_formula_number",
                env_name="PADDLEOCR_VL_SHOW_FORMULA_NUMBER",
                default=DEFAULT_PADDLEOCR_VL_SHOW_FORMULA_NUMBER,
            ),
            return_markdown_images=_override_env(
                bool,
                overrides,
                "return_markdown_images",
                env_name="PADDLEOCR_VL_RETURN_MARKDOWN_IMAGES",
                default=DEFAULT_PADDLEOCR_VL_RETURN_MARKDOWN_IMAGES,
            ),
            restructure_pages=_override_env(
                bool,
                overrides,
                "restructure_pages",
                env_name="PADDLEOCR_VL_RESTRUCTURE_PAGES",
                default=DEFAULT_PADDLEOCR_VL_RESTRUCTURE_PAGES,
            ),
            merge_tables=_override_env(
                bool,
                overrides,
                "merge_tables",
                env_name="PADDLEOCR_VL_MERGE_TABLES",
                default=DEFAULT_PADDLEOCR_VL_MERGE_TABLES,
            ),
            relevel_titles=_override_env(
                bool,
                overrides,
                "relevel_titles",
                env_name="PADDLEOCR_VL_RELEVEL_TITLES",
                default=DEFAULT_PADDLEOCR_VL_RELEVEL_TITLES,
            ),
            visualize=_override_env(
                bool,
                overrides,
                "visualize",
                env_name="PADDLEOCR_VL_VISUALIZE",
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
        page_ranges = _override_env(
            str,
            overrides,
            "page_ranges",
            "page_range",
            env_name="PADDLEOCR_VL_PAGE_RANGES",
        )
        if mode == "local" and page_ranges:
            raise ValueError(
                "PaddleOCR-VL 'page_range' only applies to "
                "PADDLEOCR_VL_API_MODE=official; local deployments do not support "
                "pageRanges"
            )

        return cls(
            api_mode=mode,
            model=str(
                _override_env(
                    str,
                    overrides,
                    "model",
                    env_name="PADDLEOCR_VL_MODEL",
                    default=DEFAULT_PADDLEOCR_VL_MODEL,
                )
            ),
            page_ranges=page_ranges,
            batch_id=_override_env(
                str,
                overrides,
                "batch_id",
                env_name="PADDLEOCR_VL_BATCH_ID",
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


def current_paddleocr_vl_options_signature(
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
        cur_size = source_file.stat().st_size
    except OSError:
        return False
    if cur_size != int(manifest.source_size_bytes):
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

    # ``current_paddleocr_vl_options_signature`` always produces a non-empty
    # value, so a non-empty manifest signature is compared against the current
    # signature (computed with the same overrides); a mismatch invalidates.
    if manifest.options_signature and (
        current_paddleocr_vl_options_signature(overrides) != manifest.options_signature
    ):
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
    "current_endpoint_signature",
    "current_paddleocr_vl_options_signature",
    "current_engine_version",
    "DEFAULT_PADDLEOCR_VL_API_MODE",
    "DEFAULT_PADDLEOCR_VL_ENGINE_VERSION",
    "DEFAULT_PADDLEOCR_VL_OFFICIAL_ENDPOINT",
    "MANIFEST_ENGINE",
    "VALID_PADDLEOCR_VL_API_MODES",
    "PaddleOCRVLParserOptions",
    "is_bundle_valid",
]
