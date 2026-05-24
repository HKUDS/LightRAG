"""Cache validation for ``*.mineru_raw/`` bundles.

Validation policy (settled in design discussion; see
``LightRAGSidecarFormat-zh.md`` related notes):

1. ``_manifest.json`` exists, parses, ``version=1.0`` ∧ ``engine=mineru``.
2. **Source size fast-path**: ``source_file.stat().st_size`` matches manifest;
   mismatch → miss without hashing.
3. **Source content_hash**: full sha256 of the current source file matches
   manifest. The size+hash pair is computed by a single-read helper so the
   stored manifest is internally self-consistent.
4. **API mode**: if the manifest recorded ``api_mode`` and it differs from
   current ``MINERU_API_MODE``, miss.
5. **Parser options**: the manifest must record an ``options_signature`` that
   matches the current effective MinerU request options. Missing signatures
   from older manifests are treated as stale.
6. **Engine version**: if ``MINERU_ENGINE_VERSION`` is set and the manifest
   recorded a non-empty one, they must match.
7. **Endpoint signature**: if the active MinerU endpoint is set and the
   manifest recorded a non-empty one, they must match.
8. **Critical file**: ``content_list.json`` must exist with matching size
   **and** sha256 — sha256 here is the final tie-breaker against silent
   corruption affecting the file the adapter depends on.
9. **Other files**: size-only verification (cheap; covers most corruption
   modes for image / middle.json / layout.pdf).

Any failed step ⇒ cache miss; the caller wipes the directory contents
(preserving the directory itself) and re-runs the download.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lightrag.constants import MINERU_RAW_DIR_SUFFIX, PARSED_DIR_SUFFIX
from lightrag.parser.external.mineru.manifest import load_manifest
from lightrag.utils import logger

DEFAULT_MINERU_API_MODE = "local"
DEFAULT_MINERU_OFFICIAL_ENDPOINT = "https://mineru.net"
DEFAULT_MINERU_MODEL_VERSION = "vlm"
DEFAULT_MINERU_LANGUAGE = "ch"
DEFAULT_MINERU_LOCAL_BACKEND = "hybrid-auto-engine"
DEFAULT_MINERU_LOCAL_PARSE_METHOD = "auto"
DEFAULT_MINERU_LOCAL_IMAGE_ANALYSIS = True
DEFAULT_MINERU_LOCAL_START_PAGE_ID = 0
DEFAULT_MINERU_LOCAL_END_PAGE_ID = 99999
DEFAULT_MINERU_ENABLE_TABLE = True
DEFAULT_MINERU_ENABLE_FORMULA = True
DEFAULT_MINERU_IS_OCR = False


def raw_dir_for_parsed_dir(parsed_dir: Path) -> Path:
    """Sibling raw dir for a given ``*.parsed`` dir.

    ``foo.parsed/`` → ``foo.mineru_raw/``. Used both at download time and at
    cache check time so the layout is canonical.
    """
    stem = parsed_dir.name
    if stem.endswith(PARSED_DIR_SUFFIX):
        stem = stem[: -len(PARSED_DIR_SUFFIX)]
    return parsed_dir.parent / f"{stem}{MINERU_RAW_DIR_SUFFIX}"


def clear_dir_contents(directory: Path) -> None:
    """Delete everything inside ``directory`` but keep ``directory`` itself."""
    if not directory.exists():
        return
    for entry in directory.iterdir():
        try:
            if entry.is_dir() and not entry.is_symlink():
                _rmtree_safe(entry)
            else:
                entry.unlink()
        except OSError:
            # Best-effort cleanup; subsequent download will overwrite.
            continue


def _rmtree_safe(directory: Path) -> None:
    import shutil

    shutil.rmtree(directory, ignore_errors=True)


def compute_size_and_hash(path: Path) -> tuple[int, str]:
    """Single-read computation of ``(size_bytes, "sha256:<hex>")``.

    Manifest writes use this so the recorded size and hash are guaranteed to
    describe the same byte stream; using two ``open()`` calls would risk a
    TOCTOU mismatch if the file changed in between.
    """
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return size, f"sha256:{h.hexdigest()}"


def _current_api_mode() -> str:
    mode = _normalize_api_mode(os.getenv("MINERU_API_MODE", DEFAULT_MINERU_API_MODE))
    return mode


def _normalize_api_mode(mode: str) -> str:
    mode = str(mode or "").strip().lower()
    return mode if mode in {"official", "local"} else DEFAULT_MINERU_API_MODE


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "[mineru_raw] %s=%r is not an integer; using %s", name, raw, default
        )
        return default


def _current_endpoint_signature() -> str:
    mode = _current_api_mode()
    if mode == "official":
        return (
            os.getenv("MINERU_OFFICIAL_ENDPOINT", DEFAULT_MINERU_OFFICIAL_ENDPOINT)
            .strip()
            .rstrip("/")
        )
    if mode == "local":
        return os.getenv("MINERU_LOCAL_ENDPOINT", "").strip().rstrip("/")
    return ""


def local_page_bounds(page_ranges: str) -> tuple[int, int]:
    raw = page_ranges.strip()
    if not raw:
        return DEFAULT_MINERU_LOCAL_START_PAGE_ID, DEFAULT_MINERU_LOCAL_END_PAGE_ID
    if "," in raw:
        raise ValueError(
            "MINERU_PAGE_RANGES with MINERU_API_MODE=local supports only a "
            "single page or simple range such as '1-10'"
        )
    if raw.isdigit():
        page = max(int(raw), 1)
        return page - 1, page - 1
    if "-" in raw:
        left, _, right = raw.partition("-")
        if left.isdigit() and right.isdigit():
            start = max(int(left), 1)
            end = max(int(right), start)
            return start - 1, end - 1
    raise ValueError(
        "MINERU_PAGE_RANGES with MINERU_API_MODE=local must be a single "
        "positive page number or simple range such as '1-10'"
    )


@dataclass(frozen=True)
class MinerUParserOptions:
    """Effective MinerU parser options used both for live requests and the
    cache signature.

    Constructed once via :meth:`from_env` so the client and the cache
    validator agree on every defaulting / normalization rule.
    """

    api_mode: str
    model_version: str
    language: str
    enable_table: bool
    enable_formula: bool
    is_ocr: bool
    page_ranges: str
    local_backend: str
    local_parse_method: str
    local_image_analysis: bool
    local_start_page_id: int
    local_end_page_id: int

    @classmethod
    def from_env(cls, *, api_mode: str | None = None) -> "MinerUParserOptions":
        mode = (
            _normalize_api_mode(api_mode)
            if api_mode is not None
            else _current_api_mode()
        )
        page_ranges = os.getenv("MINERU_PAGE_RANGES", "").strip()
        local_start = _env_int(
            "MINERU_LOCAL_START_PAGE_ID", DEFAULT_MINERU_LOCAL_START_PAGE_ID
        )
        local_end = _env_int(
            "MINERU_LOCAL_END_PAGE_ID", DEFAULT_MINERU_LOCAL_END_PAGE_ID
        )
        if mode == "local" and page_ranges:
            local_start, local_end = local_page_bounds(page_ranges)
        return cls(
            api_mode=mode,
            model_version=(
                os.getenv("MINERU_MODEL_VERSION", DEFAULT_MINERU_MODEL_VERSION).strip()
                or DEFAULT_MINERU_MODEL_VERSION
            ),
            language=(
                os.getenv("MINERU_LANGUAGE", DEFAULT_MINERU_LANGUAGE).strip()
                or DEFAULT_MINERU_LANGUAGE
            ),
            enable_table=_env_bool("MINERU_ENABLE_TABLE", DEFAULT_MINERU_ENABLE_TABLE),
            enable_formula=_env_bool(
                "MINERU_ENABLE_FORMULA", DEFAULT_MINERU_ENABLE_FORMULA
            ),
            is_ocr=_env_bool("MINERU_IS_OCR", DEFAULT_MINERU_IS_OCR),
            page_ranges=page_ranges,
            local_backend=(
                os.getenv("MINERU_LOCAL_BACKEND", DEFAULT_MINERU_LOCAL_BACKEND).strip()
                or DEFAULT_MINERU_LOCAL_BACKEND
            ),
            local_parse_method=(
                os.getenv(
                    "MINERU_LOCAL_PARSE_METHOD", DEFAULT_MINERU_LOCAL_PARSE_METHOD
                ).strip()
                or DEFAULT_MINERU_LOCAL_PARSE_METHOD
            ),
            local_image_analysis=_env_bool(
                "MINERU_LOCAL_IMAGE_ANALYSIS", DEFAULT_MINERU_LOCAL_IMAGE_ANALYSIS
            ),
            local_start_page_id=local_start,
            local_end_page_id=local_end,
        )

    def signature(self) -> str:
        return mineru_options_signature(**asdict(self))


def mineru_options_signature(
    *,
    api_mode: str,
    model_version: str = DEFAULT_MINERU_MODEL_VERSION,
    language: str = DEFAULT_MINERU_LANGUAGE,
    enable_table: bool = DEFAULT_MINERU_ENABLE_TABLE,
    enable_formula: bool = DEFAULT_MINERU_ENABLE_FORMULA,
    is_ocr: bool = DEFAULT_MINERU_IS_OCR,
    page_ranges: str = "",
    local_backend: str = DEFAULT_MINERU_LOCAL_BACKEND,
    local_parse_method: str = DEFAULT_MINERU_LOCAL_PARSE_METHOD,
    local_image_analysis: bool = DEFAULT_MINERU_LOCAL_IMAGE_ANALYSIS,
    local_start_page_id: int = DEFAULT_MINERU_LOCAL_START_PAGE_ID,
    local_end_page_id: int = DEFAULT_MINERU_LOCAL_END_PAGE_ID,
) -> str:
    mode = _normalize_api_mode(api_mode)
    payload: dict[str, Any] = {
        "signature_version": 1,
        "api_mode": mode,
        "language": str(language or "").strip() or DEFAULT_MINERU_LANGUAGE,
        "enable_table": bool(enable_table),
        "enable_formula": bool(enable_formula),
    }
    if mode == "official":
        payload.update(
            {
                "model_version": str(model_version or "").strip()
                or DEFAULT_MINERU_MODEL_VERSION,
                "is_ocr": bool(is_ocr),
                "page_ranges": str(page_ranges or "").strip(),
            }
        )
    else:
        payload.update(
            {
                "local_backend": str(local_backend or "").strip()
                or DEFAULT_MINERU_LOCAL_BACKEND,
                "local_parse_method": str(local_parse_method or "").strip()
                or DEFAULT_MINERU_LOCAL_PARSE_METHOD,
                "local_image_analysis": bool(local_image_analysis),
                "local_start_page_id": int(local_start_page_id),
                "local_end_page_id": int(local_end_page_id),
            }
        )

    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def current_mineru_options_signature() -> str:
    return MinerUParserOptions.from_env().signature()


def is_bundle_valid(raw_dir: Path, source_file: Path) -> bool:
    """Return True iff the bundle is intact and matches the current source.

    See module docstring for the full policy. Returns False on any of:
    missing manifest, malformed manifest, schema version mismatch, source
    size/hash mismatch, parser options mismatch, engine/endpoint env mismatch,
    critical file missing or corrupted, or any non-critical file size mismatch.
    """
    if not raw_dir.is_dir():
        return False

    manifest = load_manifest(raw_dir)
    if manifest is None:
        return False

    # 1. Source size fast-path
    try:
        cur_size = source_file.stat().st_size
    except OSError:
        return False
    if cur_size != int(manifest.source_size_bytes):
        return False

    # 2. Source content_hash
    _, cur_hash = compute_size_and_hash(source_file)
    if cur_hash != manifest.source_content_hash:
        return False

    # 3. API mode (only when manifest had one; old manifests remain compatible)
    cur_api_mode = _current_api_mode()
    if manifest.api_mode and cur_api_mode != manifest.api_mode:
        return False

    # 4. Parser options. Old manifests did not record this and must miss so
    # changes such as MINERU_LOCAL_BACKEND cannot silently reuse stale output.
    if not manifest.options_signature:
        return False
    if current_mineru_options_signature() != manifest.options_signature:
        return False

    # 5. Engine version (only when current env exposes one AND manifest had one)
    cur_engine_version = os.getenv("MINERU_ENGINE_VERSION", "").strip()
    if (
        cur_engine_version
        and manifest.engine_version
        and cur_engine_version != manifest.engine_version
    ):
        return False

    # 6. Endpoint signature
    cur_endpoint = _current_endpoint_signature()
    if (
        cur_endpoint
        and manifest.endpoint_signature
        and cur_endpoint != manifest.endpoint_signature
    ):
        return False

    # 7. Critical file: size + sha256
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

    # 8. Other files: size only
    for entry in manifest.files:
        ep = raw_dir / entry.path
        try:
            if ep.stat().st_size != int(entry.size):
                return False
        except OSError:
            return False

    return True


__all__ = [
    "MINERU_RAW_DIR_SUFFIX",
    "MinerUParserOptions",
    "clear_dir_contents",
    "compute_size_and_hash",
    "current_mineru_options_signature",
    "is_bundle_valid",
    "local_page_bounds",
    "mineru_options_signature",
    "raw_dir_for_parsed_dir",
]
