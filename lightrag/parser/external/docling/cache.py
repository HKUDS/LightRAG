"""Cache validation for ``*.docling_raw/`` bundles.

Validation policy (settled in
``docs/DoclingSidecarRefactorPlan-zh.md`` §4.1):

1. ``_manifest.json`` exists, parses, ``engine="docling"`` ∧ schema version
   matches.
2. **Source size fast-path**: ``source_file.stat().st_size`` matches the
   manifest; mismatch → miss without hashing.
3. **Source content_hash**: full sha256 of the current source file matches
   the manifest.
4. **Engine version**: if ``DOCLING_ENGINE_VERSION`` is set in env and the
   manifest recorded a non-empty value, they must match.
5. **Endpoint signature**: if the active ``DOCLING_ENDPOINT`` differs from
   what was recorded at parse time, miss (avoids re-using a bundle produced
   by a different docling-serve instance).
6. **Options signature**: covers every env or fixed constant that changes
   the produced bundle (OCR flags, language list, formula enrichment,
   target format and pipeline). Any change → miss.
7. **Critical file**: the main JSON must exist with matching size **and**
   sha256 — final tie-breaker against silent corruption affecting the file
   the adapter depends on.
8. **Other files**: size-only verification (cheap; covers most corruption
   modes for markdown / artifacts).

Any failed step ⇒ cache miss; the caller wipes the directory contents and
re-runs the download.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lightrag.parser.external._common import compute_size_and_hash, env_bool
from lightrag.parser.external._manifest import load_manifest
from lightrag.parser.external.docling import MANIFEST_ENGINE
from lightrag.utils import logger

# Legacy upload-path suffix. ``env.example`` historically documented
# ``DOCLING_ENDPOINT=http://host:5001/v1/convert/file/async`` (the full
# upload URL); the current client expects a base URL and appends the path
# itself. Strip the suffix so an unmodified pre-refactor ``.env`` keeps
# working instead of producing
# ``/v1/convert/file/async/v1/convert/file/async`` requests.
_LEGACY_UPLOAD_PATH_SUFFIX = "/v1/convert/file/async"
_legacy_endpoint_warned = False

# Envs that change the bytes docling-serve produces. Any change here must
# invalidate the bundle cache. ``DOCLING_BBOX_ATTRIBUTES`` is intentionally
# NOT in this list: it only affects how the adapter writes IR meta, not the
# docling bundle, so flipping it should re-emit the sidecar (which we always
# do) without forcing a re-download.
DOCLING_TUNABLE_ENVS: tuple[str, ...] = (
    "DOCLING_DO_OCR",
    "DOCLING_FORCE_OCR",
    "DOCLING_OCR_ENGINE",
    "DOCLING_OCR_PRESET",
    "DOCLING_OCR_LANG",
    "DOCLING_DO_FORMULA_ENRICHMENT",
)


def current_endpoint_signature() -> str:
    """The active docling endpoint, normalized to a base URL.

    Normalization:

    - Trims surrounding whitespace and strips trailing slashes.
    - Strips the legacy ``/v1/convert/file/async`` upload suffix if present,
      preserving backwards compatibility with the pre-refactor ``env.example``
      that documented the full upload URL.

    Returns ``""`` if ``DOCLING_ENDPOINT`` is unset — callers that need a
    real endpoint (``DoclingRawClient``) raise on empty; callers that only
    compare against a recorded manifest field (``is_bundle_valid``) silently
    skip the check when either side is empty.
    """
    global _legacy_endpoint_warned
    endpoint = os.getenv("DOCLING_ENDPOINT", "").strip().rstrip("/")
    if endpoint.endswith(_LEGACY_UPLOAD_PATH_SUFFIX):
        endpoint = endpoint[: -len(_LEGACY_UPLOAD_PATH_SUFFIX)]
        if not _legacy_endpoint_warned:
            _legacy_endpoint_warned = True
            logger.warning(
                "DOCLING_ENDPOINT still includes the legacy %r upload suffix; "
                "stripping it. Update your .env to a base URL "
                "(e.g. http://host:5001).",
                _LEGACY_UPLOAD_PATH_SUFFIX,
            )
    return endpoint


def compute_options_signature(
    *,
    tunable_env: dict[str, str],
    fixed_constants: dict[str, object],
) -> str:
    """Stable signature over user-tunable env values and fixed pipeline
    constants.

    Storing the constants in the signature means a future code change that
    flips e.g. ``image_export_mode`` from ``referenced`` to ``embedded``
    invalidates every existing cache without anyone having to remember to
    bump a version.
    """
    payload = json.dumps(
        {"env": tunable_env, "fixed": fixed_constants},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def snapshot_tunable_env(
    overrides: "Mapping[str, Any] | None" = None,
) -> dict[str, str]:
    """Read effective docling tunables so equivalent requests share a signature.

    ``overrides`` carries decoded per-file engine params (Phase 2: ``force_ocr``)
    and replaces the corresponding env value so the override feeds BOTH the live
    request and the cache signature.
    """
    overrides = overrides or {}
    force_ocr = (
        bool(overrides["force_ocr"])
        if "force_ocr" in overrides
        else env_bool("DOCLING_FORCE_OCR", True)
    )
    return {
        "DOCLING_DO_OCR": str(env_bool("DOCLING_DO_OCR", True)).lower(),
        "DOCLING_FORCE_OCR": str(force_ocr).lower(),
        "DOCLING_OCR_ENGINE": os.getenv("DOCLING_OCR_ENGINE", "auto").strip() or "auto",
        "DOCLING_OCR_PRESET": os.getenv("DOCLING_OCR_PRESET", "auto").strip() or "auto",
        "DOCLING_OCR_LANG": os.getenv("DOCLING_OCR_LANG", "").strip(),
        "DOCLING_DO_FORMULA_ENRICHMENT": str(
            env_bool("DOCLING_DO_FORMULA_ENRICHMENT", False)
        ).lower(),
    }


def is_bundle_valid(
    raw_dir: Path,
    source_file: Path,
    *,
    overrides: "Mapping[str, Any] | None" = None,
) -> bool:
    """Return True iff the bundle matches the current source + env state."""
    if not raw_dir.is_dir():
        return False

    manifest = load_manifest(raw_dir, expected_engine=MANIFEST_ENGINE)
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

    # 3. Engine version. Skip the comparison when either side is empty so
    #    operators can opt out by unsetting the env, and so bundles from
    #    earlier code that never recorded the field aren't force-invalidated.
    cur_engine_version = os.getenv("DOCLING_ENGINE_VERSION", "").strip()
    if (
        cur_engine_version
        and manifest.engine_version
        and cur_engine_version != manifest.engine_version
    ):
        return False

    # 4. Endpoint signature. Same "both non-empty to compare" rule: a bundle
    #    parsed against a different docling-serve URL must not be reused, but
    #    we don't reject the cache just because the env happens to be unset
    #    at validation time (e.g. CLI tooling that only reads the cache).
    cur_endpoint = current_endpoint_signature()
    if (
        cur_endpoint
        and manifest.endpoint_signature
        and cur_endpoint != manifest.endpoint_signature
    ):
        return False

    # 5. Options signature: only enforced if the manifest recorded one
    #    (manifests written before this commit have it empty — they are
    #    treated as stale and re-downloaded the next time the env changes).
    #
    #    Compare against the *current* fixed constants from client.py, not
    #    the copy stashed in the manifest — using the manifest's copy would
    #    always reproduce the recorded signature and silently swallow
    #    code-only changes (e.g. flipping image_export_mode or to_formats),
    #    defeating the invalidation this step is supposed to provide.
    #    Lazy import: client.py imports from cache.py.
    #
    #    When per-file overrides are requested (e.g. docling(force_ocr=true)) but
    #    the manifest predates signature recording, we cannot prove the bundle was
    #    produced with those overrides — accepting it would silently drop the
    #    user's explicit param. Treat that as a miss so the override is honored
    #    (mirrors MinerU, which misses on any absent signature). The no-override
    #    case keeps the deliberate leniency above for legacy bundles.
    if overrides and not manifest.options_signature:
        return False
    if manifest.options_signature:
        from lightrag.parser.external.docling.client import FIXED_CONSTANTS

        cur_options = compute_options_signature(
            tunable_env=snapshot_tunable_env(overrides),
            fixed_constants=FIXED_CONSTANTS,
        )
        if cur_options != manifest.options_signature:
            return False

    # 6. Critical file: size + sha256
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

    # 7. Other files: size only
    for entry in manifest.files:
        ep = raw_dir / entry.path
        try:
            if ep.stat().st_size != int(entry.size):
                return False
        except OSError:
            return False

    return True


__all__ = [
    "DOCLING_TUNABLE_ENVS",
    "compute_options_signature",
    "current_endpoint_signature",
    "is_bundle_valid",
    "snapshot_tunable_env",
]
