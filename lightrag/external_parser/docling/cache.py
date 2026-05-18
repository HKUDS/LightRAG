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
from pathlib import Path

from lightrag.external_parser._common import compute_size_and_hash
from lightrag.external_parser._manifest import load_manifest
from lightrag.external_parser.docling import MANIFEST_ENGINE

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
    """The active docling endpoint, normalized (trailing slash stripped).

    Returns ``""`` if ``DOCLING_ENDPOINT`` is unset — callers that need a
    real endpoint (``DoclingRawClient``) raise on empty; callers that only
    compare against a recorded manifest field (``is_bundle_valid``) silently
    skip the check when either side is empty.
    """
    return os.getenv("DOCLING_ENDPOINT", "").strip().rstrip("/")


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


def snapshot_tunable_env() -> dict[str, str]:
    """Read all docling tunable envs (with empty strings for missing ones)
    so the signature is deterministic."""
    return {name: os.getenv(name, "").strip() for name in DOCLING_TUNABLE_ENVS}


def is_bundle_valid(raw_dir: Path, source_file: Path) -> bool:
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
    if manifest.options_signature:
        cur_options = compute_options_signature(
            tunable_env=snapshot_tunable_env(),
            fixed_constants=manifest.extras.get("fixed_constants", {}),
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
