"""MinerU raw bundle cache.

Co-located with sidecar artifacts under ``__parsed__/<base>.mineru_raw/``,
this module provides:

- :class:`Manifest` — schema + atomic write semantics (``_manifest.json``)
- :func:`is_bundle_valid` — content-hash + size cache validation
- :class:`MinerURawClient` — downloads bundles into the raw dir

See ``docs/LightRAGSidecarFormat-zh.md`` for sidecar format and
``docs/FileProcessingConfiguration-zh.md`` for cache lifecycle.
"""

from lightrag.mineru_raw.cache import (
    MINERU_RAW_DIR_SUFFIX,
    clear_dir_contents,
    compute_size_and_hash,
    is_bundle_valid,
    raw_dir_for_parsed_dir,
)
from lightrag.mineru_raw.client import MinerURawClient
from lightrag.mineru_raw.manifest import Manifest, ManifestFile

__all__ = [
    "MINERU_RAW_DIR_SUFFIX",
    "Manifest",
    "ManifestFile",
    "MinerURawClient",
    "clear_dir_contents",
    "compute_size_and_hash",
    "is_bundle_valid",
    "raw_dir_for_parsed_dir",
]
