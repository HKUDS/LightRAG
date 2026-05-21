"""MinerU parser integration (raw client, cache, manifest, IR builder).

Public surface for the rest of the codebase. ``parse_mineru`` imports
only from this facade so the inner module layout stays free to evolve.

See ``docs/LightRAGSidecarFormat-zh.md`` for sidecar format and
``docs/FileProcessingConfiguration-zh.md`` for cache lifecycle.
"""

from lightrag.external_parser.mineru.cache import (
    MINERU_RAW_DIR_SUFFIX,
    clear_dir_contents,
    compute_size_and_hash,
    is_bundle_valid,
    raw_dir_for_parsed_dir,
)
from lightrag.external_parser.mineru.client import MinerURawClient
from lightrag.external_parser.mineru.ir_builder import MinerUIRBuilder
from lightrag.external_parser.mineru.manifest import Manifest, ManifestFile

__all__ = [
    "MINERU_RAW_DIR_SUFFIX",
    "Manifest",
    "ManifestFile",
    "MinerUIRBuilder",
    "MinerURawClient",
    "clear_dir_contents",
    "compute_size_and_hash",
    "is_bundle_valid",
    "raw_dir_for_parsed_dir",
]
