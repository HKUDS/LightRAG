"""MinerU parser integration (raw client, cache, manifest, IR builder).

Public surface for the rest of the codebase. ``parse_mineru`` imports
only from this facade so the inner module layout stays free to evolve.

See ``docs/LightRAGSidecarFormat-zh.md`` for sidecar format and
``docs/FileProcessingConfiguration-zh.md`` for cache lifecycle.
"""

from lightrag.parser.external.mineru.cache import (
    MINERU_RAW_DIR_SUFFIX,
    clear_dir_contents,
    compute_size_and_hash,
    is_bundle_valid,
    raw_dir_for_parsed_dir,
)
from lightrag.parser.external.mineru.client import MinerURawClient
from lightrag.parser.external.mineru.ir_builder import MinerUIRBuilder
from lightrag.parser.external.mineru.manifest import Manifest, ManifestFile

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

# Register with the parser registry (RFC #3197)
from lightrag.parser.external._registry import register_parser
from lightrag.parser.external.mineru.adapter import MinerUParser

register_parser("mineru", MinerUParser)
