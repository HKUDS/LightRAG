"""Docling parser integration (raw client, cache, manifest, IR adapter).

Public surface for the rest of the codebase. ``parse_docling`` imports
only from this facade so the inner module layout stays free to evolve.
"""

from lightrag.external_parser._common import (
    clear_dir_contents,
    raw_dir_for_parsed_dir as _raw_dir_for_parsed_dir,
)

DOCLING_RAW_DIR_SUFFIX = ".docling_raw"
MANIFEST_ENGINE = "docling"


def raw_dir_for_parsed_dir(parsed_dir):
    """``foo.parsed/`` → ``foo.docling_raw/`` (docling-specific binding)."""
    return _raw_dir_for_parsed_dir(parsed_dir, suffix=DOCLING_RAW_DIR_SUFFIX)


# Imported after ``MANIFEST_ENGINE`` / ``DOCLING_RAW_DIR_SUFFIX`` because
# the submodules read those constants at import time.
from lightrag.external_parser.docling.cache import (  # noqa: E402
    is_bundle_valid,
)
from lightrag.external_parser.docling.client import (  # noqa: E402
    DoclingRawClient,
)

__all__ = [
    "DOCLING_RAW_DIR_SUFFIX",
    "MANIFEST_ENGINE",
    "DoclingRawClient",
    "clear_dir_contents",
    "is_bundle_valid",
    "raw_dir_for_parsed_dir",
]
