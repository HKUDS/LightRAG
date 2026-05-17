"""Docling parser integration (raw client, cache, manifest, IR adapter).

Public surface is finalized in subsequent commits when the concrete modules
land. Until then this package exposes only the constants and shared bindings
needed by tests for the scaffolding.
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


__all__ = [
    "DOCLING_RAW_DIR_SUFFIX",
    "MANIFEST_ENGINE",
    "clear_dir_contents",
    "raw_dir_for_parsed_dir",
]
