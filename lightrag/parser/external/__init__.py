"""Adapters for external document parsing services.

Each subpackage under ``parser/external/`` integrates one external parser
(docling, mineru, ...) by handling:

- request/upload/poll choreography against the parser's HTTP API,
- on-disk caching of the raw bundle under ``<base>.<engine>_raw/``,
- normalization into LightRAG IR (``IRDoc``) for the sidecar writer.

Shared cross-engine helpers (size/hash, atomic manifest IO, safe zip
extraction, env coercion) live at this package root in private modules
prefixed ``_``. Engine-specific cache validation, manifest construction,
and IR adaptation live in each subpackage.
"""

from lightrag.parser.external._common import (
    clear_dir_contents,
    compute_size_and_hash,
    env_bool,
    env_int,
    env_json,
    raw_dir_for_parsed_dir,
)
from lightrag.parser.external._manifest import (
    MANIFEST_FILENAME,
    MANIFEST_VERSION,
    Manifest,
    ManifestFile,
    load_manifest,
    manifest_path,
    write_manifest,
)
from lightrag.parser.external._zip import safe_extract_zip

__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_VERSION",
    "Manifest",
    "ManifestFile",
    "clear_dir_contents",
    "compute_size_and_hash",
    "env_bool",
    "env_int",
    "env_json",
    "load_manifest",
    "manifest_path",
    "raw_dir_for_parsed_dir",
    "safe_extract_zip",
    "write_manifest",
]
