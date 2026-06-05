"""BaseExternalParser protocol for pluggable OCR/VLM backends.

See RFC: https://github.com/HKUDS/LightRAG/issues/3197

Every external parser engine (MinerU, Docling, PaddleOCR-VL, etc.)
implements this protocol. The pipeline dispatches through a registry
dict instead of a growing ``if engine == …`` chain.

Minimal contract:

- ``is_bundle_valid`` — cheap cache-hit check against the raw bundle
  on disk.  Returns True when the bundle is reusable (skip download).
- ``download_into`` — fetch the raw bundle from the external service
  and land it under ``raw_dir/``.  Called only on cache miss.
- ``build_ir`` — convert the raw bundle to an :class:`IRDoc`.  Called
  unconditionally after the cache/download step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lightrag.sidecar.ir import IRDoc


class BaseExternalParser(ABC):
    """Abstract base for external parser engines.

    Subclasses must set:
    - ``engine_name`` — canonical engine identifier (e.g. ``"mineru"``).
    - ``raw_dir_suffix`` — suffix for the raw bundle directory
      (e.g. ``".mineru_raw"``).
    - ``force_reparse_env`` — env var that forces a re-parse
      (e.g. ``"LIGHTRAG_FORCE_REPARSE_MINERU"``).
    """

    engine_name: str
    raw_dir_suffix: str
    force_reparse_env: str

    @abstractmethod
    def is_bundle_valid(self, raw_dir: Path, source_path: Path) -> bool:
        """Return True if the raw bundle in ``raw_dir`` is a valid
        cache hit for ``source_path``."""
        ...

    @abstractmethod
    async def download_into(
        self,
        raw_dir: Path,
        source_path: Path,
        **kwargs: object,
    ) -> None:
        """Download the raw bundle from the external service into
        ``raw_dir/``.  Called only on cache miss."""
        ...

    @abstractmethod
    def build_ir(
        self,
        raw_dir: Path,
        document_name: str,
    ) -> IRDoc:
        """Convert the raw bundle in ``raw_dir`` to an :class:`IRDoc`."""
        ...
