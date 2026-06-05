"""Docling adapter implementing BaseExternalParser."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lightrag.parser.external._base import BaseExternalParser
from lightrag.sidecar.ir import IRDoc


class DoclingParser(BaseExternalParser):
    """Docling engine adapter."""

    engine_name = "docling"
    raw_dir_suffix = ".docling_raw"
    force_reparse_env = "LIGHTRAG_FORCE_REPARSE_DOCLING"

    def is_bundle_valid(self, raw_dir: Path, source_path: Path) -> bool:
        from lightrag.parser.external.docling.cache import is_bundle_valid
        return is_bundle_valid(raw_dir, source_path)

    async def download_into(
        self,
        raw_dir: Path,
        source_path: Path,
        **kwargs: object,
    ) -> None:
        from lightrag.parser.external.docling.client import DoclingRawClient
        client = DoclingRawClient()
        upload_filename = kwargs.get("upload_filename", source_path.name)
        await client.download_into(raw_dir, source_path, upload_filename=str(upload_filename))

    def build_ir(self, raw_dir: Path, document_name: str) -> IRDoc:
        from lightrag.parser.external.docling.ir_builder import DoclingIRBuilder
        builder = DoclingIRBuilder()
        return builder.normalize_from_workdir(raw_dir, document_name=document_name)
