"""MinerU adapter implementing BaseExternalParser."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from lightrag.parser.external._base import BaseExternalParser
from lightrag.sidecar.ir import IRDoc


class MinerUParser(BaseExternalParser):
    """MinerU engine adapter."""

    engine_name = "mineru"
    raw_dir_suffix = ".mineru_raw"
    force_reparse_env = "LIGHTRAG_FORCE_REPARSE_MINERU"

    def is_bundle_valid(self, raw_dir: Path, source_path: Path) -> bool:
        from lightrag.parser.external.mineru.cache import is_bundle_valid
        return is_bundle_valid(raw_dir, source_path)

    async def download_into(
        self,
        raw_dir: Path,
        source_path: Path,
        **kwargs: object,
    ) -> None:
        from lightrag.parser.external.mineru.client import MinerURawClient
        client = MinerURawClient()
        upload_name = kwargs.get("upload_name", source_path.name)
        await client.download_into(raw_dir, source_path, upload_name=str(upload_name))

    def build_ir(self, raw_dir: Path, document_name: str) -> IRDoc:
        from lightrag.parser.external.mineru.ir_builder import MinerUIRBuilder
        builder = MinerUIRBuilder()
        return builder.normalize_from_workdir(raw_dir, document_name=document_name)
