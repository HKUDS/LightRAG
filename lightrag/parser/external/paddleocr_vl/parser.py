"""PaddleOCR-VL engine adapter (implements ExternalParserBase hooks)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lightrag.constants import PADDLEOCR_VL_RAW_DIR_SUFFIX, PARSER_ENGINE_PADDLEOCR_VL
from lightrag.parser.external._base import ExternalParserBase

if TYPE_CHECKING:
    from lightrag.sidecar.ir import IRDoc


class PaddleOCRVLParser(ExternalParserBase):
    engine_name = PARSER_ENGINE_PADDLEOCR_VL
    raw_dir_suffix = PADDLEOCR_VL_RAW_DIR_SUFFIX
    force_reparse_env = "LIGHTRAG_FORCE_REPARSE_PADDLEOCR_VL"

    def is_bundle_valid(
        self,
        raw_dir: Path,
        source_path: Path,
        *,
        engine_params: "Mapping[str, Any] | None" = None,
    ) -> bool:
        from lightrag.parser.external.paddleocr_vl import is_bundle_valid

        return is_bundle_valid(raw_dir, source_path, overrides=engine_params)

    async def download_into(
        self,
        raw_dir: Path,
        source_path: Path,
        *,
        upload_name: str,
        engine_params: "Mapping[str, Any] | None" = None,
    ) -> None:
        from lightrag.parser.external.paddleocr_vl import PaddleOCRVLRawClient

        await PaddleOCRVLRawClient(overrides=engine_params).download_into(
            raw_dir, source_path, upload_name=upload_name
        )

    def build_ir(self, raw_dir: Path, document_name: str) -> "IRDoc":
        from lightrag.parser.external.paddleocr_vl import PaddleOCRVLIRBuilder

        return PaddleOCRVLIRBuilder().normalize_from_workdir(
            raw_dir, document_name=document_name
        )

    def validate_ir(self, ir: "IRDoc", *, file_path: str, raw_dir: Path) -> None:
        if not ir.blocks:
            raise ValueError(
                f"PaddleOCR-VL IR builder produced zero blocks for {file_path} "
                f"(raw_dir={raw_dir})"
            )
