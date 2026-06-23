"""Native DOCX engine adapter (implements NativeParserBase hooks)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from lightrag.constants import PARSER_ENGINE_NATIVE
from lightrag.parser.native_base import NativeParserBase
from lightrag.utils import logger

if TYPE_CHECKING:
    from lightrag.sidecar.ir import IRDoc


class NativeDocxParser(NativeParserBase):
    """Native DOCX parser for LightRAG's production parsing path.

    ``extract_docx_blocks`` performs only heading-driven structural splitting
    (one block per DOCX heading). Block sizing is intentionally left to the
    downstream paragraph-semantic chunker, so this parser emits the
    one-heading-one-block sidecar contract that chunking consumes.
    """

    engine_name = PARSER_ENGINE_NATIVE
    sidecar_path_style = "basename_only"  # legacy native docx convention
    empty_content_label = "DOCX"

    def validate_source(self, source: Path, file_path: str) -> None:
        if not (
            source.exists() and source.is_file() and source.suffix.lower() == ".docx"
        ):
            raise ValueError(
                f"Native parser does not support pending file: {file_path}"
            )

    def extract(
        self, source: Path, *, parsed_dir: Path, asset_dir: Path, base_name: str
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
        """Extract heading-scoped DOCX blocks (sizing left to the chunker)."""

        from lightrag.parser.docx.drawing_image_extractor import (
            DrawingExtractionContext,
            load_relationships,
        )
        from lightrag.parser.docx.parse_document import extract_docx_blocks

        ctx = DrawingExtractionContext(
            docx_path=source,
            blocks_output_path=parsed_dir / f"{base_name}.blocks.jsonl",
            export_dir_name=asset_dir.name,
            export_dir_path=asset_dir,
        )
        load_relationships(ctx)
        warnings: dict[str, Any] = {}
        metadata: dict[str, Any] = {}
        blocks = extract_docx_blocks(
            str(source),
            drawing_context=ctx,
            parse_warnings=warnings,
            parse_metadata=metadata,
        )
        return blocks, warnings, metadata

    def build_ir(
        self,
        blocks: list[dict[str, Any]],
        *,
        document_name: str,
        asset_dir_name: str,
        metadata: dict[str, Any],
    ) -> "IRDoc":
        from lightrag.parser.docx.ir_builder import NativeDocxIRBuilder

        return NativeDocxIRBuilder().normalize(
            blocks,
            document_name=document_name,
            asset_dir_name=asset_dir_name,
            parse_metadata=metadata,
        )

    def surface_warnings(
        self, warnings: dict[str, Any], source: Path
    ) -> dict[str, Any] | None:
        missing = int(warnings.get("missing_paraid_count", 0) or 0)
        if missing > 0:
            # Surface once per document; affected blocks emit
            # ``positions: [{"type": "paraid", "range": null}]``.
            logger.warning(
                "[parse_native] %s: %d paragraphs lack paraId; "
                "Re-saving file in Word 2013+ to regenerate ids.",
                source.name,
                missing,
            )
            return {"missing_paraid_count": missing}
        return None
