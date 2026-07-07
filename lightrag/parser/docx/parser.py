"""Native DOCX engine adapter (implements NativeParserBase hooks)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from collections.abc import Mapping

from lightrag.constants import PARSER_ENGINE_NATIVE
from lightrag.parser.native_base import NativeExtractRuntime, NativeParserBase
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

    def wants_llm_bridge(self, engine_params: Mapping[str, Any]) -> bool:
        return bool(engine_params.get("smart_heading"))

    def extract(
        self,
        source: Path,
        *,
        parsed_dir: Path,
        asset_dir: Path,
        base_name: str,
        runtime: NativeExtractRuntime | None = None,
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
            smart_heading_runtime=runtime,
        )
        # Full smart-heading audit artifact: sits beside blocks.jsonl (the
        # clean_parsed_dir=False write keeps it), deliberately timestamp-free
        # so repeated parses stay byte-identical (I4). Popped from metadata —
        # it is not IR-builder input.
        smart_audit = metadata.pop("smart_audit", None)
        if smart_audit is not None:
            import json

            audit_path = parsed_dir / f"{base_name}.smart_audit.json"
            audit_path.write_text(
                json.dumps(smart_audit, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
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
        # Smart-heading runs surface their full compact summary (breaker
        # trips, TOC removals, demotion counts, …); the smart-off path keeps
        # its historical missing-paraId-only shape.
        smart_active = any(k.startswith(("smart_", "title_block_")) for k in warnings)
        if smart_active:
            relevant = {k: v for k, v in warnings.items() if v}
            return relevant or None
        return {"missing_paraid_count": missing} if missing > 0 else None
