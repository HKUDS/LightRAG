"""Native DOCX engine adapter (implements NativeParserBase hooks)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from collections.abc import Mapping

from lightrag.constants import PARSER_ENGINE_NATIVE
from lightrag.parser.native_base import NativeExtractRuntime, NativeParserBase
from lightrag.utils import logger

if TYPE_CHECKING:
    from lightrag.sidecar.ir import IRDoc


# Warnings whose key carries one of these prefixes are smart-heading feature
# diagnostics: finalize_parse_warnings diverts them to the sidecar
# smart_audit.json. Every other warning stays on doc_status.metadata.
_SMART_HEADING_WARNING_PREFIXES = ("smart_", "title_block_")


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
        # The smart-heading audit ledger stays in ``metadata["smart_audit"]``;
        # ``finalize_parse_warnings`` (below) merges the smart-heading warnings
        # into it and writes ``<base>.smart_audit.json`` once, after the base
        # template has computed the post-extract I4 waiver flag. It pops the key
        # before build_ir, so the IR builder never sees it.
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

    def finalize_parse_warnings(
        self,
        warnings: dict[str, Any],
        metadata: dict[str, Any],
        *,
        parsed_dir: Path,
        base_name: str,
        source: Path,
        i4_cache_disabled: bool,
    ) -> dict[str, Any] | None:
        """Divert smart-heading diagnostics to the sidecar; keep the rest.

        Smart-heading warnings (``smart_*`` / ``title_block_*``) are merged
        under a ``parse_warnings`` key into the audit ledger docx left in
        ``metadata["smart_audit"]`` and written once to
        ``<base>.smart_audit.json`` (timestamp-free + ``sort_keys`` so a
        re-parse stays byte-identical — I4). Every other warning (missing
        paraId, over-long-heading handling) is returned unconditionally so the
        pipeline keeps mirroring it onto ``doc_status.metadata``.
        """
        # I4 determinism waiver: a smart-heading LLM ran with the entity-extract
        # cache off. Record it in the audit file — the base no longer injects
        # this smart_ key into the warnings dict.
        if i4_cache_disabled:
            warnings["smart_i4_cache_disabled"] = 1

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

        smart = {
            k: v
            for k, v in warnings.items()
            if k.startswith(_SMART_HEADING_WARNING_PREFIXES) and v
        }
        other = {
            k: v
            for k, v in warnings.items()
            if not k.startswith(_SMART_HEADING_WARNING_PREFIXES) and v
        }

        # Merge into the ledger (pop before build_ir sees it) and write the
        # sidecar. Written whenever there is any audit content — ledger only,
        # warnings only, or both.
        ledger = metadata.pop("smart_audit", None)
        audit = dict(ledger) if isinstance(ledger, dict) else {}
        if smart:
            audit["parse_warnings"] = smart
        if audit:
            audit_path = parsed_dir / f"{base_name}.smart_audit.json"
            audit_path.write_text(
                json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        return other or None
