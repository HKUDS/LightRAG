"""PaddleOCR-VL manifest utilities."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lightrag.utils import logger


class Manifest:
    """Manifest for PaddleOCR-VL parsed results."""

    def __init__(
        self,
        source_filename: str,
        result_file: Path,
        options: dict[str, Any],
        endpoint_signature: str,
    ) -> None:
        self.source_filename = source_filename
        self.result_file = result_file
        self.options = options
        self.endpoint_signature = endpoint_signature
        self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to dictionary."""
        return {
            "source_filename": self.source_filename,
            "result_file": str(self.result_file),
            "options": self.options,
            "endpoint_signature": self.endpoint_signature,
            "created_at": self.created_at,
            "parser_engine": "paddleocr_vl",
        }

    def save(self, manifest_file: Path) -> None:
        """Save manifest to file."""
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"Saved PaddleOCR-VL manifest to {manifest_file}")


def write_manifest(
    raw_dir: Path,
    source_filename: str,
    result_file: Path,
    options: dict[str, Any],
    endpoint_signature: str,
) -> Manifest:
    """Write manifest for parsed results.

    Args:
        raw_dir: Directory containing parsed results
        source_filename: Original filename
        result_file: Path to the result file
        options: Parser options
        endpoint_signature: Endpoint signature for cache

    Returns:
        Manifest instance
    """
    manifest = Manifest(
        source_filename=source_filename,
        result_file=result_file,
        options=options,
        endpoint_signature=endpoint_signature,
    )
    manifest_file = raw_dir / "manifest.json"
    manifest.save(manifest_file)
    return manifest
