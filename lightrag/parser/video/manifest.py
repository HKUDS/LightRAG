"""Versioned, JSON-safe video ingestion manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MANIFEST_VERSION = "1.0"


def write_video_manifest(
    destination: Path,
    *,
    document_name: str,
    media: dict[str, Any],
    scenes: list[dict[str, Any]],
    transcript_segments: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    payload = {
        "manifest_version": MANIFEST_VERSION,
        "document_name": document_name,
        "media": media,
        "scenes": scenes,
        "transcript_segments": transcript_segments,
        "warnings": warnings,
    }
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
