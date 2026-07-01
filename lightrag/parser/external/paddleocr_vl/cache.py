"""PaddleOCR-VL cache utilities."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any


def compute_options_signature(options: dict[str, Any]) -> str:
    """Compute a stable hash of parser options for cache invalidation."""
    import json

    # Sort keys for stability
    normalized = json.dumps(options, sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def current_endpoint_signature() -> str:
    """Get current endpoint signature for cache key."""
    endpoint = os.getenv("PADDLEOCR_ENDPOINT", "http://localhost:8000")
    lang = os.getenv("PADDLEOCR_LANG", "ch")
    return f"{endpoint}|{lang}"


def snapshot_tunable_env() -> dict[str, str]:
    """Snapshot tunable environment variables for cache invalidation."""
    return {
        "PADDLEOCR_ENDPOINT": os.getenv("PADDLEOCR_ENDPOINT", ""),
        "PADDLEOCR_LANG": os.getenv("PADDLEOCR_LANG", ""),
        "PADDLEOCR_DET": os.getenv("PADDLEOCR_DET", ""),
        "PADDLEOCR_REC": os.getenv("PADDLEOCR_REC", ""),
        "PADDLEOCR_CLS": os.getenv("PADDLEOCR_CLS", ""),
    }
