"""PaddleOCR-VL raw bundle downloader.

Talks to PaddleOCR-VL API over HTTP:

- ``POST /ocr`` — base64 image upload, returns OCR results
- Supports multiple image formats (PNG, JPG, PDF pages)
- Extracts text, tables, and layout information

Pipeline constants (``lang``, ``det``, ``rec``, ``cls``) are configurable
via environment variables.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lightrag.parser.external._common import (
    env_bool,
    env_int,
    raise_for_status_with_detail,
)
from lightrag.utils import logger

if TYPE_CHECKING:
    import httpx
else:
    try:
        import httpx
    except ImportError:  # pragma: no cover
        httpx = None

# ---------------------------------------------------------------------------
# Fixed pipeline constants
# ---------------------------------------------------------------------------

DEFAULT_ENDPOINT = "http://localhost:8000"
DEFAULT_LANG = "ch"  # Chinese by default
DEFAULT_DET = True
DEFAULT_REC = True
DEFAULT_CLS = True


class PaddleOCRVLClient:
    """PaddleOCR-VL API client for document parsing.

    Construct once per parse call (cheap). Reads ``PADDLEOCR_*`` envs at
    ``__init__`` time, so callers can flip env between calls and pick up
    the new values without holding a stale instance.
    """

    def __init__(self) -> None:
        self.endpoint = os.getenv("PADDLEOCR_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/")
        self.lang = os.getenv("PADDLEOCR_LANG", DEFAULT_LANG)
        self.det = env_bool("PADDLEOCR_DET", DEFAULT_DET)
        self.rec = env_bool("PADDLEOCR_REC", DEFAULT_REC)
        self.cls = env_bool("PADDLEOCR_CLS", DEFAULT_CLS)

        if not self.endpoint:
            raise ValueError("PADDLEOCR_ENDPOINT is required")

    async def parse_file(
        self,
        client: "httpx.AsyncClient",
        source_file_path: Path,
        *,
        upload_filename: str | None = None,
    ) -> dict[str, Any]:
        """Parse a file using PaddleOCR-VL API.

        Args:
            client: httpx async client
            source_file_path: Path to the file to parse
            upload_filename: Optional filename override

        Returns:
            dict with parsed content including text, tables, layout
        """
        effective_filename = upload_filename or source_file_path.name

        # Read file and encode to base64
        with source_file_path.open("rb") as f:
            file_content = f.read()
        file_base64 = base64.b64encode(file_content).decode("utf-8")

        # Prepare request
        url = f"{self.endpoint}/ocr"
        params = {
            "det": self.det,
            "rec": self.rec,
            "cls": self.cls,
        }
        headers = {"Content-Type": "application/json"}
        payload = {"image": file_base64}

        try:
            resp = await client.post(
                url,
                params=params,
                json=payload,
                headers=headers,
                timeout=60.0,
            )
            raise_for_status_with_detail(resp, f"PaddleOCR-VL parse for {effective_filename!r}")

            result = resp.json()
            logger.info(f"PaddleOCR-VL parsed {effective_filename}: {len(result)} items")
            return result

        except Exception as e:
            logger.error(f"PaddleOCR-VL parse failed for {effective_filename}: {e}")
            raise

    async def parse_file_to_bundle(
        self,
        client: "httpx.AsyncClient",
        source_file_path: Path,
        raw_dir: Path,
        *,
        upload_filename: str | None = None,
    ) -> Path:
        """Parse file and save results to raw_dir.

        Args:
            client: httpx async client
            source_file_path: Path to the file to parse
            raw_dir: Directory to save parsed results
            upload_filename: Optional filename override

        Returns:
            Path to the main result file
        """
        effective_filename = upload_filename or source_file_path.name
        result = await self.parse_file(
            client,
            source_file_path,
            upload_filename=effective_filename,
        )

        # Save result to raw_dir
        result_file = raw_dir / f"{Path(effective_filename).stem}.json"
        raw_dir.mkdir(parents=True, exist_ok=True)

        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        logger.info(f"PaddleOCR-VL saved result to {result_file}")
        return result_file
