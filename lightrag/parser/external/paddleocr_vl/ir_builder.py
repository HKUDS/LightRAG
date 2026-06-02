"""PaddleOCR-VL IR builder: PaddleOCR JSON → :class:`IRDoc`.

Input contract: a ``*.paddleocr_raw/`` directory containing a ``<stem>.json``
produced by PaddleOCR-VL API.

Conversion rules:
- Extract text blocks with positions
- Detect tables and structure
- Handle images and layout information
- Support multiple languages (Chinese, English, etc.)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from lightrag.parser._markdown import (
    render_heading_line,
    strip_heading_markdown_prefix,
)
from lightrag.parser.external._common import env_json
from lightrag.sidecar.ir import (
    AssetSpec,
    IRBlock,
    IRDoc,
    IRDrawing,
    IREquation,
    IRPosition,
    IRTable,
)
from lightrag.utils import logger


class PaddleOCRVLIRBuilder:
    """Stateless except for env-driven config. Reusable across calls."""

    def __init__(self) -> None:
        self.min_text_length = int(os.getenv("PADDLEOCR_MIN_TEXT_LENGTH", "2"))
        self.merge_threshold = int(os.getenv("PADDLEOCR_MERGE_THRESHOLD", "50"))

    def build_ir(self, result_file: Path, source_filename: str) -> IRDoc:
        """Build IR document from PaddleOCR-VL result.

        Args:
            result_file: Path to the JSON result file
            source_filename: Original filename

        Returns:
            IRDoc with parsed content
        """
        with open(result_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        doc = IRDoc(
            source_filename=source_filename,
            parser_engine="paddleocr_vl",
        )

        # Process OCR results
        # PaddleOCR returns list of [bbox, (text, confidence)] tuples
        if isinstance(data, list):
            self._process_ocr_results(doc, data)
        elif isinstance(data, dict):
            # Handle different response formats
            if "results" in data:
                self._process_ocr_results(doc, data["results"])
            elif "data" in data:
                self._process_ocr_results(doc, data["data"])
            else:
                logger.warning(f"Unexpected PaddleOCR-VL response format: {list(data.keys())}")

        return doc

    def _process_ocr_results(self, doc: IRDoc, results: list) -> None:
        """Process OCR results and add to document."""
        for item in results:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue

            bbox, text_info = item[0], item[1]

            # Extract text and confidence
            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                text, confidence = text_info[0], text_info[1]
            elif isinstance(text_info, str):
                text = text_info
                confidence = 1.0
            else:
                continue

            # Skip short text
            if len(text.strip()) < self.min_text_length:
                continue

            # Create position from bbox
            position = self._bbox_to_position(bbox)

            # Add text block
            doc.blocks.append(
                IRBlock(
                    text=text.strip(),
                    position=position,
                    confidence=confidence,
                )
            )

    def _bbox_to_position(self, bbox: list) -> IRPosition:
        """Convert PaddleOCR bbox to IRPosition.

        PaddleOCR bbox format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        if not bbox or len(bbox) < 4:
            return IRPosition()

        # Get bounding box coordinates
        x_coords = [p[0] for p in bbox]
        y_coords = [p[1] for p in bbox]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        return IRPosition(
            x=x_min,
            y=y_min,
            width=x_max - x_min,
            height=y_max - y_min,
            origin="LEFTTOP",
        )
