"""PaddleOCR-VL parser for LightRAG."""

from __future__ import annotations

from lightrag.parser.external.paddleocr_vl.client import PaddleOCRVLClient
from lightrag.parser.external.paddleocr_vl.ir_builder import PaddleOCRVLIRBuilder

__all__ = ["PaddleOCRVLClient", "PaddleOCRVLIRBuilder"]
