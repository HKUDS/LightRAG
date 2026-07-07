"""PaddleOCR-VL external parser package."""

from __future__ import annotations

from lightrag.constants import PADDLEOCR_VL_RAW_DIR_SUFFIX
from lightrag.parser.external._common import (
    clear_dir_contents,
    compute_size_and_hash,
)
from lightrag.parser.external._common import (
    raw_dir_for_parsed_dir as _raw_dir_for_parsed_dir,
)
from lightrag.parser.external.paddleocr_vl.cache import is_bundle_valid
from lightrag.parser.external.paddleocr_vl.client import PaddleOCRVLRawClient
from lightrag.parser.external.paddleocr_vl.ir_builder import PaddleOCRVLIRBuilder


def raw_dir_for_parsed_dir(parsed_dir):
    """``foo.parsed/`` -> ``foo.paddleocr_vl_raw/``."""
    return _raw_dir_for_parsed_dir(parsed_dir, suffix=PADDLEOCR_VL_RAW_DIR_SUFFIX)


__all__ = [
    "PADDLEOCR_VL_RAW_DIR_SUFFIX",
    "PaddleOCRVLIRBuilder",
    "PaddleOCRVLRawClient",
    "clear_dir_contents",
    "compute_size_and_hash",
    "is_bundle_valid",
    "raw_dir_for_parsed_dir",
]
