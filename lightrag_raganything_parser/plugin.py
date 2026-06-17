"""Entry point for the RAG-Anything parser plugin."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from lightrag.parser.registry import ParserSpec, register_parser

from lightrag_raganything_parser import ENGINE_NAME


_RAGANYTHING_SUFFIXES = frozenset(
    {
        "pdf",
        "doc",
        "docx",
        "ppt",
        "pptx",
        "xls",
        "xlsx",
        "png",
        "jpg",
        "jpeg",
        "bmp",
        "tiff",
        "tif",
        "gif",
        "webp",
        "md",
        "txt",
        "html",
        "xhtml",
    }
)


def _raganything_configured() -> bool:
    configured_path = os.getenv("RAGANYTHING_PATH", "").strip()
    if configured_path and Path(configured_path).exists():
        return True
    return importlib.util.find_spec("raganything") is not None


def register() -> None:
    register_parser(
        ParserSpec(
            engine_name=ENGINE_NAME,
            impl="lightrag_raganything_parser.parser:RAGAnythingParser",
            suffixes=_RAGANYTHING_SUFFIXES,
            queue_group=ENGINE_NAME,
            concurrency=1,
            endpoint_configured=_raganything_configured,
            endpoint_requirement=lambda: "RAGANYTHING_PATH or installed raganything",
        )
    )


__all__ = ["register"]
