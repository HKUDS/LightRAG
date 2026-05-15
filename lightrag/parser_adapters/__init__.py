"""Parser adapters: engine-specific normalizers that produce an :class:`IRDoc`.

Each adapter handles one engine's output format quirks (mineru content_list,
docling DoclingDocument, native docx blocks, ...) and returns an IRDoc that
the spec-compliant writer (``lightrag.sidecar.writer``) consumes.
"""

from lightrag.parser_adapters.mineru import MinerUAdapter
from lightrag.parser_adapters.native_docx import NativeDocxAdapter

__all__ = ["MinerUAdapter", "NativeDocxAdapter"]
