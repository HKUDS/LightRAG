"""LightRAG native DOCX parser package.

The :mod:`parse_document` / :mod:`numbering_resolver` / :mod:`table_extractor` /
:mod:`drawing_image_extractor` / :mod:`utils` / :mod:`omml` modules ship the
upstream DOCX extraction logic verbatim (with imports localized for the new
package path). The :mod:`lightrag_adapter` module wraps them with a LightRAG
Document writer that emits ``.blocks.jsonl`` plus the table / equation /
drawing sidecars and the ``.blocks.assets`` directory.
"""

from .lightrag_adapter import parse_docx_to_lightrag_document

__all__ = ["parse_docx_to_lightrag_document"]
