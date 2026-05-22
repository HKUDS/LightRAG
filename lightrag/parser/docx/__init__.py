"""LightRAG native DOCX parser package.

The :mod:`parse_document` / :mod:`numbering_resolver` / :mod:`table_extractor` /
:mod:`drawing_image_extractor` / :mod:`utils` / :mod:`omml` modules ship the
upstream DOCX extraction logic verbatim (with imports localized for the new
package path).

The pipeline-side orchestration (extract → IR → sidecar) now lives in
:meth:`lightrag.pipeline._PipelineMixin.parse_native` so the native and
MinerU engines share one shape; see :mod:`lightrag.parser.docx.ir_builder`
for the engine IR builder.
"""

__all__: list[str] = []
