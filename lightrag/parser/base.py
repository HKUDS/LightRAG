"""Unified parser contract for native + external parser engines.

Every engine (native docx, mineru, docling, legacy, plus the internal
``reuse``/``passthrough`` format handlers) implements :class:`BaseParser`.
The pipeline dispatches through the registry
(:mod:`lightrag.parser.registry`) instead of a growing ``if engine == …``
chain.

Design notes:

- ``BaseParser`` carries *behaviour only* (the ``parse`` coroutine and any
  engine-private hooks).  Capability metadata (supported suffixes, queue
  group, endpoint requirements) lives in the registry's lightweight
  ``ParserSpec`` table so capability queries never import a parser
  implementation.
- ``ParseResult.to_dict()`` emits only semantically-present fields so the
  returned dict stays byte-for-byte compatible with the pre-refactor
  ``parse_native``/``parse_mineru``/``parse_docling`` return shapes (the
  worker reads these by ``.get(...)``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lightrag.sidecar.ir import IRDoc  # noqa: F401


@dataclass
class ResolvedSource:
    """The common parse preamble shared by every engine."""

    source_path: Path
    document_name: str
    parsed_dir: Path


@dataclass
class ParseContext:
    """Inputs handed to :meth:`BaseParser.parse`.

    Wraps the LightRAG instance plus the per-document handles so a parser
    does not receive a bare ``self``.  Convenience helpers lazily import
    pipeline-layer functions at call time, keeping this module import-cheap
    and free of import cycles.
    """

    rag: Any
    doc_id: str
    file_path: str
    content_data: dict[str, Any]
    # Set by the active pipeline batch when the user requests cancellation.
    # Native parsers pass it to their synchronous LLM bridge so a worker thread
    # does not remain blocked on a slow title-block judgment.
    pipeline_cancel_event: threading.Event | None = None

    def source_path(self, parser_engine: str) -> Path:
        """Resolve the on-disk source file for this document."""
        from lightrag.pipeline import (
            _call_source_file_resolver,
            _read_source_file,
        )

        return Path(
            _call_source_file_resolver(
                self.rag,
                self.file_path,
                source_file=_read_source_file(self.content_data),
                parser_engine=parser_engine,
            )
        )

    def resolve(self, parser_engine: str) -> ResolvedSource:
        """Resolve ``(source_path, document_name, parsed_dir)``.

        Mirrors the preamble shared by ``parse_mineru``/``parse_docling``/
        ``parse_native``: canonicalize the document name defensively (so
        direct callers may pass absolute or hint-bearing paths) and derive
        the ``__parsed__/<base>.parsed/`` output directory.
        """
        from lightrag.utils_pipeline import (
            normalize_document_file_path,
            parsed_artifact_dir_for,
        )

        source_path = self.source_path(parser_engine)
        document_name = normalize_document_file_path(self.file_path)
        if document_name == "unknown_source":
            document_name = source_path.name or f"{self.doc_id}.bin"
        parsed_dir = parsed_artifact_dir_for(
            document_name, parent_hint=source_path.parent
        )
        return ResolvedSource(source_path, document_name, parsed_dir)

    async def archive_source(self, source_path: str) -> str | None:
        """Archive the source after a successful parse + full_docs sync.

        Resolved through the pipeline module's namespace (where the function
        is imported) so existing tests that patch
        ``lightrag.pipeline.archive_docx_source_after_full_docs_sync`` keep
        intercepting it now that the call site lives in the parser layer.
        """
        import lightrag.pipeline as _pipeline

        return await _pipeline.archive_docx_source_after_full_docs_sync(source_path)


@dataclass
class ParseResult:
    """Structured parser output.

    ``to_dict`` only emits fields that carry meaning so the dict matches the
    pre-refactor return shapes exactly (no spurious ``None``/``False`` keys).
    """

    doc_id: str
    file_path: str
    parse_format: str
    content: str
    blocks_path: str = ""
    parse_engine: str | None = None
    parse_stage_skipped: bool = False
    parse_warnings: dict[str, Any] | None = None
    # LLM cache keys minted during the parse stage (docx smart_heading).
    # Fixed, first-class field: the pipeline mirrors it into
    # doc_status.metadata so adelete_by_doc_id(delete_llm_cache=True) can
    # purge parse-stage cache — there is no chunk llm_cache_list at parse
    # time to carry them.
    smartheading_llm_cache_ids: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "doc_id": self.doc_id,
            "file_path": self.file_path,
            "parse_format": self.parse_format,
            "content": self.content,
            "blocks_path": self.blocks_path,
        }
        if self.parse_engine is not None:
            out["parse_engine"] = self.parse_engine
        if self.parse_stage_skipped:
            out["parse_stage_skipped"] = True
        if self.parse_warnings:
            out["parse_warnings"] = self.parse_warnings
        if self.smartheading_llm_cache_ids:
            out["smartheading_llm_cache_ids"] = self.smartheading_llm_cache_ids
        return out


class BaseParser(ABC):
    """Abstract base for every parser engine.

    Subclasses set ``engine_name`` and implement :meth:`parse`.  Capability
    metadata (suffixes/queue group/endpoint) is declared in the registry
    ``ParserSpec``, not here.
    """

    engine_name: str

    @abstractmethod
    async def parse(self, ctx: ParseContext) -> ParseResult:
        """Parse one document and return its :class:`ParseResult`."""
        ...
