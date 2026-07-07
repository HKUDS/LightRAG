"""Shared debug LightRAG stand-in for registry-dispatched parsing.

A minimal ``LightRAG`` stand-in plus a deterministic ``datetime`` shim,
shared by the unified parser debug CLI (``lightrag/parser/cli.py``),
the golden-fixture regen script (``scripts/regen_native_docx_golden.py``),
and the byte-equivalence golden tests
(``tests/parser/docx/test_native_docx_golden.py``).

Every engine is driven the same way — ``get_parser(engine).parse(
ParseContext(rag, ...))`` — and ``ParseContext`` reads the same ``rag``
surface (``_persist_parsed_full_docs``, ``_resolve_source_file_for_parser``,
``self.full_docs``, ``self.doc_status``), so a single stand-in covers every
engine — when a parser grows a new dependency, extend this module rather
than copy-pasting parallel stubs into each call site.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class DebugFullDocs:
    """In-memory ``full_docs`` shim — captures the persisted record."""

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}

    async def upsert(self, payload: dict[str, Any]) -> None:
        self.data.update(payload)

    async def get_by_id(self, doc_id: str) -> Any:
        return self.data.get(doc_id)

    async def index_done_callback(self) -> None:
        return None


class DebugDocStatus:
    """No-op ``doc_status`` shim — the parse_* methods never read/write it."""

    async def get_by_id(self, doc_id: str) -> Any:
        return None

    async def upsert(self, data: dict[str, Any]) -> None:
        return None


def build_debug_rag(*, extract_llm_func=None):
    """Build a minimal LightRAG stand-in that exposes what a parser reads.

    The import of ``LightRAG`` is intentionally function-local: deferring
    it avoids a circular import when this helper is loaded during package
    init (the parser CLI resolves ``lightrag.parser.debug`` before
    ``lightrag`` itself is fully bound).

    A parser is driven via ``get_parser(engine).parse(ParseContext(rag, ...))``
    (CLI / golden tests / regen script). ``ParseContext`` reads these off the
    ``rag`` it is handed — every entry MUST be provided by this stand-in:

    - **methods** (rebound from :class:`LightRAG`):
        - ``_persist_parsed_full_docs(doc_id, payload)`` — async; touches
          ``self.full_docs``.
        - ``_resolve_source_file_for_parser(file_path)`` — returns the
          on-disk source path. Stubbed to identity here since the CLI / tests
          feed an already-resolved path.
    - **storages**:
        - ``self.full_docs.upsert(...)`` / ``.get_by_id(...)`` /
          ``.index_done_callback()`` — :class:`DebugFullDocs` covers all three.
        - ``self.doc_status.get_by_id(...)`` / ``.upsert(...)`` —
          :class:`DebugDocStatus` covers both.
    - **LLM surface** (``_build_global_config`` + ``llm_response_cache``):
        consumed by ``NativeParserBase._build_llm_submit`` when an engine
        param requests the LLM bridge (docx ``smart_heading``). Pass
        ``extract_llm_func`` (an async ``(prompt, **kwargs) -> str``) to
        enable it; without injection the bridge stays ``None`` and the
        algorithm hard-fails only if it actually needs the LLM.

    When a parser grows a new dependency on the ``rag`` handed to
    ``ParseContext``, extend this stand-in (and update the list above) rather
    than copy-pasting a parallel stub into the call sites.
    """
    from lightrag import LightRAG

    class _DebugRag:
        _persist_parsed_full_docs = LightRAG._persist_parsed_full_docs

        def __init__(self) -> None:
            self.full_docs = DebugFullDocs()
            self.doc_status = DebugDocStatus()
            self.llm_response_cache = None

        def _resolve_source_file_for_parser(self, file_path: str) -> str:
            return file_path

        def _build_global_config(self) -> dict[str, Any]:
            return {
                "role_llm_funcs": {"extract": extract_llm_func},
                "llm_cache_identities": {},
                "enable_llm_cache_for_entity_extract": True,
            }

    return _DebugRag()


_FROZEN_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


class FrozenDateTime(datetime):
    """Pin ``datetime.now`` so ``write_sidecar`` stamps a deterministic time."""

    @classmethod
    def now(cls, tz=None):  # noqa: D401
        return _FROZEN_NOW if tz is None else _FROZEN_NOW.astimezone(tz)
