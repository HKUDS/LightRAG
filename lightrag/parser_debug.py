"""Shared debug LightRAG stand-in for the parse_* entry points.

A minimal ``LightRAG`` stand-in plus a deterministic ``datetime`` shim,
shared by the unified parser debug CLI (``lightrag/parser_cli.py``),
the golden-fixture regen script (``scripts/regen_native_docx_golden.py``),
and the byte-equivalence golden tests
(``tests/native_parser/docx/test_native_docx_golden.py``).

All three engines (``native`` / ``mineru`` / ``docling``) read the same
``self`` surface (``_persist_parsed_full_docs``, ``_resolve_source_file_for_parser``,
``self.full_docs``, ``self.doc_status``), so a single stand-in covers every
``parse_*`` method — when one of them grows a new dependency, extend
this module rather than copy-pasting parallel stubs into each call site.
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


def build_debug_rag():
    """Build a minimal LightRAG stand-in that exposes what ``parse_*`` reads.

    The import of ``LightRAG`` is intentionally function-local: deferring
    it avoids a circular import when this helper is loaded during package
    init (the parser CLI resolves ``lightrag.parser_debug`` before
    ``lightrag`` itself is fully bound).

    LightRAG-side attributes the three ``parse_*`` methods read off ``self`` —
    every entry MUST be provided by this stand-in, or the debug CLI / golden
    tests / regen script will all break in sync:

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

    When any of the three ``LightRAG.parse_*`` methods grows a new
    dependency on ``self``, extend this stand-in (and update the list
    above) rather than copy-pasting a parallel stub into the call sites.
    """
    from lightrag import LightRAG

    class _DebugRag:
        _persist_parsed_full_docs = LightRAG._persist_parsed_full_docs
        parse_native = LightRAG.parse_native
        parse_mineru = LightRAG.parse_mineru
        parse_docling = LightRAG.parse_docling

        def __init__(self) -> None:
            self.full_docs = DebugFullDocs()
            self.doc_status = DebugDocStatus()

        def _resolve_source_file_for_parser(self, file_path: str) -> str:
            return file_path

    return _DebugRag()


_FROZEN_NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


class FrozenDateTime(datetime):
    """Pin ``datetime.now`` so ``write_sidecar`` stamps a deterministic time."""

    @classmethod
    def now(cls, tz=None):  # noqa: D401
        return _FROZEN_NOW if tz is None else _FROZEN_NOW.astimezone(tz)
