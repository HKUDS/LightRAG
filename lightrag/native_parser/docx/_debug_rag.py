"""Shared debug LightRAG stand-in for the native DOCX parse path.

Used by the package's debug CLI (``__main__.py``), the golden-fixture
regen script (``scripts/regen_native_docx_golden.py``), and the
byte-equivalence golden tests
(``tests/native_parser/docx/test_native_docx_golden.py``).

Centralising these stubs keeps the three call sites in sync when
``parse_native`` grows new dependencies on ``LightRAG`` attributes.
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
    """No-op ``doc_status`` shim — parse_native never reads/writes content."""

    async def get_by_id(self, doc_id: str) -> Any:
        return None

    async def upsert(self, data: dict[str, Any]) -> None:
        return None


def build_debug_rag():
    """Build a minimal LightRAG stand-in that exposes what ``parse_native`` reads.

    The import of ``LightRAG`` is intentionally function-local: deferring
    it avoids a circular import when this helper is loaded during package
    init (``__main__`` invocations resolve ``lightrag.native_parser.docx``
    before ``lightrag`` is fully bound).

    LightRAG-side attributes ``parse_native`` (the bound code below) currently
    reads off ``self`` — every entry MUST be provided by this stand-in, or the
    debug CLI / golden tests / regen script will all break in sync:

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

    When ``LightRAG.parse_native`` grows new dependencies on ``self``,
    extend this stand-in (and update the list above) rather than copy-pasting
    a parallel stub into the three call sites.
    """
    from lightrag import LightRAG

    class _DebugRag:
        _persist_parsed_full_docs = LightRAG._persist_parsed_full_docs
        parse_native = LightRAG.parse_native

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
