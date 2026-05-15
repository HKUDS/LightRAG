"""Shared debug LightRAG stand-in for the native DOCX parse path.

Used by the package's debug CLI (``__main__.py``), the golden-fixture
regen script (``scripts/regen_native_docx_golden.py``), and the
byte-equivalence golden tests
(``tests/parser_adapters/test_native_docx_golden.py``).

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
