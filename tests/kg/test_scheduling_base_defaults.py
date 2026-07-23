"""Base-default contract tests for the memory-bounding scheduling API (Phase 1).

A minimal, third-party-style ``DocStatusStorage`` subclass (old abstract
signatures ONLY — no new methods, no capability flags) must keep working:

* instantiable, with every new base method available as a concrete default;
* ``get_docs_by_statuses_page``: CURSOR_START → one full page ending the
  sweep; any other position → empty terminal page; ``max_failure_generation``
  ignored (one-time warning) instead of raising on the missing field;
* ``count_docs_by_statuses`` raises ``StorageCapabilityError`` (fail-closed);
* ``update_doc_status_fields`` refuses ``created_at`` and raises
  ``StorageRecordNotFoundError`` on unknown ids unless ``missing_ok``;
* ``mark_doc_failed`` default = LEGACY write side (plain FAILED upsert, no
  generation, caller-supplied ``created_at`` ignored for existing rows,
  conditional create for missing rows);
* ``ensure_processing_attempt_id`` mints once and then reuses;
* strict variants never call legacy methods with new kwargs (no TypeError);
* ``BaseKVStorage.get_by_id_strict`` base default raises capability error.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from lightrag.base import (
    CURSOR_END,
    CursorAfter,
    DocProcessingStatus,
    DocSchedulingRecord,
    DocStatus,
    DocStatusStorage,
    FailureGenerationMode,
)
from lightrag.constants import CUSTOM_CHUNK_PATCH_METADATA_KEY
from lightrag.exceptions import (
    StorageCapabilityError,
    StorageRecordNotFoundError,
)

pytestmark = pytest.mark.offline


@dataclass
class _MinimalDocStatusStorage(DocStatusStorage):
    """Third-party-style subclass: implements ONLY the legacy abstract
    surface, with the legacy signatures. Anything new must come from base
    defaults."""

    embedding_func: Any = None
    namespace: str = "test"
    workspace: str = "test"
    global_config: dict = field(default_factory=dict)
    data: dict[str, dict[str, Any]] = field(default_factory=dict)

    async def initialize(self):  # pragma: no cover - unused
        pass

    async def finalize(self):  # pragma: no cover - unused
        pass

    async def index_done_callback(self) -> None:
        pass

    async def drop(self) -> dict[str, str]:
        self.data.clear()
        return {"status": "success", "message": "dropped"}

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        row = self.data.get(id)
        return dict(row) if row is not None else None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        return [dict(self.data[i]) for i in ids if i in self.data]

    async def filter_keys(self, keys: set[str]) -> set[str]:
        return {k for k in keys if k not in self.data}

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        for key, value in data.items():
            self.data[key] = dict(value)

    async def delete(self, ids: list[str]) -> None:
        for i in ids:
            self.data.pop(i, None)

    async def is_empty(self) -> bool:
        return not self.data

    async def get_status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.data.values():
            status = str(row.get("status"))
            counts[status] = counts.get(status, 0) + 1
        return counts

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:
        return await self.get_docs_by_statuses([status])

    async def get_docs_by_statuses(
        self, statuses: list[DocStatus], strict: bool = False
    ) -> dict[str, DocProcessingStatus]:
        wanted = {s.value for s in statuses}
        out: dict[str, DocProcessingStatus] = {}
        for doc_id, row in self.data.items():
            raw_status = row.get("status")
            value = (
                raw_status.value
                if isinstance(raw_status, DocStatus)
                else str(raw_status)
            )
            if value not in wanted:
                continue
            out[doc_id] = DocProcessingStatus(
                content_summary=row.get("content_summary", ""),
                content_length=row.get("content_length", 0),
                file_path=row.get("file_path", "unknown_source"),
                status=DocStatus(value),
                created_at=row.get("created_at", ""),
                updated_at=row.get("updated_at", ""),
                track_id=row.get("track_id"),
                metadata=row.get("metadata", {}) or {},
            )
        return out

    async def get_docs_by_track_id(
        self, track_id: str
    ) -> dict[str, DocProcessingStatus]:  # pragma: no cover - unused
        return {}

    async def get_docs_paginated(
        self,
        status_filter=None,
        status_filters=None,
        page: int = 1,
        page_size: int = 50,
        sort_field: str = "updated_at",
        sort_direction: str = "desc",
    ):  # pragma: no cover - unused
        return [], 0

    async def get_all_status_counts(self) -> dict[str, int]:
        return await self.get_status_counts()

    async def get_doc_by_file_path(
        self, file_path: str
    ) -> dict[str, Any] | None:  # pragma: no cover - unused
        return None

    async def get_doc_by_file_basename(
        self, basename: str
    ) -> tuple[str, dict[str, Any]] | None:
        # Legacy signature ON PURPOSE: a strict kwarg here must never appear.
        for doc_id, row in self.data.items():
            if row.get("file_path") == basename:
                return doc_id, dict(row)
        return None

    async def get_doc_by_content_hash(
        self, content_hash: str
    ) -> tuple[str, dict[str, Any]] | None:  # pragma: no cover - unused
        return None


def _row(status: DocStatus, created_at: str = "2026-01-01T00:00:00") -> dict:
    return {
        "status": status.value,
        "content_summary": "s",
        "content_length": 1,
        "file_path": "a.pdf",
        "created_at": created_at,
        "updated_at": created_at,
        "track_id": "t1",
        "metadata": {},
    }


def _storage(**rows: dict) -> _MinimalDocStatusStorage:
    storage = _MinimalDocStatusStorage()
    storage.data.update(rows)
    return storage


def test_minimal_subclass_instantiates_with_defaults():
    storage = _storage()
    assert storage.supports_bounded_scheduling_pages is False
    assert storage.supports_failure_generation is False
    assert storage.supports_strict_doc_identity_lookup is False
    assert storage.supports_strict_point_reads is False


def test_page_default_start_is_single_terminal_page():
    async def _run():
        storage = _storage(
            d1=_row(DocStatus.PENDING),
            d2=_row(DocStatus.FAILED),
            d3=_row(DocStatus.PROCESSED),
        )
        page = await storage.get_docs_by_statuses_page(
            [DocStatus.PENDING, DocStatus.FAILED], limit=1, strict=True
        )
        # limit is ignored by the compat default: everything in one page.
        assert set(page.docs) == {"d1", "d2"}
        assert page.next_position is CURSOR_END
        record = page.docs["d1"]
        assert isinstance(record, DocSchedulingRecord)
        assert record.status is DocStatus.PENDING
        assert record.has_custom_chunk_journal is False

    asyncio.run(_run())


def test_page_default_non_start_positions_terminate_empty():
    async def _run():
        storage = _storage(d1=_row(DocStatus.PENDING))
        for position in (CURSOR_END, CursorAfter("opaque")):
            page = await storage.get_docs_by_statuses_page(
                [DocStatus.PENDING], limit=10, position=position
            )
            assert page.docs == {}
            assert page.next_position is CURSOR_END

    asyncio.run(_run())


def test_page_default_ignores_generation_filter_with_warning(caplog):
    async def _run():
        storage = _storage(d1=_row(DocStatus.FAILED))
        # Rows lack failure_generation entirely — must NOT raise, must NOT
        # filter (missing == logical 0 keeps legacy rows eligible).
        page = await storage.get_docs_by_statuses_page(
            [DocStatus.FAILED], limit=10, max_failure_generation=5
        )
        assert set(page.docs) == {"d1"}

    import logging

    with caplog.at_level(logging.WARNING, logger="lightrag"):
        logger_obj = logging.getLogger("lightrag")
        old_propagate = logger_obj.propagate
        logger_obj.propagate = True
        try:
            asyncio.run(_run())
        finally:
            logger_obj.propagate = old_propagate
    assert any("max_failure_generation" in r.message for r in caplog.records)


def test_page_default_journal_projection():
    async def _run():
        row = _row(DocStatus.PENDING)
        row["metadata"] = {CUSTOM_CHUNK_PATCH_METADATA_KEY: {"op": "x"}}
        storage = _storage(d1=row)
        page = await storage.get_docs_by_statuses_page([DocStatus.PENDING], limit=10)
        assert page.docs["d1"].has_custom_chunk_journal is True

    asyncio.run(_run())


def test_count_default_raises_capability_error():
    async def _run():
        storage = _storage()
        with pytest.raises(StorageCapabilityError):
            await storage.count_docs_by_statuses([DocStatus.PENDING])

    asyncio.run(_run())


def test_update_fields_refuses_created_at_and_missing_ids():
    async def _run():
        storage = _storage(d1=_row(DocStatus.PENDING))
        with pytest.raises(ValueError, match="created_at"):
            await storage.update_doc_status_fields(
                "d1", {"created_at": "2030-01-01T00:00:00"}
            )
        with pytest.raises(StorageRecordNotFoundError):
            await storage.update_doc_status_fields("missing", {"error_msg": "x"})
        # missing_ok best-effort path is a no-op.
        await storage.update_doc_status_fields(
            "missing", {"error_msg": "x"}, missing_ok=True
        )
        await storage.update_doc_status_fields("d1", {"error_msg": "boom"})
        assert storage.data["d1"]["error_msg"] == "boom"
        assert storage.data["d1"]["status"] == DocStatus.PENDING.value

    asyncio.run(_run())


def test_mark_doc_failed_default_is_legacy_write_side():
    async def _run():
        storage = _storage(d1=_row(DocStatus.PENDING, created_at="2026-01-01T00:00:00"))
        generation = await storage.mark_doc_failed(
            "d1",
            {
                "error_msg": "boom",
                "updated_at": "2026-02-01T00:00:00",
                # Caller-supplied created_at MUST be ignored for existing rows.
                "created_at": "2030-12-31T00:00:00",
            },
        )
        assert generation is None  # LEGACY: no generation
        row = storage.data["d1"]
        assert row["status"] == DocStatus.FAILED
        assert row["error_msg"] == "boom"
        assert row["created_at"] == "2026-01-01T00:00:00"

        # Missing row: conditional create (enqueue-time errors can fail
        # before the PENDING row landed).
        await storage.mark_doc_failed(
            "ghost", {"error_msg": "early", "created_at": "2026-03-01T00:00:00"}
        )
        assert storage.data["ghost"]["status"] == DocStatus.FAILED
        assert storage.data["ghost"]["created_at"] == "2026-03-01T00:00:00"

    asyncio.run(_run())


def test_ensure_processing_attempt_id_mints_once_then_reuses():
    async def _run():
        storage = _storage(d1=_row(DocStatus.PENDING))
        first = await storage.ensure_processing_attempt_id("d1")
        assert first
        second = await storage.ensure_processing_attempt_id("d1")
        assert second == first
        assert storage.data["d1"]["processing_attempt_id"] == first
        with pytest.raises(StorageRecordNotFoundError):
            await storage.ensure_processing_attempt_id("missing")

    asyncio.run(_run())


def test_failure_generation_defaults_are_legacy_and_capability_gated():
    async def _run():
        storage = _storage()
        assert (
            await storage.get_failure_generation_mode() is FailureGenerationMode.LEGACY
        )
        with pytest.raises(StorageCapabilityError):
            await storage.reserve_failure_generation()

    asyncio.run(_run())


def test_strict_basename_delegates_without_new_kwargs():
    async def _run():
        storage = _storage(d1=_row(DocStatus.PENDING))
        # The legacy override has the OLD signature — delegation must not
        # pass any new kwargs (a strict= kwarg would TypeError here).
        match = await storage.get_doc_by_file_basename_strict("a.pdf")
        assert match is not None and match[0] == "d1"
        assert await storage.get_doc_by_file_basename_strict("missing.pdf") is None

    asyncio.run(_run())


def test_kv_get_by_id_strict_default_raises_capability_error():
    async def _run():
        storage = _storage(d1=_row(DocStatus.PENDING))
        with pytest.raises(StorageCapabilityError):
            await storage.get_by_id_strict("d1")

    asyncio.run(_run())
