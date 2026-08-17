"""Base-contract tests for the memory-bounding scheduling API (Phase 1).

doc_status backends are all first-party: the bounded-paging / strict-batch /
strict-point / source-resolution methods are ``@abstractmethod``, so a subclass
missing any of them cannot instantiate (instantiability IS the capability
guarantee — there are no capability flags and no degraded fallback).

The methods that DO keep a concrete base default are still exercised here via a
minimal subclass that implements the abstract surface:

* ``count_docs_by_statuses`` raises ``StorageCapabilityError`` (fail-closed);
* ``update_doc_status_fields`` refuses ``created_at`` and raises
  ``StorageRecordNotFoundError`` on unknown ids unless ``missing_ok``;
* ``list_source_conflicts_page`` / ``repair_source_conflict`` raise capability
  errors until a backend overrides them.
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from dataclasses import dataclass, field, fields
from typing import Any, Sequence

import pytest

from lightrag.base import (
    CURSOR_END,
    CursorPosition,
    DocProcessingStatus,
    DocSchedulingRecord,
    DocStatus,
    DocStatusPage,
    DocStatusStorage,
    SourceAbsent,
    SourceResolution,
)
from lightrag.exceptions import (
    StorageCapabilityError,
    StorageRecordNotFoundError,
)

pytestmark = pytest.mark.offline


@dataclass
class _MinimalDocStatusStorage(DocStatusStorage):
    """First-party-style subclass implementing the full abstract surface
    (legacy methods + the mandatory scheduling methods) trivially, so the
    concrete base defaults (count/update/list/repair) can be exercised."""

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

    # NOTE: get_by_id_strict is intentionally NOT implemented — it is an
    # optional KV capability (supports_strict_point_reads defaults False), so
    # the inherited base default (raise) applies.

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

    async def get_status_counts(self) -> dict[str, int]:  # pragma: no cover
        return {}

    async def get_docs_by_status(
        self, status: DocStatus
    ) -> dict[str, DocProcessingStatus]:  # pragma: no cover - unused
        return {}

    async def get_docs_by_statuses(
        self, statuses: list[DocStatus], strict: bool = False
    ) -> dict[str, DocProcessingStatus]:  # pragma: no cover - unused
        return {}

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

    async def get_all_status_counts(self) -> dict[str, int]:  # pragma: no cover
        return {}

    async def get_doc_by_file_path(
        self, file_path: str
    ) -> dict[str, Any] | None:  # pragma: no cover - unused
        return None

    async def get_doc_by_file_basename(
        self, basename: str
    ) -> tuple[str, dict[str, Any]] | None:  # pragma: no cover - unused
        return None

    async def get_doc_by_content_hash(
        self, content_hash: str
    ) -> tuple[str, dict[str, Any]] | None:  # pragma: no cover - unused
        return None

    # Mandatory scheduling surface — trivial concrete impls so the class is
    # instantiable (behaviour is covered by the per-backend test suites).
    async def get_docs_by_statuses_page(
        self, statuses, *, limit, position=None, strict=False
    ) -> DocStatusPage:  # pragma: no cover - trivial
        return DocStatusPage(docs={}, next_position=CURSOR_END)

    async def get_docs_by_ids(
        self, doc_ids: Sequence[str], *, strict: bool = False
    ) -> dict[str, DocSchedulingRecord]:  # pragma: no cover - trivial
        return {}

    async def get_full_docs_by_ids(
        self, doc_ids: Sequence[str], *, strict: bool = False
    ) -> dict[str, DocProcessingStatus]:  # pragma: no cover - trivial
        return {}

    async def resolve_doc_source_strict(
        self, canonical_source_key: str
    ) -> SourceResolution:  # pragma: no cover - trivial
        return SourceAbsent()


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


def test_scheduling_methods_are_mandatory_abstractmethods():
    # The doc_status scheduling surface is abstract, so a backend cannot
    # instantiate without implementing every one. get_by_id_strict is NOT here
    # — it is an optional KV capability gated by supports_strict_point_reads.
    assert "get_by_id_strict" not in DocStatusStorage.__abstractmethods__
    for name in (
        "get_docs_by_statuses_page",
        "get_docs_by_ids",
        "get_full_docs_by_ids",
        "resolve_doc_source_strict",
    ):
        assert name in DocStatusStorage.__abstractmethods__, name


def test_get_by_id_strict_is_optional_capability():
    async def _run():
        storage = _storage(d1=_row(DocStatus.PENDING))
        # Not implemented + capability defaults False → conservative raise
        # (a caller must gate on the flag and fall back to a safe path).
        assert storage.supports_strict_point_reads is False
        with pytest.raises(StorageCapabilityError):
            await storage.get_by_id_strict("d1")

    asyncio.run(_run())


def test_subclass_missing_an_abstract_cannot_instantiate():
    # Re-declaring a mandatory method as abstract makes ABCMeta recompute a
    # non-empty __abstractmethods__ → instantiation raises TypeError (there is
    # no silent degraded fallback to fall back to).
    class _Incomplete(_MinimalDocStatusStorage):
        @abstractmethod
        async def resolve_doc_source_strict(
            self, canonical_source_key: str
        ) -> SourceResolution: ...

    assert "resolve_doc_source_strict" in _Incomplete.__abstractmethods__
    with pytest.raises(TypeError):
        _Incomplete()


def test_minimal_subclass_instantiates():
    storage = _MinimalDocStatusStorage()
    assert isinstance(storage, DocStatusStorage)


def test_scheduling_record_projection_is_lightweight():
    projected = {f.name for f in fields(DocSchedulingRecord)}
    assert "chunks_list" not in projected
    assert "error_msg" not in projected
    assert "content_length" not in projected


def test_count_default_raises_capability_error():
    async def _run():
        storage = _storage(d1=_row(DocStatus.PENDING))
        with pytest.raises(StorageCapabilityError):
            await storage.count_docs_by_statuses([DocStatus.PENDING])

    asyncio.run(_run())


def test_update_fields_refuses_created_at_and_missing_ids():
    async def _run():
        storage = _storage(d1=_row(DocStatus.PENDING))
        with pytest.raises(ValueError):
            await storage.update_doc_status_fields(
                "d1", {"created_at": "2027-01-01T00:00:00"}
            )
        with pytest.raises(StorageRecordNotFoundError):
            await storage.update_doc_status_fields("missing", {"status": "failed"})
        await storage.update_doc_status_fields(
            "missing", {"status": "failed"}, missing_ok=True
        )
        await storage.update_doc_status_fields("d1", {"track_id": "t2"})
        assert storage.data["d1"]["track_id"] == "t2"
        assert storage.data["d1"]["created_at"] == "2026-01-01T00:00:00"

    asyncio.run(_run())


def test_source_conflict_methods_raise_capability_error():
    async def _run():
        storage = _storage(d1=_row(DocStatus.PENDING))
        with pytest.raises(StorageCapabilityError):
            await storage.list_source_conflicts_page(limit=10)
        with pytest.raises(StorageCapabilityError):
            await storage.repair_source_conflict(
                "a.pdf",
                primary_doc_id="d1",
                expected_candidate_count=2,
                expected_candidate_fingerprint="deadbeef",
            )

    asyncio.run(_run())


def test_cursor_position_type_available():
    # Sanity: the sealed cursor type is importable for type checks.
    assert issubclass(type(CURSOR_END), CursorPosition)
