"""Coverage for PGDocStatusStorage.get_docs_by_track_id hydration resilience.

WHY: this path reimplemented the row hydration its sibling reads already share
via ``_pg_doc_processing_status_from_row`` and wrapped it in **no** try/except at
all. One row with a missing or renamed column — schema drift after an upgrade or
rollback — therefore aborted the listing for every document sharing that
track_id, surfacing as an HTTP 500 from ``/documents/track_status``.

No live PostgreSQL: ``db.query`` is mocked, which is the whole dependency of this
method.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip(
    "asyncpg",
    reason="asyncpg is required for PostgreSQL storage tests",
)

from lightrag.kg.postgres_impl import PGDocStatusStorage  # noqa: E402
from lightrag.namespace import NameSpace  # noqa: E402

pytestmark = pytest.mark.offline


def _make_storage(rows) -> PGDocStatusStorage:
    storage = PGDocStatusStorage.__new__(PGDocStatusStorage)
    storage.namespace = NameSpace.DOC_STATUS
    storage.workspace = "test_ws"
    storage.global_config = {"embedding_batch_num": 10}
    storage.db = MagicMock()
    storage.db.query = AsyncMock(return_value=rows)
    return storage


def _row(doc_id: str = "doc-1", **overrides):
    base = {
        "id": doc_id,
        "content_summary": "summary",
        "content_length": 12,
        "chunks_count": 1,
        "status": "processed",
        "file_path": "report.pdf",
        "chunks_list": "[]",
        "metadata": "{}",
        "error_msg": None,
        "track_id": "track_123",
        "content_hash": "abc123",
        "created_at": datetime(2024, 1, 1, 0, 0, 0),
        "updated_at": datetime(2024, 1, 1, 0, 0, 0),
    }
    base.update(overrides)
    return base


async def test_track_id_listing_survives_a_row_missing_a_column():
    """Schema drift: a row that predates a column the hydration reads."""
    bad = _row("doc-bad")
    del bad["content_summary"]
    storage = _make_storage([_row("doc-good"), bad])

    docs = await storage.get_docs_by_track_id("track_123")

    assert "doc-good" in docs
    assert "doc-bad" not in docs


async def test_an_unknown_column_is_ignored_rather_than_fatal():
    """PG diverges from the ``**row`` backends, by construction.

    ``_pg_doc_processing_status_from_row`` names every kwarg explicitly instead
    of splatting the row, so a column the running build does not declare is
    simply not read. The version-rollback shape that used to break
    JSON/Redis/Mongo (``TypeError: unexpected keyword argument``) never arose
    here — the other backends now match via
    ``DocProcessingStatus.from_stored``. Pinned so a refactor to
    ``DocProcessingStatus(**element)`` cannot silently import that failure
    mode into PG.
    """
    storage = _make_storage(
        [_row("old_doc"), _row("written_by_newer_build", field_from_the_future="value")]
    )

    docs = await storage.get_docs_by_track_id("track_123")

    assert set(docs) == {"old_doc", "written_by_newer_build"}


async def test_one_bad_row_does_not_hide_its_healthy_siblings():
    """track_id groups a whole upload batch — a single unusable row must not
    take the rest of the batch down with it."""
    bad = _row("doc-bad")
    del bad["chunks_count"]
    storage = _make_storage([_row(f"doc-{i}") for i in range(5)] + [bad])

    docs = await storage.get_docs_by_track_id("track_123")

    assert set(docs) == {f"doc-{i}" for i in range(5)}


async def test_hydration_decodes_json_columns_and_normalizes_file_path():
    """Delegating to the shared helper must preserve what the inline code did
    (chunks_list / metadata JSON decode, timestamp formatting) — and it also
    picks up the helper's stricter file_path fallback, which maps an EMPTY
    file_path to the sentinel where the inline version only mapped None."""
    storage = _make_storage(
        [
            _row(
                "doc-1",
                chunks_list='["chunk-a", "chunk-b"]',
                metadata='{"k": "v"}',
                file_path="",
            )
        ]
    )

    docs = await storage.get_docs_by_track_id("track_123")

    doc = docs["doc-1"]
    assert doc.chunks_list == ["chunk-a", "chunk-b"]
    assert doc.metadata == {"k": "v"}
    assert doc.file_path == "no-file-path"
    assert doc.content_hash == "abc123"
    assert doc.created_at.startswith("2024-01-01")


async def test_empty_result_set_returns_empty_dict():
    """``db.query`` returning None (no rows) must not blow up the iteration."""
    storage = _make_storage(None)

    assert await storage.get_docs_by_track_id("track_123") == {}
