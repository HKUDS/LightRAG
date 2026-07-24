"""content-hash duplicate detection is bounded (LR2 Phase 2.5).

``get_duplicate_doc_by_content_hash`` used to fall back to a full
``get_docs_by_statuses(list(DocStatus))`` scan whenever the indexed lookup
returned the current doc itself (the common case: a unique/original doc whose
own content_hash is already persisted). Phase 2.5 pushes the self-exclusion
INTO the query via ``get_doc_by_content_hash(..., exclude_doc_id=...)`` and
deletes the fallback, so the check is a single bounded lookup — verified here
against the JSON backend end-to-end, including that the fallback is gone.
"""

from __future__ import annotations

import asyncio

import pytest

from lightrag.base import DocStatus
from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils_pipeline import get_duplicate_doc_by_content_hash

pytestmark = pytest.mark.offline


class _DummyEmbeddingFunc:
    embedding_dim = 8

    async def __call__(self, texts):
        return [[0.0] * 8 for _ in texts]


@pytest.fixture(autouse=True)
def _shared():
    initialize_share_data()
    yield
    finalize_share_data()


async def _storage(tmp_path, rows: dict) -> JsonDocStatusStorage:
    storage = JsonDocStatusStorage(
        namespace="doc_status",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    await storage.initialize()
    async with storage._storage_lock:
        storage._data.update(rows)
    return storage


def _row(content_hash: str, created_at: str) -> dict:
    return {
        "content_summary": "s",
        "content_length": 3,
        "file_path": "f.txt",
        "status": DocStatus.PROCESSED,
        "created_at": created_at,
        "updated_at": created_at,
        "metadata": {},
        "error_msg": None,
        "chunks_list": [],
        "content_hash": content_hash,
    }


def test_get_doc_by_content_hash_excludes_self(tmp_path):
    async def _run():
        storage = await _storage(
            tmp_path,
            {
                "doc-early": _row("hashX", "2026-01-01T00:00:00"),
                "doc-late": _row("hashX", "2026-02-01T00:00:00"),
                "doc-solo": _row("hashY", "2026-01-01T00:00:00"),
            },
        )
        # Without exclusion the earliest holder wins.
        got = await storage.get_doc_by_content_hash("hashX")
        assert got is not None and got[0] == "doc-early"
        # Excluding the earliest yields the OTHER holder, not None.
        got = await storage.get_doc_by_content_hash("hashX", exclude_doc_id="doc-early")
        assert got is not None and got[0] == "doc-late"
        # Excluding the sole holder of a hash yields None (no other match).
        assert (
            await storage.get_doc_by_content_hash("hashY", exclude_doc_id="doc-solo")
            is None
        )

    asyncio.run(_run())


def test_duplicate_check_finds_other_and_ignores_self(tmp_path):
    async def _run():
        storage = await _storage(
            tmp_path,
            {
                "doc-orig": _row("dup", "2026-01-01T00:00:00"),
                "doc-copy": _row("dup", "2026-02-01T00:00:00"),
                "doc-unique": _row("uniq", "2026-01-01T00:00:00"),
            },
        )
        # The later copy finds the original.
        match = await get_duplicate_doc_by_content_hash(storage, "dup", "doc-copy")
        assert match is not None and match[0] == "doc-orig"
        # The original, checking itself, finds the copy (another holder exists).
        match = await get_duplicate_doc_by_content_hash(storage, "dup", "doc-orig")
        assert match is not None and match[0] == "doc-copy"
        # A unique doc (only itself holds the hash) → no duplicate.
        assert (
            await get_duplicate_doc_by_content_hash(storage, "uniq", "doc-unique")
            is None
        )

    asyncio.run(_run())


def test_duplicate_check_never_full_scans_all_statuses(tmp_path):
    """The Phase 2.5 guarantee: the dedup path no longer calls
    get_docs_by_statuses (the removed O(entire-store) fallback)."""

    async def _run():
        storage = await _storage(
            tmp_path, {"doc-unique": _row("uniq", "2026-01-01T00:00:00")}
        )

        async def _forbidden(*args, **kwargs):
            raise AssertionError(
                "get_duplicate_doc_by_content_hash must not full-scan "
                "doc_status (the removed fallback)"
            )

        storage.get_docs_by_statuses = _forbidden

        # Unique doc → the pre-Phase-2.5 code would hit the fallback here.
        assert (
            await get_duplicate_doc_by_content_hash(storage, "uniq", "doc-unique")
            is None
        )

    asyncio.run(_run())
