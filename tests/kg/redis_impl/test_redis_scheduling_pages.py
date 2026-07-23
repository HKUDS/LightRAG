"""RedisDocStatusStorage Phase 1 scheduling contract tests (offline fake).

Covers: sidecar migration on initialize (streaming build + marker published
last), per-status ZSET keyset pages with the composite per-status cursor and
consumed-position advance, O(1) ZCARD counts, atomic WATCH/MULTI writes
(status transitions move ZSET members; the basename primary index follows
the eligibility state machine — including the post-parse content-duplicate
in-place transition), reserve-before-publish with zero window in
mark_doc_failed, per-attempt idempotency, and fail-closed strict lookups.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from redis.exceptions import RedisError

from lightrag.base import (
    CURSOR_END,
    CURSOR_START,
    CursorAfter,
    DocStatus,
    FailureGenerationMode,
)
from lightrag.exceptions import (
    StorageControlPlaneError,
    StorageRecordNotFoundError,
)
from lightrag.namespace import NameSpace

from .fake_redis import FakeRedis

pytestmark = pytest.mark.offline


class _DummyEmbeddingFunc:
    embedding_dim = 1
    max_token_size = 1

    async def __call__(self, texts, **kwargs):
        return [[0.0] for _ in texts]


def _doc(
    status: str,
    file_path: str = "a.pdf",
    created_at: str = "2026-01-01T00:00:00+00:00",
    **extra,
) -> dict:
    row = {
        "content_summary": "s",
        "content_length": 10,
        "file_path": file_path,
        "status": status,
        "created_at": created_at,
        "updated_at": created_at,
        "metadata": {},
        "error_msg": None,
        "chunks_list": [],
    }
    row.update(extra)
    return row


@pytest.fixture
def storage(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(
        "lightrag.kg.redis_impl.RedisConnectionManager.get_pool",
        lambda redis_url: MagicMock(name="fake_pool"),
    )
    monkeypatch.setattr(
        "lightrag.kg.redis_impl.Redis", lambda connection_pool=None, **_: fake
    )
    from lightrag.kg.redis_impl import RedisDocStatusStorage

    instance = RedisDocStatusStorage(
        namespace=NameSpace.DOC_STATUS,
        global_config={},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    instance._initialized = True
    return instance


async def _bootstrap(storage):
    """Publish the ctrl marker the way initialize()'s migration would."""
    await storage._migrate_scheduling_sidecar()


async def _sweep_ids(storage, statuses, *, limit, max_failure_generation=None):
    ids: list[str] = []
    pages = 0
    position = CURSOR_START
    while True:
        page = await storage.get_docs_by_statuses_page(
            statuses,
            limit=limit,
            position=position,
            max_failure_generation=max_failure_generation,
            strict=True,
        )
        pages += 1
        ids.extend(page.docs.keys())
        if page.next_position is CURSOR_END:
            return ids, pages
        position = page.next_position
        assert isinstance(position, CursorAfter)
        assert pages < 100, "sweep failed to terminate"


@pytest.mark.asyncio
async def test_migration_builds_sidecar_and_publishes_marker_last(storage):
    # Pre-existing deployment: raw rows only, no sidecar.
    fake = storage._redis
    fake.store[f"{storage.final_namespace}:doc-1"] = json.dumps(
        _doc("pending", file_path="a.pdf")
    )
    fake.store[f"{storage.final_namespace}:doc-2"] = json.dumps(
        _doc("failed", file_path="b.pdf", failure_generation=7)
    )
    dup = _doc("failed", file_path="a.pdf")
    dup["metadata"] = {"is_duplicate": True}
    fake.store[f"{storage.final_namespace}:dup-1"] = json.dumps(dup)

    await _bootstrap(storage)

    assert await storage.get_failure_generation_mode() is FailureGenerationMode.ENFORCED
    # Counter calibrated to max persisted generation.
    assert await storage.reserve_failure_generation() == 8
    # ZSETs populated; duplicate marker rows are NOT in the basename index.
    assert await storage.count_docs_by_statuses([DocStatus.PENDING]) == 1
    assert await storage.count_docs_by_statuses([DocStatus.FAILED]) == 2
    match = await storage.get_doc_by_file_basename_strict("a.pdf")
    assert match is not None and match[0] == "doc-1"
    # Re-running the migration is a no-op (marker present).
    await _bootstrap(storage)
    assert await storage.count_docs_by_statuses([DocStatus.PENDING]) == 1


@pytest.mark.asyncio
async def test_missing_marker_reads_migrating_never_legacy(storage):
    assert (
        await storage.get_failure_generation_mode() is FailureGenerationMode.MIGRATING
    )
    with pytest.raises(StorageControlPlaneError):
        await storage.reserve_failure_generation()


@pytest.mark.asyncio
async def test_page_kway_merge_and_consumed_position(storage):
    await _bootstrap(storage)
    await storage.upsert(
        {
            "doc-c": _doc("pending", created_at="2026-01-01T00:00:00+00:00"),
            "doc-a": _doc("failed", created_at="2026-01-02T00:00:00+00:00"),
            "doc-b": _doc("pending", created_at="2026-01-03T00:00:00+00:00"),
        }
    )
    ids, pages = await _sweep_ids(
        storage, [DocStatus.PENDING, DocStatus.FAILED], limit=1
    )
    # Global (created_at, id) order across BOTH status streams.
    assert ids == ["doc-c", "doc-a", "doc-b"]
    assert pages >= 3


@pytest.mark.asyncio
async def test_page_prefetched_head_not_consumed(storage):
    """With limit=1 and two streams, the losing stream's prefetched head is
    NOT consumed — it must reappear on the next page (no skips)."""
    await _bootstrap(storage)
    await storage.upsert(
        {
            "doc-p": _doc("pending", created_at="2026-01-01T00:00:00+00:00"),
            "doc-f": _doc("failed", created_at="2026-01-01T00:00:00+00:00"),
        }
    )
    page1 = await storage.get_docs_by_statuses_page(
        [DocStatus.PENDING, DocStatus.FAILED], limit=1, strict=True
    )
    # (same created_at) id tie-break: doc-f < doc-p
    assert list(page1.docs) == ["doc-f"]
    page2 = await storage.get_docs_by_statuses_page(
        [DocStatus.PENDING, DocStatus.FAILED],
        limit=1,
        position=page1.next_position,
        strict=True,
    )
    assert list(page2.docs) == ["doc-p"]


@pytest.mark.asyncio
async def test_generation_filtered_page_advances_without_false_end(storage):
    await _bootstrap(storage)
    rows = {
        f"doc-{i:02d}": _doc(
            "failed",
            created_at=f"2026-01-01T00:00:0{i}+00:00",
            failure_generation=100,
        )
        for i in range(4)
    }
    rows["doc-99"] = _doc(
        "failed", created_at="2026-01-05T00:00:00+00:00", failure_generation=1
    )
    await storage.upsert(rows)
    ids, pages = await _sweep_ids(
        storage, [DocStatus.FAILED], limit=2, max_failure_generation=5
    )
    assert ids == ["doc-99"]
    assert pages >= 3


@pytest.mark.asyncio
async def test_status_transition_moves_zset_membership(storage):
    await _bootstrap(storage)
    await storage.upsert({"doc-1": _doc("pending")})
    await storage.update_doc_status_fields("doc-1", {"status": "processing"})
    assert await storage.count_docs_by_statuses([DocStatus.PENDING]) == 0
    assert await storage.count_docs_by_statuses([DocStatus.PROCESSING]) == 1
    ids, _ = await _sweep_ids(storage, [DocStatus.PROCESSING], limit=10)
    assert ids == ["doc-1"]
    with pytest.raises(ValueError, match="created_at"):
        await storage.update_doc_status_fields("doc-1", {"created_at": "2030-01-01"})
    with pytest.raises(StorageRecordNotFoundError):
        await storage.update_doc_status_fields("missing", {"status": "pending"})


@pytest.mark.asyncio
async def test_basename_index_eligibility_state_machine(storage):
    await _bootstrap(storage)
    await storage.upsert({"doc-1": _doc("pending", file_path="a.pdf")})
    assert (await storage.get_doc_by_file_basename_strict("a.pdf"))[0] == "doc-1"

    # eligible → ineligible IN PLACE: the primary row is rewritten as a
    # post-parse content duplicate — its own mapping must be released.
    dup_row = _doc("failed", file_path="a.pdf")
    dup_row["metadata"] = {"is_duplicate": True, "duplicate_kind": "content_hash"}
    await storage.upsert({"doc-1": dup_row})
    assert await storage.get_doc_by_file_basename_strict("a.pdf") is None

    # A fresh ingestion may now legitimately claim the basename.
    await storage.upsert({"doc-2": _doc("pending", file_path="a.pdf")})
    assert (await storage.get_doc_by_file_basename_strict("a.pdf"))[0] == "doc-2"

    # Deleting a NON-owning row (the duplicate marker) must not strip the
    # owner's mapping.
    await storage.delete(["doc-1"])
    assert (await storage.get_doc_by_file_basename_strict("a.pdf"))[0] == "doc-2"

    # Deleting the owner frees the name.
    await storage.delete(["doc-2"])
    assert await storage.get_doc_by_file_basename_strict("a.pdf") is None


@pytest.mark.asyncio
async def test_mark_doc_failed_atomic_reserve_and_attempt_idempotency(storage):
    await _bootstrap(storage)
    await storage.upsert(
        {
            "doc-1": _doc(
                "processing",
                created_at="2026-01-01T00:00:00+00:00",
                processing_attempt_id="attempt-1",
            )
        }
    )
    g1 = await storage.mark_doc_failed(
        "doc-1", {"error_msg": "boom", "created_at": "2030-01-01T00:00:00+00:00"}
    )
    row = await storage.get_by_id("doc-1")
    assert row["status"] == "failed"
    assert row["failure_generation"] == g1 >= 1
    assert row["failure_attempt_id"] == "attempt-1"
    assert row["created_at"] == "2026-01-01T00:00:00+00:00"  # caller's ignored
    # ZSET membership moved processing → failed atomically.
    assert await storage.count_docs_by_statuses([DocStatus.PROCESSING]) == 0
    assert await storage.count_docs_by_statuses([DocStatus.FAILED]) == 1

    # Same attempt again: idempotent, no new generation.
    assert await storage.mark_doc_failed("doc-1", {"error_msg": "boom2"}) == g1

    # New attempt: fresh, larger generation.
    await storage.update_doc_status_fields(
        "doc-1", {"status": "processing", "processing_attempt_id": "attempt-2"}
    )
    g2 = await storage.mark_doc_failed("doc-1", {"error_msg": "boom3"})
    assert g2 > g1

    # Missing row: conditional create.
    g3 = await storage.mark_doc_failed(
        "ghost",
        {
            **_doc("processing", file_path="g.pdf"),
            "processing_attempt_id": "attempt-x",
            "error_msg": "early",
        },
    )
    ghost = await storage.get_by_id("ghost")
    assert ghost["status"] == "failed" and ghost["failure_generation"] == g3 > g2


@pytest.mark.asyncio
async def test_mark_doc_failed_watch_conflict_retries_without_gap(storage):
    """A concurrent ctrl bump between read and commit forces a WATCH retry;
    the final generation reflects the interleaved reservation (no reuse)."""
    await _bootstrap(storage)
    await storage.upsert({"doc-1": _doc("processing", processing_attempt_id="a1")})
    fake = storage._redis

    original = FakeRedis.hgetall
    bumped = {"done": False}

    async def hgetall_with_interleaving(self, key):
        result = await original(self, key)
        if not bumped["done"] and key.endswith(":ctrl"):
            bumped["done"] = True
            # Interleave a concurrent reservation AFTER this read: bump the
            # version so the WATCH transaction conflicts and retries.
            self.hashes[key]["failure_generation_counter"] = str(
                int(result.get("failure_generation_counter") or 0) + 1
            )
            self._bump(key)
        return result

    fake.hgetall = hgetall_with_interleaving.__get__(fake)
    try:
        generation = await storage.mark_doc_failed("doc-1", {"error_msg": "x"})
    finally:
        fake.hgetall = original.__get__(fake)
    # The interleaved reservation took 1, so ours is 2 — never a duplicate.
    assert generation == 2


@pytest.mark.asyncio
async def test_count_and_strict_lookup_fail_closed(storage):
    await _bootstrap(storage)
    await storage.upsert({"doc-1": _doc("pending")})
    fake = storage._redis

    fake.fail_next["zcard"] = RedisError("boom")
    with pytest.raises(RedisError):
        await storage.count_docs_by_statuses([DocStatus.PENDING])

    fake.fail_next["get"] = RedisError("boom")
    with pytest.raises(RedisError):
        await storage.get_doc_by_file_basename_strict("a.pdf")
    # The legacy method keeps the swallow-and-None behaviour.
    fake.fail_next["get"] = RedisError("boom")
    assert await storage.get_doc_by_file_basename("a.pdf") is None

    # Broken invariant (index points at a missing row) raises for strict.
    fake.store[storage._basename_key("ghost.pdf")] = "doc-ghost"
    with pytest.raises(StorageControlPlaneError):
        await storage.get_doc_by_file_basename_strict("ghost.pdf")


@pytest.mark.asyncio
async def test_ensure_processing_attempt_id(storage):
    await _bootstrap(storage)
    await storage.upsert({"doc-1": _doc("pending")})
    first = await storage.ensure_processing_attempt_id("doc-1")
    assert first and (await storage.ensure_processing_attempt_id("doc-1")) == first
    with pytest.raises(StorageRecordNotFoundError):
        await storage.ensure_processing_attempt_id("missing")


@pytest.mark.asyncio
async def test_malformed_cursor_raises_control_plane_error(storage):
    await _bootstrap(storage)
    with pytest.raises(StorageControlPlaneError):
        await storage.get_docs_by_statuses_page(
            [DocStatus.PENDING], limit=1, position=CursorAfter("[1,2]")
        )


@pytest.mark.asyncio
async def test_drop_clears_sidecar_but_keeps_counter(storage):
    await _bootstrap(storage)
    await storage.upsert({"doc-1": _doc("pending", file_path="a.pdf")})
    g1 = await storage.reserve_failure_generation()
    await storage.drop()
    assert await storage.count_docs_by_statuses([DocStatus.PENDING]) == 0
    assert await storage.get_doc_by_file_basename_strict("a.pdf") is None
    # Counter survives the destructive clear: monotonic, never reused.
    assert await storage.reserve_failure_generation() == g1 + 1
