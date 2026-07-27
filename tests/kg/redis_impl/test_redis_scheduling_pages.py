"""RedisDocStatusStorage Phase 1 scheduling contract tests (offline fake).

Covers: derived-sidecar rebuild on initialize (streaming build into a temp
keyspace + atomic switch), per-status ZSET keyset pages with the composite
per-status cursor and consumed-position advance, O(1) ZCARD counts, atomic
WATCH/MULTI writes (status transitions move ZSET members; the source multimap
follows the member-level eligibility state machine — including the post-parse
content-duplicate in-place transition), the typed conflict-aware source
resolver (Absent/Unique/Conflict with stale self-heal and early stop), the
strict batch read, the operator conflict listing + CAS repair, and fail-closed
strict lookups.
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
    SourceAbsent,
    SourceConflict,
    SourceUnique,
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


def _dup(status: str, file_path: str = "a.pdf", **extra) -> dict:
    row = _doc(status, file_path=file_path, **extra)
    row["metadata"] = {"is_duplicate": True}
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
    """Run the derived-sidecar rebuild the way initialize() would."""
    await storage._rebuild_scheduling_sidecar()


async def _sweep_ids(storage, statuses, *, limit):
    ids: list[str] = []
    pages = 0
    position = CURSOR_START
    while True:
        page = await storage.get_docs_by_statuses_page(
            statuses,
            limit=limit,
            position=position,
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
async def test_rebuild_builds_sidecar_from_primary_rows(storage):
    # Pre-existing deployment: raw rows only, no derived sidecar.
    fake = storage._redis
    fake.store[f"{storage.final_namespace}:doc-1"] = json.dumps(
        _doc("pending", file_path="a.pdf")
    )
    fake.store[f"{storage.final_namespace}:doc-2"] = json.dumps(
        _doc("failed", file_path="b.pdf")
    )
    fake.store[f"{storage.final_namespace}:dup-1"] = json.dumps(
        _dup("failed", file_path="a.pdf")
    )

    await _bootstrap(storage)

    # Status ZSETs populated from the actual rows.
    assert await storage.count_docs_by_statuses([DocStatus.PENDING]) == 1
    assert await storage.count_docs_by_statuses([DocStatus.FAILED]) == 2
    # Source multimap built; duplicate marker rows are NOT indexed.
    resolved = await storage.resolve_doc_source_strict("a.pdf")
    assert isinstance(resolved, SourceUnique) and resolved.doc_id == "doc-1"
    # No leftover rebuild temp keyspace after the atomic switch.
    assert not any(
        k.startswith(f"{storage._sched_prefix}_rebuild:") for k in fake.zsets
    )
    assert not any(k.startswith(f"{storage._sched_prefix}_rebuild:") for k in fake.sets)
    # Re-running the rebuild is a no-op (index already present).
    await _bootstrap(storage)
    assert await storage.count_docs_by_statuses([DocStatus.PENDING]) == 1


@pytest.mark.asyncio
async def test_publish_refuses_to_clobber_a_concurrent_writer(storage, monkeypatch):
    """Fix-proof: the switch used to DELETE the official keys and RENAME the
    snapshot over them unconditionally, so a write that landed after its row was
    scanned (it had already maintained the OFFICIAL index in the same
    transaction) was silently reverted — leaving the doc in the wrong status ZSET
    and out of the sweep. The publish must now refuse instead of losing it."""
    fake = storage._redis
    fake.store[f"{storage.final_namespace}:doc-1"] = json.dumps(
        _doc("pending", file_path="a.pdf")
    )

    # Land the racing write through the SCAN the rebuild itself drives, so this
    # test exercises whatever publish logic exists rather than asserting a
    # particular helper is present. The primary keyspace is scanned twice: once
    # by the pre-lock "are there rows to build from" probe, then by the snapshot
    # loop — the write has to land during the second one, after the snapshot has
    # already captured its slot.
    primary_pattern = f"{storage.final_namespace}:*"
    real_scan = fake.scan
    primary_scans = {"n": 0}

    async def scan_with_racing_write(
        cursor: int = 0, match: str = "", count: int = 1000
    ):
        result = await real_scan(cursor, match=match, count=count)
        if match == primary_pattern:
            primary_scans["n"] += 1
            if primary_scans["n"] == 2:
                # Another worker enqueues doc-new: its write commits the row and
                # the OFFICIAL sidecar entry in one transaction.
                await storage.upsert({"doc-new": _doc("pending", file_path="c.pdf")})
        return result

    monkeypatch.setattr(fake, "scan", scan_with_racing_write)

    # Old behaviour: no exception — the snapshot was published over the live
    # index. New behaviour: refuse, because the write cannot be preserved.
    with pytest.raises(StorageControlPlaneError, match="stale snapshot"):
        await storage._rebuild_scheduling_sidecar()

    # Either way, the concurrent write must still be in the official index.
    pending_members = fake.zsets[f"{storage._sched_prefix}:status:pending"]
    assert any("doc-new" in member for member in pending_members)


@pytest.mark.asyncio
async def test_sidecar_not_ready_raises_never_empty(storage):
    # Simulate a not-yet-initialized instance: strict reads must refuse rather
    # than read the empty index as confirmed absence.
    storage._initialized = False
    with pytest.raises(StorageControlPlaneError):
        await storage.count_docs_by_statuses([DocStatus.PENDING])
    with pytest.raises(StorageControlPlaneError):
        await storage.get_docs_by_statuses_page([DocStatus.PENDING], limit=1)
    with pytest.raises(StorageControlPlaneError):
        await storage.resolve_doc_source_strict("a.pdf")


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
async def test_upsert_with_a_live_docstatus_enum_is_discoverable(storage):
    """Fix-proof: production callers (e.g. ``apipeline_enqueue_documents``)
    write ``{"status": DocStatus.PENDING}`` — the live enum member, not its
    ``.value`` string. ``DocStatus`` mixes in ``str`` so ``json.dumps``
    happily persists the primary row as ``"pending"``, but the sidecar used
    to key its ZSET off a plain ``str()`` call, which goes through
    ``Enum.__str__`` and produces ``"DocStatus.PENDING"`` instead — filing
    the brand-new doc under a ZSET no page/count query ever reads. Every
    test above this one passes a pre-lowered plain string and would not have
    caught this."""
    await _bootstrap(storage)
    await storage.upsert({"doc-1": _doc(DocStatus.PENDING, file_path="a.pdf")})

    assert (await storage.get_by_id("doc-1"))["status"] == "pending"
    assert await storage.count_docs_by_statuses([DocStatus.PENDING]) == 1
    ids, _ = await _sweep_ids(storage, [DocStatus.PENDING], limit=10)
    assert ids == ["doc-1"]

    # A transition written with the live enum, too, must move the membership
    # rather than leaving it stranded under the mis-keyed bucket.
    await storage.update_doc_status_fields("doc-1", {"status": DocStatus.PROCESSING})
    assert await storage.count_docs_by_statuses([DocStatus.PENDING]) == 0
    assert await storage.count_docs_by_statuses([DocStatus.PROCESSING]) == 1


@pytest.mark.asyncio
async def test_atomic_write_retries_on_watch_conflict(storage):
    """A concurrent bump of the doc key between the WATCH read and EXEC forces
    a retry; the write still lands and the sidecar stays consistent."""
    await _bootstrap(storage)
    await storage.upsert({"doc-1": _doc("pending")})
    fake = storage._redis
    main_key = f"{storage.final_namespace}:doc-1"

    original = FakeRedis.get
    tripped = {"done": False}

    async def get_with_interleaving(self, key):
        result = await original(self, key)
        if not tripped["done"] and key == main_key:
            tripped["done"] = True
            # Interleave a concurrent write AFTER the WATCH snapshot read.
            self._bump(key)
        return result

    fake.get = get_with_interleaving.__get__(fake)
    try:
        await storage.update_doc_status_fields("doc-1", {"status": "processing"})
    finally:
        fake.get = original.__get__(fake)
    assert tripped["done"] is True
    assert (await storage.get_by_id("doc-1"))["status"] == "processing"
    assert await storage.count_docs_by_statuses([DocStatus.PROCESSING]) == 1
    assert await storage.count_docs_by_statuses([DocStatus.PENDING]) == 0


@pytest.mark.asyncio
async def test_atomic_write_transport_failure_propagates(storage):
    """A transport failure at EXEC propagates (not swallowed as WatchError);
    the transaction commits nothing."""
    await _bootstrap(storage)
    await storage.upsert({"doc-1": _doc("pending")})
    fake = storage._redis
    fake.fail_next["execute"] = RedisError("boom")
    with pytest.raises(RedisError):
        await storage.update_doc_status_fields("doc-1", {"status": "processing"})
    assert (await storage.get_by_id("doc-1"))["status"] == "pending"


@pytest.mark.asyncio
async def test_source_multimap_eligibility_state_machine(storage):
    await _bootstrap(storage)
    await storage.upsert({"doc-1": _doc("pending", file_path="a.pdf")})
    resolved = await storage.resolve_doc_source_strict("a.pdf")
    assert isinstance(resolved, SourceUnique) and resolved.doc_id == "doc-1"

    # eligible → ineligible IN PLACE: the primary row is rewritten as a
    # post-parse content duplicate — its own membership must be released.
    await storage.upsert(
        {"doc-1": _dup("failed", file_path="a.pdf", duplicate_kind="content_hash")}
    )
    assert isinstance(await storage.resolve_doc_source_strict("a.pdf"), SourceAbsent)

    # A fresh ingestion may now legitimately claim the basename.
    await storage.upsert({"doc-2": _doc("pending", file_path="a.pdf")})
    resolved = await storage.resolve_doc_source_strict("a.pdf")
    assert isinstance(resolved, SourceUnique) and resolved.doc_id == "doc-2"

    # Deleting a NON-owning row (the duplicate marker) must not strip the
    # surviving primary's membership.
    await storage.delete(["doc-1"])
    resolved = await storage.resolve_doc_source_strict("a.pdf")
    assert isinstance(resolved, SourceUnique) and resolved.doc_id == "doc-2"

    # Deleting the primary frees the name.
    await storage.delete(["doc-2"])
    assert isinstance(await storage.resolve_doc_source_strict("a.pdf"), SourceAbsent)


@pytest.mark.asyncio
async def test_resolve_source_conflict(storage):
    await _bootstrap(storage)
    await storage.upsert(
        {
            "doc-1": _doc("pending", file_path="a.pdf"),
            "doc-2": _doc("failed", file_path="a.pdf"),
        }
    )
    resolved = await storage.resolve_doc_source_strict("a.pdf")
    assert isinstance(resolved, SourceConflict)
    assert resolved.candidate_count is None  # only proved "at least two"
    assert set(resolved.sample_doc_ids) == {"doc-1", "doc-2"}


@pytest.mark.asyncio
async def test_resolve_source_stale_member_self_heals(storage):
    await _bootstrap(storage)
    await storage.upsert({"doc-1": _doc("pending", file_path="a.pdf")})
    # Inject a stale member whose primary row does not exist.
    fake = storage._redis
    set_key = storage._basename_key("a.pdf")
    fake.sets[set_key].add("gone")

    resolved = await storage.resolve_doc_source_strict("a.pdf")
    assert isinstance(resolved, SourceUnique) and resolved.doc_id == "doc-1"
    # The stale member was self-healed out of the set.
    assert "gone" not in fake.sets.get(set_key, set())


@pytest.mark.asyncio
async def test_resolve_source_stops_after_two_valid_candidates(storage):
    await _bootstrap(storage)
    await storage.upsert(
        {
            "doc-a": _doc("pending", file_path="a.pdf"),
            "doc-b": _doc("failed", file_path="a.pdf"),
        }
    )
    fake = storage._redis
    set_key = storage._basename_key("a.pdf")
    # A third (stale) member that sorts LAST: if the resolver stopped after two
    # valid candidates it is never examined, so it is not self-healed.
    fake.sets[set_key].add("zzz-stale")

    resolved = await storage.resolve_doc_source_strict("a.pdf")
    assert isinstance(resolved, SourceConflict)
    assert set(resolved.sample_doc_ids) == {"doc-a", "doc-b"}
    assert "zzz-stale" in fake.sets.get(set_key, set())


@pytest.mark.asyncio
async def test_get_docs_by_ids_present_and_missing(storage):
    await _bootstrap(storage)
    await storage.upsert({"doc-1": _doc("pending"), "doc-2": _doc("failed")})
    result = await storage.get_docs_by_ids(["doc-1", "doc-2", "ghost"], strict=True)
    assert set(result) == {"doc-1", "doc-2"}  # missing omitted, confirmed absent
    assert result["doc-2"].status is DocStatus.FAILED
    assert not hasattr(result["doc-1"], "chunks_list")


@pytest.mark.asyncio
async def test_conflict_listing_is_bounded_and_resumable(storage, monkeypatch):
    """Fix-proof: the listing used to materialize EVERY canonical key, validate
    every one with card>=2 across the whole workspace, build the complete
    conflict list and only then slice to `limit` — redoing all of it per page.
    It must now stop early and validate only what it actually surfaces."""
    await _bootstrap(storage)
    rows = {}
    # 40 non-conflicting sources plus 4 conflicting ones.
    for i in range(40):
        rows[f"solo-{i:03d}"] = _doc("pending", file_path=f"solo-{i:03d}.pdf")
    for i in range(4):
        rows[f"conf-{i}-a"] = _doc("pending", file_path=f"dup-{i}.pdf")
        rows[f"conf-{i}-b"] = _doc("failed", file_path=f"dup-{i}.pdf")
    await storage.upsert(rows)

    validated: list[str] = []
    real_valid = storage._valid_primary_ids

    async def counting_valid_primary_ids(redis, canonical):
        validated.append(canonical)
        return await real_valid(redis, canonical)

    monkeypatch.setattr(storage, "_valid_primary_ids", counting_valid_primary_ids)

    # Force a small batch so "stop once limit is reached" is reachable.
    monkeypatch.setattr(type(storage), "_CONFLICT_SCAN_BATCH", 8)

    page = await storage.list_source_conflicts_page(limit=1)
    assert len(page.conflicts) >= 1
    assert isinstance(page.next_position, CursorAfter)
    # Bounded work: it stopped instead of validating all 44 source keys. Only
    # keys surviving the SCARD>=2 prefilter are ever validated.
    assert len(validated) <= 8
    assert all(name.startswith("dup-") for name in validated)

    # Resuming from the cursor keeps making progress and terminates.
    seen = {c.canonical_source_key for c in page.conflicts}
    position = page.next_position
    for _ in range(20):
        nxt = await storage.list_source_conflicts_page(limit=1, position=position)
        seen.update(c.canonical_source_key for c in nxt.conflicts)
        if nxt.next_position is CURSOR_END:
            break
        position = nxt.next_position
    else:
        raise AssertionError("conflict paging failed to terminate")
    assert seen == {f"dup-{i}.pdf" for i in range(4)}


@pytest.mark.asyncio
async def test_get_full_docs_by_ids_present_and_missing(storage):
    await _bootstrap(storage)
    await storage.upsert({"doc-1": _doc("pending"), "doc-2": _doc("failed")})
    result = await storage.get_full_docs_by_ids(
        ["doc-1", "doc-2", "ghost"], strict=True
    )
    # Missing id omitted (confirmed absent); present ids hydrated.
    assert set(result) == {"doc-1", "doc-2"}
    # status is the raw str value (DocStatus is a str-enum) -> use ==.
    assert result["doc-2"].status == DocStatus.FAILED
    # FULL projection: fields the lightweight scheduling record omits.
    assert result["doc-1"].content_summary == "s"
    assert result["doc-1"].content_length == 10
    assert result["doc-1"].chunks_list == []


@pytest.mark.asyncio
async def test_get_full_docs_by_ids_strict_transport_error_raises(storage):
    await _bootstrap(storage)
    await storage.upsert({"doc-1": _doc("pending")})
    fake = storage._redis
    # Transport failure at the batched pipeline read fails the WHOLE call —
    # never a partial mapping.
    fake.fail_next["execute"] = RedisError("boom")
    with pytest.raises(RedisError):
        await storage.get_full_docs_by_ids(["doc-1"], strict=True)


@pytest.mark.asyncio
async def test_get_full_docs_by_ids_relaxed_skips_undecodable_row(storage):
    await _bootstrap(storage)
    await storage.upsert({"doc-1": _doc("pending")})
    fake = storage._redis
    # An undecodable primary row: relaxed mode logs + skips it, returning the
    # good ones.
    fake.store[f"{storage.final_namespace}:bad"] = "not-json{"
    result = await storage.get_full_docs_by_ids(["doc-1", "bad"], strict=False)
    assert set(result) == {"doc-1"}
    assert result["doc-1"].content_summary == "s"


@pytest.mark.asyncio
async def test_list_and_repair_source_conflict(storage):
    await _bootstrap(storage)
    await storage.upsert(
        {
            "doc-1": _doc("pending", file_path="a.pdf"),
            "doc-2": _doc("failed", file_path="a.pdf"),
            "doc-3": _doc("pending", file_path="a.pdf"),
            "solo": _doc("pending", file_path="b.pdf"),
        }
    )
    page = await storage.list_source_conflicts_page(limit=10)
    assert len(page.conflicts) == 1
    conflict = page.conflicts[0]
    assert conflict.canonical_source_key == "a.pdf"
    assert conflict.candidate_count == 3
    assert set(conflict.sample_doc_ids) == {"doc-1", "doc-2", "doc-3"}

    # dry-run reports without mutating.
    dry = await storage.repair_source_conflict(
        "a.pdf",
        primary_doc_id="doc-2",
        expected_candidate_count=0,
        expected_candidate_fingerprint="ignored-in-dry-run",
    )
    assert dry.committed is False
    assert dry.candidate_count == 3
    assert isinstance(await storage.resolve_doc_source_strict("a.pdf"), SourceConflict)

    # primary_doc_id must be a current candidate.
    with pytest.raises(ValueError):
        await storage.repair_source_conflict(
            "a.pdf",
            primary_doc_id="not-a-candidate",
            expected_candidate_count=dry.candidate_count,
            expected_candidate_fingerprint=dry.fingerprint,
            dry_run=False,
        )

    # stale expectation → CAS failure.
    with pytest.raises(StorageControlPlaneError):
        await storage.repair_source_conflict(
            "a.pdf",
            primary_doc_id="doc-2",
            expected_candidate_count=99,
            expected_candidate_fingerprint=dry.fingerprint,
            dry_run=False,
        )

    # correct expectation → commit; resolver returns the chosen primary.
    result = await storage.repair_source_conflict(
        "a.pdf",
        primary_doc_id="doc-2",
        expected_candidate_count=dry.candidate_count,
        expected_candidate_fingerprint=dry.fingerprint,
        dry_run=False,
    )
    assert result.committed is True
    resolved = await storage.resolve_doc_source_strict("a.pdf")
    assert isinstance(resolved, SourceUnique) and resolved.doc_id == "doc-2"
    # losers demoted to duplicates, content intact.
    for loser in ("doc-1", "doc-3"):
        row = await storage.get_by_id(loser)
        assert row["metadata"]["is_duplicate"] is True
        assert row["metadata"]["original_doc_id"] == "doc-2"
    # no more conflicts.
    page = await storage.list_source_conflicts_page(limit=10)
    assert page.conflicts == ()


@pytest.mark.asyncio
async def test_count_and_strict_lookup_fail_closed(storage):
    await _bootstrap(storage)
    await storage.upsert({"doc-1": _doc("pending")})
    fake = storage._redis

    fake.fail_next["zcard"] = RedisError("boom")
    with pytest.raises(RedisError):
        await storage.count_docs_by_statuses([DocStatus.PENDING])

    # The typed resolver propagates transport errors (never a false Absent).
    fake.fail_next["sscan"] = RedisError("boom")
    with pytest.raises(RedisError):
        await storage.resolve_doc_source_strict("a.pdf")
    # The legacy method keeps the swallow-and-None behaviour.
    fake.fail_next["sscan"] = RedisError("boom")
    assert await storage.get_doc_by_file_basename("a.pdf") is None


@pytest.mark.asyncio
async def test_malformed_cursor_raises_control_plane_error(storage):
    await _bootstrap(storage)
    with pytest.raises(StorageControlPlaneError):
        await storage.get_docs_by_statuses_page(
            [DocStatus.PENDING], limit=1, position=CursorAfter("[1,2]")
        )


@pytest.mark.asyncio
async def test_drop_clears_rows_and_sidecar(storage):
    await _bootstrap(storage)
    await storage.upsert({"doc-1": _doc("pending", file_path="a.pdf")})
    await storage.drop()
    assert await storage.count_docs_by_statuses([DocStatus.PENDING]) == 0
    assert isinstance(await storage.resolve_doc_source_strict("a.pdf"), SourceAbsent)


@pytest.mark.asyncio
async def test_the_rebuild_switch_is_bounded_not_one_transaction_per_workspace(
    storage, monkeypatch
):
    """Fix-proof: the switch listed EVERY official and temp key into Python lists
    and queued one DELETE/RENAME per key into a single MULTI, so both the client
    allocation and the server-side transaction buffer were O(total_docs) — during
    the very initialization this phase exists to bound. It must now cost
    O(_PUBLISH_BATCH)."""
    fake = storage._redis
    total = 5 * storage._PUBLISH_BATCH
    for index in range(total):
        fake.store[f"{storage.final_namespace}:doc-{index:05d}"] = json.dumps(
            _doc("pending", file_path=f"f{index}.pdf")
        )

    # Widest single round trip the SWITCH issues. Counted by RENAME because only
    # the switch renames — the snapshot build's own batches are bounded by its
    # SCAN count and are not what regressed.
    widest = {"renames": 0}
    real_pipeline = fake.pipeline

    def counting_pipeline(transaction: bool = True):
        pipe = real_pipeline(transaction=transaction)
        real_execute = pipe.execute

        async def _execute(raise_on_error: bool = True):
            renames = sum(1 for op in pipe._ops if op[0] == "rename")
            widest["renames"] = max(widest["renames"], renames)
            return await real_execute(raise_on_error=raise_on_error)

        pipe.execute = _execute
        return pipe

    monkeypatch.setattr(fake, "pipeline", counting_pipeline)

    await storage._rebuild_scheduling_sidecar()

    assert 0 < widest["renames"] <= storage._PUBLISH_BATCH, (
        f"a single round trip queued {widest['renames']} renames for {total} "
        "docs; the switch is not bounded"
    )
    # ...and it really did publish everything.
    members = fake.zsets[f"{storage._sched_prefix}:status:pending"]
    assert len(members) == total
    assert len(fake.sets) == total  # one basename set per document
    assert not [k for k in fake.zsets if "_rebuild:" in k]
    assert not [k for k in fake.sets if "_rebuild:" in k]


@pytest.mark.asyncio
async def test_publish_survives_a_duplicate_scan_return(storage, monkeypatch):
    """Fix-proof: SCAN's own contract allows an element to be returned more than
    once during one iteration, and the bounded publish hands SCAN a moving
    target ON PURPOSE — each RENAME deletes from the very pattern being walked,
    which is exactly the shape that provokes a repeat. A later batch in the
    SAME pass re-handed a key an earlier batch already renamed away must not
    crash the publish with an unhandled ResponseError("no such key")."""
    fake = storage._redis
    monkeypatch.setattr(storage, "_PUBLISH_BATCH", 1)
    for index in range(3):
        fake.store[f"{storage.final_namespace}:doc-{index}"] = json.dumps(
            _doc("pending", file_path=f"f{index}.pdf")
        )

    real_scan = fake.scan
    basename_pattern = f"{storage._sched_prefix}_rebuild:basename:*"
    state = {"first_key": None, "injected": False}

    async def duplicating_scan(cursor=0, match="", count=1000):
        next_cursor, keys = await real_scan(cursor, match=match, count=count)
        if match == basename_pattern:
            if cursor == 0 and keys and state["first_key"] is None:
                state["first_key"] = keys[0]
            elif (
                cursor != 0
                and keys
                and not state["injected"]
                and state["first_key"] is not None
            ):
                # The documented SCAN behaviour: hand back a key a PRIOR call
                # in this very pass already renamed away.
                state["injected"] = True
                keys = [state["first_key"], *keys]
        return next_cursor, keys

    monkeypatch.setattr(fake, "scan", duplicating_scan)

    await storage._rebuild_scheduling_sidecar()  # must not raise

    # Nothing lost, nothing left behind, the repeat was silently absorbed.
    assert not [k for k in fake.zsets if "_rebuild:" in k]
    assert not [k for k in fake.sets if "_rebuild:" in k]
    assert len(fake.sets) == 3  # one basename set per document
    assert len(fake.zsets[f"{storage._sched_prefix}:status:pending"]) == 3


@pytest.mark.asyncio
async def test_a_half_published_index_is_rebuilt_not_trusted(storage):
    """The cost of a non-atomic switch: a rebuilder that died mid-publish leaves
    some official keys switched and the rest absent, which "any status key exists"
    reads as healthy. The leftover temp keyspace is what distinguishes it, so a
    later startup must rebuild instead of serving the half-published index."""
    fake = storage._redis
    for index in range(3):
        fake.store[f"{storage.final_namespace}:doc-{index}"] = json.dumps(
            _doc("pending", file_path=f"f{index}.pdf")
        )
    # One document's entry published, the other two still in the temp keyspace —
    # exactly the state a killed rebuilder leaves.
    fake.zsets[f"{storage._sched_prefix}:status:pending"] = {"x|doc-0"}
    fake.zsets[f"{storage._sched_prefix}_rebuild:status:pending"] = {"x|doc-1"}

    await storage._rebuild_scheduling_sidecar()

    # Rebuilt from the rows: all three documents are schedulable again.
    members = fake.zsets[f"{storage._sched_prefix}:status:pending"]
    assert len(members) == 3
    assert not [k for k in fake.zsets if "_rebuild:" in k]
