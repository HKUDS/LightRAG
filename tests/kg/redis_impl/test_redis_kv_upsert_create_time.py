"""Regression tests: RedisKVStorage replacement upserts preserve create_time.

Issue #3870: an update whose payload carries business fields only used to drop
the storage-managed ``create_time``. Redis cannot read one JSON field, so the
value is serialized ``create_time``-first and the previous timestamp is
recovered with a bounded ``GETRANGE`` prefix read -- see
``BaseKVStorage.upsert`` for the contract these tests pin, and
``_dumps_create_time_first`` for why the ordering is only an optimization.

Runs against the in-memory FakeRedis stand-in (no live Redis required);
``tests/kg/redis_impl/test_redis_kv_create_time_integration.py`` re-checks the
same invariants against a real server.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.namespace import NameSpace

from .fake_redis import FakeRedis

pytestmark = pytest.mark.offline


class _DummyEmbeddingFunc:
    embedding_dim = 1
    max_token_size = 1

    async def __call__(self, texts, **kwargs):
        return [[0.0] for _ in texts]


@pytest.fixture(autouse=True)
def _shared():
    initialize_share_data()
    yield
    finalize_share_data()


@pytest.fixture(autouse=True)
def _reset_eviction_cache():
    from lightrag.kg import redis_impl

    cache = getattr(redis_impl, "_eviction_checked", None)
    if cache is not None:
        cache.clear()
    yield
    if cache is not None:
        cache.clear()


@pytest.fixture
def fake(monkeypatch):
    instance = FakeRedis()
    monkeypatch.setattr(
        "lightrag.kg.redis_impl.RedisConnectionManager.get_pool",
        lambda redis_url: MagicMock(name="fake_pool"),
    )
    monkeypatch.setattr(
        "lightrag.kg.redis_impl.Redis", lambda connection_pool=None, **_: instance
    )
    return instance


def _kv_storage(workspace: str = "ct-ws"):
    from lightrag.kg.redis_impl import RedisKVStorage

    return RedisKVStorage(
        namespace=NameSpace.KV_STORE_ENTITY_CHUNKS,
        global_config={},
        embedding_func=_DummyEmbeddingFunc(),
        workspace=workspace,
    )


def _stored(fake: FakeRedis, storage, key: str) -> dict:
    return json.loads(fake.store[f"{storage.final_namespace}:{key}"])


@pytest.mark.asyncio
async def test_replacement_upsert_preserves_create_time(fake):
    storage = _kv_storage()
    await storage.initialize()

    with patch("time.time", return_value=1_700_000_000):
        await storage.upsert({"E": {"chunk_ids": ["c1"], "count": 1}})

    created = await storage.get_by_id("E")
    assert created is not None
    assert created["create_time"] == 1_700_000_000
    assert created["update_time"] == 1_700_000_000

    with patch("time.time", return_value=1_700_000_100):
        await storage.upsert({"E": {"chunk_ids": ["c1", "c2"], "count": 2}})

    updated = await storage.get_by_id("E")
    assert updated is not None
    assert updated["create_time"] == 1_700_000_000
    assert updated["update_time"] == 1_700_000_100
    assert updated["chunk_ids"] == ["c1", "c2"]
    assert updated["count"] == 2

    persisted = _stored(fake, storage, "E")
    assert persisted["create_time"] == 1_700_000_000
    assert persisted["update_time"] == 1_700_000_100


@pytest.mark.asyncio
async def test_value_is_serialized_create_time_first(fake):
    """The prefix read only works while create_time leads the value."""
    storage = _kv_storage()
    await storage.initialize()

    with patch("time.time", return_value=1_700_000_000):
        await storage.upsert({"E": {"chunk_ids": ["c1"], "count": 1}})

    raw = fake.store[f"{storage.final_namespace}:E"]
    assert next(iter(json.loads(raw))) == "create_time"
    assert raw.startswith('{"create_time"')


@pytest.mark.asyncio
async def test_update_reads_prefix_and_never_pulls_whole_value(fake):
    """The optimization itself: GETRANGE only, no full GET.

    Without this the fix still behaves correctly -- the fallback full read
    produces the same timestamps -- so only a command-level assertion can tell
    the fast path apart from a silently dead one.
    """
    storage = _kv_storage()
    await storage.initialize()

    with patch("time.time", return_value=1_700_000_000):
        await storage.upsert({"E": {"chunk_ids": ["c1"], "count": 1}})

    fake.command_counts.clear()
    with patch("time.time", return_value=1_700_000_100):
        await storage.upsert({"E": {"chunk_ids": ["c1", "c2"], "count": 2}})

    assert fake.command_counts["getrange"] == 1
    assert fake.command_counts["get"] == 0


@pytest.mark.asyncio
async def test_insert_reads_prefix_only(fake):
    """A brand-new key must not trigger the legacy full-read fallback."""
    storage = _kv_storage()
    await storage.initialize()

    fake.command_counts.clear()
    with patch("time.time", return_value=1_700_000_300):
        await storage.upsert({"new": {"chunk_ids": ["c9"], "count": 1}})

    assert fake.command_counts["getrange"] == 1
    assert fake.command_counts["get"] == 0
    row = await storage.get_by_id("new")
    assert row["create_time"] == 1_700_000_300
    assert row["update_time"] == 1_700_000_300


@pytest.mark.asyncio
async def test_legacy_row_with_trailing_create_time_falls_back_to_full_read(fake):
    """A row written before the ordering existed keeps its real create_time."""
    storage = _kv_storage()
    await storage.initialize()

    key = f"{storage.final_namespace}:legacy"
    fake.store[key] = json.dumps(
        {
            "chunk_ids": ["c1"],
            "count": 1,
            "create_time": 1_650_000_000,
            "update_time": 1_650_000_000,
        }
    )

    fake.command_counts.clear()
    with patch("time.time", return_value=1_700_000_200):
        await storage.upsert({"legacy": {"chunk_ids": ["c1", "c2"], "count": 2}})

    assert fake.command_counts["get"] == 1  # the fallback ran
    row = await storage.get_by_id("legacy")
    assert row["create_time"] == 1_650_000_000
    assert row["update_time"] == 1_700_000_200

    # ...and the row is rewritten in the prefix layout, so the next update
    # takes the fast path.
    fake.command_counts.clear()
    with patch("time.time", return_value=1_700_000_400):
        await storage.upsert({"legacy": {"chunk_ids": ["c3"], "count": 1}})
    assert fake.command_counts["get"] == 0
    assert _stored(fake, storage, "legacy")["create_time"] == 1_650_000_000


@pytest.mark.asyncio
async def test_legacy_row_missing_create_time_keeps_zero(fake):
    storage = _kv_storage()
    await storage.initialize()

    key = f"{storage.final_namespace}:legacy"
    fake.store[key] = json.dumps(
        {
            "chunk_ids": ["c1"],
            "count": 1,
            "update_time": 1_600_000_000,
            "_id": "legacy",
        }
    )

    with patch("time.time", return_value=1_700_000_200):
        await storage.upsert({"legacy": {"chunk_ids": ["c1", "c2"], "count": 2}})

    row = await storage.get_by_id("legacy")
    assert row is not None
    assert row["create_time"] == 0
    assert row["update_time"] == 1_700_000_200
    assert _stored(fake, storage, "legacy")["create_time"] == 0


@pytest.mark.asyncio
async def test_legacy_float_create_time_is_normalized(fake):
    """An older release could store time.time() unrounded."""
    storage = _kv_storage()
    await storage.initialize()

    key = f"{storage.final_namespace}:f"
    fake.store[key] = json.dumps({"chunk_ids": ["c1"], "create_time": 1_650_000_000.75})

    with patch("time.time", return_value=1_700_000_200):
        await storage.upsert({"f": {"chunk_ids": ["c2"]}})

    stored = _stored(fake, storage, "f")
    assert stored["create_time"] == 1_650_000_000
    assert isinstance(stored["create_time"], int)


@pytest.mark.asyncio
async def test_corrupt_row_records_zero_and_warns(fake, caplog):
    """Undecodable storage is corruption, not a legacy shape: say so."""
    storage = _kv_storage()
    await storage.initialize()

    fake.store[f"{storage.final_namespace}:c"] = "}} not json {{"

    logger = logging.getLogger("lightrag")
    previous = logger.propagate
    logger.propagate = True  # lightrag's logger does not propagate by default
    try:
        with caplog.at_level(logging.WARNING, logger="lightrag"):
            with patch("time.time", return_value=1_700_000_200):
                await storage.upsert({"c": {"chunk_ids": ["c1"]}})
    finally:
        logger.propagate = previous

    assert _stored(fake, storage, "c")["create_time"] == 0
    assert any("not decodable JSON" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_caller_supplied_create_time_ignored_on_update(fake):
    storage = _kv_storage()
    await storage.initialize()

    with patch("time.time", return_value=1_700_000_000):
        await storage.upsert({"E": {"chunk_ids": ["c1"]}})
    with patch("time.time", return_value=1_700_000_100):
        await storage.upsert({"E": {"chunk_ids": ["c2"], "create_time": 1}})

    assert _stored(fake, storage, "E")["create_time"] == 1_700_000_000


@pytest.mark.asyncio
async def test_row_deleted_between_the_two_reads_is_an_insert(fake, monkeypatch):
    """The fallback's second read can find the key already gone."""
    storage = _kv_storage()
    await storage.initialize()

    key = f"{storage.final_namespace}:gone"
    # Trailing create_time forces the two-phase path.
    fake.store[key] = json.dumps({"x": 1, "create_time": 1_650_000_000})

    original_apply = fake._apply

    def apply_and_vanish(op):
        result = original_apply(op)
        if op[0] == "getrange" and op[1] == key:
            fake.store.pop(key, None)
        return result

    monkeypatch.setattr(fake, "_apply", apply_and_vanish)

    with patch("time.time", return_value=1_700_000_500):
        await storage.upsert({"gone": {"x": 2}})

    stored = _stored(fake, storage, "gone")
    assert stored["create_time"] == 1_700_000_500
    assert stored["update_time"] == 1_700_000_500


# ---------------------------------------------------------------------------
# Concurrency: the insert race (issue #3870 follow-up)
# ---------------------------------------------------------------------------
#
# Resolving the stored timestamp and writing the value are two round trips, so
# a second writer can slip in between them. Only the INSERT outcome is
# contended -- two writers that both see the key absent would each stamp their
# own clock -- which is why a presumed-insert goes out as ``SET NX``.
#
# The interleaving is forced by giving the second storage a stale "absent"
# resolution, which is exactly what it would have read a moment before the
# first writer's SET landed.


def _hide_key_from_the_next_prefix_read(fake: FakeRedis, full_key: str):
    """Model the interleaving at the SERVER, not through storage internals.

    The classification read lands before the other writer's ``SET`` (so it
    reports "absent") and the repair read lands after it (so it sees the
    winner). Driving that through the fake keeps the test independent of how
    the storage happens to structure its reads.
    """
    real_apply = fake._apply
    state = {"hidden": False}

    def apply(op):
        if not state["hidden"] and op[0] == "getrange" and op[1] == full_key:
            state["hidden"] = True
            fake.command_counts["getrange"] += 1  # it did reach the server
            return ""
        return real_apply(op)

    fake._apply = apply
    return state


@pytest.mark.asyncio
async def test_concurrent_first_insert_keeps_the_earliest_create_time(fake):
    """A later writer must not move create_time off the real first creation."""
    worker_a = _kv_storage()
    worker_b = _kv_storage()
    await worker_a.initialize()
    await worker_b.initialize()

    with patch("time.time", return_value=100):
        await worker_a.upsert({"K": {"chunk_ids": ["c1"], "count": 1}})

    # Worker B's classification read lands before A's SET became visible.
    _hide_key_from_the_next_prefix_read(fake, f"{worker_b.final_namespace}:K")
    with patch("time.time", return_value=200):
        await worker_b.upsert({"K": {"chunk_ids": ["c2"], "count": 2}})

    row = _stored(fake, worker_a, "K")
    assert row["create_time"] == 100, "the first creation must win"
    assert row["update_time"] == 200
    # B's business value still lands -- only the timestamp is adopted.
    assert row["chunk_ids"] == ["c2"]
    assert row["count"] == 2


@pytest.mark.asyncio
async def test_insert_race_repair_stays_bounded(fake):
    """Losing the NX race costs one prefix read and one SET, never a full GET."""
    worker_a = _kv_storage()
    worker_b = _kv_storage()
    await worker_a.initialize()
    await worker_b.initialize()

    with patch("time.time", return_value=100):
        await worker_a.upsert({"K": {"x": 1}})

    _hide_key_from_the_next_prefix_read(fake, f"{worker_b.final_namespace}:K")
    fake.command_counts.clear()
    with patch("time.time", return_value=200):
        await worker_b.upsert({"K": {"x": 2}})

    assert fake.command_counts["getrange"] == 2  # classify + repair
    assert fake.command_counts["get"] == 0
    assert fake.command_counts["set"] == 2  # refused NX + repairing SET
    assert _stored(fake, worker_b, "K")["create_time"] == 100


@pytest.mark.asyncio
async def test_uncontended_insert_does_no_repair_round(fake):
    """The race handling must not cost anything when there is no race."""
    storage = _kv_storage()
    await storage.initialize()

    fake.command_counts.clear()
    with patch("time.time", return_value=300):
        await storage.upsert({"fresh": {"x": 1}})

    assert fake.command_counts["getrange"] == 1
    assert fake.command_counts["set"] == 1
    row = _stored(fake, storage, "fresh")
    assert row["create_time"] == 300
    assert row["update_time"] == 300


@pytest.mark.asyncio
async def test_concurrent_updates_agree_without_coordination(fake):
    """An existing row needs no NX: every writer derives the same timestamp.

    This is why only the insert is made atomic -- pinning it keeps a future
    change from "fixing" the update path with coordination it does not need.
    """
    worker_a = _kv_storage()
    worker_b = _kv_storage()
    await worker_a.initialize()
    await worker_b.initialize()

    with patch("time.time", return_value=100):
        await worker_a.upsert({"K": {"x": 1}})

    # Both workers resolve the same stored row, then write in either order.
    with patch("time.time", return_value=200):
        await worker_a.upsert({"K": {"x": 2}})
        await worker_b.upsert({"K": {"x": 3}})

    row = _stored(fake, worker_b, "K")
    assert row["create_time"] == 100
    assert row["x"] == 3
