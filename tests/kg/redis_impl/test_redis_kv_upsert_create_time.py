"""Regression tests: RedisKVStorage replacement upserts preserve create_time.

Issue #3870: an update whose payload carries business fields only used to drop
the storage-managed ``create_time``. Redis cannot read one JSON field, so the
value is written ``create_time``-first and ``_CREATE_TIME_UPSERT_LUA`` recovers
the previous timestamp from a bounded ``GETRANGE`` prefix -- in the same atomic
step as the write, which is what keeps a concurrent insert or delete from
moving it. See ``BaseKVStorage.upsert`` for the contract these tests pin.

Runs against the in-memory FakeRedis stand-in (no live Redis required). The
fake models the script in Python rather than running Lua, so the script's own
semantics and the two races are checked against a real server in
``tests/kg/redis_impl/test_redis_kv_create_time_integration.py``.
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
async def test_update_is_one_script_call_and_never_pulls_the_whole_value(fake):
    """The optimization itself: one round trip, a prefix read, no full GET.

    Without this the fix still behaves correctly -- the legacy full-read
    fallback produces the same timestamps -- so only a command-level
    assertion can tell the fast path apart from a silently dead one.
    """
    storage = _kv_storage()
    await storage.initialize()

    with patch("time.time", return_value=1_700_000_000):
        await storage.upsert({"E": {"chunk_ids": ["c1"], "count": 1}})

    fake.command_counts.clear()
    with patch("time.time", return_value=1_700_000_100):
        await storage.upsert({"E": {"chunk_ids": ["c1", "c2"], "count": 2}})

    assert fake.command_counts["script"] == 1
    assert fake.command_counts["getrange"] == 1  # inside the script
    assert fake.command_counts["get"] == 0


@pytest.mark.asyncio
async def test_insert_is_one_script_call(fake):
    """A brand-new key must not trigger the legacy hint round."""
    storage = _kv_storage()
    await storage.initialize()

    fake.command_counts.clear()
    with patch("time.time", return_value=1_700_000_300):
        await storage.upsert({"new": {"chunk_ids": ["c9"], "count": 1}})

    assert fake.command_counts["script"] == 1
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

    # needs_hint, then a full read, then the write.
    assert fake.command_counts["get"] == 1
    assert fake.command_counts["script"] == 2
    row = await storage.get_by_id("legacy")
    assert row["create_time"] == 1_650_000_000
    assert row["update_time"] == 1_700_000_200

    # ...and the row is rewritten in the prefix layout, so the next update
    # takes the fast path.
    fake.command_counts.clear()
    with patch("time.time", return_value=1_700_000_400):
        await storage.upsert({"legacy": {"chunk_ids": ["c3"], "count": 1}})
    assert fake.command_counts["get"] == 0
    assert fake.command_counts["script"] == 1
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
async def test_empty_value_row_records_zero_not_a_new_timestamp(fake, caplog):
    """An empty value is a stored row with no usable timestamp, not an insert.

    ``GETRANGE`` answers ``''`` for a missing key and for a key holding an
    empty string alike, so classifying on the prefix alone would fabricate a
    ``create_time`` for a row that already existed -- the very direction issue
    #3870 is about, and inconsistent with the corrupt-value row next door.
    """
    storage = _kv_storage()
    await storage.initialize()

    fake.store[f"{storage.final_namespace}:empty"] = ""

    logger = logging.getLogger("lightrag")
    previous = logger.propagate
    logger.propagate = True  # lightrag's logger does not propagate by default
    try:
        with caplog.at_level(logging.WARNING, logger="lightrag"):
            with patch("time.time", return_value=1_700_000_700):
                await storage.upsert({"empty": {"x": 1}})
    finally:
        logger.propagate = previous

    stored = _stored(fake, storage, "empty")
    assert stored["create_time"] == 0
    assert stored["update_time"] == 1_700_000_700
    assert any("not decodable JSON" in record.message for record in caplog.records)
    # It took the hint round, like every other unusable value.
    assert fake.command_counts["get"] == 1


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
async def test_legacy_row_deleted_before_the_hint_read_stamps_this_write(fake):
    """The only two-step branch left, and its absent case.

    A legacy-shaped row answers ``needs_hint``; if it is gone by the time the
    hint read runs, there is no earlier creation to preserve, so the write's
    own clock stands.
    """
    storage = _kv_storage()
    await storage.initialize()

    key = f"{storage.final_namespace}:gone"
    # Trailing create_time forces the hint round.
    fake.store[key] = json.dumps({"x": 1, "create_time": 1_650_000_000})

    real_apply = fake._apply

    def apply_and_vanish(op):
        result = real_apply(op)
        if op[0] == "script" and op[1] == key and result[0] == "needs_hint":
            fake.store.pop(key, None)
        return result

    fake._apply = apply_and_vanish

    with patch("time.time", return_value=1_700_000_500):
        await storage.upsert({"gone": {"x": 2}})

    stored = _stored(fake, storage, "gone")
    assert stored["create_time"] == 1_700_000_500
    assert stored["update_time"] == 1_700_000_500


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------
#
# The timestamp decision and the write are ONE atomic step inside
# _CREATE_TIME_UPSERT_LUA, which is what closes both races: two writers
# racing a first insert, and a delete() landing between a writer's read and
# its write. Neither window can be reproduced here -- FakeRedis cannot run
# Lua, and its model of the script is a single indivisible step, exactly like
# the real one. What these tests pin is the storage's side of the protocol:
# that it lets the server decide and then adopts that decision. The races
# themselves are measured against a real server in
# test_redis_kv_create_time_integration.py.


@pytest.mark.asyncio
async def test_second_writer_of_a_new_key_keeps_the_first_timestamp(fake):
    """Whoever creates the row owns create_time; later writers adopt it."""
    worker_a = _kv_storage()
    worker_b = _kv_storage()
    await worker_a.initialize()
    await worker_b.initialize()

    with patch("time.time", return_value=100):
        await worker_a.upsert({"K": {"chunk_ids": ["c1"], "count": 1}})

    fake.command_counts.clear()
    with patch("time.time", return_value=200):
        await worker_b.upsert({"K": {"chunk_ids": ["c2"], "count": 2}})

    # The server answered "kept", so no hint round was needed.
    assert fake.command_counts["script"] == 1
    assert fake.command_counts["get"] == 0

    row = _stored(fake, worker_b, "K")
    assert row["create_time"] == 100, "the first creation must win"
    assert row["update_time"] == 200
    # B's business value still lands -- only the timestamp is adopted.
    assert row["chunk_ids"] == ["c2"]
    assert row["count"] == 2


@pytest.mark.asyncio
async def test_the_callers_payload_ends_up_matching_storage(fake):
    """The dict the caller passed in carries what the SERVER decided.

    The optimistic value the loop stamps would otherwise be wrong for every
    update, which callers that reuse the dict (or log it) would see.
    """
    storage = _kv_storage()
    await storage.initialize()

    with patch("time.time", return_value=100):
        await storage.upsert({"K": {"x": 1}})

    payload = {"x": 2}
    with patch("time.time", return_value=200):
        await storage.upsert({"K": payload})

    assert payload["create_time"] == 100
    assert payload["update_time"] == 200
    assert payload["create_time"] == _stored(fake, storage, "K")["create_time"]


@pytest.mark.asyncio
async def test_concurrent_updates_agree_without_coordination(fake):
    """Two writers updating an existing row cannot disagree.

    Every writer derives the timestamp from the same stored row, so the
    outcome does not depend on who writes last.
    """
    worker_a = _kv_storage()
    worker_b = _kv_storage()
    await worker_a.initialize()
    await worker_b.initialize()

    with patch("time.time", return_value=100):
        await worker_a.upsert({"K": {"x": 1}})

    with patch("time.time", return_value=200):
        await worker_a.upsert({"K": {"x": 2}})
        await worker_b.upsert({"K": {"x": 3}})

    row = _stored(fake, worker_b, "K")
    assert row["create_time"] == 100
    assert row["x"] == 3


@pytest.mark.asyncio
async def test_infinite_create_time_does_not_abort_the_upsert(fake):
    """A stored ``1e309`` must take the 0 fallback, not raise OverflowError."""
    storage = _kv_storage()
    await storage.initialize()

    fake.store[f"{storage.final_namespace}:inf"] = json.dumps(
        {"x": 1, "create_time": json.loads("1e309")}
    )

    with patch("time.time", return_value=1_700_000_600):
        await storage.upsert({"inf": {"x": 2}})

    assert _stored(fake, storage, "inf")["create_time"] == 0
