"""Integration tests: create_time preservation against a REAL Redis.

The unit tests run on the FakeRedis stand-in, which can only prove the fake's
own semantics. Two things here need the real server (issue #3870):

* ``GETRANGE`` returns an empty string — not nil — for a missing key, which is
  how ``upsert`` recognizes an insert;
* the prefix actually MATCHES what ``json.dumps`` writes. A pattern that is
  too strict still produces correct timestamps (the full-read fallback covers
  it) while silently sending every update down that fallback, so the
  optimization is asserted through ``INFO commandstats``.

Opt-in — these tests write and DELETE keys, so they never run against the URI
configured for a real deployment. They read ``LIGHTRAG_TEST_REDIS_URI`` and
are skipped when it is unset::

    docker run -d --name lr-redis-test -p 16379:6379 redis:latest

    LIGHTRAG_TEST_REDIS_URI=redis://localhost:16379 \\
        ./scripts/test.sh tests/kg/redis_impl/test_redis_kv_create_time_integration.py \\
        --run-integration
"""

from __future__ import annotations

import asyncio
import json
import os
import time

import pytest
from unittest.mock import patch

from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.namespace import NameSpace

pytestmark = [pytest.mark.integration, pytest.mark.requires_db]

_WORKSPACE = "ittest_create_time"


class _DummyEmbeddingFunc:
    embedding_dim = 1
    max_token_size = 1

    async def __call__(self, texts, **kwargs):
        return [[0.0] for _ in texts]


@pytest.fixture
async def storage():
    uri = os.getenv("LIGHTRAG_TEST_REDIS_URI")
    if not uri:
        pytest.skip("Redis not configured for tests (LIGHTRAG_TEST_REDIS_URI not set)")

    previous = os.environ.get("REDIS_URI")
    os.environ["REDIS_URI"] = uri

    from lightrag.kg import redis_impl
    from lightrag.kg.redis_impl import RedisKVStorage

    cache = getattr(redis_impl, "_eviction_checked", None)
    if cache is not None:
        cache.clear()

    initialize_share_data()
    kv = RedisKVStorage(
        namespace=NameSpace.KV_STORE_ENTITY_CHUNKS,
        global_config={},
        embedding_func=_DummyEmbeddingFunc(),
        workspace=_WORKSPACE,
    )
    await kv.initialize()
    await kv.drop()
    try:
        yield kv
    finally:
        await kv.drop()
        await kv.finalize()
        finalize_share_data()
        if previous is None:
            os.environ.pop("REDIS_URI", None)
        else:
            os.environ["REDIS_URI"] = previous


async def _raw(storage, key: str) -> str:
    async with storage._get_redis_connection() as redis:
        return await redis.get(f"{storage.final_namespace}:{key}")


@pytest.mark.asyncio
async def test_value_is_stored_create_time_first(storage):
    await storage.upsert({"E": {"chunk_ids": ["c1"], "count": 1}})

    raw = await _raw(storage, "E")
    assert raw.startswith('{"create_time"')
    assert next(iter(json.loads(raw))) == "create_time"


@pytest.mark.asyncio
async def test_prefix_read_matches_what_the_writer_produced(storage):
    """The regex must match real json.dumps output, spacing included."""
    from lightrag.kg.redis_impl import (
        _CREATE_TIME_PREFIX_BYTES,
        _CREATE_TIME_PREFIX_RE,
    )

    await storage.upsert({"E": {"chunk_ids": ["c1"], "count": 1}})
    created = json.loads(await _raw(storage, "E"))["create_time"]

    async with storage._get_redis_connection() as redis:
        prefix = await redis.getrange(
            f"{storage.final_namespace}:E", 0, _CREATE_TIME_PREFIX_BYTES - 1
        )
    match = _CREATE_TIME_PREFIX_RE.match(prefix)
    assert match is not None, f"prefix did not match: {prefix!r}"
    assert int(match.group(1)) == created


@pytest.mark.asyncio
async def test_getrange_on_a_missing_key_is_empty(storage):
    """Redis answers a missing key with "", which upsert reads as an insert."""
    async with storage._get_redis_connection() as redis:
        assert (
            await redis.getrange(f"{storage.final_namespace}:__absent__", 0, 63) == ""
        )


@pytest.mark.asyncio
async def test_update_never_pulls_the_whole_value_back(storage):
    await storage.upsert({"E": {"chunk_ids": ["c1"], "count": 1}})

    async with storage._get_redis_connection() as redis:
        await redis.config_resetstat()
        await storage.upsert({"E": {"chunk_ids": ["c1", "c2"], "count": 2}})
        stats = await redis.info("commandstats")

    assert stats.get("cmdstat_getrange", {}).get("calls", 0) >= 1
    assert stats.get("cmdstat_get", {}).get("calls", 0) == 0


@pytest.mark.asyncio
async def test_replacement_upsert_preserves_create_time(storage):
    await storage.upsert({"E": {"chunk_ids": ["c1"], "count": 1}})
    created = (await storage.get_by_id("E"))["create_time"]

    time.sleep(1.1)  # update_time must be observably later
    await storage.upsert({"E": {"chunk_ids": ["c1", "c2"], "count": 2}})

    row = await storage.get_by_id("E")
    assert row["create_time"] == created
    assert row["update_time"] > created
    assert row["chunk_ids"] == ["c1", "c2"]
    assert row["count"] == 2


@pytest.mark.asyncio
async def test_caller_supplied_create_time_is_ignored_on_update(storage):
    await storage.upsert({"E": {"chunk_ids": ["c1"]}})
    created = (await storage.get_by_id("E"))["create_time"]

    await storage.upsert({"E": {"chunk_ids": ["c2"], "create_time": 1}})

    assert (await storage.get_by_id("E"))["create_time"] == created


@pytest.mark.asyncio
async def test_legacy_row_with_trailing_create_time_is_recovered_and_repaired(storage):
    async with storage._get_redis_connection() as redis:
        await redis.set(
            f"{storage.final_namespace}:L",
            json.dumps({"chunk_ids": ["c1"], "count": 1, "create_time": 1_650_000_000}),
        )

    await storage.upsert({"L": {"chunk_ids": ["c1", "c2"], "count": 2}})

    assert (await storage.get_by_id("L"))["create_time"] == 1_650_000_000
    # Rewritten in the prefix layout, so the next update takes the fast path.
    assert (await _raw(storage, "L")).startswith('{"create_time"')


@pytest.mark.asyncio
async def test_legacy_row_missing_create_time_records_zero(storage):
    async with storage._get_redis_connection() as redis:
        await redis.set(
            f"{storage.final_namespace}:M", json.dumps({"chunk_ids": ["c1"]})
        )

    await storage.upsert({"M": {"chunk_ids": ["c2"]}})

    assert (await storage.get_by_id("M"))["create_time"] == 0


@pytest.mark.asyncio
async def test_legacy_float_create_time_is_normalized(storage):
    async with storage._get_redis_connection() as redis:
        await redis.set(
            f"{storage.final_namespace}:F",
            json.dumps({"x": 1, "create_time": 1_650_000_000.75}),
        )

    await storage.upsert({"F": {"x": 2}})

    stored = json.loads(await _raw(storage, "F"))
    assert stored["create_time"] == 1_650_000_000
    assert isinstance(stored["create_time"], int)


@pytest.mark.asyncio
async def test_empty_value_row_records_zero(storage):
    """Redis answers GETRANGE with "" for a missing key AND an empty value.

    Only a server-side EXISTS separates them, so this pins that an existing
    empty row is treated as a stored row with an unknown timestamp instead of
    a first creation.
    """
    async with storage._get_redis_connection() as redis:
        await redis.set(f"{storage.final_namespace}:empty", "")
        assert await redis.exists(f"{storage.final_namespace}:empty") == 1
        assert await redis.getrange(f"{storage.final_namespace}:empty", 0, 63) == ""

    with patch("time.time", return_value=1_700_000_700):
        await storage.upsert({"empty": {"x": 1}})

    row = await storage.get_by_id("empty")
    assert row["create_time"] == 0
    assert row["update_time"] == 1_700_000_700


@pytest.mark.asyncio
async def test_corrupt_row_records_zero(storage):
    async with storage._get_redis_connection() as redis:
        await redis.set(f"{storage.final_namespace}:C", "}} not json {{")

    await storage.upsert({"C": {"chunk_ids": ["c1"]}})

    assert (await storage.get_by_id("C"))["create_time"] == 0


@pytest.mark.asyncio
async def test_batch_update_preserves_every_create_time(storage):
    ids = [f"B{i}" for i in range(50)]
    await storage.upsert({doc_id: {"chunk_ids": ["c"], "count": 1} for doc_id in ids})
    rows = await storage.get_by_ids(ids)
    created = rows[0]["create_time"]

    time.sleep(1.1)
    await storage.upsert(
        {doc_id: {"chunk_ids": ["c", "d"], "count": 2} for doc_id in ids}
    )

    rows = await storage.get_by_ids(ids)
    assert [row["create_time"] for row in rows] == [created] * len(ids)
    assert all(row["update_time"] > created for row in rows)


# ---------------------------------------------------------------------------
# Concurrency: the insert race
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_nx_refuses_an_existing_key(storage):
    """The server primitive the insert race relies on."""
    async with storage._get_redis_connection() as redis:
        key = f"{storage.final_namespace}:nx"
        assert await redis.set(key, "first", nx=True) is True
        assert await redis.set(key, "second", nx=True) is None
        assert await redis.get(key) == "first"


@pytest.mark.asyncio
async def test_upsert_racing_a_delete_never_resurrects_the_old_timestamp(storage):
    """The delete race, which only server-side atomicity can close.

    If the row exists afterwards, the upsert's write was the last one on that
    key, so the row is a new incarnation and must carry the upsert's own
    clock. A client-side read-then-write leaves a window where the read sees
    the pre-delete row, the delete lands, and the write puts the removed
    incarnation's timestamp back -- and nothing ever repairs it, because every
    later update preserves what it finds.

    Measured against the read-then-write predecessor of this code, the window
    was hit in roughly a quarter of the rounds below; the script closes it
    because its read and write are one step.
    """
    rounds = 40
    resurrected = []
    for index in range(rounds):
        key = f"race-{index}"
        with patch("time.time", return_value=1_700_000_000):
            await storage.upsert({key: {"round": 1}})
        first = (await storage.get_by_id(key))["create_time"]

        with patch("time.time", return_value=1_700_000_500):
            await asyncio.gather(
                storage.upsert({key: {"round": 2}}), storage.delete([key])
            )

        row = await storage.get_by_id(key)
        if row is not None and row["create_time"] == first:
            resurrected.append(key)

    assert not resurrected, (
        f"{len(resurrected)}/{rounds} rows came back carrying the deleted "
        f"incarnation's create_time"
    )


@pytest.mark.asyncio
async def test_concurrent_inserts_converge_on_one_create_time(storage):
    """Unforced concurrency: many writers, one first-creation timestamp."""
    from lightrag.kg.redis_impl import RedisKVStorage

    others = [
        RedisKVStorage(
            namespace=NameSpace.KV_STORE_ENTITY_CHUNKS,
            global_config={},
            embedding_func=_DummyEmbeddingFunc(),
            workspace=_WORKSPACE,
        )
        for _ in range(4)
    ]
    for other in others:
        await other.initialize()
    workers = [storage, *others]

    try:
        await asyncio.gather(
            *(
                worker.upsert({"R": {"writer": index}})
                for index, worker in enumerate(workers)
            )
        )
        settled = (await storage.get_by_id("R"))["create_time"]

        # Whatever won, it must not move afterwards.
        time.sleep(1.1)
        await asyncio.gather(
            *(
                worker.upsert({"R": {"writer": index, "round": 2}})
                for index, worker in enumerate(workers)
            )
        )
        row = await storage.get_by_id("R")
        assert row["create_time"] == settled
        assert row["update_time"] > settled
    finally:
        for other in others:
            await other.finalize()


# Raw stored values, chosen so each side of the prefix pattern is exercised:
# compact and spaced separators, zero, a negative value, and four shapes the
# pattern must REFUSE (tail position, float, null, quoted) plus non-JSON.
_PREFIX_SAMPLES = [
    '{"create_time":1650000000,"x":1}',
    '{"create_time": 1650000000, "x": 1}',
    '{"create_time":0}',
    '{"create_time": -5, "x": 1}',
    '{"x":1,"create_time":1650000000}',
    '{"create_time": 1.5, "x": 1}',
    '{"create_time": null}',
    '{"create_time": "1650000000"}',
    "not json at all",
]


@pytest.mark.parametrize("raw", _PREFIX_SAMPLES, ids=range(len(_PREFIX_SAMPLES)))
@pytest.mark.asyncio
async def test_lua_prefix_pattern_agrees_with_the_python_regex(storage, raw):
    """The two patterns are one rule written twice; keep them in step.

    A Lua pattern that refused a prefix the Python regex accepts would send
    every update down the legacy read; one that accepted more would take a
    timestamp the caller never normalized.
    """
    from lightrag.kg.redis_impl import (
        _CREATE_TIME_PREFIX_BYTES,
        _CREATE_TIME_PREFIX_RE,
        _CREATE_TIME_UPSERT_LUA,
    )

    key = f"{storage.final_namespace}:pattern"
    expected = _CREATE_TIME_PREFIX_RE.match(raw[:_CREATE_TIME_PREFIX_BYTES])

    async with storage._get_redis_connection() as redis:
        await redis.set(key, raw)
        script = redis.register_script(_CREATE_TIME_UPSERT_LUA)
        outcome, value = await script(
            keys=[key],
            args=['{"x":2}', "", 1_700_000_000, _CREATE_TIME_PREFIX_BYTES],
        )

    if expected is None:
        assert outcome == "needs_hint"
        assert value == ""
    else:
        assert outcome == "kept"
        assert value == expected.group(1)
