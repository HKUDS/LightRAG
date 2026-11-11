"""Integration tests: create_time preservation against a REAL OpenSearch.

The unit tests can only pin the bulk action shape; the semantics of the
``scripted_upsert`` painless script (issue #3870) are a server behavior, so
they are verified here against a live cluster.

Opt-in — these tests CREATE and DROP an index, so they never run against the
hosts configured for a real deployment. They read
``LIGHTRAG_TEST_OPENSEARCH_HOSTS`` and are skipped when it is unset::

    docker run -d --name lr-os-test -p 19200:9200 \\
        -e "discovery.type=single-node" -e "DISABLE_SECURITY_PLUGIN=true" \\
        -e "DISABLE_INSTALL_DEMO_CONFIG=true" \\
        opensearchproject/opensearch:3

    LIGHTRAG_TEST_OPENSEARCH_HOSTS=localhost:19200 \\
        ./scripts/test.sh tests/kg/opensearch_impl/test_opensearch_kv_create_time_integration.py \\
        --run-integration

Validated on OpenSearch 3.6.0.
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from lightrag.base import normalize_kv_create_time
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data

pytestmark = [pytest.mark.integration, pytest.mark.requires_db]

_WORKSPACE = "ittest_create_time"


class _DummyEmbeddingFunc:
    embedding_dim = 1
    max_token_size = 1

    async def __call__(self, texts, **kwargs):
        return [[0.0] for _ in texts]


@pytest.fixture
async def storage():
    hosts = os.getenv("LIGHTRAG_TEST_OPENSEARCH_HOSTS")
    if not hosts:
        pytest.skip(
            "OpenSearch not configured for tests "
            "(LIGHTRAG_TEST_OPENSEARCH_HOSTS not set)"
        )

    previous = {
        key: os.environ.get(key)
        for key in ("OPENSEARCH_HOSTS", "OPENSEARCH_USE_SSL", "OPENSEARCH_USER")
    }
    os.environ["OPENSEARCH_HOSTS"] = hosts
    os.environ["OPENSEARCH_USE_SSL"] = os.getenv(
        "LIGHTRAG_TEST_OPENSEARCH_USE_SSL", "false"
    )
    os.environ["OPENSEARCH_USER"] = os.getenv("LIGHTRAG_TEST_OPENSEARCH_USER", "")

    from lightrag.kg.opensearch_impl import OpenSearchKVStorage

    initialize_share_data()
    kv = OpenSearchKVStorage(
        namespace="entity_chunks",
        global_config={
            "embedding_batch_num": 1,
            "max_graph_nodes": 10,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        embedding_func=_DummyEmbeddingFunc(),
        workspace=_WORKSPACE,
    )
    await kv.initialize()
    await kv.drop()
    await kv.initialize()
    try:
        yield kv
    finally:
        await kv.drop()
        await kv.finalize()
        finalize_share_data()
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


async def _write(storage, payload):
    await storage.upsert(payload)
    await storage.index_done_callback()


@pytest.mark.asyncio
async def test_insert_stamps_both_timestamps(storage):
    before = int(time.time())
    await _write(storage, {"E": {"chunk_ids": ["c1"], "count": 1}})

    row = await storage.get_by_id("E")
    assert row["create_time"] >= before
    assert row["update_time"] >= before


@pytest.mark.asyncio
async def test_replacement_upsert_preserves_create_time(storage):
    await _write(storage, {"E": {"chunk_ids": ["c1"], "count": 1}})
    created = (await storage.get_by_id("E"))["create_time"]

    time.sleep(1.1)  # update_time must be observably later
    await _write(storage, {"E": {"chunk_ids": ["c1", "c2"], "count": 2}})

    row = await storage.get_by_id("E")
    assert row["create_time"] == created
    assert row["update_time"] > created
    # The business value is replaced, not merged.
    assert row["chunk_ids"] == ["c1", "c2"]
    assert row["count"] == 2
    # The scripted-upsert sentinel is an implementation detail of the write.
    assert "__lightrag_kv_new" not in row


@pytest.mark.asyncio
async def test_replacement_drops_fields_the_payload_omits(storage):
    """A scripted upsert must replace the source, not partially merge it."""
    await _write(storage, {"E": {"chunk_ids": ["c1"], "count": 1, "extra": "x"}})
    await _write(storage, {"E": {"chunk_ids": ["c1"], "count": 1}})

    assert "extra" not in await storage.get_by_id("E")


@pytest.mark.asyncio
async def test_caller_supplied_create_time_is_ignored_on_update(storage):
    await _write(storage, {"E": {"chunk_ids": ["c1"], "count": 1}})
    created = (await storage.get_by_id("E"))["create_time"]

    await _write(storage, {"E": {"chunk_ids": ["c2"], "count": 1, "create_time": 1}})

    assert (await storage.get_by_id("E"))["create_time"] == created


@pytest.mark.asyncio
async def test_legacy_row_without_create_time_records_zero(storage):
    """A row written before the field existed keeps the 0/unknown meaning."""
    await storage.client.index(
        index=storage._index_name,
        id="L",
        body={"chunk_ids": ["c1"], "count": 1, "__mirrored_id": "L"},
        refresh=True,
    )

    await _write(storage, {"L": {"chunk_ids": ["c1", "c2"], "count": 2}})

    assert (await storage.get_by_id("L"))["create_time"] == 0


@pytest.mark.asyncio
async def test_null_create_time_records_zero(storage):
    await storage.client.index(
        index=storage._index_name,
        id="N",
        body={"chunk_ids": ["c1"], "create_time": None, "__mirrored_id": "N"},
        refresh=True,
    )

    await _write(storage, {"N": {"chunk_ids": ["c2"]}})

    assert (await storage.get_by_id("N"))["create_time"] == 0


@pytest.mark.asyncio
async def test_delete_then_upsert_stamps_a_fresh_create_time(storage):
    await _write(storage, {"E": {"chunk_ids": ["c1"], "count": 1}})
    created = (await storage.get_by_id("E"))["create_time"]

    time.sleep(1.1)
    await storage.delete(["E"])
    await storage.index_done_callback()
    await _write(storage, {"E": {"chunk_ids": ["new"], "count": 1}})

    assert (await storage.get_by_id("E"))["create_time"] > created


@pytest.mark.asyncio
async def test_many_small_upserts_share_one_flush(storage):
    """The buffering pattern this backend exists for (issue #2785)."""
    ids = [f"B{i}" for i in range(5)]
    for doc_id in ids:
        await storage.upsert({doc_id: {"chunk_ids": ["c"], "count": 1}})
    await storage.index_done_callback()

    rows = await storage.get_by_ids(ids)
    assert all(row is not None for row in rows)
    created = rows[0]["create_time"]

    time.sleep(1.1)
    for doc_id in ids:
        await storage.upsert({doc_id: {"chunk_ids": ["c", "d"], "count": 2}})
    await storage.index_done_callback()

    rows = await storage.get_by_ids(ids)
    assert [row["create_time"] for row in rows] == [created] * len(ids)
    assert all(row["update_time"] > created for row in rows)


# Shapes a LightRAG release (or external tooling) could have left in a stored
# row. ``True`` is deliberately absent: a document with a boolean
# ``create_time`` makes dynamic mapping type the field BOOLEAN, and the mapper
# then refuses the long the repair writes -- loudly, as a failed bulk item. No
# release ever wrote a boolean there, so the equivalence claim is scoped to
# shapes that can actually occur.
_NORMALIZATION_CASES = [
    1650000000,
    1650000000.75,
    -1.5,
    "1650000000",
    " 1650000000 ",
    "1.5",
    "broken",
    None,
    [],
    0,
]


@pytest.mark.parametrize("stored_value", _NORMALIZATION_CASES, ids=repr)
@pytest.mark.asyncio
async def test_script_normalization_matches_the_python_helper(storage, stored_value):
    """The painless script must answer exactly ``normalize_kv_create_time``.

    Two implementations of one rule drift unless something pins them
    together; a row whose ``create_time`` stays a float or a string is enough
    to make the LLM-cache ordering in ``operate.py`` compare ``str`` with
    ``int`` and raise. Each case gets a virgin index (the fixture drops and
    recreates it) because dynamic mapping types the field from the first
    document it sees and would otherwise reject the later shapes.
    """
    await storage.client.index(
        index=storage._index_name,
        id="N",
        body={"x": 1, "create_time": stored_value, "__mirrored_id": "N"},
        refresh=True,
    )

    await _write(storage, {"N": {"x": 2}})

    row = await storage.get_by_id("N")
    assert row["create_time"] == normalize_kv_create_time(stored_value)
    assert isinstance(row["create_time"], int)


@pytest.mark.asyncio
async def test_legacy_float_row_is_repaired_to_an_int(storage):
    """The shape is fixed on the row's next write, not only on read."""
    await storage.client.index(
        index=storage._index_name,
        id="F",
        body={"x": 1, "create_time": 1_650_000_000.75, "__mirrored_id": "F"},
        refresh=True,
    )

    await _write(storage, {"F": {"x": 2}})

    persisted = await storage.client.get(index=storage._index_name, id="F")
    assert persisted["_source"]["create_time"] == 1_650_000_000
    assert isinstance(persisted["_source"]["create_time"], int)


@pytest.mark.asyncio
async def test_concurrent_first_insert_keeps_the_earliest_create_time(storage):
    """The first creation wins even when two writers flush the same new id.

    The script needs no extra coordination for this: the server applies the
    two update actions one after another, and the second one finds the
    document already there (no sentinel) and restores the timestamp the first
    one wrote.
    """
    from lightrag.kg.opensearch_impl import OpenSearchKVStorage

    other = OpenSearchKVStorage(
        namespace="entity_chunks",
        global_config={
            "embedding_batch_num": 1,
            "max_graph_nodes": 10,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        embedding_func=_DummyEmbeddingFunc(),
        workspace=_WORKSPACE,
    )
    await other.initialize()
    try:
        await storage.upsert({"R": {"writer": "a"}})
        time.sleep(1.1)  # the second writer's estimate is strictly later
        await other.upsert({"R": {"writer": "b"}})

        await asyncio.gather(storage.index_done_callback(), other.index_done_callback())
        settled = (await storage.get_by_id("R"))["create_time"]

        # Whoever landed first owns the timestamp, and it must not move.
        time.sleep(1.1)
        await _write(storage, {"R": {"writer": "a", "round": 2}})
        row = await storage.get_by_id("R")
        assert row["create_time"] == settled
        assert row["update_time"] > settled
    finally:
        await other.finalize()
