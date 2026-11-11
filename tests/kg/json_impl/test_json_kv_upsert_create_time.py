"""Regression tests: JsonKVStorage replacement upserts preserve create_time.

Fixes https://github.com/HKUDS/LightRAG/issues/3870 — callers that replace a
KV value with business fields only must not discard the storage-managed
create_time. update_time advances; legacy rows missing create_time keep the
0/unknown convention rather than inventing an original timestamp. See
``BaseKVStorage.upsert`` for the contract these tests pin.

Unlike the remote backends this one needs no I/O to preserve the field -- the
previous value is already in shared memory -- but on a multi-worker
deployment that memory is a ``Manager().dict()`` proxy, so the number of
subscripts is the cost that matters and is asserted here too.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.namespace import NameSpace

pytestmark = pytest.mark.offline


class _DummyEmbeddingFunc:
    embedding_dim = 1
    max_token_size = 1

    async def __call__(self, texts, **kwargs):
        return [[0.0] for _ in texts]


@pytest.fixture(autouse=True)
def setup_shared_data():
    initialize_share_data()
    yield
    finalize_share_data()


def _make_storage(tmp_path, workspace="ct-ws"):
    return JsonKVStorage(
        namespace=NameSpace.KV_STORE_ENTITY_CHUNKS,
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace=workspace,
    )


@pytest.mark.asyncio
async def test_replacement_upsert_preserves_create_time(tmp_path):
    storage = _make_storage(tmp_path)
    await storage.initialize()

    with patch("time.time", return_value=1_700_000_000):
        await storage.upsert({"E": {"chunk_ids": ["c1"], "count": 1}})

    created = await storage.get_by_id("E")
    assert created is not None
    assert created["create_time"] == 1_700_000_000
    assert created["update_time"] == 1_700_000_000
    assert created["chunk_ids"] == ["c1"]
    assert created["count"] == 1

    with patch("time.time", return_value=1_700_000_100):
        await storage.upsert({"E": {"chunk_ids": ["c1", "c2"], "count": 2}})

    updated = await storage.get_by_id("E")
    assert updated is not None
    assert updated["create_time"] == 1_700_000_000
    assert updated["update_time"] == 1_700_000_100
    assert updated["chunk_ids"] == ["c1", "c2"]
    assert updated["count"] == 2
    # Business fields are replaced, not merged with the previous value.
    assert "llm_cache_list" not in updated

    await storage.index_done_callback()
    persisted = storage._data["E"]
    assert persisted["create_time"] == 1_700_000_000
    assert persisted["update_time"] == 1_700_000_100


@pytest.mark.asyncio
async def test_legacy_row_missing_create_time_keeps_zero(tmp_path):
    storage = _make_storage(tmp_path)
    await storage.initialize()

    # Simulate a legacy row written before create_time existed.
    storage._data["legacy"] = {
        "chunk_ids": ["c1"],
        "count": 1,
        "update_time": 1_600_000_000,
        "_id": "legacy",
    }

    with patch("time.time", return_value=1_700_000_200):
        await storage.upsert({"legacy": {"chunk_ids": ["c1", "c2"], "count": 2}})

    row = await storage.get_by_id("legacy")
    assert row is not None
    assert row["create_time"] == 0
    assert row["update_time"] == 1_700_000_200
    assert storage._data["legacy"]["create_time"] == 0


@pytest.mark.asyncio
async def test_insert_stamps_both_timestamps(tmp_path):
    storage = _make_storage(tmp_path)
    await storage.initialize()

    with patch("time.time", return_value=1_700_000_300):
        await storage.upsert({"new": {"chunk_ids": ["c9"], "count": 1}})

    row = await storage.get_by_id("new")
    assert row is not None
    assert row["create_time"] == 1_700_000_300
    assert row["update_time"] == 1_700_000_300


class _CountingDict(dict):
    """Stand-in for the multiprocess ``Manager().dict()`` proxy.

    Every subscript on the real proxy is a separate IPC round trip, and
    ``__getitem__`` ships the whole value back. Counting them is the only way
    to pin "one lookup per key" -- the timestamps are identical either way.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = {"get": 0, "getitem": 0, "contains": 0}

    def get(self, key, default=None):
        self.calls["get"] += 1
        return super().get(key, default)

    def __getitem__(self, key):
        self.calls["getitem"] += 1
        return super().__getitem__(key)

    def __contains__(self, key):
        self.calls["contains"] += 1
        return super().__contains__(key)


@pytest.mark.asyncio
async def test_update_uses_one_lookup_per_key(tmp_path):
    storage = _make_storage(tmp_path)
    await storage.initialize()

    counting = _CountingDict()
    storage._data = counting

    with patch("time.time", return_value=1_700_000_000):
        await storage.upsert({"A": {"x": 1}, "B": {"x": 1}})
    with patch("time.time", return_value=1_700_000_100):
        counting.calls.update({"get": 0, "getitem": 0, "contains": 0})
        await storage.upsert({"A": {"x": 2}, "B": {"x": 2}})

    assert counting.calls["get"] == 2
    assert counting.calls["getitem"] == 0
    assert counting.calls["contains"] == 0
    assert counting["A"]["create_time"] == 1_700_000_000
    assert counting["A"]["update_time"] == 1_700_000_100


@pytest.mark.asyncio
async def test_null_create_time_normalized_to_zero(tmp_path):
    """A hand-edited JSON file can carry an explicit null."""
    storage = _make_storage(tmp_path)
    await storage.initialize()

    storage._data["legacy"] = {"x": 1, "create_time": None}
    with patch("time.time", return_value=1_700_000_200):
        await storage.upsert({"legacy": {"x": 2}})

    assert storage._data["legacy"]["create_time"] == 0
    assert storage._data["legacy"]["update_time"] == 1_700_000_200


@pytest.mark.asyncio
async def test_legacy_float_create_time_is_normalized(tmp_path):
    """An older release could store time.time() unrounded."""
    storage = _make_storage(tmp_path)
    await storage.initialize()

    storage._data["f"] = {"x": 1, "create_time": 1_650_000_000.75}
    with patch("time.time", return_value=1_700_000_200):
        await storage.upsert({"f": {"x": 2}})

    stored = storage._data["f"]
    assert stored["create_time"] == 1_650_000_000
    assert isinstance(stored["create_time"], int)


@pytest.mark.asyncio
async def test_caller_supplied_create_time_ignored_on_update(tmp_path):
    storage = _make_storage(tmp_path)
    await storage.initialize()

    with patch("time.time", return_value=1_700_000_000):
        await storage.upsert({"E": {"x": 1}})
    with patch("time.time", return_value=1_700_000_100):
        await storage.upsert({"E": {"x": 2, "create_time": 1}})

    assert storage._data["E"]["create_time"] == 1_700_000_000


@pytest.mark.asyncio
async def test_infinite_create_time_does_not_abort_the_upsert(tmp_path):
    """JSON ``1e309`` decodes to float infinity, and ``int(inf)`` overflows.

    The helper's documented fallback is 0; before OverflowError was handled
    one hand-edited row aborted the whole upsert instead.
    """
    storage = _make_storage(tmp_path)
    await storage.initialize()

    storage._data["inf"] = {"x": 1, "create_time": json.loads("1e309")}
    with patch("time.time", return_value=1_700_000_600):
        await storage.upsert({"inf": {"x": 2}})

    assert storage._data["inf"]["create_time"] == 0
    assert storage._data["inf"]["update_time"] == 1_700_000_600
