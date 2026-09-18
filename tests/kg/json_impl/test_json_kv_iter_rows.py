"""``JsonKVStorage.iter_rows``: the enumeration surface (scenario 16 in
docs/design/ConfigurationStorage.md) on the JSON backend.

Rows come out shaped like ``get_by_ids`` returns them, ``_id`` included and
deep-copied; the key list is one snapshot and each page re-reads its rows, so
a row deleted mid-scan is skipped rather than served stale.
"""

from __future__ import annotations

import pytest

from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _shared():
    initialize_share_data()
    yield
    finalize_share_data()


async def _storage(tmp_path, workspace="iterws"):
    storage = JsonKVStorage(
        namespace="config",
        workspace=workspace,
        global_config={"working_dir": str(tmp_path)},
        embedding_func=None,
    )
    await storage.initialize()
    return storage


async def test_every_row_is_yielded_across_pages_with_the_point_read_shape(tmp_path):
    storage = await _storage(tmp_path)
    await storage.upsert(
        {f"k{i:02d}": {"value": {"n": i}, "nested": [i]} for i in range(7)}
    )

    rows = [row async for row in storage.iter_rows(page_size=3)]

    assert sorted(r["_id"] for r in rows) == [f"k{i:02d}" for i in range(7)]
    by_id = {r["_id"]: r for r in rows}
    point = await storage.get_by_id("k03")
    assert by_id["k03"] == point
    # Deep-copied: mutating a yielded row must not reach the shared dict.
    by_id["k03"]["nested"].append("x")
    assert (await storage.get_by_id("k03"))["nested"] == [3]


async def test_an_empty_namespace_yields_nothing(tmp_path):
    storage = await _storage(tmp_path)
    assert [row async for row in storage.iter_rows()] == []


async def test_a_row_deleted_mid_scan_is_skipped(tmp_path):
    storage = await _storage(tmp_path)
    await storage.upsert({f"k{i}": {"value": {}} for i in range(4)})

    seen = []
    async for row in storage.iter_rows(page_size=2):
        seen.append(row["_id"])
        if len(seen) == 1:
            # Delete a row from a page not yet read.
            await storage.delete(["k3"])

    assert "k3" not in seen
    assert len(seen) == 3


async def test_an_uninitialized_storage_raises_instead_of_yielding_nothing(tmp_path):
    from lightrag.exceptions import StorageNotInitializedError

    storage = JsonKVStorage(
        namespace="config",
        workspace="never-initialized",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=None,
    )
    with pytest.raises(StorageNotInitializedError):
        async for _ in storage.iter_rows():
            pass
