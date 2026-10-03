"""A keyed lock asked for the same key twice must take it once.

Each key's lock is not reentrant, so before duplicates were dropped a context
such as ``get_storage_keyed_lock(sorted([src, tgt]))`` for a self-loop edge
(``src == tgt``) waited forever on the lock it already held (issue #4148).
"""

import asyncio

import pytest

import lightrag.kg.shared_storage as shared_storage
from lightrag.kg.shared_storage import (
    _get_combined_key,
    finalize_share_data,
    get_storage_keyed_lock,
    initialize_share_data,
)

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data(1)
    yield
    finalize_share_data()


@pytest.mark.offline
async def test_duplicate_keys_are_acquired_once_and_released():
    keyed = shared_storage._storage_keyed_lock
    combined = _get_combined_key("GraphDB", "A")

    async def take():
        async with get_storage_keyed_lock(["A", "A"], namespace="GraphDB"):
            assert keyed._async_lock_count[combined] == 1
            return True

    assert await asyncio.wait_for(take(), timeout=2)
    assert combined not in keyed._async_lock
    assert combined not in keyed._async_lock_count


@pytest.mark.offline
async def test_deduplicated_key_still_excludes_other_holders():
    entered = asyncio.Event()
    release = asyncio.Event()
    order: list[str] = []

    async def self_loop_writer():
        async with get_storage_keyed_lock(["A", "B", "A"], namespace="GraphDB"):
            order.append("self-loop")
            entered.set()
            await release.wait()

    async def other_writer():
        await entered.wait()
        async with get_storage_keyed_lock("A", namespace="GraphDB"):
            order.append("other")

    first = asyncio.create_task(self_loop_writer())
    second = asyncio.create_task(other_writer())
    await asyncio.wait_for(entered.wait(), timeout=2)
    for _ in range(5):
        await asyncio.sleep(0)
    assert order == ["self-loop"]  # "A" is still held

    release.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=2)
    assert order == ["self-loop", "other"]
