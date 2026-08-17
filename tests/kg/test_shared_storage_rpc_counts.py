"""RPC-count regression tests for the shared_storage Manager-proxy reductions.

Covers three optimizations, all of which collapse per-element proxy access to a
single slice/snapshot under multi-worker mode:

- ``set_all_update_flags`` / ``clear_all_update_flags`` slice the flag ListProxy
  once instead of re-indexing DictProxy+ListProxy per flag.
- ``get_namespace_data`` caches the created namespace dict per process, so a
  hot-path hit skips the internal lock and the __contains__/__getitem__ RPCs.
- The pipeline_status endpoint materializes ``history_messages`` with a slice
  (verified via a real Manager end-to-end + cross-fork cache lifetime).

Counting stand-ins mirror the ``tests/kg/test_reservation_primitives.py``
technique. Single-process mode exercises the same code paths with plain
containers (0 RPC); the real-Manager cases pin genuine proxy behavior.
"""

import os
import sys

import pytest

import lightrag.kg.shared_storage as ss
from lightrag.exceptions import PipelineNotInitializedError
from lightrag.kg.shared_storage import (
    clear_all_update_flags,
    finalize_share_data,
    get_final_namespace,
    get_namespace_data,
    get_update_flag,
    initialize_pipeline_status,
    initialize_share_data,
    set_all_update_flags,
)

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Counting stand-ins
# ---------------------------------------------------------------------------


class _CountingValue:
    """ValueProxy-shaped fake counting .value get/set."""

    def __init__(self, initial=False):
        self._v = initial
        self.get_calls = 0
        self.set_calls = 0

    @property
    def value(self):
        self.get_calls += 1
        return self._v

    @value.setter
    def value(self, v):
        self.set_calls += 1
        self._v = v


class _CountingList(list):
    """ListProxy-shaped fake distinguishing slice vs index getitem."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slice_getitem_calls = 0
        self.index_getitem_calls = 0
        self.len_calls = 0

    def __getitem__(self, key):
        if isinstance(key, slice):
            self.slice_getitem_calls += 1
        else:
            self.index_getitem_calls += 1
        return super().__getitem__(key)

    def __len__(self):
        self.len_calls += 1
        return super().__len__()


class _CountingDict(dict):
    """DictProxy-shaped fake counting contains/getitem/setitem."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contains_calls = 0
        self.getitem_calls = 0
        self.setitem_calls = 0

    def __contains__(self, key):
        self.contains_calls += 1
        return super().__contains__(key)

    def __getitem__(self, key):
        self.getitem_calls += 1
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        self.setitem_calls += 1
        return super().__setitem__(key, value)


@pytest.fixture
def single_process_shared_data():
    finalize_share_data()
    initialize_share_data(1)
    yield
    finalize_share_data()


# ---------------------------------------------------------------------------
# update flags: one slice, no per-index getitem, no __len__
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_all_update_flags_slices_once(single_process_shared_data):
    ws = "w"
    await get_update_flag("ns", workspace=ws)
    await get_update_flag("ns", workspace=ws)
    await get_update_flag("ns", workspace=ws)

    final_ns = get_final_namespace("ns", ws)
    values = [_CountingValue(), _CountingValue(), _CountingValue()]
    ss._update_flags[final_ns] = _CountingList(values)
    counting_list = ss._update_flags[final_ns]

    await set_all_update_flags("ns", workspace=ws)

    assert counting_list.slice_getitem_calls == 1
    assert counting_list.index_getitem_calls == 0
    assert counting_list.len_calls == 0
    assert all(v.set_calls == 1 for v in values)
    assert all(v._v is True for v in values)


@pytest.mark.asyncio
async def test_clear_all_update_flags_slices_once(single_process_shared_data):
    ws = "w"
    await get_update_flag("ns", workspace=ws)
    await get_update_flag("ns", workspace=ws)

    final_ns = get_final_namespace("ns", ws)
    values = [_CountingValue(True), _CountingValue(True)]
    ss._update_flags[final_ns] = _CountingList(values)
    counting_list = ss._update_flags[final_ns]

    await clear_all_update_flags("ns", workspace=ws)

    assert counting_list.slice_getitem_calls == 1
    assert counting_list.index_getitem_calls == 0
    assert counting_list.len_calls == 0
    assert all(v.set_calls == 1 for v in values)
    assert all(v._v is False for v in values)


# ---------------------------------------------------------------------------
# get_namespace_data: cold create, hot hit skips shared-dict access
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_namespace_cache_hot_hit_skips_shared_dict(single_process_shared_data):
    counting = _CountingDict()
    ss._shared_dicts = counting

    first = await get_namespace_data("ns", workspace="w")
    # cold path: one contains (miss) + one setitem (create) + one getitem (read)
    assert counting.contains_calls == 1
    assert counting.setitem_calls == 1

    counting.contains_calls = 0
    counting.getitem_calls = 0
    counting.setitem_calls = 0

    second = await get_namespace_data("ns", workspace="w")
    assert second is first
    # hot path: served entirely from the per-process cache
    assert counting.contains_calls == 0
    assert counting.getitem_calls == 0
    assert counting.setitem_calls == 0


@pytest.mark.asyncio
async def test_namespace_cache_isolates_workspaces(single_process_shared_data):
    a = await get_namespace_data("ns", workspace="w1")
    b = await get_namespace_data("ns", workspace="w2")
    assert a is not b
    assert ss._namespace_data_cache[get_final_namespace("ns", "w1")] is a
    assert ss._namespace_data_cache[get_final_namespace("ns", "w2")] is b


@pytest.mark.asyncio
async def test_namespace_cache_preserves_pipeline_not_initialized(
    single_process_shared_data,
):
    # Uncreated pipeline_status must still raise (cache only holds created NS).
    with pytest.raises(PipelineNotInitializedError):
        await get_namespace_data("pipeline_status", workspace="w")

    # After init it resolves and is cached.
    await initialize_pipeline_status(workspace="w")
    ps = await get_namespace_data("pipeline_status", workspace="w")
    assert ss._namespace_data_cache[get_final_namespace("pipeline_status", "w")] is ps


@pytest.mark.asyncio
async def test_namespace_cache_rebuilt_after_finalize():
    finalize_share_data()
    initialize_share_data(1)
    try:
        old = await get_namespace_data("ns", workspace="w")
    finally:
        finalize_share_data()
    assert ss._namespace_data_cache is None

    initialize_share_data(1)
    try:
        assert ss._namespace_data_cache == {}
        new = await get_namespace_data("ns", workspace="w")
        assert new is not old
    finally:
        finalize_share_data()


# ---------------------------------------------------------------------------
# Real Manager: genuine ListProxy slice + cross-fork cache lifetime
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_history_messages_slice_real_manager(tmp_path):
    finalize_share_data()
    initialize_share_data(2)
    try:
        await initialize_pipeline_status(workspace="w")
        ps = await get_namespace_data("pipeline_status", workspace="w")
        async with ss.get_internal_lock():
            ps["history_messages"].extend([f"line {i}" for i in range(5)])

        # Mirror the endpoint transform: one copy() then a slice on the nested
        # ListProxy (not list(proxy)).
        snapshot = ps.copy()
        history = snapshot["history_messages"][:]
        assert history == [f"line {i}" for i in range(5)]
        assert isinstance(history, list)
    finally:
        finalize_share_data()


@pytest.mark.skipif(
    not hasattr(os, "fork"), reason="cross-fork cache test requires os.fork"
)
@pytest.mark.filterwarnings("ignore:.*fork.*may lead to deadlocks:DeprecationWarning")
@pytest.mark.asyncio
async def test_namespace_cache_survives_fork_real_manager():
    # Simulate gunicorn master-preload -> worker-fork: parent warms the cache,
    # a forked child writes through the *cached* proxy, parent sees the write.
    # The child touches the proxy synchronously (a DictProxy setitem needs no
    # event loop); multiprocessing's ForkAwareLocal re-establishes the child's
    # own Manager connection on first use after the fork.
    finalize_share_data()
    initialize_share_data(2)
    try:
        final_ns = get_final_namespace("fork_ns", "w")
        ns = await get_namespace_data("fork_ns", workspace="w")
        ns["seed"] = "parent"  # ensure the namespace exists server-side
        assert ss._namespace_data_cache[final_ns] is ns

        pid = os.fork()
        if pid == 0:  # child
            code = 0
            try:
                cached = ss._namespace_data_cache[final_ns]
                cached["child_key"] = "child_wrote"
            except Exception:
                code = 1
            finally:
                os._exit(code)

        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0

        parent_ns = await get_namespace_data("fork_ns", workspace="w")
        assert parent_ns.get("child_key") == "child_wrote"
    finally:
        finalize_share_data()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
