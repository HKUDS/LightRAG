"""Test battery for the server-side atomic keyed-lock holder table.

Covers the guarantees the ``KeyedHolderTable`` redesign was built for:

* kill safety — a SIGKILLed holder in another process never deadlocks the
  keyed lock (the reason the client-held ``manager.RLock()`` guard was
  removed);
* RPC budget — one Manager RPC per uncontended acquire and one per release;
* reclaim CAS — one winner per dead record under thread contention;
* real-liveness mutual exclusion under stress;
* the ``start_delta`` clock-adjustment-safe process identity (Linux /proc
  tick track and non-Linux sandwich-sampled psutil track, via injected time
  sources);
* zombie deadness, the psutil-less fallback, the PID-aware ``_my_start_id``
  cache, spawn-start-method compatibility, and the /health lock-status
  semantics.

Direct ``KeyedHolderTable()`` instances (no Manager in between) are used
where a test must forge internal state that the production proxy refuses to
accept (the server always stamps the true ``start_delta``) or where counting
server-side calls requires same-process visibility.
"""

import asyncio
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
import time
import uuid

import pytest

import lightrag.kg.shared_storage as shared_storage
from lightrag.kg.shared_storage import (
    KeyedHolderTable,
    _holder_dead,
    _pid_alive,
    _start_delta,
    finalize_share_data,
    get_keyed_lock_status,
    get_storage_keyed_lock,
    initialize_share_data,
)

pytestmark = pytest.mark.offline


def _dead_pid() -> int:
    """A confirmed-dead PID: spawn a child, SIGKILL it, reap it."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    pid = proc.pid
    proc.kill()
    proc.wait()
    return pid


def _seed_direct(table: KeyedHolderTable, key: str, record: dict) -> None:
    """Forge an arbitrary holder record on a DIRECT instance (unit-layer only:
    such records cannot be injected through the proxy, the server would stamp
    the true start_delta)."""
    with table._lock:
        table._holders[key] = dict(record)


# ---------------------------------------------------------------------------
# Kill safety (the core regression this redesign exists for)
# ---------------------------------------------------------------------------


def _kill_safety_holder(acquired_evt) -> None:  # pragma: no cover - child process
    async def _run():
        ctx = shared_storage.get_storage_keyed_lock("victim", namespace="killsafe")
        await ctx.__aenter__()
        acquired_evt.set()
        await asyncio.sleep(120)  # hold until SIGKILLed

    asyncio.run(_run())


def _kill_safety_contender() -> None:  # pragma: no cover - child process
    async def _run():
        async with asyncio.timeout(15):
            async with shared_storage.get_storage_keyed_lock(
                "victim", namespace="killsafe"
            ):
                pass

    try:
        asyncio.run(_run())
    except (TimeoutError, asyncio.TimeoutError):
        os._exit(2)
    os._exit(0)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs the fork start method")
@pytest.mark.filterwarnings("ignore:.*fork.*:DeprecationWarning")
def test_sigkilled_holder_never_deadlocks_other_processes():
    """A worker SIGKILLed while holding a keyed lock must not deadlock another
    process's acquire: the next try_acquire confirms the owner dead and
    reclaims the record. (The old manager.RLock() registry guard hung forever
    here.) The contender runs in a terminable child with its own timeout AND a
    parent-side join timeout, so a regression fails the test instead of
    hanging the suite."""
    finalize_share_data()
    initialize_share_data(2)
    try:
        ctx = multiprocessing.get_context("fork")
        acquired = ctx.Event()
        holder = ctx.Process(target=_kill_safety_holder, args=(acquired,), daemon=True)
        holder.start()
        assert acquired.wait(10), "holder child failed to acquire the keyed lock"

        os.kill(holder.pid, signal.SIGKILL)
        holder.join(10)  # reap (zombie deadness is covered separately anyway)

        contender = ctx.Process(target=_kill_safety_contender, daemon=True)
        contender.start()
        contender.join(30)
        if contender.is_alive():
            contender.kill()
            contender.join(5)
            pytest.fail("contender deadlocked on a SIGKILLed holder's keyed lock")
        assert contender.exitcode == 0
    finally:
        finalize_share_data()


# ---------------------------------------------------------------------------
# Zombie deadness
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shared_storage.psutil is None, reason="needs psutil")
def test_zombie_holder_is_confirmed_dead():
    """A SIGKILLed-but-not-reaped child (its wedged parent has not called
    wait()) is a zombie: it executes no code and cannot be using the lock, so
    liveness must report dead — os.kill(pid, 0) alone would report it alive
    and stall reclaim until the reap."""
    psutil = shared_storage.psutil
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        os.kill(proc.pid, signal.SIGKILL)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if psutil.Process(proc.pid).status() == psutil.STATUS_ZOMBIE:
                break
            time.sleep(0.01)
        else:
            pytest.fail("child never became a zombie")

        assert _pid_alive(proc.pid) is False
        assert shared_storage._process_alive(proc.pid, None) is False
        # Lock reclaim ultimately rides on _holder_dead:
        assert (
            _holder_dead({"owner_pid": proc.pid, "lease_id": "z", "start_delta": None})
            is True
        )
    finally:
        proc.wait()  # reap


# ---------------------------------------------------------------------------
# RPC budget and proxy single-winner
# ---------------------------------------------------------------------------


async def test_uncontended_cycle_is_one_rpc_each_way(monkeypatch):
    """Uncontended acquire = exactly one try_acquire RPC; release = exactly one
    release RPC (the whole point of moving the check-and-set server-side)."""
    finalize_share_data()
    initialize_share_data(2)
    try:
        calls = []
        original = shared_storage.BaseProxy._callmethod

        def counting(self, methodname, args=(), kwds=None):
            calls.append(methodname)
            return original(self, methodname, args, kwds or {})

        monkeypatch.setattr(shared_storage._HolderTableProxy, "_callmethod", counting)

        ctx = get_storage_keyed_lock("k", namespace="rpc")
        await ctx.__aenter__()
        assert calls == ["try_acquire"]
        calls.clear()
        await ctx.__aexit__(None, None, None)
        assert calls == ["release"]
    finally:
        finalize_share_data()


def test_proxy_grant_is_single_winner_and_owner_checked():
    finalize_share_data()
    initialize_share_data(2)
    try:
        table = shared_storage._keyed_holder_table
        assert table.try_acquire("w", {"owner_pid": os.getpid(), "lease_id": "a"})
        assert (
            table.try_acquire("w", {"owner_pid": os.getpid(), "lease_id": "b"}) is False
        )
        assert table.release("w", "b") is False  # not the owner
        assert table.release("w", "a") is True
        assert table.release("w", "a") is False  # idempotent second release
    finally:
        finalize_share_data()


# ---------------------------------------------------------------------------
# Reclaim CAS: exactly one winner per dead record
# ---------------------------------------------------------------------------


def test_dead_record_reclaim_has_exactly_one_winner():
    """N threads race to reclaim the same confirmed-dead record; the lease_id
    CAS under the server lock lets exactly one try_acquire return True."""
    table = KeyedHolderTable()
    key = "cas"
    _seed_direct(
        table,
        key,
        {"owner_pid": _dead_pid(), "lease_id": "dead-lease", "start_delta": None},
    )

    n = 16
    barrier = threading.Barrier(n)
    results = []
    results_lock = threading.Lock()

    def contender(i: int) -> None:
        record = {"owner_pid": os.getpid(), "lease_id": f"lease-{i}"}
        barrier.wait()
        won = table.try_acquire(key, record)
        with results_lock:
            results.append(won)

    threads = [threading.Thread(target=contender, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert len(results) == n
    assert sum(results) == 1  # exactly one winner, no double grant


# ---------------------------------------------------------------------------
# Real-liveness mutual exclusion stress (no liveness mocks)
# ---------------------------------------------------------------------------


def test_mutual_exclusion_stress_with_real_liveness():
    table = KeyedHolderTable()
    key = "stress"
    active = 0
    max_active = 0
    meta = threading.Lock()

    def worker() -> None:
        nonlocal active, max_active
        for _ in range(40):
            record = {"owner_pid": os.getpid(), "lease_id": uuid.uuid4().hex}
            while not table.try_acquire(key, record):
                time.sleep(0.0002)
            with meta:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.0005)  # widen the critical section
            with meta:
                active -= 1
            assert table.release(key, record["lease_id"]) is True

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert max_active == 1  # never two holders at once
    assert table.holders_snapshot() == {}


def test_no_unbounded_growth_over_distinct_keys():
    """The table holds a record only while the key is HELD — distinct keys do
    not accumulate (tighter than the old three bookkeeping tables)."""
    table = KeyedHolderTable()
    for i in range(2000):
        record = {"owner_pid": os.getpid(), "lease_id": f"l{i}"}
        assert table.try_acquire(f"key-{i}", record)
        assert table.release(f"key-{i}", f"l{i}")
    assert table.holders_snapshot() == {}


# ---------------------------------------------------------------------------
# start_delta semantics (injected time sources)
# ---------------------------------------------------------------------------


def test_start_delta_baseline_and_forged_reuse_self():
    """Stamp-then-recompute for the same live process is equal → alive; a
    forged smaller stamp simulates PID reuse → dead. Platform-independent (the
    forge exceeds both the Linux any-difference criterion and the non-Linux
    1s tolerance)."""
    d0 = _start_delta(os.getpid())
    if d0 is None:
        pytest.skip("no process start identity available on this platform")
    assert _start_delta(os.getpid()) == d0  # deterministic recompute
    me = {"owner_pid": os.getpid(), "lease_id": "x"}
    assert _holder_dead({**me, "start_delta": d0}) is False
    assert _holder_dead({**me, "start_delta": d0 - 2}) is True


def test_start_delta_common_mode_offset_cancels(monkeypatch):
    """A wall-clock step applied to ALL reads of one complete sample (the
    common-mode case) must not change the delta or the verdict."""
    monkeypatch.setattr(sys, "platform", "darwin")
    base = {os.getpid(): 1000.0, 4242: 1234.5}
    state = {"offset": 0.0}
    monkeypatch.setattr(
        shared_storage, "_read_create_time", lambda pid: base[pid] + state["offset"]
    )

    clean = shared_storage._start_delta(4242)
    assert clean == pytest.approx(234.5)
    state["offset"] = 500.0  # NTP/manual step between samples
    assert shared_storage._start_delta(4242) == pytest.approx(clean)

    monkeypatch.setattr(shared_storage, "_pid_alive", lambda pid: True)
    record = {"owner_pid": 4242, "lease_id": "x", "start_delta": clean}
    assert shared_storage._holder_dead(record) is False


def test_start_delta_polluted_window_returns_none_and_never_kills(monkeypatch):
    """A clock adjustment landing INSIDE the sampling window breaks the anchor
    sandwich (a0 != a1) on every retry → None → conservatively alive."""
    monkeypatch.setattr(sys, "platform", "darwin")
    state = {"clock": 1000.0}

    def jumping_read(pid):
        state["clock"] += 100.0  # the clock moves between every read
        return state["clock"]

    monkeypatch.setattr(shared_storage, "_read_create_time", jumping_read)
    assert shared_storage._start_delta(4242) is None

    monkeypatch.setattr(shared_storage, "_pid_alive", lambda pid: True)
    record = {"owner_pid": 4242, "lease_id": "x", "start_delta": 1.0}
    assert shared_storage._holder_dead(record) is False  # polluted → never dead


def test_start_delta_retry_recovers_from_one_polluted_sample(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    # First triple polluted (anchor mismatch), second clean.
    reads = iter([1000.0, 1234.5, 1600.0, 2000.0, 2234.5, 2000.0])
    monkeypatch.setattr(shared_storage, "_read_create_time", lambda pid: next(reads))
    assert shared_storage._start_delta(4242) == pytest.approx(234.5)


def test_holder_dead_non_linux_one_sided_tolerance(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(shared_storage, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(shared_storage, "_start_delta", lambda pid: 100.0)

    def record(d0):
        return {"owner_pid": 4242, "lease_id": "x", "start_delta": d0}

    assert shared_storage._holder_dead(record(100.0)) is False  # same identity
    # exactly at tolerance (d1 == d0 + 1.0) → not dead; just beyond → dead
    assert shared_storage._holder_dead(record(99.0)) is False
    assert shared_storage._holder_dead(record(98.9)) is True
    # Backwards clock step + reuse: recomputed delta SMALLER than the stamp.
    # The one-sided criterion deliberately judges alive (documented liveness
    # gap, never a double-hold).
    assert shared_storage._holder_dead(record(150.0)) is False


def test_holder_dead_linux_any_tick_difference(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(shared_storage, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(shared_storage, "_start_delta", lambda pid: 100)

    def record(d0):
        return {"owner_pid": 4242, "lease_id": "x", "start_delta": d0}

    assert shared_storage._holder_dead(record(100)) is False
    assert shared_storage._holder_dead(record(99)) is True  # ticks differ = dead
    assert shared_storage._holder_dead(record(101)) is True  # even when smaller


@pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="Linux /proc tick track"
)
def test_start_delta_linux_never_touches_wall_clock(monkeypatch):
    """Even with psutil installed, the Linux track reads /proc ticks only —
    downgrading the monotonic tick identity to wall-clock sampling would be a
    pure regression."""
    wall_reads = []
    monkeypatch.setattr(
        shared_storage,
        "_read_create_time",
        lambda pid: wall_reads.append(pid) or 0.0,
    )
    assert _start_delta(os.getpid()) == 0  # self vs self: zero ticks
    assert wall_reads == []


def test_holder_dead_none_identity_falls_back_to_pid_probe(monkeypatch):
    # Dead pid → dead regardless of any delta.
    dead = _dead_pid()
    assert _holder_dead({"owner_pid": dead, "lease_id": "x", "start_delta": None})
    # Live pid without a stamped identity → alive, and the delta is never
    # even sampled.
    samples = []
    monkeypatch.setattr(
        shared_storage, "_start_delta", lambda pid: samples.append(pid) or 0
    )
    assert (
        shared_storage._holder_dead(
            {"owner_pid": os.getpid(), "lease_id": "x", "start_delta": None}
        )
        is False
    )
    assert samples == []
    # No owner pid at all → never dead.
    assert shared_storage._holder_dead({"lease_id": "x"}) is False


def test_grant_stamps_identity_and_live_rejection_stamps_nothing(monkeypatch):
    """start_delta sampling happens only on grant paths: the grant stamps the
    candidate once; a poll rejected by a live holder recomputes the HOLDER's
    identity (PID-reuse check) but never samples the rejected candidate."""
    table = KeyedHolderTable()
    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        sampled = []
        real = shared_storage._start_delta

        def counting(pid):
            sampled.append(pid)
            return real(pid)

        monkeypatch.setattr(shared_storage, "_start_delta", counting)

        assert table.try_acquire("k", {"owner_pid": holder.pid, "lease_id": "l1"})
        assert sampled == [holder.pid]  # grant stamped the candidate once
        stamped = table.holders_snapshot()["k"]
        assert "start_delta" in stamped

        sampled.clear()
        rejected = table.try_acquire("k", {"owner_pid": os.getpid(), "lease_id": "l2"})
        assert rejected is False
        assert os.getpid() not in sampled  # candidate never stamped
        assert sampled == [holder.pid]  # only the holder's reuse recompute
    finally:
        holder.kill()
        holder.wait()


# ---------------------------------------------------------------------------
# _my_start_id PID-aware cache (fork inheritance fix)
# ---------------------------------------------------------------------------


def test_my_start_id_cache_is_pid_aware(monkeypatch):
    """A cache warmed by the parent must be recomputed when the PID changes —
    a bare 'already computed' flag survives fork and makes a worker publish
    reservation records with the MASTER's identity."""
    monkeypatch.setattr(
        shared_storage, "_read_proc_starttime", lambda pid: f"token-{pid}"
    )
    # Simulate the post-fork state: cache warmed under a different PID.
    monkeypatch.setattr(shared_storage, "_MY_START_ID_CACHE", "parent-token")
    monkeypatch.setattr(shared_storage, "_MY_START_ID_PID", os.getpid() - 1)
    assert shared_storage._my_start_id() == f"token-{os.getpid()}"
    # And it stays cached for the CURRENT pid.
    monkeypatch.setattr(
        shared_storage, "_read_proc_starttime", lambda pid: "should-not-be-read"
    )
    assert shared_storage._my_start_id() == f"token-{os.getpid()}"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="real /proc assertion")
@pytest.mark.filterwarnings("ignore:.*fork.*:DeprecationWarning")
def test_my_start_id_recomputed_after_real_fork():
    parent_id = shared_storage._my_start_id()  # warm the cache BEFORE forking
    assert parent_id is not None

    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        code = 1
        try:
            os.close(read_fd)
            child_id = shared_storage._my_start_id()
            expected = shared_storage._read_proc_starttime(os.getpid())
            if child_id == expected and child_id != parent_id:
                code = 0
            os.write(write_fd, b"1" if code == 0 else b"0")
        finally:
            os._exit(code)

    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as fh:
        verdict = fh.read(1)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert verdict == b"1"

    # Parent's own cache is untouched by the child's recompute.
    assert shared_storage._my_start_id() == parent_id


# ---------------------------------------------------------------------------
# Minimal-install (no psutil) fallback
# ---------------------------------------------------------------------------


def test_pid_alive_fallback_without_psutil(monkeypatch):
    monkeypatch.setattr(shared_storage, "psutil", None)
    assert shared_storage._pid_alive(os.getpid()) is True
    assert shared_storage._pid_alive(_dead_pid()) is False
    # No psutil → no wall-clock identity source either.
    assert shared_storage._read_create_time(os.getpid()) is None
    monkeypatch.setattr(sys, "platform", "darwin")
    assert shared_storage._start_delta(os.getpid()) is None


@pytest.mark.skipif(
    shared_storage.psutil is None, reason="needs psutil to observe the zombie"
)
def test_pid_alive_fallback_keeps_zombies_alive(monkeypatch):
    """Without psutil the historical os.kill(pid, 0) behavior is preserved:
    zombies count as alive (dead-only reclaim then waits for the reap)."""
    psutil = shared_storage.psutil
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        os.kill(proc.pid, signal.SIGKILL)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if psutil.Process(proc.pid).status() == psutil.STATUS_ZOMBIE:
                break
            time.sleep(0.01)
        else:
            pytest.fail("child never became a zombie")
        monkeypatch.setattr(shared_storage, "psutil", None)
        assert shared_storage._pid_alive(proc.pid) is True
    finally:
        proc.wait()


# ---------------------------------------------------------------------------
# Spawn start-method compatibility of the custom manager type
# ---------------------------------------------------------------------------


def test_custom_manager_type_works_under_spawn():
    """The class and its registration are module-level, so a spawn-started
    Manager server can import them — pin that with a real spawn server."""
    ctx = multiprocessing.get_context("spawn")
    manager = shared_storage._LightRAGManager(ctx=ctx)
    manager.start()
    try:
        table = manager.KeyedHolderTable()
        record = {"owner_pid": os.getpid(), "lease_id": "spawn-lease"}
        assert table.try_acquire("k", record) is True
        snapshot = table.holders_snapshot()
        assert snapshot["k"]["owner_pid"] == os.getpid()
        assert table.holder_count() == 1
        assert table.release("k", "spawn-lease") is True
        assert table.holder_count() == 0
    finally:
        manager.shutdown()


# ---------------------------------------------------------------------------
# /health lock status semantics
# ---------------------------------------------------------------------------


async def test_lock_status_counts_currently_held_keys():
    """total_mp_locks now means 'currently held keys' (server-side count);
    pending_mp_cleanup is fixed at 0 — keys preserved, value semantics new."""
    finalize_share_data()
    initialize_share_data(2)
    try:
        async with get_storage_keyed_lock(["k1", "k2"], namespace="status"):
            status = shared_storage._storage_keyed_lock.get_lock_status()
            assert status["total_mp_locks"] == 2
            assert status["pending_mp_cleanup"] == 0

            public = get_keyed_lock_status()  # delegates to get_lock_status
            assert public["total_mp_locks"] == 2
            assert public["pending_mp_cleanup"] == 0
            assert public["process_id"] == os.getpid()

        assert (
            shared_storage._storage_keyed_lock.get_lock_status()["total_mp_locks"] == 0
        )

        cleanup = shared_storage.cleanup_keyed_lock()
        assert cleanup["cleanup_performed"]["mp_cleaned"] == 0  # compat, always 0
    finally:
        finalize_share_data()
