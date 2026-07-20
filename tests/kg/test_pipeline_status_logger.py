"""Unit tests for the operation-scoped ``PipelineStatusLogger``.

Coverage:

  - history-handle caching: the first write fetches
    ``pipeline_status.get("history_messages")`` exactly once; later writes
    reuse the cached handle (the counting double proves the ``.get()`` is
    cached — it does NOT claim to count total Manager round-trips);
  - cache recovery: a failed/None fetch is retried on the next write, a
    failed ``extend`` drops the cache, and messages of the failed call are
    never re-sent;
  - history identity: in-place resets (``del h[:]`` / ``h[:] = [...]``) keep
    the cached handle live, while replacing the list object orphans it (the
    documented limitation, pinned both ways);
  - never-raise contract for ``log`` / ``append``, including a failing
    diagnostic logger;
  - real Manager proxies: a fork-inherited cached ListProxy stays writable in
    the child and visible in the parent, and concurrent cached-handle writers
    lose no messages and never tear a multi-message group.
"""

import logging
import multiprocessing as mp

import pytest

from lightrag.kg.shared_storage import PipelineStatusLogger

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Test doubles.
# ---------------------------------------------------------------------------


class _CountingStatus:
    """Mapping-shaped double counting ``.get("history_messages")`` calls.

    Each such call is the fetch the logger is supposed to cache. Set
    ``history`` to None to simulate a missing/late-initialized key, and
    ``fail_next_gets`` to make the next N fetches raise.
    """

    def __init__(self, history=()):
        self.get_calls = 0
        self.latest = None
        self.latest_writes = []
        self.history = list(history)
        self.fail_next_gets = 0

    def __setitem__(self, key, value):
        if key == "latest_message":
            self.latest = value
            self.latest_writes.append(value)

    def get(self, key, default=None):
        if key != "history_messages":
            return default
        self.get_calls += 1
        if self.fail_next_gets > 0:
            self.fail_next_gets -= 1
            raise RuntimeError("get boom")
        return self.history


class _FlakyHistory(list):
    """List whose ``extend`` raises for the first ``fail_times`` calls."""

    def __init__(self, *args, fail_times=1):
        super().__init__(*args)
        self.fail_times = fail_times

    def extend(self, iterable):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("extend boom")
        super().extend(iterable)


class _RaisingSetitemStatus(_CountingStatus):
    def __setitem__(self, key, value):
        raise RuntimeError("setitem boom")


# ---------------------------------------------------------------------------
# History-handle caching.
# ---------------------------------------------------------------------------


def test_construction_is_fetch_free():
    status = _CountingStatus()
    PipelineStatusLogger(status)
    assert status.get_calls == 0
    assert status.latest_writes == []


def test_first_write_fetches_history_once_then_reuses_cached_handle():
    status = _CountingStatus()
    status_logger = PipelineStatusLogger(status)
    status_logger.log("m0")
    assert status.get_calls == 1
    for i in range(1, 11):
        status_logger.log(f"m{i}")
    assert status.get_calls == 1  # cached handle, no re-fetch
    assert status.history == [f"m{i}" for i in range(11)]
    assert status.latest == "m10"


def test_append_writes_history_only():
    status = _CountingStatus()
    status_logger = PipelineStatusLogger(status)
    status_logger.append("a", "b")
    assert status.latest is None
    assert status.history == ["a", "b"]
    assert status.get_calls == 1


def test_multi_message_log_sets_latest_to_last_and_appends_in_order():
    status = _CountingStatus()
    status_logger = PipelineStatusLogger(status)
    status_logger.log("a", "b", "c")
    assert status.latest == "c"
    assert status.history == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Cache recovery.
# ---------------------------------------------------------------------------


def test_get_failure_is_retried_next_write_without_resending_lost_message():
    status = _CountingStatus()
    status.fail_next_gets = 1
    status_logger = PipelineStatusLogger(status)
    status_logger.log("lost")  # fetch raises; latest still written
    assert status.get_calls == 1
    assert status.history == []
    assert status.latest == "lost"
    status_logger.log("kept")  # fetch retried and now cached
    assert status.get_calls == 2
    assert status.history == ["kept"]  # "lost" is NOT re-sent
    status_logger.log("more")
    assert status.get_calls == 2


def test_get_returning_none_is_not_cached_and_retried_after_late_init():
    status = _CountingStatus()
    status.history = None  # key missing / not yet initialized
    status_logger = PipelineStatusLogger(status)
    status_logger.log("lost")
    assert status.get_calls == 1
    assert status.latest == "lost"
    status.history = []  # late initialization
    status_logger.log("kept")
    assert status.get_calls == 2
    assert status.history == ["kept"]


def test_extend_failure_drops_cache_and_refetches_without_resending():
    history = _FlakyHistory(fail_times=1)
    status = _CountingStatus()
    status.history = history
    status_logger = PipelineStatusLogger(status)
    status_logger.log("lost")  # extend raises → cache dropped, no retry
    assert status.get_calls == 1
    assert list(history) == []
    status_logger.log("kept")  # re-fetches the handle
    assert status.get_calls == 2
    assert list(history) == ["kept"]  # "lost" is NOT re-sent


# ---------------------------------------------------------------------------
# History identity across in-place resets (repo invariant: never replaced).
# ---------------------------------------------------------------------------


def test_in_place_clear_keeps_cached_handle_live():
    status = _CountingStatus()
    status_logger = PipelineStatusLogger(status)
    status_logger.log("a")
    # Reservation-style reset (acquire_processing_reservation).
    del status.history[:]
    status_logger.log("b")
    assert status.history == ["b"]
    assert status.get_calls == 1  # no re-fetch needed


def test_in_place_slice_assignment_keeps_cached_handle_live():
    status = _CountingStatus()
    status_logger = PipelineStatusLogger(status)
    status_logger.log("a")
    # Document-routes-style reset.
    status.history[:] = ["preset"]
    status_logger.log("c")
    assert status.history == ["preset", "c"]
    assert status.get_calls == 1


def test_replacing_history_object_orphans_the_cache():
    """Pins the documented limitation: replacing ``history_messages`` (which
    violates the in-place-only repo invariant) leaves the logger writing to
    the orphaned list."""
    status = _CountingStatus()
    status_logger = PipelineStatusLogger(status)
    status_logger.log("a")
    old_history = status.history
    status.history = []  # invariant violation: replaced, not reset in place
    status_logger.log("b")
    assert old_history == ["a", "b"]  # went to the orphan
    assert status.history == []


# ---------------------------------------------------------------------------
# Never-raise contract.
# ---------------------------------------------------------------------------


def test_none_status_all_methods_noop():
    status_logger = PipelineStatusLogger(None)
    status_logger.log("x")
    status_logger.append("x")


def test_no_messages_is_noop_and_fetch_free():
    status = _CountingStatus()
    status_logger = PipelineStatusLogger(status)
    status_logger.log()
    status_logger.append()
    assert status.get_calls == 0
    assert status.latest is None


def test_never_raises_when_setitem_fails_but_still_appends_history():
    status = _RaisingSetitemStatus()
    status_logger = PipelineStatusLogger(status)
    status_logger.log("a", "b")  # must not raise
    assert status.history == ["a", "b"]


def test_never_raises_even_when_diagnostic_logging_itself_fails(monkeypatch):
    def _boom_debug(*_args, **_kwargs):
        raise RuntimeError("logging subsystem down")

    monkeypatch.setattr(logging.getLogger("lightrag"), "debug", _boom_debug)
    status = _CountingStatus()
    status.fail_next_gets = 1
    status_logger = PipelineStatusLogger(status)
    # Both the fetch AND its diagnostic logging fail — still must not raise.
    status_logger.log("a")


# ---------------------------------------------------------------------------
# Real Manager DictProxy / ListProxy backing path.
# ---------------------------------------------------------------------------


def _fork_ctx():
    try:
        return mp.get_context("fork")
    except (ValueError, RuntimeError):
        return None


_FORK = _fork_ctx()
_skip_no_fork = pytest.mark.skipif(
    _FORK is None, reason="fork start method unavailable (Manager proxy sharing test)"
)


def _child_logs(status_logger, message):
    status_logger.log(message)


@_skip_no_fork
def test_fork_inherited_cached_proxy_writes_visible_in_parent():
    """A logger whose history handle was cached BEFORE the fork keeps working
    in the child (BaseProxy re-registers its connection after fork) and the
    child's writes are visible to the parent."""
    mgr = _FORK.Manager()
    try:
        status = mgr.dict({"latest_message": "", "history_messages": mgr.list()})
        status_logger = PipelineStatusLogger(status)
        status_logger.log("parent-1")  # warms the cache in the parent
        proc = _FORK.Process(target=_child_logs, args=(status_logger, "child-1"))
        proc.start()
        proc.join(timeout=30)
        assert proc.exitcode == 0
        assert list(status["history_messages"]) == ["parent-1", "child-1"]
        assert status["latest_message"] == "child-1"
    finally:
        mgr.shutdown()


def _mp_cached_writer(status, prefix, n):
    # One logger per process: after the first call every write goes through
    # the cached ListProxy, so this pins the cached-handle path specifically.
    status_logger = PipelineStatusLogger(status)
    for i in range(n):
        status_logger.log(f"{prefix}-{i}-begin", f"{prefix}-{i}-end")


@_skip_no_fork
def test_concurrent_cached_writers_lose_no_messages_and_groups_stay_intact():
    """Concurrent multi-message ``extend`` on cached ListProxies from several
    processes never drops a message NOR tears a group."""
    mgr = _FORK.Manager()
    try:
        status = mgr.dict({"latest_message": "", "history_messages": mgr.list()})
        n_writers, per_writer = 4, 50
        procs = [
            _FORK.Process(target=_mp_cached_writer, args=(status, f"w{w}", per_writer))
            for w in range(n_writers)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=30)
            assert p.exitcode == 0

        history = list(status["history_messages"])
        assert len(history) == n_writers * per_writer * 2
        expected = {
            f"w{w}-{i}-{end}"
            for w in range(n_writers)
            for i in range(per_writer)
            for end in ("begin", "end")
        }
        assert set(history) == expected
        for idx, msg in enumerate(history):
            if msg.endswith("-begin"):
                assert history[idx + 1] == msg[: -len("-begin")] + "-end", (
                    f"group torn at {idx}: {history[idx : idx + 2]}"
                )
    finally:
        mgr.shutdown()
