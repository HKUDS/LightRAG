"""Unit tests for the lock-free ``append_pipeline_log`` status-log helper.

Stage-1 coverage (helper only — depends on NO call-site conversion):
  - semantics: single / multi / set_latest=False / no-op inputs;
  - both backing paths: plain dict/list AND a real Manager DictProxy/ListProxy;
  - never-raise contract, probed at each of __setitem__ / .get / .extend, and
    even when the diagnostic logging itself fails;
  - the "single extend is indivisible on the Manager backing list" assumption,
    pinned by a real multi-process concurrent-writer test (no message loss).

The helper must NEVER raise: a raising status-log write would mask the real
exception at call sites shaped like ``except ...: append_pipeline_log(...); raise``.
"""

import logging
import multiprocessing as mp

import pytest

from lightrag.kg.shared_storage import append_pipeline_log

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Fakes that raise at a specific operation, to probe the never-raise contract.
# ---------------------------------------------------------------------------


class _RaisingHistory(list):
    def extend(self, iterable):  # noqa: D401 - test double
        raise RuntimeError("extend boom")


class _FakeStatus:
    """Mapping-shaped status double; can be told to raise on a given op.

    Not a ``dict`` subclass so that setup never routes through the overridden
    ``__setitem__`` we may want to make raise.
    """

    def __init__(self, history=None, raise_on=()):
        self.latest = None
        self.latest_set = False
        self._history = history if history is not None else []
        self._raise_on = set(raise_on)

    def __setitem__(self, key, value):
        if "setitem" in self._raise_on:
            raise RuntimeError("setitem boom")
        if key == "latest_message":
            self.latest = value
            self.latest_set = True

    def get(self, key, default=None):
        if "get" in self._raise_on:
            raise RuntimeError("get boom")
        if key == "history_messages":
            return self._history
        return default


# ---------------------------------------------------------------------------
# Semantics — plain dict/list backing path.
# ---------------------------------------------------------------------------


def test_none_status_is_noop():
    # Must not raise; nothing to assert beyond "returns cleanly".
    append_pipeline_log(None, "x")


def test_no_messages_is_noop():
    status = {"latest_message": "keep", "history_messages": ["keep"]}
    append_pipeline_log(status)
    assert status == {"latest_message": "keep", "history_messages": ["keep"]}


def test_single_message_sets_latest_and_appends():
    status = {"latest_message": "", "history_messages": []}
    append_pipeline_log(status, "a")
    assert status["latest_message"] == "a"
    assert status["history_messages"] == ["a"]


def test_multiple_messages_latest_is_last_and_all_appended_in_order():
    status = {"latest_message": "", "history_messages": []}
    append_pipeline_log(status, "a", "b", "c")
    assert status["latest_message"] == "c"
    assert status["history_messages"] == ["a", "b", "c"]


def test_set_latest_false_appends_without_touching_latest():
    status = {"latest_message": "prev", "history_messages": ["x"]}
    append_pipeline_log(status, "y", "z", set_latest=False)
    assert status["latest_message"] == "prev"
    assert status["history_messages"] == ["x", "y", "z"]


def test_missing_history_writes_latest_and_skips_append():
    status = {"latest_message": ""}  # no history_messages key
    append_pipeline_log(status, "only-latest")
    assert status["latest_message"] == "only-latest"
    assert "history_messages" not in status


def test_none_history_writes_latest_and_skips_append():
    status = {"latest_message": "", "history_messages": None}
    append_pipeline_log(status, "hi")
    assert status["latest_message"] == "hi"
    assert status["history_messages"] is None


# ---------------------------------------------------------------------------
# Never-raise contract — probed at each operation independently.
# ---------------------------------------------------------------------------


def test_never_raises_when_setitem_fails_but_still_appends_history():
    status = _FakeStatus(raise_on={"setitem"})
    append_pipeline_log(status, "a", "b")  # must not raise
    # latest failed, but history still got the whole group.
    assert status.latest_set is False
    assert status._history == ["a", "b"]


def test_never_raises_when_get_fails_but_still_writes_latest():
    status = _FakeStatus(raise_on={"get"})
    append_pipeline_log(status, "a", "b")  # must not raise
    assert status.latest == "b"  # latest still written


def test_never_raises_when_extend_fails_but_still_writes_latest():
    status = _FakeStatus(history=_RaisingHistory(), raise_on=set())
    append_pipeline_log(status, "a")  # must not raise
    assert status.latest == "a"
    assert list(status._history) == []  # extend raised → nothing appended


def test_never_raises_even_when_diagnostic_logging_itself_fails(monkeypatch):
    # Break only the "lightrag" logger's .debug (not the global logging
    # subsystem, which pytest's own plugins rely on).
    def _boom_debug(*_args, **_kwargs):
        raise RuntimeError("logging subsystem down")

    monkeypatch.setattr(logging.getLogger("lightrag"), "debug", _boom_debug)
    status = _FakeStatus(raise_on={"setitem"})
    # Both the write AND its diagnostic logging fail — still must not raise.
    append_pipeline_log(status, "a")


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


@_skip_no_fork
def test_real_listproxy_extend_and_latest():
    mgr = _FORK.Manager()
    try:
        status = mgr.dict({"latest_message": "", "history_messages": mgr.list()})
        append_pipeline_log(status, "a", "b", "c")
        assert status["latest_message"] == "c"
        assert list(status["history_messages"]) == ["a", "b", "c"]
    finally:
        mgr.shutdown()


def _mp_writer(status, prefix, n):
    # Each call writes a two-message group; a single ``extend`` must append the
    # pair atomically so nothing gets wedged between the begin and its end.
    for i in range(n):
        append_pipeline_log(status, f"{prefix}-{i}-begin", f"{prefix}-{i}-end")


@_skip_no_fork
def test_concurrent_multiprocess_writers_lose_no_messages_and_groups_stay_intact():
    """Pins the atomicity assumption: concurrent multi-message ``extend`` on a
    real Manager ListProxy from several processes never drops a message NOR tears
    a group (each ``(begin, end)`` pair stays contiguous, begin then end)."""
    mgr = _FORK.Manager()
    try:
        status = mgr.dict({"latest_message": "", "history_messages": mgr.list()})
        n_writers, per_writer = 4, 50
        procs = [
            _FORK.Process(target=_mp_writer, args=(status, f"w{w}", per_writer))
            for w in range(n_writers)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=30)
            assert p.exitcode == 0

        history = list(status["history_messages"])
        # No loss: every message of every group present exactly once.
        assert len(history) == n_writers * per_writer * 2
        expected = {
            f"w{w}-{i}-{end}"
            for w in range(n_writers)
            for i in range(per_writer)
            for end in ("begin", "end")
        }
        assert set(history) == expected
        # No tearing: each "*-begin" is immediately followed by its matching
        # "*-end" — impossible unless ``extend`` appends the group indivisibly.
        for idx, msg in enumerate(history):
            if msg.endswith("-begin"):
                assert history[idx + 1] == msg[: -len("-begin")] + "-end", (
                    f"group torn at {idx}: {history[idx : idx + 2]}"
                )
    finally:
        mgr.shutdown()
