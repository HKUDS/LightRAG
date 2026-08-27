"""Coverage for the per-write ordering token shared by the file-backed vdbs.

``__created_at__`` is whole seconds (``int(time.time())``), so two writes
inside the same second tie — and a tie used to resolve as "the redo replay
proceeds", letting a stale record overwrite a genuinely newer durable row
(PR #3709 review). ``lightrag.kg.write_seq`` replaces it as the ordering
field; these tests pin its guarantees: the token never repeats or regresses,
it decides whenever both rows carry it, and the comparison falls back to whole
seconds — the pre-token behavior, ties included — only when a row predates it.
"""

import pytest

from lightrag.kg import write_seq
from lightrag.kg.write_seq import (
    WRITE_SEQ_FIELD,
    next_write_seq,
    row_is_strictly_newer,
)


def _row(created_at: int, seq: int | None = None) -> dict:
    row = {"__created_at__": created_at, "content": "c"}
    if seq is not None:
        row[WRITE_SEQ_FIELD] = seq
    return row


@pytest.mark.offline
def test_successive_tokens_strictly_increase():
    tokens = [next_write_seq() for _ in range(100)]
    assert tokens == sorted(tokens)
    assert len(set(tokens)) == len(tokens), "a repeated token cannot order writes"


@pytest.mark.offline
def test_a_frozen_or_backwards_clock_still_yields_ordered_tokens(monkeypatch):
    """A coarse clock (Windows' ~15ms granularity) hands out the same reading
    twice, and an NTP step hands out an earlier one; neither may produce a
    token that ties with or precedes one already issued."""
    readings = iter([500, 500, 400, 300])
    monkeypatch.setattr(write_seq.time, "time_ns", lambda: next(readings))
    monkeypatch.setattr(write_seq, "_last_seq", 499)

    tokens = [next_write_seq() for _ in range(4)]
    assert tokens == [500, 501, 502, 503]


@pytest.mark.offline
def test_the_token_decides_when_both_rows_carry_one():
    """Fix-proof (PR #3709 review): whole seconds used to be compared first,
    so a clock stepped backward across a second boundary made the *later* write
    carry the *smaller* ``__created_at__`` and be declared older — its higher
    token never read. That discarded the guarantee ``next_write_seq`` exists
    for, and let a stale redo row replay over a newer durable one."""
    stepped_back = _row(99, 501)  # written after, on a clock stepped backward
    ours = _row(100, 500)
    assert row_is_strictly_newer(stepped_back, ours) is True
    assert row_is_strictly_newer(ours, stepped_back) is False


@pytest.mark.offline
def test_whole_seconds_decide_only_when_a_row_predates_the_token():
    """The fallback still orders a legacy row against a tokened one, and a
    newer second wins there — it is the only evidence such a pair has."""
    assert row_is_strictly_newer(_row(101), _row(100, 999)) is True
    assert row_is_strictly_newer(_row(100, 999), _row(101)) is False


@pytest.mark.offline
def test_the_token_breaks_a_same_second_tie():
    assert row_is_strictly_newer(_row(100, 2), _row(100, 1)) is True
    assert row_is_strictly_newer(_row(100, 1), _row(100, 2)) is False


@pytest.mark.offline
def test_equal_tokens_from_two_processes_stay_a_tie():
    """The bump that keeps tokens distinct is process-local, so two processes
    reading one clock tick stamp the *same* token — reachable only on a coarse
    clock, and only by two writers racing on one id (which the backends'
    single-writer invariant excludes). Pin that such a pair resolves to the
    pre-token behavior rather than to an arbitrary winner: ordering it would
    take a generation shared across processes."""
    ours = _row(100, 5)
    theirs = _row(100, 5)  # a second process, same clock tick
    assert row_is_strictly_newer(theirs, ours) is False
    assert row_is_strictly_newer(ours, theirs) is False


@pytest.mark.offline
def test_a_missing_token_is_not_read_as_older():
    """A row written before the token existed carries none. Defaulting it to 0
    would declare it older than any freshly stamped row and let a replay
    overwrite it, so such a same-second pair stays a tie (False)."""
    assert row_is_strictly_newer(_row(100), _row(100, 5)) is False
    assert row_is_strictly_newer(_row(100, 5), _row(100)) is False


@pytest.mark.offline
def test_an_absent_row_never_supersedes():
    assert row_is_strictly_newer(None, _row(100, 1)) is False
