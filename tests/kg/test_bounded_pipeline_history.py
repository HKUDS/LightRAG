"""The bounded pipeline status history funnel (LR2 §10.3).

Before this, the only bound on ``pipeline_status["history_messages"]`` was a
hard-coded 10000→5000 trim sitting *inside the document extraction loop*. Two
holes followed from that, and both are pinned here as fix-proof:

  - a writer that never reaches that loop was unbounded. Deletions, ``/scan``,
    manual resets and ``clear_documents`` all log per item and none of them runs
    the extraction loop, so their history grew for the lifetime of the process;
  - no message had a size cap, so N messages × unbounded size is still
    unbounded — one call site appends a whole ``traceback.format_exc()``, and any
    message interpolating an exception string or a file list can be arbitrarily
    large. A count alone bounds nothing.

The repo-wide guard test matters as much as the numeric ones: the bound is only
real if every write goes through the funnel, so a re-introduced raw
``["history_messages"].append(...)`` is a regression even though nothing fails.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from lightrag.constants import (
    PIPELINE_HISTORY_MAX_MESSAGES,
    PIPELINE_HISTORY_MESSAGE_MAX_BYTES,
)
from lightrag.kg import shared_storage
from lightrag.kg.shared_storage import (
    PipelineStatusLogger,
    append_pipeline_history,
    clamp_pipeline_history_message,
)

pytestmark = pytest.mark.offline

_INTERVAL = shared_storage._HISTORY_TRIM_CHECK_INTERVAL


@pytest.fixture(autouse=True)
def _reset_trim_counter():
    """The capacity check is amortized through a process-local counter; a test
    must not inherit another test's position in that cycle."""
    shared_storage._history_appends_since_trim = 0
    yield
    shared_storage._history_appends_since_trim = 0


# ---------------------------------------------------------------------------
# Fix-proof: the ring capacity, enforced for every writer.
# ---------------------------------------------------------------------------


def test_a_writer_that_never_reaches_the_extraction_loop_is_still_bounded():
    """Shaped like a deletion / clear job: it only logs, it never processes a
    document, so the old in-loop trim never ran for it."""
    history = []
    status = {"history_messages": history}

    total = PIPELINE_HISTORY_MAX_MESSAGES + 3 * _INTERVAL
    for i in range(total):
        append_pipeline_history(status, f"deleting document {i}")

    assert len(history) <= PIPELINE_HISTORY_MAX_MESSAGES + _INTERVAL
    # A ring, not a head-truncation: the newest line survives, the oldest is gone.
    assert history[-1] == f"deleting document {total - 1}"
    assert "deleting document 0" not in history


def test_a_group_write_advances_the_bound_by_its_own_length():
    """The counter counts messages, not calls — otherwise a call site passing a
    group of N would multiply the overshoot by N."""
    history = []
    status = {"history_messages": history}

    group = 8
    for i in range(0, PIPELINE_HISTORY_MAX_MESSAGES + 3 * _INTERVAL, group):
        append_pipeline_history(status, *[f"msg {i}.{k}" for k in range(group)])

    assert len(history) <= PIPELINE_HISTORY_MAX_MESSAGES + _INTERVAL


def test_trimming_never_rebinds_the_shared_list_object():
    """``history_messages`` is a Manager ListProxy that every worker appends to
    and that PipelineStatusLogger caches; replacing it would orphan both."""
    history = []
    status = {"history_messages": history}

    for i in range(PIPELINE_HISTORY_MAX_MESSAGES + 2 * _INTERVAL):
        append_pipeline_history(status, f"line {i}")

    assert status["history_messages"] is history


# ---------------------------------------------------------------------------
# Fix-proof: the per-message byte cap.
# ---------------------------------------------------------------------------


def test_a_traceback_sized_message_is_capped():
    history = []
    traceback_text = "Traceback (most recent call last):\n" + (
        '  File "lightrag/pipeline.py", line 1, in _process\n    raise RuntimeError\n'
        * 400
    )
    assert len(traceback_text.encode()) > 5 * PIPELINE_HISTORY_MESSAGE_MAX_BYTES

    append_pipeline_history({"history_messages": history}, traceback_text)

    stored = history[0]
    assert len(stored.encode("utf-8")) <= PIPELINE_HISTORY_MESSAGE_MAX_BYTES
    # The head identifies what happened; the marker names the size that was cut
    # so an operator knows to go to the server log for the rest.
    assert stored.startswith("Traceback (most recent call last):")
    assert f"[truncated, {len(traceback_text.encode())} bytes]" in stored


def test_a_cjk_message_is_cut_on_a_byte_boundary():
    """Counting characters would overshoot the byte budget by 3x on CJK, and
    slicing bytes carelessly would leave a mojibake half-character."""
    message = "文档解析失败：" + "中" * 5000

    stored = clamp_pipeline_history_message(message)

    assert len(stored.encode("utf-8")) <= PIPELINE_HISTORY_MESSAGE_MAX_BYTES
    assert stored.startswith("文档解析失败：")
    assert "�" not in stored  # no split character


def test_a_message_at_the_cap_is_stored_verbatim():
    message = "x" * PIPELINE_HISTORY_MESSAGE_MAX_BYTES

    assert clamp_pipeline_history_message(message) == message
    assert "truncated" not in clamp_pipeline_history_message("short line")


# ---------------------------------------------------------------------------
# Fix-proof: the funnel is the only writer.
# ---------------------------------------------------------------------------


def test_no_raw_history_append_survives_in_the_package():
    """A single missed call site is an unbounded leak that no test would notice."""
    package = Path(shared_storage.__file__).resolve().parents[1]
    raw_append = re.compile(r'\["history_messages"\]\s*\.\s*append\(')

    offenders = []
    for path in sorted(package.rglob("*.py")):
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if raw_append.search(line) and not line.lstrip().startswith("#"):
                offenders.append(f"{path.relative_to(package)}:{lineno}")

    assert offenders == [], (
        "route these through append_pipeline_history (LR2 §10.3): "
        + ", ".join(offenders)
    )


# ---------------------------------------------------------------------------
# The never-raise contract (call sites are `except ...: log(...); raise`).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status",
    [None, {}, {"history_messages": None}],
    ids=["none-status", "missing-key", "late-initialized-key"],
)
def test_a_missing_history_is_skipped_not_raised(status):
    append_pipeline_history(status, "message")


def test_a_failing_extend_is_swallowed():
    class _DeadProxy(list):
        def extend(self, _iterable):
            raise ConnectionRefusedError("manager is gone")

    append_pipeline_history({"history_messages": _DeadProxy()}, "message")


def test_a_failing_len_does_not_fail_the_append():
    """The capacity check is best effort; losing it must not lose the message."""

    class _UnmeasurableProxy(list):
        def __len__(self):
            raise BrokenPipeError("manager is gone")

    history = _UnmeasurableProxy()
    shared_storage._history_appends_since_trim = _INTERVAL - 1

    append_pipeline_history({"history_messages": history}, "message")

    assert list.__len__(history) == 1


def test_a_non_string_message_is_coerced_not_dropped():
    history = []

    append_pipeline_history({"history_messages": history}, RuntimeError("boom"), 42)

    assert history == ["boom", "42"]


# ---------------------------------------------------------------------------
# PipelineStatusLogger shares the same bounds (it is the hot path).
# ---------------------------------------------------------------------------


def test_the_lock_free_logger_clamps_and_trims_too():
    history = []
    status = {"history_messages": history}
    logger = PipelineStatusLogger(status)

    for i in range(PIPELINE_HISTORY_MAX_MESSAGES + 2 * _INTERVAL):
        logger.log(f"entity {i}")
    logger.log("失败" * 5000)

    assert len(history) <= PIPELINE_HISTORY_MAX_MESSAGES + _INTERVAL
    assert status["history_messages"] is history
    assert len(history[-1].encode("utf-8")) <= PIPELINE_HISTORY_MESSAGE_MAX_BYTES


def test_latest_message_is_clamped_as_well():
    """A single slot, but it is re-serialized on every status poll, so an
    unclamped line is shipped to the UI for as long as it stays the latest."""
    status = {"history_messages": []}

    PipelineStatusLogger(status).log("x" * (4 * PIPELINE_HISTORY_MESSAGE_MAX_BYTES))

    assert (
        len(status["latest_message"].encode("utf-8"))
        <= PIPELINE_HISTORY_MESSAGE_MAX_BYTES
    )
