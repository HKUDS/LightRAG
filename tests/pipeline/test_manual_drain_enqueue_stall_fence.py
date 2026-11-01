"""DRAIN_TO_IDLE's wait on in-flight enqueues is bounded (LR2 §7.2 step 5).

The manual drain waits for ``pending_enqueues`` to reach 0 so every pre-freeze
enqueue lands its documents in the drain's cohort. That wait used to be
unbounded: it reaps confirmed-dead reservation tokens, but a token whose owner
process is very much alive and simply never releases held the drain forever —
``busy`` latched, nothing in flight, and no ``recovery_required`` fence for
``POST /documents/recovery/force_reset`` to clear, so only a process restart got
the workspace back.

The canonical way in was a bg task that kept its own reservation while driving
``apipeline_process_enqueue_documents`` (closed at the source in the API layer —
see ``tests/api/routes/test_admission_released_before_processing.py``). This
fence is the backstop: an in-flight set that does not change for the whole
bounded window fences the workspace with ``manual_drain_enqueue_stalled``
instead of waiting again.

The freeze admits no new reservation, so the set can only shrink — which is why
"unchanged" is the right stall signal and any change resets the count.
"""

from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag import pipeline as pipeline_module
from lightrag.exceptions import PipelineRecoveryRequiredError
from lightrag.base import CURSOR_START
from lightrag.kg.shared_storage import (
    SOURCE_REPAIR_RESERVATION_KIND,
    acquire_enqueue_reservation,
    finalize_share_data,
    get_namespace_data,
    get_namespace_lock,
    initialize_share_data,
    release_token_set_reservation,
)
from lightrag.pipeline import (
    PipelineNextDecision,
    PipelineNextStep,
    _ManualDrainProgress,
)
from lightrag.utils import EmbeddingFunc, Tokenizer

pytestmark = pytest.mark.offline

_STALL_ROUNDS = 3


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


async def _dummy_llm(*args, **kwargs) -> str:
    return "ok"


@pytest.fixture(autouse=True)
def _shared():
    initialize_share_data()
    yield
    finalize_share_data()


@pytest.fixture(autouse=True)
def _short_stall_window(monkeypatch):
    """Shrink the 10-minute window so the bound is testable in milliseconds.

    The constant is read inside the tracker, so patching the module attribute is
    enough; the poll sleep is zeroed for the same reason.
    """
    monkeypatch.setattr(
        pipeline_module, "_MANUAL_DRAIN_ENQUEUE_STALL_ROUNDS", _STALL_ROUNDS
    )
    monkeypatch.setattr(pipeline_module, "_MANUAL_DRAIN_POLL_SECONDS", 0)


async def _build_rag(tmp_path) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"stall-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    return rag


async def _status_handles(rag):
    return (
        await get_namespace_data("pipeline_status", workspace=rag.workspace),
        get_namespace_lock("pipeline_status", workspace=rag.workspace),
    )


async def _reserve(
    pipeline_status, lock, token: str, *, kind: str | None = None
) -> None:
    result = await acquire_enqueue_reservation(
        pipeline_status, lock, token=token, reject_when=(), weight=1, kind=kind
    )
    assert result.acquired


async def _drain_wait_round(rag, pipeline_status, lock, progress):
    """One CONTINUE_DRAIN_WAIT refetch, exactly as the run performs it."""
    return await rag._refetch_for_decision(
        PipelineNextDecision(PipelineNextStep.CONTINUE_DRAIN_WAIT),
        (),
        CURSOR_START,
        None,
        token="busy-token",
        pipeline_status=pipeline_status,
        pipeline_status_lock=lock,
        drain_progress=progress,
    )


# ---------------------------------------------------------------------------
# The tracker
# ---------------------------------------------------------------------------


def test_an_unchanged_token_set_runs_out_of_rounds():
    progress = _ManualDrainProgress()
    tokens = {"a": {}, "b": {}}
    assert [progress.observe_enqueue(tokens) for _ in range(_STALL_ROUNDS)] == [
        True,
        True,
        False,
    ]


def test_a_shrinking_token_set_never_runs_out():
    """A release is a live producer finishing, so the evidence starts over."""
    progress = _ManualDrainProgress()
    live = {"a": {}, "b": {}, "c": {}}
    for _ in range(_STALL_ROUNDS * 4):
        assert progress.observe_enqueue(live) is True
        if len(live) > 1:
            live.pop(sorted(live)[0])
        else:
            live = {f"t-{uuid4().hex[:4]}": {}}


def test_the_two_drain_waits_do_not_reset_each_other():
    """One drain interleaves both waits — a doc published by a finishing enqueue
    puts the run back into sweeping — so shared evidence would hide either
    stall."""
    progress = _ManualDrainProgress()
    for _ in range(_STALL_ROUNDS - 1):
        assert progress.observe_enqueue({"a": {}}) is True
        assert progress.observe({"doc-1"}) is True
    assert progress.observe_enqueue({"a": {}}) is False


# ---------------------------------------------------------------------------
# The fence
# ---------------------------------------------------------------------------


def test_a_stalled_in_flight_set_fences_the_workspace(tmp_path):
    """Fix-proof: this wait had no bound at all — the run spun on it forever."""

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            pipeline_status, lock = await _status_handles(rag)
            token = uuid4().hex
            await _reserve(pipeline_status, lock, token)
            progress = _ManualDrainProgress()

            for _ in range(_STALL_ROUNDS - 1):
                docs, _statuses, _cursor = await _drain_wait_round(
                    rag, pipeline_status, lock, progress
                )
                assert docs == {}
                async with lock:
                    assert not pipeline_status.get("recovery_required")

            with pytest.raises(PipelineRecoveryRequiredError, match="drain stalled"):
                await _drain_wait_round(rag, pipeline_status, lock, progress)

            async with lock:
                fence = pipeline_status.get("recovery_required")
            assert fence["kind"] == "manual_drain_enqueue_stalled"
            # The message has to name the way out and the blocker.
            assert "force_reset" in fence["message"]
            assert token in fence["operation_record"]["scope"]
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_an_enqueue_that_finishes_keeps_the_drain_waiting(tmp_path):
    """A producer that lands its documents must never be fenced, however long
    the drain has already waited for its siblings."""

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            pipeline_status, lock = await _status_handles(rag)
            slow = uuid4().hex
            finishing = uuid4().hex
            await _reserve(pipeline_status, lock, slow)
            await _reserve(pipeline_status, lock, finishing)
            progress = _ManualDrainProgress()

            for _ in range(_STALL_ROUNDS - 1):
                await _drain_wait_round(rag, pipeline_status, lock, progress)

            # One of them lands: the set changed, so the evidence resets.
            await release_token_set_reservation(
                rag.workspace, tokens_key="pending_enqueue_tokens", token=finishing
            )
            for _ in range(_STALL_ROUNDS - 1):
                await _drain_wait_round(rag, pipeline_status, lock, progress)

            async with lock:
                assert not pipeline_status.get("recovery_required")
                assert pipeline_status["pending_enqueues"] == 1
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_an_empty_in_flight_set_is_not_a_stall(tmp_path):
    """``pending_enqueues == 0`` is the drain's exit, not its blocker: the
    decision never returns CONTINUE_DRAIN_WAIT then, and a refetch that somehow
    does must not fence on an empty set."""

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            pipeline_status, lock = await _status_handles(rag)
            progress = _ManualDrainProgress()
            for _ in range(_STALL_ROUNDS * 2):
                await _drain_wait_round(rag, pipeline_status, lock, progress)
            async with lock:
                assert not pipeline_status.get("recovery_required")
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_a_set_that_changed_since_the_poll_is_not_fenced(tmp_path):
    """The stall evidence is read under an EARLIER lock hold than the write that
    fences on it. A producer that releases in that gap has just proved the drain
    can advance, so the fence must not be written from the stale snapshot — it
    would refuse every mutation on a workspace that had already recovered, and
    only a manual force_reset undoes that."""

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            pipeline_status, lock = await _status_handles(rag)
            finishing = uuid4().hex
            await _reserve(pipeline_status, lock, finishing)
            async with lock:
                observed = dict(pipeline_status.get("pending_enqueue_tokens") or {})
            assert set(observed) == {finishing}

            # The gap: the producer lands its documents and releases.
            await release_token_set_reservation(
                rag.workspace, tokens_key="pending_enqueue_tokens", token=finishing
            )

            # Fencing is now attempted on the stale snapshot — and must not
            # happen, nor unwind the run.
            await rag._fence_stalled_enqueue_drain(observed, pipeline_status, lock)

            async with lock:
                assert not pipeline_status.get("recovery_required")
                assert pipeline_status["pending_enqueues"] == 0
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_a_shrink_since_the_poll_is_not_fenced_either(tmp_path):
    """Same rule when the set only shrank: one of several producers finishing is
    forward progress, and the drain goes on waiting for the rest."""

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            pipeline_status, lock = await _status_handles(rag)
            slow = uuid4().hex
            finishing = uuid4().hex
            await _reserve(pipeline_status, lock, slow)
            await _reserve(pipeline_status, lock, finishing)
            async with lock:
                observed = dict(pipeline_status.get("pending_enqueue_tokens") or {})

            # Spend all but the last round of the window on the full set, with
            # the SAME tracker the drain uses — so what follows tests the
            # tracker's state, not a fresh one.
            progress = _ManualDrainProgress()
            for _ in range(_STALL_ROUNDS - 1):
                await _drain_wait_round(rag, pipeline_status, lock, progress)

            await release_token_set_reservation(
                rag.workspace, tokens_key="pending_enqueue_tokens", token=finishing
            )
            await rag._fence_stalled_enqueue_drain(observed, pipeline_status, lock)

            async with lock:
                assert not pipeline_status.get("recovery_required")
                assert set(pipeline_status["pending_enqueue_tokens"]) == {slow}

            # The shrink resets the evidence: the remaining token must run out a
            # WHOLE new window. Without the reset, the very next round would be
            # the window's last and would fence here.
            for _ in range(_STALL_ROUNDS - 1):
                await _drain_wait_round(rag, pipeline_status, lock, progress)
            async with lock:
                assert not pipeline_status.get("recovery_required")
            with pytest.raises(PipelineRecoveryRequiredError, match="drain stalled"):
                await _drain_wait_round(rag, pipeline_status, lock, progress)
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# What the fence message may say, and to whom
# ---------------------------------------------------------------------------


def test_the_fence_message_carries_no_token_or_pid(tmp_path):
    """``recovery_message``, ``latest_message`` and ``history_messages`` all
    carry this text to the API unfiltered, and the /documents/pipeline_status
    projection drops the whole internal fence record precisely so reservation
    tokens and owner pids do not leave the process. The message must not put
    them back; the server log carries them instead."""

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            pipeline_status, lock = await _status_handles(rag)
            token = uuid4().hex
            await _reserve(pipeline_status, lock, token)
            progress = _ManualDrainProgress()

            for _ in range(_STALL_ROUNDS - 1):
                await _drain_wait_round(rag, pipeline_status, lock, progress)
            with pytest.raises(PipelineRecoveryRequiredError) as excinfo:
                await _drain_wait_round(rag, pipeline_status, lock, progress)

            async with lock:
                fence = pipeline_status.get("recovery_required")
                latest = pipeline_status.get("latest_message") or ""
                history = " ".join(pipeline_status.get("history_messages") or [])
            pid = str(os.getpid())
            for surface in (fence["message"], latest, history, str(excinfo.value)):
                assert token not in surface
                assert pid not in surface
            # The internal record still names the blockers for support: it is
            # dropped from the API response, not surfaced.
            assert token in fence["operation_record"]["scope"]
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_the_fence_message_says_a_repair_guard_is_not_dropped(tmp_path):
    """force_reset keeps a source-conflict repair's guard, so the message must
    not promise it drops ``these reservations`` and send the operator straight
    back to /documents/reprocess_failed — the retry would be blocked by the
    retained guard and fence again a whole window later."""

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            pipeline_status, lock = await _status_handles(rag)
            await _reserve(pipeline_status, lock, uuid4().hex)
            await _reserve(
                pipeline_status,
                lock,
                f"source-repair-{uuid4().hex}",
                kind=SOURCE_REPAIR_RESERVATION_KIND,
            )
            progress = _ManualDrainProgress()

            for _ in range(_STALL_ROUNDS - 1):
                await _drain_wait_round(rag, pipeline_status, lock, progress)
            with pytest.raises(PipelineRecoveryRequiredError):
                await _drain_wait_round(rag, pipeline_status, lock, progress)

            async with lock:
                message = pipeline_status.get("recovery_required")["message"]
            assert "1 upload/insert, 1 source-conflict repair guard(s)" in message
            assert "does NOT drop" in message
            assert "restart the process holding it" in message
            # The unconditional promise must be gone in this case.
            assert "AND drops these reservations" not in message
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
