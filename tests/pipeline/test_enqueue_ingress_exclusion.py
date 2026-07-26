"""The enqueue write side is mutually exclusive with a manual retry (LR2 §9.2).

``apipeline_enqueue_documents`` checked the ingress fences ONCE at entry and then
validated, deduped and wrote storage — a span with many awaits in it. That check
takes no reservation, so it left the call invisible to DRAIN_TO_IDLE: a manual
retry could raise ``manual_freeze_requested``, observe ``pending_enqueues == 0``,
declare strict idle and start the exclusive FAILED→PENDING reset while this
enqueue was still on its way to ``full_docs.upsert`` / ``doc_status.upsert``.
With admission disabled (the default) there was no second check at all; with it
enabled, the last-line reservation passed ``reject_when=()`` on purpose.

The fix is the reservation, not another check: it is taken after dedup and
before the FIRST write of the critical section, the fences are re-evaluated
atomically when it is newly minted, and the drain then waits for it. A caller
that already holds a reservation (an endpoint took one before parsing the body)
is deliberately NOT re-fenced — §9.2 lets a request admitted before a freeze
finish, and the drain is already waiting for that token.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.kg.shared_storage import (
    acquire_enqueue_reservation,
    finalize_share_data,
    get_namespace_data,
    get_namespace_lock,
    initialize_share_data,
)
from lightrag.utils import EmbeddingFunc, Tokenizer

pytestmark = pytest.mark.offline


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


async def _build_rag(tmp_path, *, capacity: int = 0) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"excl-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        max_parallel_insert=1,
        max_pending_documents=capacity,
    )
    await rag.initialize_storages()
    return rag


async def _status_handles(rag):
    return (
        await get_namespace_data("pipeline_status", workspace=rag.workspace),
        get_namespace_lock("pipeline_status", workspace=rag.workspace),
    )


def _freeze_during_dedup(rag, pipeline_status, lock):
    """Raise the manual freeze from INSIDE the enqueue's critical section.

    ``filter_keys`` runs after the entry fence check and before every write, so
    this lands the freeze exactly in the window the entry check cannot cover.
    """
    original = rag.doc_status.filter_keys

    async def _filter_keys(keys):
        async with lock:
            pipeline_status["manual_freeze_requested"] = True
        return await original(keys)

    rag.doc_status.filter_keys = _filter_keys


def _spy_on_first_write(rag, pipeline_status, lock) -> dict:
    """Record what the drain would see at the first storage write."""
    seen: dict = {}
    original = rag.full_docs.upsert

    async def _upsert(data):
        async with lock:
            seen["pending_enqueues"] = pipeline_status.get("pending_enqueues", 0)
        return await original(data)

    rag.full_docs.upsert = _upsert
    return seen


# ---------------------------------------------------------------------------
# The freeze must refuse a write that has not happened yet
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("capacity", [0, 10], ids=["admission-off", "admission-on"])
def test_a_freeze_raised_mid_enqueue_refuses_the_write(tmp_path, capacity):
    """Fix-proof: with admission off there was no second check at all, and with
    it on the last-line reservation used ``reject_when=()``. Either way an SDK
    enqueue wrote its rows into the exclusive-reset window."""

    async def _run():
        rag = await _build_rag(tmp_path, capacity=capacity)
        try:
            pipeline_status, lock = await _status_handles(rag)
            _freeze_during_dedup(rag, pipeline_status, lock)

            with pytest.raises(RuntimeError, match="manual retry"):
                await rag.apipeline_enqueue_documents(
                    "hello world", file_paths="report.pdf", track_id="t-1"
                )

            # Nothing at all was written — not the PENDING row, not a dup record.
            assert dict(rag.doc_status._data) == {}
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_the_freeze_refusal_also_covers_the_duplicate_records(tmp_path):
    """The duplicate ``dup-*`` FAILED rows are written BEFORE the primary ones,
    so a guard placed after them would still land doc_status writes inside the
    exclusive-reset window — and a fresh FAILED row is exactly what that reset
    is paging over."""

    async def _run():
        rag = await _build_rag(tmp_path, capacity=10)
        try:
            # An existing row so the second enqueue is a pure duplicate: its only
            # write is the dup record.
            await rag.apipeline_enqueue_documents(
                "hello world", file_paths="report.pdf", track_id="t-1"
            )
            before = dict(rag.doc_status._data)

            pipeline_status, lock = await _status_handles(rag)
            _freeze_during_dedup(rag, pipeline_status, lock)

            with pytest.raises(RuntimeError, match="manual retry"):
                await rag.apipeline_enqueue_documents(
                    "different body", file_paths="report.pdf", track_id="t-2"
                )

            assert dict(rag.doc_status._data) == before  # no dup-* row landed
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# ... and an enqueue already past the fence must be VISIBLE to the drain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("capacity", [0, 10], ids=["admission-off", "admission-on"])
def test_the_enqueue_holds_a_reservation_when_it_writes(tmp_path, capacity):
    """The other half of the fix, and the reason a mere re-check would not do:
    DRAIN_TO_IDLE proves idleness from ``pending_enqueues``, so an enqueue that
    passed the fence has to be counted there until its last write. Otherwise the
    drain concludes idle and the reset runs alongside it."""

    async def _run():
        rag = await _build_rag(tmp_path, capacity=capacity)
        try:
            pipeline_status, lock = await _status_handles(rag)
            seen = _spy_on_first_write(rag, pipeline_status, lock)

            await rag.apipeline_enqueue_documents(
                "hello world", file_paths="report.pdf", track_id="t-1"
            )

            assert seen["pending_enqueues"] == 1
            # ...and released afterwards, so it cannot wedge a later freeze.
            assert pipeline_status.get("pending_enqueues", 0) == 0
            assert dict(pipeline_status.get("pending_enqueue_tokens", {})) == {}
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# The deliberate asymmetry (LR2 §9.2)
# ---------------------------------------------------------------------------


def test_a_caller_admitted_before_the_freeze_is_allowed_to_finish(tmp_path):
    """An endpoint reserves — fences and all — before it reads the request body.
    Re-fencing its re-weight would refuse a request the freeze protocol promises
    to let finish, and the drain is already waiting for that token anyway.

    Scope note: this covers the RE-WEIGHT. The entry fence check still refuses a
    token-holding caller when the freeze is already up before this call — a
    separate §9.2 inconsistency, not touched here."""

    async def _run():
        rag = await _build_rag(tmp_path, capacity=10)
        try:
            pipeline_status, lock = await _status_handles(rag)
            result = await acquire_enqueue_reservation(
                pipeline_status,
                lock,
                token="endpoint-token",
                reject_when=(),
                weight=1,
                capacity=10,
                active_count=0,
            )
            assert result.acquired is True

            # The freeze appears after the entry check, in the same window the
            # refusal tests above use — but this caller was already admitted.
            _freeze_during_dedup(rag, pipeline_status, lock)

            await rag.apipeline_enqueue_documents(
                "hello world",
                file_paths="report.pdf",
                track_id="t-1",
                admission_token="endpoint-token",
            )
            assert len(dict(rag.doc_status._data)) == 1
            # The caller's token is still the caller's to release.
            assert "endpoint-token" in pipeline_status["pending_enqueue_tokens"]
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_scan_enqueues_are_exempt_from_the_fence_and_the_reservation(tmp_path):
    """§9.1: the scan owns the manual operation, so its own enqueues must not
    self-block. Its exclusion is structural — it holds ``scanning_exclusive``
    across the whole enqueue span — not reservation-based."""

    async def _run():
        rag = await _build_rag(tmp_path, capacity=1)
        try:
            pipeline_status, lock = await _status_handles(rag)
            seen = _spy_on_first_write(rag, pipeline_status, lock)
            async with lock:
                pipeline_status["manual_freeze_requested"] = True
                pipeline_status["scanning_exclusive"] = True

            await rag.apipeline_enqueue_documents(
                ["a", "b"],
                file_paths=["a.txt", "b.txt"],
                track_id="t-scan",
                from_scan=True,
            )
            # Two docs written past a capacity of 1, and no reservation taken.
            assert len(dict(rag.doc_status._data)) == 2
            assert seen["pending_enqueues"] == 0
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
