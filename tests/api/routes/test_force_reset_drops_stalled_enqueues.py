"""``force_reset`` must clear what the fence it clears was actually blocked on.

For ``manual_drain_enqueue_stalled`` the blocker is not the fence — it is the
in-flight enqueue reservation set the drain waited out its whole bounded window
on. Dropping the fence and leaving that set behind changes nothing an operator
can observe: ``/documents/scan`` and ``/documents/clear`` both refuse on
``pending_enqueues``, and a re-issued ``/documents/reprocess_failed`` starts a
fresh ``_ManualDrainProgress``, waits out the window again and fences again — a
process restart stays the only exit, which is the dead end the bounded wait
exists to remove.

Every OTHER fence kind keeps the set: it is not owner-held state the reset is
abandoning. A healthy upload/insert may hold a token for reasons unrelated to
the fence being cleared. The rule is that force_reset clears exactly what the
fence message told the operator it would clear.

And within the enqueue-stall kind, only ORDINARY enqueues are dropped. A
source-conflict repair parks a weighted-0 token in the same set to exclude
clear/delete, scan classification and the manual reset for the whole span in
which it re-reads the candidate set and demotes the losers — dropping that guard
would re-open all three against a coroutine that may still resume. Weight cannot
tell them apart (an enqueue whose documents all dedup away re-weights to 0 too),
so the kind is stamped into the reservation at acquire time.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from uuid import uuid4

import numpy as np
import pytest

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

from lightrag import LightRAG  # noqa: E402
from lightrag.kg.shared_storage import (  # noqa: E402
    SOURCE_REPAIR_RESERVATION_KIND,
    acquire_enqueue_reservation,
    fence_workspace_for_recovery,
    finalize_share_data,
    get_namespace_data,
    get_namespace_lock,
    initialize_pipeline_status,
    initialize_share_data,
)
from lightrag.utils import EmbeddingFunc, Tokenizer  # noqa: E402

DocumentManager = _document_routes.DocumentManager
ForceResetRecoveryRequest = _document_routes.ForceResetRecoveryRequest
_acquire_destructive_busy = _document_routes._acquire_destructive_busy
_reserve_enqueue_slot = _document_routes._reserve_enqueue_slot

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


async def _build_rag(tmp_path) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"fr-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    await initialize_pipeline_status(workspace=rag.workspace)
    return rag


def _force_reset_endpoint(rag, tmp_path):
    doc_manager = DocumentManager(str(tmp_path / "inputs"))
    router = _document_routes.create_document_routes(rag, doc_manager)
    return [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == "force_reset_recovery"
    ][-1]


async def _status_handles(rag):
    return (
        await get_namespace_data("pipeline_status", workspace=rag.workspace),
        get_namespace_lock("pipeline_status", workspace=rag.workspace),
    )


async def _fence(rag, kind: str) -> None:
    pipeline_status, lock = await _status_handles(rag)
    await fence_workspace_for_recovery(
        pipeline_status,
        lock,
        kind=kind,
        message=f"{kind} for test",
        operation_record={"scope": "t"},
    )


def test_force_reset_drops_the_reservations_the_enqueue_fence_waited_on(tmp_path):
    """Fix-proof: with the fence cleared but the token left in place,
    ``pending_enqueues`` stayed 1 and ``_acquire_destructive_busy`` was still
    refused — the workspace was exactly as stuck as before the reset."""

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            pipeline_status, _ = await _status_handles(rag)
            stuck = uuid4().hex
            assert await _reserve_enqueue_slot(rag, stuck)
            await _fence(rag, "manual_drain_enqueue_stalled")

            endpoint = _force_reset_endpoint(rag, tmp_path)
            response = await endpoint(ForceResetRecoveryRequest(confirm=True))

            assert response.status == "reset"

            # The behavioural assertion first: the documented recovery path has
            # to be OPEN again. ``_acquire_destructive_busy`` refuses on
            # ``pending_enqueues``, so a reset that only dropped the fence fails
            # right here rather than on a field that merely reports the drop.
            acquired, reason = await _acquire_destructive_busy(
                rag,
                uuid4().hex,
                kind="clear",
                operation_record={"kind": "clear"},
            )
            assert acquired, reason

            assert pipeline_status.get("pending_enqueues") == 0
            assert dict(pipeline_status.get("pending_enqueue_tokens") or {}) == {}
            assert response.dropped_enqueue_reservations == 1
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.parametrize(
    "kind",
    [
        # The two self-fenced drain kinds, plus a dead-owner fence: that one
        # takes its kind from the dead reservation's ``owner_kind``
        # (``_dead_reservation_updates``), which for a destructive job is
        # "clear" / "delete".
        "manual_drain_stalled",
        "manual_drain_blocked",
        "clear",
    ],
)
def test_force_reset_keeps_reservations_for_every_other_fence_kind(tmp_path, kind):
    """A token held by a healthy producer (or a source-conflict repair's
    weighted-0 guard) is not this reset's to abandon: only the enqueue-stall
    fence names the reservation set as its blocker."""

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            pipeline_status, _ = await _status_handles(rag)
            live = uuid4().hex
            assert await _reserve_enqueue_slot(rag, live)
            await _fence(rag, kind)

            endpoint = _force_reset_endpoint(rag, tmp_path)
            response = await endpoint(ForceResetRecoveryRequest(confirm=True))

            assert response.status == "reset"
            assert response.dropped_enqueue_reservations == 0
            assert pipeline_status.get("pending_enqueues") == 1
            assert live in dict(pipeline_status.get("pending_enqueue_tokens") or {})
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


async def _reserve_repair_guard(rag, token: str) -> None:
    """Register what ``_repair_ingress_reservation`` registers: a weighted-0
    token labelled as a source-conflict repair."""
    pipeline_status, lock = await _status_handles(rag)
    result = await acquire_enqueue_reservation(
        pipeline_status,
        lock,
        token=token,
        reject_when=(),
        weight=0,
        capacity=0,
        kind=SOURCE_REPAIR_RESERVATION_KIND,
    )
    assert result.acquired


def test_force_reset_keeps_a_source_repair_guard_it_cannot_safely_drop(tmp_path):
    """The repair's guard is the one token in this set that must outlive the
    reset: clear/delete, scan classification and the manual reset all race its
    candidate re-read and demotion span. An operational wedge is recoverable;
    corrupted source ownership is not."""

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            pipeline_status, _ = await _status_handles(rag)
            stuck = uuid4().hex
            guard = f"source-repair-{uuid4().hex}"
            assert await _reserve_enqueue_slot(rag, stuck)
            await _reserve_repair_guard(rag, guard)
            await _fence(rag, "manual_drain_enqueue_stalled")

            endpoint = _force_reset_endpoint(rag, tmp_path)
            response = await endpoint(ForceResetRecoveryRequest(confirm=True))

            assert response.status == "reset"
            # The ordinary enqueue went; the guard stayed, and the count mirrors
            # what is left rather than the reset's intent.
            tokens = dict(pipeline_status.get("pending_enqueue_tokens") or {})
            assert list(tokens) == [guard]
            assert pipeline_status.get("pending_enqueues") == 1
            assert response.dropped_enqueue_reservations == 1
            assert response.retained_enqueue_reservations == 1

            # And the exclusion the guard exists for still holds.
            acquired, reason = await _acquire_destructive_busy(
                rag,
                uuid4().hex,
                kind="clear",
                operation_record={"kind": "clear"},
            )
            assert acquired is False
            assert "in flight" in (reason or "")
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
