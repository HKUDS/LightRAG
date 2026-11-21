"""An enqueue-stall fence must not void the work of the holders it named.

``manual_drain_enqueue_stalled`` is raised BY the in-flight enqueue set: a
manual retry's DRAIN_TO_IDLE gave up waiting for those reservations. The bound
that raises it accepts a false positive — a producer that was alive and merely
slow — and that producer then comes back and finishes its enqueue.

If the recovery fence refused it like any other mutation, the false positive
would cost the payload: ``/documents/text`` and ``/documents/texts`` carry their
content in the request's background task, so unlike an upload there is no file
in INPUT/ for the next ``/documents/scan`` to rediscover, and the client was
already told 200. So a token the snapshot shows REGISTERED is let through THIS
fence kind — and only this one: every other kind means a worker may have left
storage half-committed, where a new write is exactly what must not happen.

The rows it writes still cannot be processed until an operator clears the fence
(the processing reservation keeps refusing), so nothing about the fence's
purpose is weakened — only the data loss is removed.
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
from lightrag.base import DocStatus  # noqa: E402
from lightrag.exceptions import PipelineReservationConflictError  # noqa: E402
from lightrag.kg.shared_storage import (  # noqa: E402
    MANUAL_DRAIN_ENQUEUE_STALL_FENCE,
    fence_workspace_for_recovery,
    finalize_share_data,
    get_namespace_data,
    get_namespace_lock,
    initialize_share_data,
)
from lightrag.utils import EmbeddingFunc, Tokenizer  # noqa: E402

_release_enqueue_slot = _document_routes._release_enqueue_slot
_reserve_enqueue_slot = _document_routes._reserve_enqueue_slot
pipeline_index_texts = _document_routes.pipeline_index_texts

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
        workspace=f"stall-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        max_parallel_insert=1,
        max_pending_documents=capacity,
    )
    await rag.initialize_storages()
    # The fence stops the run before processing; the test is about the enqueue.
    rag.apipeline_process_enqueue_documents = _noop_process
    return rag


async def _noop_process(*args, **kwargs) -> None:
    return None


async def _status_handles(rag):
    return (
        await get_namespace_data("pipeline_status", workspace=rag.workspace),
        get_namespace_lock("pipeline_status", workspace=rag.workspace),
    )


async def _fence(rag, kind: str) -> None:
    pipeline_status, lock = await _status_handles(rag)
    await fence_workspace_for_recovery(
        pipeline_status, lock, kind=kind, message=f"{kind} for test"
    )


@pytest.mark.parametrize("capacity", [0, 10], ids=["admission-off", "admission-on"])
def test_an_admitted_text_still_lands_under_the_stall_fence(tmp_path, capacity):
    """Fix-proof: the fence used to refuse this enqueue, and the text — which
    exists nowhere but this task — was gone after a 200."""

    async def _run():
        rag = await _build_rag(tmp_path, capacity=capacity)
        try:
            token = uuid4().hex
            assert await _reserve_enqueue_slot(rag, token)
            # The drain gives up on exactly this reservation.
            await _fence(rag, MANUAL_DRAIN_ENQUEUE_STALL_FENCE)

            try:
                await pipeline_index_texts(
                    rag,
                    ["a payload that lives only in this task"],
                    file_sources=["slow.txt"],
                    track_id="t-stalled",
                    admission_token=token,
                )
            finally:
                await _release_enqueue_slot(rag, token)

            pending = await rag.doc_status.get_docs_by_statuses([DocStatus.PENDING])
            assert len(pending) == 1

            # The fence itself is untouched — clearing it stays the operator's
            # call, and nothing may be processed until they make it.
            pipeline_status, lock = await _status_handles(rag)
            async with lock:
                fence = pipeline_status.get("recovery_required")
            assert fence["kind"] == MANUAL_DRAIN_ENQUEUE_STALL_FENCE
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


@pytest.mark.parametrize(
    "kind",
    [
        # A worker died mid-clear: storage may be half-committed, and a new
        # write is precisely what must not happen next.
        "clear",
        # The other two drain fences: raised about document rows, not about
        # this reservation, so they are not its to pass.
        "manual_drain_stalled",
        "manual_drain_blocked",
    ],
)
def test_every_other_fence_kind_still_refuses_the_same_holder(tmp_path, kind):
    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            token = uuid4().hex
            assert await _reserve_enqueue_slot(rag, token)
            await _fence(rag, kind)

            with pytest.raises(PipelineReservationConflictError):
                await pipeline_index_texts(
                    rag,
                    ["refused"],
                    file_sources=["refused.txt"],
                    track_id="t-refused",
                    admission_token=token,
                )

            pending = await rag.doc_status.get_docs_by_statuses([DocStatus.PENDING])
            assert len(pending) == 0
        finally:
            await _release_enqueue_slot(rag, token)
            await rag.finalize_storages()

    asyncio.run(_run())


def test_a_holder_the_fence_never_knew_is_not_exempt(tmp_path):
    """The exemption is read from the snapshot's registered set, never from the
    caller's word: a token that is not registered buys nothing."""

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            await _fence(rag, MANUAL_DRAIN_ENQUEUE_STALL_FENCE)

            with pytest.raises(PipelineReservationConflictError):
                await pipeline_index_texts(
                    rag,
                    ["not admitted"],
                    file_sources=["stranger.txt"],
                    track_id="t-stranger",
                    admission_token=uuid4().hex,
                )

            pending = await rag.doc_status.get_docs_by_statuses([DocStatus.PENDING])
            assert len(pending) == 0
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
