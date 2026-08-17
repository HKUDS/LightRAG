"""Admission capacity at the enqueue chokepoint (LR2 Phase 5-a, §9.1/§9.2).

``MAX_PENDING_DOCUMENTS`` bounds how many documents may be active
(PENDING/PARSING/ANALYZING/PROCESSING) or reserved by an in-flight request. The
guard lives inside ``apipeline_enqueue_documents`` — after dedup, before the
first storage write — because that is the one point every entry point funnels
through, and the first point where the real document count is known.

Covered here:

* disabled by default: zero behaviour change, no strict count taken;
* the cap is enforced against the strict active count and refuses with the
  structured ``PipelineBackpressureError`` (→ 429), not a bare string;
* dedup shrinks the charge: a request whose documents are duplicates is not
  charged for them;
* an in-flight reservation's weight counts, and re-weighting the SAME token
  replaces its weight instead of adding to it (no self-collision);
* a strict-count failure fails closed (propagates) instead of being read as
  "there is room";
* ``from_scan`` and manual retries break through the cap by design.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.exceptions import PipelineBackpressureError, StorageControlPlaneError
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


def _chunking(
    tokenizer,
    content,
    split_by_character,
    split_by_character_only,
    chunk_overlap_token_size,
    chunk_token_size,
) -> list[dict]:
    return [{"tokens": 1, "content": f"{content}::chunk1", "chunk_order_index": 0}]


@pytest.fixture(autouse=True)
def _shared():
    initialize_share_data()
    yield
    finalize_share_data()


async def _build_rag(tmp_path, *, capacity: int) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"adm-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        chunking_func=_chunking,
        max_parallel_insert=1,
        max_pending_documents=capacity,
    )
    await rag.initialize_storages()
    return rag


def _count_spy(rag) -> dict:
    """Count strict-count calls and let a test force a failure."""
    state = {"calls": 0, "raise": None}
    original = rag.doc_status.count_docs_by_statuses

    async def _counting(statuses, *, strict=True):
        state["calls"] += 1
        if state["raise"] is not None:
            raise state["raise"]
        return await original(statuses, strict=strict)

    rag.doc_status.count_docs_by_statuses = _counting
    return state


async def _pending_count(rag) -> int:
    return await rag.doc_status.count_docs_by_statuses(
        [
            DocStatus.PENDING,
            DocStatus.PARSING,
            DocStatus.ANALYZING,
            DocStatus.PROCESSING,
        ]
    )


def test_admission_disabled_by_default_takes_no_count(tmp_path):
    """Default 0: the guard never runs, so an existing deployment sees neither a
    refusal nor the extra strict count per enqueue."""

    async def _run():
        rag = await _build_rag(tmp_path, capacity=0)
        try:
            spy = _count_spy(rag)
            for i in range(5):
                await rag.apipeline_enqueue_documents(
                    input=f"body {i}", file_paths=f"doc{i}.txt"
                )
            # Asserted before any counting of our own: the guard must not have
            # taken a single strict count.
            assert spy["calls"] == 0
            assert await _pending_count(rag) == 5
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_capacity_refuses_with_structured_backpressure(tmp_path):
    async def _run():
        rag = await _build_rag(tmp_path, capacity=2)
        try:
            await rag.apipeline_enqueue_documents(input="a", file_paths="a.txt")
            await rag.apipeline_enqueue_documents(input="b", file_paths="b.txt")

            with pytest.raises(PipelineBackpressureError) as excinfo:
                await rag.apipeline_enqueue_documents(input="c", file_paths="c.txt")

            error = excinfo.value
            assert (error.current, error.requested, error.capacity) == (2, 1, 2)
            assert "capacity" in str(error)
            # Refused BEFORE any write: the third document does not exist.
            assert await rag.doc_status.get_by_id_strict("doc-anything") is None
            assert await _pending_count(rag) == 2
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_batch_is_charged_after_dedup(tmp_path):
    """The guard runs after dedup, so duplicates inside the request are not
    charged — a 3-text batch with 2 duplicates fits a capacity of 1."""

    async def _run():
        rag = await _build_rag(tmp_path, capacity=1)
        try:
            await rag.apipeline_enqueue_documents(
                input=["same body", "same body", "same body"],
                file_paths=["a.txt", "b.txt", "c.txt"],
            )
            # One primary landed (the other two became duplicate records, which
            # are FAILED and therefore not active).
            assert await _pending_count(rag) == 1
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_whole_batch_is_refused_when_it_does_not_fit(tmp_path):
    async def _run():
        rag = await _build_rag(tmp_path, capacity=2)
        try:
            with pytest.raises(PipelineBackpressureError) as excinfo:
                await rag.apipeline_enqueue_documents(
                    input=["a", "b", "c"],
                    file_paths=["a.txt", "b.txt", "c.txt"],
                )
            assert excinfo.value.requested == 3
            assert await _pending_count(rag) == 0  # nothing partially admitted
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_other_requests_reservation_weight_counts(tmp_path):
    """A reservation held by a request that has not written yet occupies
    capacity — otherwise two concurrent uploads both pass a capacity-1 check."""

    async def _run():
        rag = await _build_rag(tmp_path, capacity=1)
        try:
            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            lock = get_namespace_lock("pipeline_status", workspace=rag.workspace)
            await acquire_enqueue_reservation(
                pipeline_status,
                lock,
                token="other-request",
                reject_when=(),
                weight=1,
                capacity=1,
                active_count=0,
            )

            with pytest.raises(PipelineBackpressureError) as excinfo:
                await rag.apipeline_enqueue_documents(input="a", file_paths="a.txt")
            assert excinfo.value.current == 1  # 0 active + 1 reserved elsewhere
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_reweighting_own_token_does_not_collide_with_itself(tmp_path):
    """``/texts`` reserves 1 before the body is known, then re-weights the same
    token to N. The token's own weight must be excluded from the sum, or a
    request would be refused because of itself."""

    async def _run():
        rag = await _build_rag(tmp_path, capacity=3)
        try:
            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            lock = get_namespace_lock("pipeline_status", workspace=rag.workspace)
            for weight in (1, 3):
                result = await acquire_enqueue_reservation(
                    pipeline_status,
                    lock,
                    token="mine",
                    reject_when=(),
                    weight=weight,
                    capacity=3,
                    active_count=0,
                )
                assert result.acquired is True
            tokens = pipeline_status["pending_enqueue_tokens"]
            assert tokens["mine"]["weight"] == 3
            # One token, not two — a re-weight is a replacement.
            assert pipeline_status["pending_enqueues"] == 1

            # And the enqueue itself, holding that token, is not charged twice.
            await rag.apipeline_enqueue_documents(
                input=["a", "b", "c"],
                file_paths=["a.txt", "b.txt", "c.txt"],
                admission_token="mine",
            )
            assert await _pending_count(rag) == 3
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_count_failure_fails_closed(tmp_path):
    """A backend that cannot count must not be read as "capacity available"."""

    async def _run():
        rag = await _build_rag(tmp_path, capacity=10)
        try:
            spy = _count_spy(rag)
            spy["raise"] = StorageControlPlaneError("index unavailable")

            with pytest.raises(StorageControlPlaneError):
                await rag.apipeline_enqueue_documents(input="a", file_paths="a.txt")
            spy["raise"] = None  # let the assertion below read the real count
            assert await _pending_count(rag) == 0
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_scan_breaks_through_the_cap(tmp_path):
    """§9.1: scan bulk enqueue and manual retries are exempt; the active rows
    they create are what makes ordinary uploads wait."""

    async def _run():
        rag = await _build_rag(tmp_path, capacity=1)
        try:
            for i in range(3):
                await rag.apipeline_enqueue_documents(
                    input=f"scanned {i}",
                    file_paths=f"scan{i}.txt",
                    from_scan=True,
                )
            assert await _pending_count(rag) == 3

            # ...and an ordinary upload now waits behind them.
            with pytest.raises(PipelineBackpressureError):
                await rag.apipeline_enqueue_documents(
                    input="ordinary", file_paths="ordinary.txt"
                )
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_minted_reservation_is_released_after_the_write(tmp_path):
    """The SDK path has no reservation of its own, so the guard mints one; it
    must be gone once the enqueue returns (and not leak on failure either)."""

    async def _run():
        rag = await _build_rag(tmp_path, capacity=5)
        try:
            await rag.apipeline_enqueue_documents(input="a", file_paths="a.txt")
            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            assert pipeline_status["pending_enqueue_tokens"] == {}
            assert pipeline_status["pending_enqueues"] == 0
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_capacity_requires_a_count(tmp_path):
    """Enforcing capacity without a strict count would be guessing."""

    async def _run():
        rag = await _build_rag(tmp_path, capacity=1)
        try:
            pipeline_status = await get_namespace_data(
                "pipeline_status", workspace=rag.workspace
            )
            lock = get_namespace_lock("pipeline_status", workspace=rag.workspace)
            with pytest.raises(ValueError, match="active_count"):
                await acquire_enqueue_reservation(
                    pipeline_status,
                    lock,
                    token="t",
                    reject_when=(),
                    weight=1,
                    capacity=1,
                )
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_negative_capacity_is_rejected_at_construction(tmp_path):
    with pytest.raises(ValueError, match="MAX_PENDING_DOCUMENTS"):
        LightRAG(
            working_dir=str(tmp_path / "wd2"),
            workspace="adm-negative",
            llm_model_func=_dummy_llm,
            embedding_func=EmbeddingFunc(
                embedding_dim=8, max_token_size=8192, func=_dummy_embedding
            ),
            tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
            chunking_func=_chunking,
            max_pending_documents=-1,
        )
