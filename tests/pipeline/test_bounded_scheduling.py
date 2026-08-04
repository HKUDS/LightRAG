"""Bounded scheduling acceptance tests (LR2 Phase 2).

The scheduler sweeps the doc_status backlog through keyset pages of
``PIPELINE_SCHEDULING_PAGE_SIZE`` records (hydrated per page via
``get_full_docs_by_ids``) instead of materializing every PENDING/orphan row
at once, so memory grows with page-size + inflight, not with the backlog.

These cover the §12-Phase2 acceptance list:

* a huge PENDING backlog drains page-by-page (cursor advances, every doc is
  processed — the tail is never starved);
* a page-query failure does NOT advance the cursor and re-arms auto-rescan,
  so the remaining backlog is recovered by the next run;
* ``PIPELINE_SCHEDULING_PAGE_SIZE=0`` collapses to the legacy single scan.

(The AUTO sweep never carries FAILED post-Phase-3; the manual FAILED reset is
paged separately by ``_next_failed_page`` — see ``test_manual_exclusive_reset``.)
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import CURSOR_START, DocStatus
from lightrag.kg.shared_storage import get_pipeline_ingress
from lightrag.utils import EmbeddingFunc, Tokenizer, compute_mdhash_id

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


class _Extract:
    def __init__(self, fail: bool = False):
        self.fail = fail

    async def __call__(self, chunks, *args, **kwargs):
        if self.fail:
            raise RuntimeError("extract fail sentinel")
        return [({}, {}) for _ in chunks]


async def _build_rag(tmp_path, *, page_size: int) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"bs-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        chunking_func=_chunking,
        max_parallel_insert=1,
        pipeline_scheduling_page_size=page_size,
    )
    await rag.initialize_storages()
    rag._process_extract_entities = _Extract()
    return rag


async def _status_of(rag: LightRAG, doc_id: str) -> str:
    row = await rag.doc_status.get_by_id(doc_id)
    raw = (row or {}).get("status")
    return raw.value if isinstance(raw, DocStatus) else str(raw or "<missing>")


async def _enqueue_n(rag: LightRAG, n: int) -> list[str]:
    ids = []
    for i in range(n):
        name = f"doc{i}.txt"
        await rag.apipeline_enqueue_documents(input=f"body {i}", file_paths=name)
        ids.append(compute_mdhash_id(name, prefix="doc-"))
    return ids


def _count_page_calls(rag):
    """Wrap the two scheduling reads, returning (counters, restore)."""
    counts = {"page": 0, "scan": 0}
    orig_page = rag.doc_status.get_docs_by_statuses_page
    orig_scan = rag.doc_status.get_docs_by_statuses

    async def counting_page(statuses, *, limit, position=CURSOR_START, strict=False):
        counts["page"] += 1
        return await orig_page(statuses, limit=limit, position=position, strict=strict)

    async def counting_scan(statuses, strict=False):
        counts["scan"] += 1
        return await orig_scan(statuses, strict=strict)

    rag.doc_status.get_docs_by_statuses_page = counting_page
    rag.doc_status.get_docs_by_statuses = counting_scan
    return counts


def test_bounded_sweep_processes_all_docs_across_pages(tmp_path):
    """A backlog larger than one page drains page-by-page: the cursor advances
    (>1 page fetched) and EVERY doc reaches PROCESSED — the tail behind the
    first pages is never starved."""

    async def _run():
        rag = await _build_rag(tmp_path, page_size=2)
        try:
            ids = await _enqueue_n(rag, 5)
            counts = _count_page_calls(rag)

            await rag.apipeline_process_enqueue_documents()

            for doc_id in ids:
                assert await _status_of(rag, doc_id) == DocStatus.PROCESSED.value
            # 5 docs / page_size 2 → the SCHEDULING sweep needed multiple pages,
            # proving it did NOT materialize the whole backlog at once. (Note:
            # get_docs_by_statuses may still be called by the per-doc
            # content-hash dedup fallback in utils_pipeline — a separate,
            # pre-existing scan outside Phase 2's scheduling scope — so we
            # assert on the page count, the scheduling-path evidence.)
            assert counts["page"] >= 3, counts
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_page_size_zero_uses_legacy_single_scan(tmp_path):
    """PIPELINE_SCHEDULING_PAGE_SIZE=0 disables paging: one get_docs_by_statuses
    scan, no page fetch — byte-for-byte the pre-Phase-2 behaviour."""

    async def _run():
        rag = await _build_rag(tmp_path, page_size=0)
        try:
            ids = await _enqueue_n(rag, 3)
            counts = _count_page_calls(rag)

            await rag.apipeline_process_enqueue_documents()

            for doc_id in ids:
                assert await _status_of(rag, doc_id) == DocStatus.PROCESSED.value
            assert counts["page"] == 0, "paging off → page API never called"
            assert counts["scan"] >= 1, "legacy single scan drove the sweep"
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_sweep_page_failure_rearms_auto_and_recovers(tmp_path):
    """A page-query failure mid-sweep must NOT advance the cursor and must
    re-arm auto-rescan; the un-swept docs stay PENDING and a clean re-run
    finishes every one (no doc stranded, none double-processed)."""

    async def _run():
        rag = await _build_rag(tmp_path, page_size=1)
        try:
            ids = await _enqueue_n(rag, 3)
            ingress = await get_pipeline_ingress(rag.workspace)
            ingress.consume_auto_rescan()  # clear enqueue-era noise

            orig_page = rag.doc_status.get_docs_by_statuses_page
            calls = {"n": 0}

            async def flaky_page(
                statuses, *, limit, position=CURSOR_START, strict=False
            ):
                calls["n"] += 1
                if calls["n"] == 2:  # fail the 2nd page (a CONTINUE_SWEEP_PAGE)
                    raise ConnectionError("backend page failure")
                return await orig_page(
                    statuses, limit=limit, position=position, strict=strict
                )

            rag.doc_status.get_docs_by_statuses_page = flaky_page

            with pytest.raises(ConnectionError, match="backend page failure"):
                await rag.apipeline_process_enqueue_documents()

            # The failure re-armed auto-rescan (remaining backlog recovery).
            assert ingress.counts()["auto_rescan_pending"] is True
            # At least one doc is still PENDING (not stranded terminal/lost).
            statuses = [await _status_of(rag, d) for d in ids]
            assert DocStatus.PENDING.value in statuses, statuses

            # Clean re-run drains everything.
            rag.doc_status.get_docs_by_statuses_page = orig_page
            await rag.apipeline_process_enqueue_documents()
            for doc_id in ids:
                assert await _status_of(rag, doc_id) == DocStatus.PROCESSED.value
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_feeder_epoch_full_gate(tmp_path):
    """The single-epoch bound: _feeder_epoch_full is True once inflight +
    routing reaches the page size, and always False when paging is off."""

    async def _run():
        rag = await _build_rag(tmp_path, page_size=2)
        try:
            from lightrag.pipeline import _BatchRunContext

            ctx = _BatchRunContext(
                pipeline_status={},
                pipeline_status_lock=asyncio.Lock(),
                semaphore=asyncio.Semaphore(1),
                total_files=0,
                parse_queues={},
                parser_specs={},
                q_analyze=asyncio.Queue(),
                q_process=asyncio.Queue(),
                pipeline_cancel_event=None,
            )
            assert rag._feeder_epoch_full(ctx) is False  # empty
            ctx.inflight_doc_ids.update({"a", "b"})
            assert rag._feeder_epoch_full(ctx) is True  # at cap (>= page_size 2)

            rag.pipeline_scheduling_page_size = 0  # paging off → never full
            assert rag._feeder_epoch_full(ctx) is False
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
