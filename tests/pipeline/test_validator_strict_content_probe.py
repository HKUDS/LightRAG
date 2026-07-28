"""The batch consistency validator only DELETES on a confirmed content absence
(LR2 §5.4 / §7.2).

``_validate_and_fix_document_consistency`` deletes the ``doc_status`` row of an
active document whose ``full_docs`` content is gone. That verdict comes from a
point read, and a point read that answers "best effort" turns a storage blip into
data loss: OpenSearch's KV ``get_by_id`` returns ``None`` when its index is not
ready or transiently missing (its own contract comment says so), so a validator
reading ``None`` as "content missing" deletes every live scheduling row it swept
during the outage. Those rows are the ONLY record that the documents were queued.

So the probe is strict where the backend offers it, and a backend that cannot
confirm keeps the row out of the batch instead of deleting it. Same rule as the
manual reset's ``_reset_failed_page`` and scan's ``_confirm_full_docs_absent``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import DocStatus
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


def _chunking(tokenizer, content, *a, **k) -> list[dict]:
    return [{"tokens": 1, "content": f"{content}::chunk1", "chunk_order_index": 0}]


async def _build_rag(tmp_path) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"vscp-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        chunking_func=_chunking,
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    return rag


async def _seed_active_row(rag: LightRAG, name: str) -> str:
    """An interrupted (PROCESSING) row — the shape the validator judges."""
    doc_id = compute_mdhash_id(name, prefix="doc-")
    now = datetime.now(timezone.utc).isoformat()
    await rag.doc_status.upsert(
        {
            doc_id: {
                "status": DocStatus.PROCESSING,
                "content_summary": "body",
                "content_length": 4,
                "chunks_count": 0,
                "chunks_list": [],
                "created_at": now,
                "updated_at": now,
                "file_path": name,
                "track_id": "t",
                "error_msg": "",
                "metadata": {},
            }
        }
    )
    return doc_id


async def _validate(rag: LightRAG, doc_ids: list[str]) -> dict:
    from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock

    status = await get_namespace_data("pipeline_status", workspace=rag.workspace)
    lock = get_namespace_lock("pipeline_status", workspace=rag.workspace)
    hydrated = await rag.doc_status.get_full_docs_by_ids(doc_ids, strict=True)
    return await rag._validate_and_fix_document_consistency(hydrated, status, lock)


def test_unconfirmable_miss_keeps_the_row(tmp_path):
    """A backend WITHOUT strict point reads cannot tell "content gone" from "read
    failed", so the row is kept and excluded from the batch — never deleted."""

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            doc_id = await _seed_active_row(rag, "blip.txt")

            # A best-effort backend: no strict point reads, and a failure that
            # surfaces as a plain miss (the OpenSearch index-not-ready shape).
            rag.full_docs.supports_strict_point_reads = False

            async def best_effort_miss(_doc_id):
                return None

            rag.full_docs.get_by_id = best_effort_miss

            survivors = await _validate(rag, [doc_id])

            # Not routed this batch...
            assert doc_id not in survivors
            # ...and crucially still in doc_status: the queue record survived.
            row = await rag.doc_status.get_by_id(doc_id)
            assert row is not None
            raw = row["status"]
            value = raw.value if isinstance(raw, DocStatus) else raw
            assert value == DocStatus.PROCESSING.value
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_confirmed_absence_still_deletes(tmp_path):
    """The other half of the contract: with a strict point read that CONFIRMS the
    content is gone, the inconsistent row is still deleted. Failing closed must
    not turn into never cleaning up."""

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            doc_id = await _seed_active_row(rag, "really-gone.txt")

            rag.full_docs.supports_strict_point_reads = True

            async def confirmed_absent(_doc_id):
                return None

            rag.full_docs.get_by_id_strict = confirmed_absent

            survivors = await _validate(rag, [doc_id])

            assert doc_id not in survivors
            assert await rag.doc_status.get_by_id(doc_id) is None
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_strict_read_failure_propagates(tmp_path):
    """A strict point read that RAISES must not be swallowed into a miss: the
    batch aborts and the rows stay for the next run."""

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            doc_id = await _seed_active_row(rag, "flaky.txt")

            rag.full_docs.supports_strict_point_reads = True

            async def boom(_doc_id):
                raise ConnectionError("full_docs transport failure")

            rag.full_docs.get_by_id_strict = boom

            with pytest.raises(ConnectionError, match="transport failure"):
                await _validate(rag, [doc_id])

            assert await rag.doc_status.get_by_id(doc_id) is not None
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_content_is_read_once_per_document(tmp_path):
    """The consistency verdict and the PENDING-reset payload come from ONE read.

    Two reads could disagree (the content vanishing in between made the validator
    reset a row it had just judged consistent) and doubled the round trips per
    swept document."""

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            doc_id = await _seed_active_row(rag, "counted.txt")
            await rag.full_docs.upsert({doc_id: {"content": "body"}})

            rag.full_docs.supports_strict_point_reads = True
            calls: list[str] = []
            real_strict = rag.full_docs.get_by_id_strict

            async def counting(target):
                calls.append(target)
                return await real_strict(target)

            rag.full_docs.get_by_id_strict = counting

            survivors = await _validate(rag, [doc_id])

            # Content present → reset to PENDING and carried forward.
            assert doc_id in survivors
            row = await rag.doc_status.get_by_id(doc_id)
            raw = row["status"]
            value = raw.value if isinstance(raw, DocStatus) else raw
            assert value == DocStatus.PENDING.value

            assert calls == [doc_id]
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
