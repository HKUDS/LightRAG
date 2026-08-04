"""A source-conflict demotion survives every later doc_status write (LR2 §5.5).

``metadata.is_duplicate`` is not a display field: it is the ONLY thing that makes
a row ineligible as a primary candidate for its canonical source (every backend's
``_basename_of`` returns None for it). So an operator's source-conflict repair
lives or dies with that key surviving.

It did not. Both metadata rebuilds dropped it, and a repair deliberately keeps the
demoted row's content AND status — so it stays a perfectly ordinary document that
the pipeline will happily touch:

* the manual FAILED→PENDING retry (``_build_pending_reset_update`` →
  ``doc_status_reset_metadata``) rebuilt metadata from a directives whitelist;
* every status transition (``doc_status_transition_metadata`` →
  ``doc_status_metadata_carry_over``) rebuilt it from a carry-over whitelist.

Either one handed the canonical source straight back to the demoted row, putting
the key back into conflict — deterministically, with no concurrency involved. And
the repair tool cannot undo that: ``repair_source_conflict`` only demotes, and it
refuses a ``primary_doc_id`` that is not a current candidate, so the resurrected
conflict has to be repaired from scratch every time.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import CURSOR_START, DocStatus, SourceConflict, SourceUnique
from lightrag.kg.shared_storage import (
    finalize_share_data,
    get_namespace_data,
    get_namespace_lock,
    initialize_share_data,
    make_owner_record,
)
from lightrag.utils import EmbeddingFunc, Tokenizer

pytestmark = pytest.mark.offline

_SOURCE = "collision.pdf"
_TOKEN = "owner-A"


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
        workspace=f"demote-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    return rag


def _row(doc_id: str, status: DocStatus, seconds: int) -> dict:
    """A primary-candidate row: real content, no duplicate marker."""
    stamp = datetime(2026, 1, 1, 0, 0, seconds, tzinfo=timezone.utc).isoformat()
    return {
        "status": status,
        "content_summary": f"summary-{doc_id}",
        "content_length": 12,
        "chunks_count": 0,
        "chunks_list": [],
        "created_at": stamp,
        "updated_at": stamp,
        "file_path": _SOURCE,
        "track_id": "t",
        "metadata": {"process_options": "R!"},
    }


async def _seed_conflict(rag, loser_status: DocStatus) -> tuple[str, str]:
    """Two primaries on one canonical source, then repair keeping ``doc-keep``.

    Written straight to storage: enqueue refuses a colliding basename by design,
    so a historical collision is the only way this state exists.
    """
    keep, loser = "doc-keep", "doc-loser"
    await rag.doc_status.upsert(
        {
            keep: _row(keep, DocStatus.PROCESSED, 1),
            loser: _row(loser, loser_status, 2),
        }
    )
    # The reset probes full_docs for content; a demoted row keeps its own.
    await rag.full_docs.upsert(
        {
            keep: {"content": "keep body", "file_path": _SOURCE},
            loser: {"content": "loser body", "file_path": _SOURCE},
        }
    )
    assert isinstance(
        await rag.doc_status.resolve_doc_source_strict(_SOURCE), SourceConflict
    )

    dry = await rag.doc_status.repair_source_conflict(
        _SOURCE,
        primary_doc_id=keep,
        expected_candidate_count=0,
        expected_candidate_fingerprint="",
        dry_run=True,
    )
    committed = await rag.doc_status.repair_source_conflict(
        _SOURCE,
        primary_doc_id=keep,
        expected_candidate_count=dry.candidate_count,
        expected_candidate_fingerprint=dry.fingerprint,
        dry_run=False,
    )
    assert committed.committed is True
    resolution = await rag.doc_status.resolve_doc_source_strict(_SOURCE)
    assert isinstance(resolution, SourceUnique) and resolution.doc_id == keep
    return keep, loser


async def _status_handles(rag, token: str):
    status = await get_namespace_data("pipeline_status", workspace=rag.workspace)
    lock = get_namespace_lock("pipeline_status", workspace=rag.workspace)
    status["history_messages"] = []
    status.update({"busy": True, "busy_owner": make_owner_record(token, "processing")})
    return status, lock


def test_a_manual_failed_retry_does_not_undo_the_repair(tmp_path):
    """Fix-proof: the operator repairs the conflict, then anyone clicking "retry
    failed documents" (or running /scan, which drives the same reset) handed the
    canonical source back to the demoted row."""

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            keep, loser = await _seed_conflict(rag, DocStatus.FAILED)
            status, lock = await _status_handles(rag, _TOKEN)

            # The production reset sweep, over the page that contains the
            # demoted row.
            docs, _ = await rag._next_failed_page(CURSOR_START)
            assert loser in docs
            reset = await rag._reset_failed_page(docs, _TOKEN, status, lock)
            assert reset == 1  # it really was reset, not skipped

            row = await rag.doc_status.get_by_id(loser)
            assert row["status"] == DocStatus.PENDING  # retried, as asked
            assert row["metadata"]["is_duplicate"] is True  # but still demoted
            assert row["metadata"]["original_doc_id"] == keep
            assert row["metadata"]["process_options"] == "R!"  # directives kept

            resolution = await rag.doc_status.resolve_doc_source_strict(_SOURCE)
            assert isinstance(resolution, SourceUnique)
            assert resolution.doc_id == keep
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_a_status_transition_does_not_undo_the_repair(tmp_path):
    """The wider half: a demoted row in any non-terminal status is resumed by the
    ordinary AUTO sweep, and its first transition rebuilt metadata from the
    carry-over whitelist — so the demotion did not even need a manual retry to
    disappear."""

    async def _run():
        from lightrag.pipeline import _BatchRunContext
        from lightrag.parser.registry import parser_specs_snapshot

        rag = await _build_rag(tmp_path)
        try:
            keep, loser = await _seed_conflict(rag, DocStatus.PENDING)
            status, lock = await _status_handles(rag, _TOKEN)

            hydrated = await rag.doc_status.get_full_docs_by_ids([loser], strict=True)
            ctx = _BatchRunContext(
                pipeline_status=status,
                pipeline_status_lock=lock,
                semaphore=asyncio.Semaphore(1),
                total_files=1,
                parse_queues={"native": asyncio.Queue()},
                parser_specs=parser_specs_snapshot(),
                q_analyze=asyncio.Queue(),
                q_process=asyncio.Queue(),
                run_owner_token=_TOKEN,
            )

            await rag._upsert_doc_status_transition(
                doc_id=loser,
                status=DocStatus.PROCESSED,
                status_doc=hydrated[loser],
                file_path=_SOURCE,
                ctx=ctx,
            )

            row = await rag.doc_status.get_by_id(loser)
            assert row["status"] == DocStatus.PROCESSED
            assert row["metadata"]["is_duplicate"] is True
            assert row["metadata"]["original_doc_id"] == keep

            resolution = await rag.doc_status.resolve_doc_source_strict(_SOURCE)
            assert isinstance(resolution, SourceUnique)
            assert resolution.doc_id == keep
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())


def test_the_repair_tool_cannot_undo_a_resurrection_itself(tmp_path):
    """Why the two tests above are load-bearing rather than cosmetic: there is no
    supported way back. ``repair_source_conflict`` refuses a primary that is not a
    current candidate, so once a demoted row is re-promoted the only exit is
    repairing the conflict again from scratch."""

    async def _run():
        rag = await _build_rag(tmp_path)
        try:
            _keep, loser = await _seed_conflict(rag, DocStatus.FAILED)
            with pytest.raises(ValueError):
                await rag.doc_status.repair_source_conflict(
                    _SOURCE,
                    primary_doc_id=loser,  # demoted: no longer a candidate
                    expected_candidate_count=1,
                    expected_candidate_fingerprint="whatever",
                    dry_run=True,
                )
        finally:
            await rag.finalize_storages()

    asyncio.run(_run())
