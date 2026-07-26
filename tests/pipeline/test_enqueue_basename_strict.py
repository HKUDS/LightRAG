"""Enqueue's filename dedup resolves the source STRICTLY.

Step 3a of ``apipeline_enqueue_documents`` used ``get_doc_by_file_basename``,
which is best-effort by contract: several backends swallow transport errors and
return ``None``, and enqueue reads "no match" as "not a duplicate". A storage
blip therefore admitted a second row for a filename that already existed. The
same call also returned SOME primary on a historical basename collision, silently
attaching the new document to an arbitrary one.

Both are now routed through ``resolve_doc_source_strict``: failures raise, and a
collision is refused explicitly instead of guessed.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import DocStatus, SourceAbsent, SourceConflict, SourceUnique
from lightrag.exceptions import StorageControlPlaneError
from lightrag.utils import EmbeddingFunc, Tokenizer
from lightrag.utils_pipeline import resolve_existing_doc_source

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


async def _build_rag(tmp_path) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"basename-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    return rag


async def _rows(rag) -> dict:
    return dict(rag.doc_status._data)


# ---------------------------------------------------------------------------
# resolve_existing_doc_source
# ---------------------------------------------------------------------------


class _RaisingDocStatus:
    """Stands in for a backend whose identity query fails."""

    def __init__(self):
        self.calls = 0

    async def resolve_doc_source_strict(self, canonical_source_key):
        self.calls += 1
        raise StorageControlPlaneError("backend down")


@pytest.mark.asyncio
async def test_resolver_propagates_backend_failure():
    doc_status = _RaisingDocStatus()
    with pytest.raises(StorageControlPlaneError):
        await resolve_existing_doc_source(doc_status, "report.pdf")
    assert doc_status.calls == 1


@pytest.mark.asyncio
async def test_resolver_treats_unknown_source_as_absent_without_querying():
    doc_status = _RaisingDocStatus()
    assert isinstance(
        await resolve_existing_doc_source(doc_status, "unknown_source"), SourceAbsent
    )
    assert doc_status.calls == 0


# ---------------------------------------------------------------------------
# enqueue behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enqueue_refuses_when_the_identity_query_fails(tmp_path, monkeypatch):
    """Fix-proof: the swallowed error used to look like "not a duplicate", so the
    document was admitted and a second row for an existing filename was written.
    Enqueue must now fail instead of admitting it."""
    rag = await _build_rag(tmp_path)
    try:

        async def _boom(canonical_source_key):
            raise StorageControlPlaneError("doc_status unavailable")

        monkeypatch.setattr(rag.doc_status, "resolve_doc_source_strict", _boom)

        with pytest.raises(StorageControlPlaneError):
            await rag.apipeline_enqueue_documents(
                "hello world", file_paths="report.pdf", track_id="t-1"
            )

        # Nothing was admitted on the failure.
        assert await _rows(rag) == {}
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_enqueue_admits_a_new_basename(tmp_path):
    rag = await _build_rag(tmp_path)
    try:
        await rag.apipeline_enqueue_documents(
            "hello world", file_paths="report.pdf", track_id="t-1"
        )
        rows = await _rows(rag)
        assert len(rows) == 1
        (row,) = rows.values()
        assert row["status"] == DocStatus.PENDING
        assert row["file_path"] == "report.pdf"
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_enqueue_marks_a_unique_existing_basename_as_duplicate(tmp_path):
    rag = await _build_rag(tmp_path)
    try:
        await rag.apipeline_enqueue_documents(
            "hello world", file_paths="report.pdf", track_id="t-1"
        )
        original_id = next(iter(await _rows(rag)))

        await rag.apipeline_enqueue_documents(
            "different body", file_paths="report.pdf", track_id="t-2"
        )
        rows = await _rows(rag)
        dup_rows = {k: v for k, v in rows.items() if k.startswith("dup-")}
        assert len(dup_rows) == 1
        (dup,) = dup_rows.values()
        assert dup["status"] == DocStatus.FAILED
        assert dup["metadata"]["duplicate_kind"] == "filename"
        assert dup["metadata"]["original_doc_id"] == original_id
        assert "File name already exists" in dup["error_msg"]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_enqueue_refuses_an_unresolved_basename_conflict(tmp_path, monkeypatch):
    """A historical collision leaves several primaries on one basename. Attaching
    the new document to an arbitrary one hides the collision, so enqueue refuses
    and records WHY, with the bounded candidate sample."""
    rag = await _build_rag(tmp_path)
    try:

        async def _conflict(canonical_source_key):
            return SourceConflict(
                candidate_count=2, sample_doc_ids=("doc-old-a", "doc-old-b")
            )

        monkeypatch.setattr(rag.doc_status, "resolve_doc_source_strict", _conflict)

        await rag.apipeline_enqueue_documents(
            "hello world", file_paths="report.pdf", track_id="t-1"
        )

        rows = await _rows(rag)
        # Refused: only the trackable conflict record, no PENDING work.
        assert all(k.startswith("dup-") for k in rows), rows
        assert len(rows) == 1
        (record,) = rows.values()
        assert record["status"] == DocStatus.FAILED
        assert record["metadata"]["duplicate_kind"] == "filename_conflict"
        # No original is asserted; the sample is surfaced instead.
        assert "doc-old-a" in record["error_msg"]
        assert "doc-old-b" in record["error_msg"]
        assert "Repair the conflict by doc id" in record["error_msg"]
        assert "repair required" in record["content_summary"]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_enqueue_uses_the_strict_resolver_not_the_legacy_lookup(
    tmp_path, monkeypatch
):
    """Guard against a regression to the best-effort path: the legacy
    basename lookup must not be what decides the duplicate."""
    rag = await _build_rag(tmp_path)
    try:
        seen = {"strict": 0, "legacy": 0}
        original = rag.doc_status.resolve_doc_source_strict

        async def _counting_strict(canonical_source_key):
            seen["strict"] += 1
            return await original(canonical_source_key)

        async def _counting_legacy(basename):
            seen["legacy"] += 1
            return None

        monkeypatch.setattr(
            rag.doc_status, "resolve_doc_source_strict", _counting_strict
        )
        monkeypatch.setattr(
            rag.doc_status, "get_doc_by_file_basename", _counting_legacy
        )

        await rag.apipeline_enqueue_documents(
            "hello world", file_paths="report.pdf", track_id="t-1"
        )

        assert seen["strict"] >= 1
        assert seen["legacy"] == 0
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_unique_resolution_reuses_the_page_projection(tmp_path):
    """SourceUnique already carries status/track_id, so the duplicate record is
    built without a second read of the original row."""
    rag = await _build_rag(tmp_path)
    try:
        await rag.apipeline_enqueue_documents(
            "hello world", file_paths="report.pdf", track_id="t-orig"
        )
        resolution = await rag.doc_status.resolve_doc_source_strict("report.pdf")
        assert isinstance(resolution, SourceUnique)
        assert resolution.doc.track_id == "t-orig"
        assert resolution.doc.status is DocStatus.PENDING
    finally:
        await rag.finalize_storages()
