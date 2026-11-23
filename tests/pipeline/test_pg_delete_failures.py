"""PostgreSQL delete failures must keep the document's deletion retry state.

The real ``PGKVStorage.delete`` / ``PGDocStatusStorage.delete`` run against a
database seam that fails; the rest of the pipeline uses the JSON / NetworkX
defaults. Before the fix both deletes logged the error and returned, so
``adelete_by_doc_id`` reported success over a delete that never happened.
"""

from uuid import uuid4

import pytest

pytest.importorskip("asyncpg")

from lightrag.kg.postgres_impl import (  # noqa: E402
    PGDocStatusStorage,
    PGKVStorage,
)

from .test_delete_fail_closed import _build_rag, _chunk_ids, _ingest  # noqa: E402

pytestmark = pytest.mark.offline


class _FailingDB:
    async def _run_with_retry(self, operation):
        raise RuntimeError("delete unavailable")


def _failing_pg_delete(cls, namespace: str, workspace: str):
    storage = object.__new__(cls)
    storage.namespace = namespace
    storage.workspace = workspace
    storage._max_delete_records_per_batch = 0
    storage.db = _FailingDB()
    return storage.delete


@pytest.mark.asyncio
async def test_pg_chunk_delete_failure_preserves_retry_state(tmp_path):
    rag = await _build_rag(tmp_path, f"pg-chunks-{uuid4().hex[:8]}")
    original_delete = rag.text_chunks.delete
    try:
        doc_id = await _ingest(rag)
        chunk_ids = await _chunk_ids(rag, doc_id)
        rag.text_chunks.delete = _failing_pg_delete(
            PGKVStorage, rag.text_chunks.namespace, rag.workspace
        )

        result = await rag.adelete_by_doc_id(doc_id)

        assert result.status == "fail"
        row = await rag.doc_status.get_by_id(doc_id)
        assert row is not None
        assert row["metadata"]["deletion_failure_stage"] == "delete_chunks"
        assert await rag.full_docs.get_by_id(doc_id) is not None
        assert all(await rag.text_chunks.get_by_ids(chunk_ids))

        rag.text_chunks.delete = original_delete
        retry = await rag.adelete_by_doc_id(doc_id)
        assert retry.status == "success"
        assert await rag.doc_status.get_by_id(doc_id) is None
        assert await rag.full_docs.get_by_id(doc_id) is None
        assert not any(await rag.text_chunks.get_by_ids(chunk_ids))
    finally:
        rag.text_chunks.delete = original_delete
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_pg_doc_status_delete_failure_keeps_document_intact(tmp_path):
    rag = await _build_rag(tmp_path, f"pg-status-{uuid4().hex[:8]}")
    original_delete = rag.doc_status.delete
    try:
        doc_id = await _ingest(rag)
        rag.doc_status.delete = _failing_pg_delete(
            PGDocStatusStorage, rag.doc_status.namespace, rag.workspace
        )

        result = await rag.adelete_by_doc_id(doc_id)

        # The swallowed failure used to report success, then delete full_docs
        # while the status row survived -- the state the doc_status-first
        # ordering in adelete_by_doc_id exists to avoid.
        assert result.status == "fail"
        assert await rag.doc_status.get_by_id(doc_id) is not None
        assert await rag.full_docs.get_by_id(doc_id) is not None

        rag.doc_status.delete = original_delete
        retry = await rag.adelete_by_doc_id(doc_id)
        assert retry.status == "success"
        assert await rag.doc_status.get_by_id(doc_id) is None
        assert await rag.full_docs.get_by_id(doc_id) is None
    finally:
        rag.doc_status.delete = original_delete
        await rag.finalize_storages()
