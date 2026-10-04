"""Mongo KV failures must retain the document's deletion retry state."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pymongo.errors import PyMongoError

from lightrag.kg.mongo_impl import MongoKVStorage

from .test_delete_fail_closed import _build_rag, _chunk_ids, _ingest

pytestmark = pytest.mark.offline


@pytest.mark.asyncio
async def test_mongo_chunk_delete_failure_preserves_retry_state(tmp_path):
    rag = await _build_rag(tmp_path, f"mongo-delete-{uuid4().hex[:8]}")
    original_delete = rag.text_chunks.delete
    try:
        doc_id = await _ingest(rag)
        chunk_ids = await _chunk_ids(rag, doc_id)
        mongo = MongoKVStorage("text_chunks", {}, None, rag.workspace)
        mongo._data = SimpleNamespace(
            delete_many=AsyncMock(side_effect=PyMongoError("delete unavailable"))
        )
        rag.text_chunks.delete = mongo.delete

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
