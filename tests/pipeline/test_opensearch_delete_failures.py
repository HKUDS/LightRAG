"""Per-item OpenSearch deletion failures must stop the document purge."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

import lightrag.kg.opensearch_impl as opensearch_module
from lightrag.kg.opensearch_impl import (
    OpenSearchDocStatusStorage,
    OpenSearchGraphStorage,
)

from .test_delete_fail_closed import _build_rag, _chunk_ids, _ingest

pytestmark = pytest.mark.offline


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["remove_edges", "remove_nodes"])
async def test_failed_graph_bulk_delete_preserves_document(
    tmp_path, monkeypatch, method
):
    rag = await _build_rag(tmp_path, f"os-delete-{uuid4().hex[:8]}")
    graph = rag.chunk_entity_relation_graph
    original_delete = getattr(graph, method)
    try:
        doc_id = await _ingest(rag)
        chunk_ids = await _chunk_ids(rag, doc_id)
        failure = {
            "delete": {"_id": "failed-id", "status": 503, "error": "unavailable"}
        }
        storage = OpenSearchGraphStorage("graph", {}, None, rag.workspace)
        storage._indices_ready = True
        storage.client = SimpleNamespace(
            bulk=AsyncMock(return_value={"errors": True, "items": [failure]}),
            delete_by_query=AsyncMock(return_value={"failures": []}),
        )
        monkeypatch.setattr(
            opensearch_module,
            "_run_chunked_async_bulk",
            AsyncMock(return_value=(0, [failure])),
        )
        setattr(graph, method, getattr(storage, method))

        result = await rag.adelete_by_doc_id(doc_id)

        assert result.status == "fail"
        assert await rag.doc_status.get_by_id(doc_id) is not None
        assert await rag.full_docs.get_by_id(doc_id) is not None
        assert all(await rag.text_chunks.get_by_ids(chunk_ids))

        setattr(graph, method, original_delete)
        retry = await rag.adelete_by_doc_id(doc_id)
        assert retry.status == "success"
        assert await rag.doc_status.get_by_id(doc_id) is None
    finally:
        setattr(graph, method, original_delete)
        await rag.finalize_storages()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ["item", "transport"])
async def test_failed_status_delete_preserves_retry_record(
    tmp_path, monkeypatch, failure_kind
):
    rag = await _build_rag(tmp_path, f"os-status-delete-{uuid4().hex[:8]}")
    original_delete = rag.doc_status.delete
    try:
        doc_id = await _ingest(rag)
        storage = OpenSearchDocStatusStorage("doc_status", {}, None, rag.workspace)
        storage._index_ready = True
        storage.client = SimpleNamespace()
        failure = {"delete": {"_id": doc_id, "status": 503, "error": "unavailable"}}
        bulk = AsyncMock(return_value=(0, [failure]))
        if failure_kind == "transport":
            bulk.side_effect = opensearch_module.OpenSearchException("unavailable")
        monkeypatch.setattr(opensearch_module, "_run_chunked_async_bulk", bulk)
        rag.doc_status.delete = storage.delete

        result = await rag.adelete_by_doc_id(doc_id)

        assert result.status == "fail"
        assert await rag.doc_status.get_by_id(doc_id) is not None

        rag.doc_status.delete = original_delete
        retry = await rag.adelete_by_doc_id(doc_id)
        assert retry.status == "success"
        assert await rag.doc_status.get_by_id(doc_id) is None
    finally:
        rag.doc_status.delete = original_delete
        await rag.finalize_storages()
