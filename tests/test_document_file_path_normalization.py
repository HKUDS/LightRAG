import sys

import pytest

sys.argv = sys.argv[:1]

from lightrag.api.routers.document_routes import (  # noqa: E402
    DocStatusResponse,
    normalize_file_path,
    pipeline_index_texts,
)
from lightrag.base import DocStatus  # noqa: E402


class DummyRAG:
    def __init__(self):
        self.enqueued_calls = []
        self.processed = False

    async def apipeline_enqueue_documents(self, input, file_paths=None, track_id=None):
        self.enqueued_calls.append(
            {"input": input, "file_paths": file_paths, "track_id": track_id}
        )

    async def apipeline_process_enqueue_documents(self):
        self.processed = True


@pytest.mark.asyncio
async def test_pipeline_index_texts_normalizes_missing_file_sources():
    rag = DummyRAG()

    await pipeline_index_texts(
        rag,
        texts=["alpha"],
        file_sources=[None],
        track_id="track-1",
    )

    assert rag.enqueued_calls == [
        {
            "input": ["alpha"],
            "file_paths": ["unknown_source"],
            "track_id": "track-1",
        }
    ]
    assert rag.processed is True


def test_doc_status_response_uses_non_null_unknown_source():
    response = DocStatusResponse(
        id="doc-1",
        content_summary="summary",
        content_length=5,
        status=DocStatus.PENDING,
        created_at="2026-03-19T00:00:00+00:00",
        updated_at="2026-03-19T00:00:00+00:00",
        file_path=normalize_file_path(None),
    )

    assert response.file_path == "unknown_source"
