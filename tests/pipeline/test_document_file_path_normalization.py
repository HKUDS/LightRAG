import sys

import pytest

sys.argv = sys.argv[:1]

from lightrag.api.routers.document_routes import (  # noqa: E402
    DocStatusResponse,
    normalize_file_path,
    pipeline_index_texts,
)
from lightrag.base import DocStatus  # noqa: E402
from lightrag.constants import PROCESS_OPTION_CHUNK_FIXED  # noqa: E402
from lightrag.pipeline import _PipelineMixin  # noqa: E402

pytestmark = pytest.mark.offline


class DummyRAG:
    def __init__(self):
        self.enqueued_calls = []
        self.processed = False
        # _resolve_text_chunking reads addon_params; {} -> default chunker config.
        self.addon_params = {}

    async def apipeline_enqueue_documents(
        self,
        input,
        file_paths=None,
        track_id=None,
        process_options=None,
        chunk_options=None,
    ):
        self.enqueued_calls.append(
            {
                "input": input,
                "file_paths": file_paths,
                "track_id": track_id,
                "process_options": process_options,
                "chunk_options": chunk_options,
            }
        )

    async def apipeline_process_enqueue_documents(self):
        self.processed = True


class CaptureDocStatus:
    def __init__(self):
        self.upserts = []

    async def upsert(self, data):
        self.upserts.append(data)


class DummyPipeline(_PipelineMixin):
    def __init__(self):
        self.doc_status = CaptureDocStatus()


class CaptureKV:
    def __init__(self):
        self.upserts = []

    async def filter_keys(self, keys):
        return set(keys)

    async def get_by_id(self, id):
        return None

    async def upsert(self, data):
        self.upserts.append(data)


@pytest.mark.asyncio
async def test_pipeline_index_texts_rejects_missing_file_sources():
    rag = DummyRAG()

    with pytest.raises(ValueError, match="valid file source"):
        await pipeline_index_texts(
            rag,
            texts=["alpha"],
            file_sources=[None],
            track_id="track-1",
        )

    assert rag.enqueued_calls == []
    assert rag.processed is False


@pytest.mark.asyncio
async def test_pipeline_index_texts_normalizes_file_sources_to_basename():
    rag = DummyRAG()

    await pipeline_index_texts(
        rag,
        texts=["alpha"],
        file_sources=["/tmp/source/alpha.txt"],
        track_id="track-1",
    )

    assert len(rag.enqueued_calls) == 1
    call = rag.enqueued_calls[0]
    assert call["input"] == ["alpha"]
    assert call["file_paths"] == ["alpha.txt"]
    assert call["track_id"] == "track-1"
    assert call["process_options"] == PROCESS_OPTION_CHUNK_FIXED
    # No chunking config supplied -> default F snapshot from addon_params.
    assert isinstance(call["chunk_options"], dict)
    assert "fixed_token" in call["chunk_options"]
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


@pytest.mark.asyncio
async def test_error_document_enqueue_canonicalizes_file_path_before_upsert():
    rag = DummyPipeline()

    await rag.apipeline_enqueue_error_documents(
        [
            {
                "file_path": "/tmp/uploads/report.[native-Fi].pdf",
                "error_description": "bad file",
                "original_error": "parse failed",
            }
        ],
        track_id="track-1",
    )

    saved = next(iter(rag.doc_status.upserts[0].values()))
    assert saved["file_path"] == "report.pdf"


@pytest.mark.asyncio
async def test_custom_chunks_use_canonical_unknown_source_before_upsert():
    from lightrag import LightRAG
    from lightrag.kg.shared_storage import (
        initialize_pipeline_status,
        initialize_share_data,
    )

    rag = LightRAG.__new__(LightRAG)
    rag.workspace = ""
    rag.full_docs = CaptureKV()
    rag.text_chunks = CaptureKV()
    rag.chunks_vdb = CaptureKV()
    rag.doc_status = CaptureKV()
    rag.tokenizer = type("Tokenizer", (), {"encode": lambda self, text: [text]})()
    initialize_share_data()
    await initialize_pipeline_status(rag.workspace)

    # ainsert_custom_chunks now passes pipeline_status / lock down to extraction
    # (see #3352); accept and ignore them here. Empty results -> no KG merge, so
    # this test still exercises only the file_path canonicalization path. It also
    # gates reprocessing on doc_status and flushes via _insert_done_with_cleanup,
    # so stub both here.
    async def _process_extract_entities(chunks, *args, **kwargs):
        return []

    async def _insert_done_with_cleanup():
        return None

    rag._process_extract_entities = _process_extract_entities
    rag._insert_done_with_cleanup = _insert_done_with_cleanup

    await rag.ainsert_custom_chunks("full text", ["chunk text"], doc_id="doc-1")

    assert rag.full_docs.upserts[0]["doc-1"]["file_path"] == "unknown_source"
    chunk = next(iter(rag.text_chunks.upserts[0].values()))
    assert chunk["file_path"] == "unknown_source"
