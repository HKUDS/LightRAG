import pytest

from lightrag.base import DocStatus
from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage
from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.namespace import NameSpace

pytestmark = pytest.mark.offline


class _DummyEmbeddingFunc:
    embedding_dim = 1
    max_token_size = 1

    async def __call__(self, texts, **kwargs):
        return [[0.0] for _ in texts]


def _doc(status: str, file_path: str) -> dict:
    return {
        "content_summary": f"{status} summary",
        "content_length": 10,
        "file_path": file_path,
        "status": status,
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00",
        "metadata": {},
        "error_msg": None,
    }


@pytest.fixture(autouse=True)
def setup_shared_data():
    initialize_share_data()
    yield
    finalize_share_data()


@pytest.mark.asyncio
async def test_get_docs_paginated_with_status_filters(tmp_path):
    storage = JsonDocStatusStorage(
        namespace="doc_status",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    await storage.initialize()

    async with storage._storage_lock:
        storage._data.update(
            {
                "doc-1": _doc("preprocessed", "a.pdf"),
                "doc-2": _doc("parsing", "b.pdf"),
                "doc-3": _doc("analyzing", "c.pdf"),
                "doc-4": _doc("processed", "d.pdf"),
            }
        )

    docs, total = await storage.get_docs_paginated(
        status_filter=DocStatus.PROCESSED,
        status_filters=[
            DocStatus.PREPROCESSED,
            DocStatus.PARSING,
            DocStatus.ANALYZING,
        ],
        page=1,
        page_size=10,
        sort_field="id",
        sort_direction="asc",
    )

    assert total == 3
    assert [doc_id for doc_id, _ in docs] == ["doc-1", "doc-2", "doc-3"]
    assert [doc.status for _, doc in docs] == [
        DocStatus.PREPROCESSED,
        DocStatus.PARSING,
        DocStatus.ANALYZING,
    ]


@pytest.mark.asyncio
async def test_doc_status_upsert_preserves_caller_file_path(tmp_path):
    storage = JsonDocStatusStorage(
        namespace=NameSpace.DOC_STATUS,
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    await storage.initialize()

    await storage.upsert(
        {
            "doc-1": _doc(
                DocStatus.PENDING.value,
                "/tmp/uploads/report.[native-Fi].pdf",
            )
        }
    )

    assert (await storage.get_by_id("doc-1"))["file_path"] == (
        "/tmp/uploads/report.[native-Fi].pdf"
    )
    assert await storage.get_doc_by_file_basename("report.pdf") is None


@pytest.mark.asyncio
async def test_doc_status_basename_lookup_requires_canonical_stored_path(tmp_path):
    storage = JsonDocStatusStorage(
        namespace=NameSpace.DOC_STATUS,
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    await storage.initialize()

    async with storage._storage_lock:
        storage._data["doc-1"] = _doc(
            DocStatus.PROCESSED.value,
            "report.[native].pdf",
        )

    assert await storage.get_doc_by_file_basename("report.pdf") is None


@pytest.mark.asyncio
async def test_json_kv_upsert_preserves_caller_file_paths(tmp_path):
    full_docs = JsonKVStorage(
        namespace=NameSpace.KV_STORE_FULL_DOCS,
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    text_chunks = JsonKVStorage(
        namespace=NameSpace.KV_STORE_TEXT_CHUNKS,
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    await full_docs.initialize()
    await text_chunks.initialize()

    await full_docs.upsert(
        {
            "doc-1": {
                "content": "full text",
                "file_path": "/tmp/uploads/report.[native-Fi].pdf",
            }
        }
    )
    await text_chunks.upsert(
        {
            "chunk-1": {
                "content": "chunk text",
                "tokens": 2,
                "chunk_order_index": 0,
                "full_doc_id": "doc-1",
                "file_path": "/tmp/uploads/report.[native-Fi].pdf",
            }
        }
    )

    assert (await full_docs.get_by_id("doc-1"))["file_path"] == (
        "/tmp/uploads/report.[native-Fi].pdf"
    )
    assert (await text_chunks.get_by_id("chunk-1"))["file_path"] == (
        "/tmp/uploads/report.[native-Fi].pdf"
    )
