"""Unit tests verifying that JsonDocStatusStorage read methods return deep copies to prevent internal state mutation of nested status fields."""

import pytest
from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage
from lightrag.kg.shared_storage import initialize_share_data, finalize_share_data

pytestmark = pytest.mark.offline


class _DummyEmbeddingFunc:
    embedding_dim = 1
    max_token_size = 1

    async def __call__(self, texts, **kwargs):
        return [[0.0] for _ in texts]


@pytest.fixture(autouse=True)
def setup_shared_data():
    initialize_share_data()
    yield
    finalize_share_data()


@pytest.mark.asyncio
async def test_get_by_id_returns_copy(tmp_path):
    storage = JsonDocStatusStorage(
        namespace="doc_status",
        workspace="test",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
    )
    await storage.initialize()

    await storage.upsert(
        {
            "doc1": {
                "status": "pending",
                "file_path": "test.pdf",
                "metadata": {"key": "original"},
            }
        }
    )

    doc = await storage.get_by_id("doc1")
    assert doc is not None
    assert doc["status"] == "pending"

    # Mutate returned dictionary top-level and nested fields
    doc["status"] = "modified_externally"
    doc["metadata"]["key"] = "modified_nested"
    doc["chunks_list"] = ["chunk_1"]

    # Verify internal storage was not mutated
    doc_after = await storage.get_by_id_strict("doc1")
    assert doc_after is not None
    assert doc_after["status"] == "pending"
    assert doc_after["metadata"]["key"] == "original"
    assert doc_after["chunks_list"] == []

@pytest.mark.asyncio
async def test_get_doc_by_file_path_returns_copy(tmp_path):
    storage = JsonDocStatusStorage(
        namespace="doc_status",
        workspace="test",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
    )
    await storage.initialize()

    await storage.upsert(
        {
            "doc2": {
                "status": "pending",
                "file_path": "report.pdf",
                "metadata": {"key": "original"},
            }
        }
    )

    doc = await storage.get_doc_by_file_path("report.pdf")
    assert doc is not None
    assert doc["status"] == "pending"

    # Mutate returned dictionary top-level and nested fields
    doc["status"] = "modified_externally"
    doc["metadata"]["key"] = "modified_nested"

    # Verify internal storage was not mutated
    doc_after = await storage.get_by_id_strict("doc2")
    assert doc_after is not None
    assert doc_after["status"] == "pending"
    assert doc_after["metadata"]["key"] == "original"
@pytest.mark.asyncio
async def test_get_by_ids_returns_deep_copy(tmp_path):
    """
    Locks out the bug where get_by_ids returned shallow copies, allowing external callers
    to mutate nested metadata or chunks_list in internal storage.
    """
    storage = JsonDocStatusStorage(
        namespace="doc_status",
        workspace="test",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
    )
    await storage.initialize()

    await storage.upsert(
        {
            "doc3": {
                "status": "pending",
                "file_path": "data.pdf",
                "metadata": {"kg_purge": {"phase": "prepared"}},
                "chunks_list": ["c1", "c2"],
            }
        }
    )

    docs = await storage.get_by_ids(["doc3"])
    assert len(docs) == 1 and docs[0] is not None
    docs[0]["metadata"]["kg_purge"]["phase"] = "corrupted"
    docs[0]["chunks_list"].append("c3")

    stored = await storage.get_by_id_strict("doc3")
    assert stored is not None
    assert stored["metadata"]["kg_purge"]["phase"] == "prepared"
    assert stored["chunks_list"] == ["c1", "c2"]
