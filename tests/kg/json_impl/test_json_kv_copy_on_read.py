"""Unit tests verifying that JsonKVStorage read methods return deep copies so a
caller mutating a nested field (e.g. text_chunks' llm_cache_list, updated
in-place by lightrag.utils.update_chunk_cache_list) cannot corrupt internal
storage state without going through upsert."""

import pytest

from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.shared_storage import initialize_share_data, finalize_share_data
from lightrag.namespace import NameSpace

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
async def test_get_by_id_returns_deep_copy(tmp_path):
    storage = JsonKVStorage(
        namespace=NameSpace.KV_STORE_TEXT_CHUNKS,
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    await storage.initialize()

    await storage.upsert(
        {
            "chunk1": {
                "content": "hello",
                "llm_cache_list": ["cache-a"],
                "metadata": {"key": "original"},
            }
        }
    )

    doc = await storage.get_by_id("chunk1")
    assert doc is not None

    # Mutate the returned nested list/dict in place, mirroring
    # update_chunk_cache_list's ``chunk_data["llm_cache_list"].extend(...)``.
    doc["llm_cache_list"].append("cache-b")
    doc["metadata"]["key"] = "modified_nested"

    stored = await storage.get_by_id("chunk1")
    assert stored is not None
    assert stored["llm_cache_list"] == ["cache-a"]
    assert stored["metadata"]["key"] == "original"


@pytest.mark.asyncio
async def test_get_by_ids_returns_deep_copy(tmp_path):
    storage = JsonKVStorage(
        namespace=NameSpace.KV_STORE_TEXT_CHUNKS,
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    await storage.initialize()

    await storage.upsert(
        {
            "chunk2": {
                "content": "hello",
                "llm_cache_list": ["cache-a"],
                "metadata": {"key": "original"},
            }
        }
    )

    docs = await storage.get_by_ids(["chunk2"])
    assert len(docs) == 1 and docs[0] is not None
    docs[0]["llm_cache_list"].append("cache-b")
    docs[0]["metadata"]["key"] = "modified_nested"

    stored = await storage.get_by_id("chunk2")
    assert stored is not None
    assert stored["llm_cache_list"] == ["cache-a"]
    assert stored["metadata"]["key"] == "original"


@pytest.mark.asyncio
async def test_update_chunk_cache_list_does_not_corrupt_storage_on_read(tmp_path):
    """
    Regression test for the real vulnerable caller: update_chunk_cache_list
    reads a chunk, appends to its llm_cache_list in place, THEN calls
    upsert(). Before the deep-copy fix, that in-place append (in
    single-process mode) mutated the same list object living in
    JsonKVStorage's internal ``self._data``, bypassing ``_storage_lock``
    entirely — any concurrent reader holding an earlier ``get_by_id`` result
    for the same chunk would see the mutation appear underneath it, without
    ever calling get_by_id again.
    """
    from lightrag.utils import update_chunk_cache_list

    storage = JsonKVStorage(
        namespace=NameSpace.KV_STORE_TEXT_CHUNKS,
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    await storage.initialize()

    await storage.upsert(
        {
            "chunk3": {
                "content": "hello",
                "llm_cache_list": ["cache-a"],
            }
        }
    )

    # A snapshot taken before update_chunk_cache_list runs must stay frozen —
    # it must not observe the later cache-key append just because it holds a
    # reference to the same (would-be-aliased) list object.
    snapshot = await storage.get_by_id("chunk3")

    await update_chunk_cache_list("chunk3", storage, ["cache-b"])

    assert snapshot["llm_cache_list"] == ["cache-a"]

    stored = await storage.get_by_id("chunk3")
    assert stored is not None
    assert stored["llm_cache_list"] == ["cache-a", "cache-b"]
