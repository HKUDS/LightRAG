"""Unit tests verifying escaping of double quotes and backslashes in Milvus query filter expressions."""

import pytest
from unittest.mock import MagicMock
from lightrag.kg.milvus_impl import MilvusVectorDBStorage
from lightrag.kg.shared_storage import initialize_share_data, finalize_share_data

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def setup_shared_data():
    initialize_share_data()
    yield
    finalize_share_data()


def _make_milvus_storage():
    mock_embedding_func = MagicMock()
    mock_embedding_func.embedding_dim = 128
    storage = MilvusVectorDBStorage(
        namespace="test_entities",
        workspace="test_workspace",
        global_config={
            "embedding_batch_num": 100,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.3,
            },
        },
        embedding_func=mock_embedding_func,
        meta_fields={"content", "created_at"},
    )
    storage._client = MagicMock()
    return storage


@pytest.mark.asyncio
async def test_get_by_id_escapes_special_characters():
    storage = _make_milvus_storage()
    storage._client.query.return_value = [{"id": 'doc"1', "content": "test"}]

    result = await storage.get_by_id('doc"1')
    assert result == {"id": 'doc"1', "content": "test"}

    storage._client.query.assert_called_once()
    _, kwargs = storage._client.query.call_args
    assert kwargs["filter"] == 'id == "doc\\"1"'


@pytest.mark.asyncio
async def test_get_by_ids_escapes_special_characters():
    storage = _make_milvus_storage()
    storage._client.query.return_value = [{"id": 'doc"1', "content": "test"}]

    result = await storage.get_by_ids(['doc"1', "doc\\2"])
    assert result == [{"id": 'doc"1', "content": "test"}, None]

    storage._client.query.assert_called_once()
    _, kwargs = storage._client.query.call_args
    assert kwargs["filter"] == 'id in ["doc\\"1", "doc\\\\2"]'


@pytest.mark.asyncio
async def test_get_vectors_by_ids_escapes_special_characters():
    storage = _make_milvus_storage()
    storage._client.query.return_value = [{"id": 'vec"1', "vector": [0.1, 0.2]}]

    result = await storage.get_vectors_by_ids(['vec"1'])
    assert result == {'vec"1': [0.1, 0.2]}

    storage._client.query.assert_called_once()
    _, kwargs = storage._client.query.call_args
    assert kwargs["filter"] == 'id in ["vec\\"1"]'


@pytest.mark.asyncio
async def test_delete_entity_relation_escapes_special_characters():
    storage = _make_milvus_storage()
    storage._client.query.return_value = [{"id": "rel1"}]

    await storage.delete_entity_relation('ENTITY "A"')

    storage._client.query.assert_called_once()
    _, kwargs = storage._client.query.call_args
    assert (
        kwargs["filter"] == 'src_id == "ENTITY \\"A\\"" or tgt_id == "ENTITY \\"A\\""'
    )
