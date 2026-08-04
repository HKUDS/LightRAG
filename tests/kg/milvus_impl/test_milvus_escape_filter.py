"""Milvus filter escaping and structured-value boundary tests."""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from lightrag.kg.milvus_impl import MilvusVectorDBStorage, _escape_milvus_str
from lightrag.kg.shared_storage import initialize_share_data, finalize_share_data

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def setup_shared_data():
    initialize_share_data()
    yield
    finalize_share_data()


def _make_milvus_storage(
    *,
    namespace: str = "test_entities",
    meta_fields: set[str] | None = None,
):
    mock_embedding_func = MagicMock()
    mock_embedding_func.embedding_dim = 128
    storage = MilvusVectorDBStorage(
        namespace=namespace,
        workspace="test_workspace",
        global_config={
            "embedding_batch_num": 100,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.3,
            },
        },
        embedding_func=mock_embedding_func,
        meta_fields=meta_fields or {"content", "created_at"},
    )
    storage._client = MagicMock()
    return storage


@pytest.mark.parametrize(
    ("raw", "escaped"),
    [
        ('doc"1', 'doc\\"1'),
        ("doc\\2", "doc\\\\2"),
        ("line\nbreak", "line\\nbreak"),
        ("column\tbreak", "column\\tbreak"),
        ("carriage\rreturn", "carriage\\rreturn"),
    ],
)
def test_escape_milvus_str_covers_filter_literal_escapes(raw, escaped):
    assert _escape_milvus_str(raw) == escaped


@pytest.mark.asyncio
async def test_get_by_id_escapes_special_characters():
    storage = _make_milvus_storage()
    raw_id = 'doc"\\1\nline\tcolumn\rreturn'
    storage._client.query.return_value = [{"id": raw_id, "content": "test"}]

    result = await storage.get_by_id(raw_id)
    assert result == {"id": raw_id, "content": "test"}

    storage._client.query.assert_called_once()
    _, kwargs = storage._client.query.call_args
    assert kwargs["filter"] == 'id == "doc\\"\\\\1\\nline\\tcolumn\\rreturn"'


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

    await storage.delete_entity_relation('ENTITY "A"\\PATH\nLINE\tCOLUMN\rRETURN')

    storage._client.query.assert_called_once()
    _, kwargs = storage._client.query.call_args
    assert (
        kwargs["filter"]
        == 'src_id == "ENTITY \\"A\\"\\\\PATH\\nLINE\\tCOLUMN\\rRETURN" '
        'or tgt_id == "ENTITY \\"A\\"\\\\PATH\\nLINE\\tCOLUMN\\rRETURN"'
    )


@pytest.mark.asyncio
async def test_upsert_preserves_structured_values_without_filter_escaping():
    storage = _make_milvus_storage(
        namespace="test_relationships",
        meta_fields={"content", "src_id", "tgt_id", "created_at"},
    )
    raw_id = 'rel"\\1\nline\tcolumn\rreturn'
    raw_src = 'ENTITY "A"\\PATH\nLINE'
    raw_tgt = "TARGET\tCOLUMN\rRETURN"

    await storage.upsert(
        {
            raw_id: {
                "content": "relationship content",
                "src_id": raw_src,
                "tgt_id": raw_tgt,
            }
        }
    )

    pending = storage._pending_vector_docs[raw_id].source
    assert pending["id"] == raw_id
    assert pending["src_id"] == raw_src
    assert pending["tgt_id"] == raw_tgt

    storage.embedding_func = AsyncMock(
        return_value=np.zeros((1, storage.embedding_func.embedding_dim))
    )
    await storage.index_done_callback()

    persisted = storage._client.upsert.call_args.kwargs["data"][0]
    assert persisted["id"] == raw_id
    assert persisted["src_id"] == raw_src
    assert persisted["tgt_id"] == raw_tgt


@pytest.mark.asyncio
async def test_delete_preserves_primary_keys_without_filter_escaping():
    storage = _make_milvus_storage()
    raw_id = 'doc"\\1\nline\tcolumn\rreturn'

    await storage.delete([raw_id])
    await storage.index_done_callback()

    storage._client.delete.assert_called_once_with(
        collection_name=storage.final_namespace,
        pks=[raw_id],
    )
