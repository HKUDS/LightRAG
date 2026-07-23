from unittest.mock import AsyncMock

import numpy as np
import pytest

from lightrag import LightRAG, QueryParam
from lightrag.kg import STORAGE_IMPLEMENTATIONS, STORAGES
from lightrag.kg.factory import get_storage_class
from lightrag.kg.noop_vector_db_impl import NoopVectorDBStorage
from lightrag.tools.rebuild_vdb import (
    rebuild_chunks_vdb,
    rebuild_entities_vdb,
    rebuild_relationships_vdb,
)
from lightrag.utils import EmbeddingFunc, compute_mdhash_id


class FailingEmbedding:
    def __init__(self) -> None:
        self.call_count = 0

    async def __call__(self, texts: list[str], **_: object) -> np.ndarray:
        self.call_count += 1
        raise AssertionError("NoopVectorDBStorage must not call embedding_func")


class DeterministicEmbedding:
    async def __call__(self, texts: list[str], **_: object) -> np.ndarray:
        vector = np.arange(1, 9, dtype=np.float32)
        return np.tile(vector, (len(texts), 1))


def make_custom_kg_data() -> dict[str, list[dict[str, object]]]:
    return {
        "chunks": [
            {
                "content": "Alice knows Bob.",
                "source_id": "source-1",
                "file_path": "example.txt",
            }
        ],
        "entities": [
            {
                "entity_name": "Alice",
                "entity_type": "PERSON",
                "description": "A person",
                "source_id": "source-1",
                "file_path": "example.txt",
            },
            {
                "entity_name": "Bob",
                "entity_type": "PERSON",
                "description": "Another person",
                "source_id": "source-1",
                "file_path": "example.txt",
            },
        ],
        "relationships": [
            {
                "src_id": "Alice",
                "tgt_id": "Bob",
                "description": "Alice knows Bob",
                "keywords": "knows",
                "weight": 1.0,
                "source_id": "source-1",
                "file_path": "example.txt",
            }
        ],
    }


def make_storage(embedding: FailingEmbedding) -> NoopVectorDBStorage:
    return NoopVectorDBStorage(
        namespace="test_vectors",
        workspace="test_workspace",
        global_config={},
        embedding_func=EmbeddingFunc(
            embedding_dim=8,
            max_token_size=512,
            func=embedding,
        ),
        meta_fields={"content"},
    )


def test_noop_vector_storage_is_registered() -> None:
    assert (
        "NoopVectorDBStorage"
        in STORAGE_IMPLEMENTATIONS["VECTOR_STORAGE"]["implementations"]
    )
    assert STORAGES["NoopVectorDBStorage"] == ".kg.noop_vector_db_impl"
    assert get_storage_class("NoopVectorDBStorage") is NoopVectorDBStorage


@pytest.mark.offline
@pytest.mark.asyncio
async def test_noop_vector_storage_contract_never_embeds() -> None:
    embedding = FailingEmbedding()
    storage = make_storage(embedding)

    await storage.initialize()
    await storage.upsert({"id-1": {"content": "alpha"}})
    await storage.delete(["id-1"])
    await storage.delete_entity("Entity")
    await storage.delete_entity_relation("Entity")
    await storage.index_done_callback()
    await storage.drop_pending_index_ops()

    assert await storage.get_by_id("id-1") is None
    assert await storage.get_by_ids(["id-1", "id-2"]) == [None, None]
    assert await storage.get_vectors_by_ids(["id-1"]) == {}
    assert await storage.drop() == {
        "status": "success",
        "message": "Noop vector storage contains no data",
    }
    assert embedding.call_count == 0

    with pytest.raises(RuntimeError, match="lightrag-rebuild-vdb"):
        await storage.query("question", top_k=5)

    await storage.finalize()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_graph_only_custom_kg_insertion_preserves_graph(tmp_path) -> None:
    embedding = FailingEmbedding()
    rag = LightRAG(
        working_dir=str(tmp_path),
        vector_storage="NoopVectorDBStorage",
        llm_model_func=AsyncMock(return_value=""),
        embedding_func=EmbeddingFunc(
            embedding_dim=8,
            max_token_size=512,
            func=embedding,
        ),
    )
    await rag.initialize_storages()

    await rag.ainsert_custom_kg(make_custom_kg_data())

    assert await rag.chunk_entity_relation_graph.has_node("Alice")
    assert await rag.chunk_entity_relation_graph.has_node("Bob")
    assert await rag.chunk_entity_relation_graph.has_edge("Alice", "Bob")
    assert embedding.call_count == 0

    await rag.finalize_storages()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_graph_only_lightrag_initializes_without_embedding_func(tmp_path) -> None:
    rag = LightRAG(
        working_dir=str(tmp_path),
        vector_storage="NoopVectorDBStorage",
        llm_model_func=AsyncMock(return_value=""),
        embedding_func=None,
    )

    await rag.initialize_storages()

    assert isinstance(rag.entities_vdb, NoopVectorDBStorage)
    assert isinstance(rag.relationships_vdb, NoopVectorDBStorage)
    assert isinstance(rag.chunks_vdb, NoopVectorDBStorage)

    await rag.finalize_storages()


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize("query_method", ["aquery", "aquery_data", "aquery_llm"])
async def test_graph_only_public_query_apis_fail_loudly(
    tmp_path, query_method: str
) -> None:
    rag = LightRAG(
        working_dir=str(tmp_path),
        vector_storage="NoopVectorDBStorage",
        llm_model_func=AsyncMock(return_value=""),
        embedding_func=None,
    )
    await rag.initialize_storages()

    with pytest.raises(
        RuntimeError,
        match="NoopVectorDBStorage.*persistent vector storage.*lightrag-rebuild-vdb",
    ):
        await getattr(rag, query_method)(
            "question",
            QueryParam(mode="naive"),
        )

    await rag.finalize_storages()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_graph_only_data_rebuilds_into_nano_vector_storage(tmp_path) -> None:
    graph_only_rag = LightRAG(
        working_dir=str(tmp_path),
        vector_storage="NoopVectorDBStorage",
        llm_model_func=AsyncMock(return_value=""),
        embedding_func=None,
    )
    await graph_only_rag.initialize_storages()
    await graph_only_rag.ainsert_custom_kg(make_custom_kg_data())
    await graph_only_rag.finalize_storages()

    embedding = DeterministicEmbedding()
    indexed_rag = LightRAG(
        working_dir=str(tmp_path),
        vector_storage="NanoVectorDBStorage",
        llm_model_func=AsyncMock(return_value=""),
        embedding_func=EmbeddingFunc(
            embedding_dim=8,
            max_token_size=512,
            func=embedding,
        ),
    )
    await indexed_rag.initialize_storages()

    entity_stats = await rebuild_entities_vdb(
        indexed_rag.chunk_entity_relation_graph,
        indexed_rag.entities_vdb,
        indexed_rag._build_global_config(),
    )
    relationship_stats = await rebuild_relationships_vdb(
        indexed_rag.chunk_entity_relation_graph,
        indexed_rag.relationships_vdb,
        indexed_rag._build_global_config(),
    )
    chunk_stats = await rebuild_chunks_vdb(
        indexed_rag.text_chunks,
        indexed_rag.chunks_vdb,
    )

    entity_ids = [
        compute_mdhash_id("Alice", prefix="ent-"),
        compute_mdhash_id("Bob", prefix="ent-"),
    ]
    relationship_id = compute_mdhash_id("AliceBob", prefix="rel-")
    chunk_id = compute_mdhash_id("Alice knows Bob.", prefix="chunk-")

    assert entity_stats["rebuilt"] == 2
    assert relationship_stats["rebuilt"] == 1
    assert chunk_stats["rebuilt"] == 1
    assert all(
        record is not None
        for record in await indexed_rag.entities_vdb.get_by_ids(entity_ids)
    )
    assert await indexed_rag.relationships_vdb.get_by_id(relationship_id) is not None
    assert await indexed_rag.chunks_vdb.get_by_id(chunk_id) is not None

    await indexed_rag.finalize_storages()
