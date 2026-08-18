from unittest.mock import AsyncMock

import numpy as np
import pytest

from lightrag import LightRAG, QueryParam
from lightrag.base import DocStatus
from lightrag.kg import STORAGE_IMPLEMENTATIONS, STORAGES
from lightrag.kg.factory import get_storage_class
from lightrag.kg.noop_vector_db_impl import NoopVectorDBStorage
from lightrag.tools.rebuild_vdb import (
    rebuild_chunks_vdb,
    rebuild_entities_vdb,
    rebuild_relationships_vdb,
)
from lightrag.utils import EmbeddingFunc, compute_mdhash_id


EXTRACTION_RESULT = "\n".join(
    [
        "entity<|#|>Alice<|#|>person<|#|>A person",
        "entity<|#|>Bob<|#|>person<|#|>Another person",
        "relation<|#|>Alice<|#|>Bob<|#|>knows<|#|>Alice knows Bob",
        "<|COMPLETE|>",
    ]
)


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
    assert NoopVectorDBStorage.requires_embedding_func is False
    assert NoopVectorDBStorage.persists_vectors is False


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
async def test_graph_only_query_reports_vector_storage_failure(tmp_path) -> None:
    rag = LightRAG(
        working_dir=str(tmp_path),
        vector_storage="NoopVectorDBStorage",
        llm_model_func=AsyncMock(return_value=""),
        embedding_func=None,
    )
    await rag.initialize_storages()

    result = await rag.aquery_llm("question", QueryParam(mode="naive"))

    assert result["status"] == "failure"
    assert "NoopVectorDBStorage" in result["message"]
    assert "persistent vector storage" in result["message"]
    assert "lightrag-rebuild-vdb" in result["message"]
    await rag.finalize_storages()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_graph_only_bypass_query_remains_available(tmp_path) -> None:
    rag = LightRAG(
        working_dir=str(tmp_path),
        vector_storage="NoopVectorDBStorage",
        llm_model_func=AsyncMock(return_value="direct answer"),
        embedding_func=None,
    )
    await rag.initialize_storages()

    result = await rag.aquery_llm(
        "question",
        QueryParam(mode="bypass", stream=False),
    )

    assert result["llm_response"]["content"] == "direct answer"
    await rag.finalize_storages()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_graph_only_normal_ingestion_rebuilds_into_nano_vector_storage(
    tmp_path,
) -> None:
    workspace = f"graph-only-{tmp_path.name}"
    failing_embedding = FailingEmbedding()
    graph_only_rag = LightRAG(
        working_dir=str(tmp_path),
        workspace=workspace,
        vector_storage="NoopVectorDBStorage",
        llm_model_func=AsyncMock(return_value=EXTRACTION_RESULT),
        embedding_func=EmbeddingFunc(
            embedding_dim=8,
            max_token_size=512,
            func=failing_embedding,
        ),
        entity_extract_max_gleaning=0,
        # EXTRACTION_RESULT is delimiter-formatted, so the parser must be too.
        # The field otherwise defaults from ENTITY_EXTRACTION_USE_JSON, which
        # env.example ships as true -- leaving it ambient makes the assertions
        # below depend on the developer's .env.
        entity_extraction_use_json=False,
    )
    await graph_only_rag.initialize_storages()
    await graph_only_rag.ainsert("Alice knows Bob.", file_paths="example.txt")

    processed_docs = await graph_only_rag.doc_status.get_docs_by_status(
        DocStatus.PROCESSED
    )
    assert len(processed_docs) == 1
    doc_id, doc_status = next(iter(processed_docs.items()))
    chunk_ids = doc_status.chunks_list
    assert await graph_only_rag.full_docs.get_by_id(doc_id) is not None
    assert len(chunk_ids) == 1
    assert await graph_only_rag.text_chunks.get_by_id(chunk_ids[0]) is not None
    assert await graph_only_rag.chunk_entity_relation_graph.has_node("Alice")
    assert await graph_only_rag.chunk_entity_relation_graph.has_node("Bob")
    assert await graph_only_rag.chunk_entity_relation_graph.has_edge("Alice", "Bob")
    assert failing_embedding.call_count == 0
    await graph_only_rag.finalize_storages()

    indexed_rag = LightRAG(
        working_dir=str(tmp_path),
        workspace=workspace,
        vector_storage="NanoVectorDBStorage",
        llm_model_func=AsyncMock(return_value=""),
        embedding_func=EmbeddingFunc(
            embedding_dim=8,
            max_token_size=512,
            func=DeterministicEmbedding(),
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
    reopened_docs = await indexed_rag.doc_status.get_docs_by_status(DocStatus.PROCESSED)

    assert doc_id in reopened_docs
    assert reopened_docs[doc_id].chunks_list == chunk_ids
    assert entity_stats["rebuilt"] == 2
    assert relationship_stats["rebuilt"] == 1
    assert chunk_stats["rebuilt"] == 1
    assert all(
        record is not None
        for record in await indexed_rag.entities_vdb.get_by_ids(entity_ids)
    )
    assert await indexed_rag.relationships_vdb.get_by_id(relationship_id) is not None
    assert await indexed_rag.chunks_vdb.get_by_id(chunk_ids[0]) is not None

    await indexed_rag.finalize_storages()
