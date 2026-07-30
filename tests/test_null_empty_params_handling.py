import asyncio
import pytest
from lightrag.rerank import aggregate_chunk_scores
import networkx as nx
from lightrag.operate import _format_relation_edge_label
from lightrag.utils import (
    pick_by_weighted_polling,
    pick_by_vector_similarity,
    generate_reference_list_from_chunks,
)
from lightrag.kg.networkx_impl import NetworkXStorage


def test_aggregate_chunk_scores_with_none_and_non_dict():
    # Contains None and non-dict items alongside valid chunk result
    chunk_results = [None, "invalid", {"index": 0, "relevance_score": 0.85}]
    doc_indices = [0]
    num_original_docs = 1
    result = aggregate_chunk_scores(chunk_results, doc_indices, num_original_docs)
    assert result == [{"index": 0, "relevance_score": 0.85}]

    assert aggregate_chunk_scores(None, [0], 1) == []
    assert aggregate_chunk_scores([], None, 1) == []


def test_format_relation_edge_label_empty_and_none():
    assert _format_relation_edge_label([]) == "N/A->N/A"
    assert _format_relation_edge_label(["A"]) == "A->N/A"
    assert _format_relation_edge_label(()) == "N/A->N/A"
    assert _format_relation_edge_label(None) == "N/A->N/A"


def test_pick_by_weighted_polling_null_elements_and_none_chunks():
    entities_or_relations = [
        None,
        {"sorted_chunks": None},
        {"sorted_chunks": ["c1", "c2"]},
    ]
    result = pick_by_weighted_polling(entities_or_relations, max_related_chunks=2)
    assert result == ["c1", "c2"]


@pytest.mark.asyncio
async def test_pick_by_vector_similarity_null_and_none_chunks():
    entity_info = [None, {"sorted_chunks": None}]
    result = await pick_by_vector_similarity(
        query="test",
        text_chunks_storage=None,
        chunks_vdb=None,
        num_of_chunks=2,
        entity_info=entity_info,
        embedding_func=None,
    )
    assert result == []


def test_generate_reference_list_from_chunks_with_none():
    chunks = [None, {"file_path": "doc1.txt"}]
    ref_list, updated_chunks = generate_reference_list_from_chunks(chunks)
    assert len(ref_list) == 1
    assert ref_list[0]["file_path"] == "doc1.txt"
    assert len(updated_chunks) == 1
    assert updated_chunks[0]["reference_id"] == "1"


@pytest.mark.asyncio
async def test_networkx_storage_null_edges(tmp_path):
    storage = NetworkXStorage(
        namespace="test",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=None,
        workspace="",
    )
    storage._storage_lock = asyncio.Lock()
    storage._graph = nx.Graph()
    storage.storage_updated = type("Value", (), {"value": False})()

    await storage.upsert_edges_batch([])
    assert storage._graph.number_of_edges() == 0
    # Malformed / None edges in batch operations
    await storage.upsert_edges_batch([None, ("A",), ("A", "B", {"weight": 1.0})])
    assert await storage.has_edge("A", "B") is True

    await storage.remove_edges([None, ("A",), ("A", "B")])
    assert await storage.has_edge("A", "B") is False
