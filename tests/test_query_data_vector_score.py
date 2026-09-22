"""``/query/data`` reports each record's vector-search score as ``vector_score``.

The vector stores return a ``distance`` with every hit; retrieval used to drop
it, so a client that wanted to threshold or rerank had to re-embed the results
it had just been given. The score must reach the user format for chunks,
entities and relationships, stay ``None`` for records reached only through the
graph, and survive the dedup steps that keep a graph-derived copy first.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lightrag.base import QueryParam
from lightrag.operate import (
    _carry_vector_scores,
    _get_edge_data,
    _get_node_data,
    _get_vector_context,
    _merge_all_chunks,
)
from lightrag.utils import convert_to_user_format

pytestmark = pytest.mark.offline


class _VDB:
    cosine_better_than_threshold = 0.2

    def __init__(self, results):
        self._results = results

    async def query(self, *args, **kwargs):
        return self._results


def _graph(nodes: dict, edges: dict) -> SimpleNamespace:
    return SimpleNamespace(
        get_nodes_batch=AsyncMock(return_value=nodes),
        node_degrees_batch=AsyncMock(return_value={name: 1 for name in nodes}),
        get_nodes_edges_batch=AsyncMock(
            return_value={name: list(edges) for name in nodes}
        ),
        get_edges_batch=AsyncMock(return_value=edges),
        edge_degrees_batch=AsyncMock(return_value={pair: 1 for pair in edges}),
    )


async def test_vector_context_keeps_backend_distance():
    vdb = _VDB(
        [
            {"content": "a", "id": "chunk-1", "file_path": "a.txt", "distance": 0.83},
            {"content": "b", "id": "chunk-2", "file_path": "b.txt"},
        ]
    )
    chunks = await _get_vector_context("q", vdb, QueryParam(), query_embedding=[0.1])
    assert [c["vector_score"] for c in chunks] == [0.83, None]


async def test_node_data_scores_vector_entities_only():
    vdb = _VDB([{"entity_name": "Alpha", "distance": 0.71}])
    graph = _graph(
        nodes={"Alpha": {"entity_type": "CONCEPT", "description": "d"}},
        edges={("Alpha", "Beta"): {"weight": 1.0, "description": "rel"}},
    )
    entities, relations = await _get_node_data(
        "kw", graph, vdb, QueryParam(), query_embedding=[0.1]
    )
    assert entities[0]["vector_score"] == 0.71
    # Relations here come from graph expansion, not from a vector search.
    assert all(r.get("vector_score") is None for r in relations)


async def test_edge_data_scores_vector_relations():
    vdb = _VDB([{"src_id": "Alpha", "tgt_id": "Beta", "distance": 0.64}])
    graph = _graph(
        nodes={"Alpha": {"description": "a"}, "Beta": {"description": "b"}},
        edges={("Alpha", "Beta"): {"weight": 1.0, "description": "rel"}},
    )
    relations, entities = await _get_edge_data(
        "kw", graph, vdb, QueryParam(), query_embedding=[0.1]
    )
    assert relations[0]["vector_score"] == 0.64
    assert all(e.get("vector_score") is None for e in entities)


def test_carry_vector_scores_fills_graph_copy_without_mutating_it():
    graph_copy = {"entity_name": "Alpha"}
    merged = _carry_vector_scores(
        [graph_copy, {"entity_name": "Gamma"}],
        [{"entity_name": "Alpha", "vector_score": 0.5}],
        key=lambda e: e["entity_name"],
    )
    assert [e.get("vector_score") for e in merged] == [0.5, None]
    assert "vector_score" not in graph_copy


async def test_merge_all_chunks_keeps_score_when_entity_copy_wins(monkeypatch):
    vector_chunks = [
        {
            "content": "v",
            "file_path": "v.txt",
            "chunk_id": "chunk-v",
            "vector_score": 0.9,
        },
        {
            "content": "s",
            "file_path": "s.txt",
            "chunk_id": "chunk-s",
            "vector_score": 0.4,
        },
    ]
    # The entity path returns chunk-s first, so its copy wins the round-robin dedup.
    monkeypatch.setattr(
        "lightrag.operate._find_related_text_unit_from_entities",
        AsyncMock(
            return_value=[{"content": "s", "file_path": "s.txt", "chunk_id": "chunk-s"}]
        ),
    )
    merged = await _merge_all_chunks(
        filtered_entities=[{"entity_name": "Alpha"}],
        filtered_relations=[],
        vector_chunks=vector_chunks,
        text_chunks_db=SimpleNamespace(global_config={}),
        query_param=QueryParam(),
    )
    assert {c["chunk_id"]: c["vector_score"] for c in merged} == {
        "chunk-v": 0.9,
        "chunk-s": 0.4,
    }


def test_convert_to_user_format_exposes_vector_score():
    entity = {"entity_name": "Alpha", "vector_score": 0.7}
    relation = {"src_id": "Alpha", "tgt_id": "Beta", "vector_score": 0.6}
    data = convert_to_user_format(
        entities_context=[{"entity": "Alpha"}, {"entity": "Gamma"}],
        relations_context=[{"entity1": "Alpha", "entity2": "Beta"}],
        chunks=[
            {"content": "c", "chunk_id": "chunk-1", "vector_score": 0.8},
            {"content": "d", "chunk_id": "chunk-2"},
        ],
        references=[],
        query_mode="mix",
        entity_id_to_original={"Alpha": entity},
        relation_id_to_original={("Alpha", "Beta"): relation},
    )["data"]
    assert [e["vector_score"] for e in data["entities"]] == [0.7, None]
    assert data["relationships"][0]["vector_score"] == 0.6
    assert [c["vector_score"] for c in data["chunks"]] == [0.8, None]
