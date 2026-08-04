"""Unit tests for /graph3d/edge_info and /graph3d/entity_edges endpoints."""

import importlib
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
graph3d_routes = importlib.import_module("lightrag.api.routers.graph3d_routes")
sys.argv = _original_argv

pytestmark = pytest.mark.offline


class _StubGraph:
    """Minimal stand-in for chunk_entity_relation_graph (NetworkX semantics)."""

    def __init__(self):
        # Undirected: stored once as ("A", "B")
        self._edges = {
            ("A", "B"): {"description": "A relates to B", "keywords": "x, y", "weight": 0.8},
        }

    async def get_edge(self, src, tgt):
        return self._edges.get((src, tgt))

    async def get_node_edges(self, name):
        return [(s, t) for (s, t) in self._edges if s == name or t == name]


class _StubRag:
    working_dir = "/tmp"
    chunk_entity_relation_graph = _StubGraph()


def _client():
    app = FastAPI()
    app.include_router(graph3d_routes.create_graph3d_routes(_StubRag()))
    return TestClient(app)


def test_edge_info_returns_relationship_details():
    resp = _client().get("/graph3d/edge_info", params={"src": "A", "tgt": "B"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["description"] == "A relates to B"
    assert body["keywords"] == "x, y"
    assert body["weight"] == 0.8


def test_edge_info_tries_reversed_order_for_undirected_graph():
    resp = _client().get("/graph3d/edge_info", params={"src": "B", "tgt": "A"})
    assert resp.status_code == 200
    assert resp.json()["description"] == "A relates to B"


def test_edge_info_missing_pair_returns_404():
    resp = _client().get("/graph3d/edge_info", params={"src": "A", "tgt": "ZZZ"})
    assert resp.status_code == 404


def test_entity_edges_returns_neighbors_with_details_sorted_by_weight():
    resp = _client().get("/graph3d/entity_edges", params={"name": "A"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["neighbors"][0]["name"] == "B"
    assert body["neighbors"][0]["weight"] == 0.8


def test_entity_edges_unknown_entity_returns_empty_list():
    resp = _client().get("/graph3d/entity_edges", params={"name": "NOPE"})
    assert resp.status_code == 200
    assert resp.json() == {"entity": "NOPE", "neighbors": [], "total": 0}
