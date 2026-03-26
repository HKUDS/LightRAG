from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from lightrag.api.graph_workbench import (
    LOW_WEIGHT_EDGE_THRESHOLD,
    query_graph_workbench,
)

pytestmark = pytest.mark.offline


class _DummyRAG:
    def __init__(
        self,
        graph_payload: dict[str, Any],
        runtime_max_graph_nodes: int | None = None,
        backend_max_graph_nodes: int | None = None,
    ) -> None:
        self.graph_payload = graph_payload
        self.max_graph_nodes = runtime_max_graph_nodes
        self.chunk_entity_relation_graph = SimpleNamespace(
            global_config=(
                {"max_graph_nodes": backend_max_graph_nodes}
                if backend_max_graph_nodes is not None
                else {}
            )
        )
        self.last_graph_call: dict[str, Any] | None = None

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int, max_nodes: int
    ) -> dict[str, Any]:
        self.last_graph_call = {
            "node_label": node_label,
            "max_depth": max_depth,
            "max_nodes": max_nodes,
        }
        return self.graph_payload


def _node(node_id: str, entity_type: str, description: str = "") -> dict[str, Any]:
    return {
        "id": node_id,
        "labels": [entity_type],
        "properties": {
            "entity_type": entity_type,
            "description": description,
        },
    }


def _edge(
    edge_id: str,
    source: str,
    target: str,
    relation_type: str,
    keywords: str = "",
    weight: float = 1.0,
    source_id: str = "",
    file_path: str = "",
) -> dict[str, Any]:
    return {
        "id": edge_id,
        "type": relation_type,
        "source": source,
        "target": target,
        "properties": {
            "relation_type": relation_type,
            "keywords": keywords,
            "weight": weight,
            "source_id": source_id,
            "file_path": file_path,
        },
    }


@pytest.mark.asyncio
async def test_query_bounded_base_graph_filtering_and_node_filtering():
    rag = _DummyRAG(
        graph_payload={
            "nodes": [
                _node("n1", "PERSON"),
                _node("n2", "ORGANIZATION"),
                _node("n3", "PERSON"),
            ],
            "edges": [],
            "is_truncated": False,
        },
        runtime_max_graph_nodes=2,
    )

    result = await query_graph_workbench(
        rag,
        {
            "scope": {"label": "*", "max_depth": 2, "max_nodes": 10},
            "node_filters": {"entity_types": ["PERSON"]},
        },
    )

    assert rag.last_graph_call == {"node_label": "*", "max_depth": 2, "max_nodes": 2}
    assert result["truncation"]["effective_max_nodes"] == 2
    assert [node["id"] for node in result["data"]["nodes"]] == ["n1", "n3"]


@pytest.mark.asyncio
async def test_query_v1_and_or_semantics_for_group_and_field_and_array_or():
    rag = _DummyRAG(
        graph_payload={
            "nodes": [
                _node("n1", "PERSON", "founder"),
                _node("n2", "ORGANIZATION", "founder team"),
                _node("n3", "LOCATION", "hq"),
            ],
            "edges": [
                _edge(
                    "e1",
                    "n1",
                    "n2",
                    "owns",
                    keywords="equity stake",
                    weight=0.7,
                    source_id="doc-1",
                    file_path="/a.md",
                ),
                _edge(
                    "e2",
                    "n2",
                    "n3",
                    "located_in",
                    keywords="hq",
                    weight=0.9,
                    source_id="doc-2",
                    file_path="/b.md",
                ),
            ],
            "is_truncated": False,
        }
    )

    result = await query_graph_workbench(
        rag,
        {
            "scope": {"label": "*", "max_depth": 2, "max_nodes": 100},
            "node_filters": {
                "entity_types": ["PERSON", "ORGANIZATION"],
                "description_query": "founder",
            },
            "edge_filters": {
                "relation_types": ["owns", "acquires"],
                "keyword_query": "equity",
                "weight_min": 0.5,
            },
            "source_filters": {
                "source_id_query": "doc-1",
                "file_paths": ["/a.md", "/x.md"],
            },
        },
    )

    assert {node["id"] for node in result["data"]["nodes"]} == {"n1", "n2"}
    assert [edge["id"] for edge in result["data"]["edges"]] == ["e1"]


@pytest.mark.asyncio
async def test_truncation_flags_when_base_graph_already_truncated():
    rag = _DummyRAG(
        graph_payload={
            "nodes": [_node("n1", "PERSON"), _node("n2", "PERSON")],
            "edges": [],
            "is_truncated": True,
        }
    )

    result = await query_graph_workbench(
        rag,
        {"scope": {"label": "*", "max_depth": 1, "max_nodes": 10}},
    )

    assert result["truncation"]["was_truncated_before_filtering"] is True
    assert result["truncation"]["was_truncated_after_filtering"] is True


@pytest.mark.asyncio
async def test_truncation_flags_when_only_after_filtering_truncated():
    rag = _DummyRAG(
        graph_payload={
            "nodes": [_node("n1", "PERSON"), _node("n2", "PERSON"), _node("n3", "PERSON")],
            "edges": [],
            "is_truncated": False,
        },
        runtime_max_graph_nodes=2,
    )

    result = await query_graph_workbench(
        rag,
        {"scope": {"label": "*", "max_depth": 1, "max_nodes": 10}},
    )

    assert result["truncation"]["was_truncated_before_filtering"] is False
    assert result["truncation"]["was_truncated_after_filtering"] is True
    assert len(result["data"]["nodes"]) == 2


@pytest.mark.asyncio
async def test_effective_max_nodes_never_exceeds_runtime_or_backend_limit():
    rag = _DummyRAG(
        graph_payload={"nodes": [_node("n1", "PERSON")], "edges": [], "is_truncated": False},
        runtime_max_graph_nodes=50,
        backend_max_graph_nodes=20,
    )

    result = await query_graph_workbench(
        rag,
        {"scope": {"label": "*", "max_depth": 1, "max_nodes": 100}},
    )

    assert result["truncation"]["requested_max_nodes"] == 100
    assert result["truncation"]["effective_max_nodes"] == 20
    assert rag.last_graph_call == {"node_label": "*", "max_depth": 1, "max_nodes": 20}


@pytest.mark.asyncio
async def test_query_time_filters_support_mixed_timezone_and_naive_boundaries():
    rag = _DummyRAG(
        graph_payload={
            "nodes": [_node("n1", "PERSON"), _node("n2", "ORGANIZATION")],
            "edges": [
                _edge(
                    "e1",
                    "n1",
                    "n2",
                    "works_for",
                    source_id="doc-1",
                    file_path="/a.md",
                )
            ],
            "is_truncated": False,
        }
    )
    rag.graph_payload["edges"][0]["properties"]["time"] = "2026-01-01T00:00:00"

    result = await query_graph_workbench(
        rag,
        {
            "scope": {"label": "*", "max_depth": 1, "max_nodes": 10},
            "source_filters": {
                "time_from": "2026-01-01T00:00:00+00:00",
                "time_to": "2025-12-31T16:00:00-08:00",
            },
        },
    )

    assert [edge["id"] for edge in result["data"]["edges"]] == ["e1"]


@pytest.mark.asyncio
async def test_query_highlight_matches_is_ignored_and_not_counted_as_filtering_applied():
    rag = _DummyRAG(
        graph_payload={
            "nodes": [_node("n1", "PERSON"), _node("n2", "ORGANIZATION")],
            "edges": [_edge("e1", "n1", "n2", "works_for", weight=1.0)],
            "is_truncated": False,
        }
    )

    result = await query_graph_workbench(
        rag,
        {
            "scope": {"label": "*", "max_depth": 1, "max_nodes": 10},
            "view_options": {"highlight_matches": True},
        },
    )

    assert result["meta"]["filtering_applied"] is False
    assert "view_options.highlight_matches" in result["meta"]["ignored_filter_groups"]


@pytest.mark.asyncio
async def test_query_hide_low_weight_edges_uses_explicit_threshold_semantics():
    rag = _DummyRAG(
        graph_payload={
            "nodes": [_node("n1", "PERSON"), _node("n2", "ORGANIZATION")],
            "edges": [
                _edge("e-low", "n1", "n2", "works_for", weight=LOW_WEIGHT_EDGE_THRESHOLD),
                _edge(
                    "e-high",
                    "n1",
                    "n2",
                    "works_for",
                    weight=LOW_WEIGHT_EDGE_THRESHOLD + 0.01,
                ),
            ],
            "is_truncated": False,
        }
    )

    result = await query_graph_workbench(
        rag,
        {
            "scope": {"label": "*", "max_depth": 1, "max_nodes": 10},
            "view_options": {"hide_low_weight_edges": True},
        },
    )

    assert [edge["id"] for edge in result["data"]["edges"]] == ["e-high"]
