import importlib
import sys
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lightrag.base import DeletionResult

pytestmark = pytest.mark.offline


class _DummyRAG:
    def __init__(
        self,
        entity_delete_status: str = "success",
        relation_delete_status: str = "success",
        relation_delete_status_code: int = 200,
        relation_delete_message: str | None = None,
        stale_entity_edit: bool = False,
        stale_relation_edit: bool = False,
        stale_merge: bool = False,
    ):
        self.entity_delete_status = entity_delete_status
        self.relation_delete_status = relation_delete_status
        self.relation_delete_status_code = relation_delete_status_code
        self.relation_delete_message = relation_delete_message
        self.stale_entity_edit = stale_entity_edit
        self.stale_relation_edit = stale_relation_edit
        self.stale_merge = stale_merge
        self.last_graph_call: dict[str, Any] | None = None
        self.last_deleted_entity: str | None = None
        self.last_deleted_relation: tuple[str, str] | None = None
        self.last_deleted_relation_request: dict[str, Any] | None = None
        self.last_entity_edit_request: dict[str, Any] | None = None
        self.last_relation_edit_request: dict[str, Any] | None = None
        self.last_merge_request: dict[str, Any] | None = None
        self.last_merge_suggestions_request: Any = None

    async def get_knowledge_graph(
        self, node_label: str, max_depth: int, max_nodes: int
    ) -> dict[str, Any]:
        self.last_graph_call = {
            "node_label": node_label,
            "max_depth": max_depth,
            "max_nodes": max_nodes,
        }
        return {
            "nodes": [
                {
                    "id": node_label,
                    "labels": ["ORGANIZATION"],
                    "properties": {
                        "entity_type": "ORGANIZATION",
                        "description": "Tesla company",
                    },
                    "graph_data": {
                        "description": "Tesla company",
                        "entity_type": "ORGANIZATION",
                        "aliases": ["Tesla Motors"],
                    },
                },
                {
                    "id": "Elon Musk",
                    "labels": ["PERSON"],
                    "properties": {"entity_type": "PERSON", "description": "Founder"},
                    "graph_data": {
                        "description": "Founder",
                        "entity_type": "PERSON",
                    },
                },
            ],
            "edges": [
                {
                    "id": "rel-1",
                    "source": "Elon Musk",
                    "target": node_label,
                    "type": "owns",
                    "properties": {
                        "relation_type": "owns",
                        "keywords": "ownership",
                        "weight": 1.0,
                        "source_id": "chunk-1",
                        "file_path": "docs/a.txt",
                    },
                    "graph_data": {
                        "description": "Elon Musk owns shares in Tesla",
                        "keywords": "ownership",
                        "weight": 1.0,
                        "source_id": "chunk-1",
                        "file_path": "docs/a.txt",
                    },
                }
            ],
            "is_truncated": False,
        }

    async def adelete_by_entity(self, entity_name: str) -> DeletionResult:
        self.last_deleted_entity = entity_name
        return DeletionResult(
            status=self.entity_delete_status,  # type: ignore[arg-type]
            doc_id="legacy-doc-id",
            message=f"Deleted entity {entity_name}",
            status_code=200,
            file_path=None,
        )

    async def adelete_by_relation(
        self,
        source_entity: str,
        target_entity: str,
        expected_revision_token: str | None = None,
    ) -> DeletionResult:
        self.last_deleted_relation = (source_entity, target_entity)
        self.last_deleted_relation_request = {
            "source_entity": source_entity,
            "target_entity": target_entity,
            "expected_revision_token": expected_revision_token,
        }
        return DeletionResult(
            status=self.relation_delete_status,  # type: ignore[arg-type]
            doc_id="legacy-doc-id",
            message=(
                self.relation_delete_message
                if self.relation_delete_message is not None
                else f"Deleted relation {source_entity}->{target_entity}"
            ),
            status_code=self.relation_delete_status_code,
            file_path=None,
        )

    async def aedit_entity(
        self,
        entity_name: str,
        updated_data: dict[str, Any],
        allow_rename: bool,
        allow_merge: bool,
        expected_revision_token: str | None = None,
    ) -> dict[str, Any]:
        self.last_entity_edit_request = {
            "entity_name": entity_name,
            "updated_data": dict(updated_data),
            "allow_rename": allow_rename,
            "allow_merge": allow_merge,
            "expected_revision_token": expected_revision_token,
        }
        if self.stale_entity_edit:
            raise ValueError("Stale entity revision token")
        return {
            "entity_name": updated_data.get("entity_name", entity_name),
            "description": updated_data.get("description", "updated"),
        }

    async def aedit_relation(
        self,
        source_entity: str,
        target_entity: str,
        updated_data: dict[str, Any],
        expected_revision_token: str | None = None,
    ) -> dict[str, Any]:
        self.last_relation_edit_request = {
            "source_entity": source_entity,
            "target_entity": target_entity,
            "updated_data": dict(updated_data),
            "expected_revision_token": expected_revision_token,
        }
        if self.stale_relation_edit:
            raise ValueError("Stale relation revision token")
        return {
            "src_entity": source_entity,
            "tgt_entity": target_entity,
            "graph_data": dict(updated_data),
        }

    async def amerge_entities(
        self,
        source_entities: list[str],
        target_entity: str,
        merge_strategy: dict[str, Any] | None = None,
        target_entity_data: dict[str, Any] | None = None,
        expected_revision_tokens: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        self.last_merge_request = {
            "source_entities": list(source_entities),
            "target_entity": target_entity,
            "merge_strategy": dict(merge_strategy or {}),
            "target_entity_data": dict(target_entity_data or {}),
            "expected_revision_tokens": dict(expected_revision_tokens or {}),
        }
        if self.stale_merge:
            raise ValueError("Stale merge revision token")
        return {
            "entity_name": target_entity,
            "merged_from": list(source_entities),
        }

    async def aget_merge_suggestions(self, request: Any) -> list[dict[str, Any]]:
        self.last_merge_suggestions_request = request
        return [
            {
                "target_entity": "Tesla",
                "source_entities": ["Tesla Inc.", "Tesla Motors"],
                "score": 0.87,
                "reasons": [
                    {"code": "name_similarity", "score": 0.94},
                    {"code": "shared_neighbors", "score": 0.73},
                ],
            }
        ]


class _DummyRAGNoMergeSupport:
    async def get_knowledge_graph(
        self, node_label: str, max_depth: int, max_nodes: int
    ) -> dict[str, Any]:
        return {"nodes": [], "edges": [], "is_truncated": False}


def _build_graph_client(monkeypatch, rag):
    monkeypatch.setattr(sys, "argv", [sys.argv[0]])

    graph_routes = importlib.import_module("lightrag.api.routers.graph_routes")
    graph_routes = importlib.reload(graph_routes)

    app = FastAPI()
    app.include_router(graph_routes.create_graph_routes(rag, api_key=None))
    return TestClient(app)


@pytest.fixture
def graph_client(monkeypatch):
    rag = _DummyRAG()
    with _build_graph_client(monkeypatch, rag) as client:
        yield client, rag


def test_get_graphs_route_remains_backward_compatible(graph_client):
    client, rag = graph_client

    response = client.get(
        "/graphs",
        params={"label": "Tesla", "max_depth": 2, "max_nodes": 128},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["nodes"][0]["id"] == "Tesla"
    assert body["is_truncated"] is False
    assert rag.last_graph_call == {
        "node_label": "Tesla",
        "max_depth": 2,
        "max_nodes": 128,
    }


def test_get_graphs_route_rejects_blank_label(graph_client):
    client, rag = graph_client

    response = client.get(
        "/graphs",
        params={"label": "   ", "max_depth": 2, "max_nodes": 128},
    )

    assert response.status_code == 422
    assert "label cannot be empty" in response.json()["detail"]
    assert rag.last_graph_call is None


def test_graph_query_accepts_v1_filter_shape_and_returns_meta_truncation(graph_client):
    client, rag = graph_client

    response = client.post(
        "/graph/query",
        json={
            "scope": {
                "label": "Tesla",
                "max_depth": 2,
                "max_nodes": 128,
                "only_matched_neighborhood": True,
            },
            "node_filters": {
                "entity_types": ["PERSON", "ORGANIZATION"],
                "name_query": "tesla",
                "degree_min": 1,
                "degree_max": 50,
                "isolated_only": False,
            },
            "edge_filters": {
                "relation_types": ["owns", "acquires"],
                "keyword_query": "ownership",
                "weight_min": 0.2,
                "weight_max": 5.0,
            },
            "source_filters": {
                "source_id_query": "chunk-1",
                "file_paths": ["docs/a.txt"],
            },
            "view_options": {
                "show_nodes_only": False,
                "show_edges_only": False,
                "hide_low_weight_edges": True,
                "hide_empty_description": True,
                "highlight_matches": True,
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert "meta" in body
    assert "truncation" in body
    assert body["meta"]["filter_semantics"] == {
        "group_operator": "AND",
        "field_operator": "AND",
        "array_operator": "OR",
        "version": "v1",
    }
    assert body["truncation"]["requested_max_nodes"] == 128
    assert "was_truncated_before_filtering" in body["truncation"]
    assert "was_truncated_after_filtering" in body["truncation"]
    assert body["meta"]["execution_mode"] == "base_graph_only_placeholder"
    assert body["meta"]["filtering_applied"] is True
    assert body["meta"]["ignored_filter_groups"] == ["view_options.highlight_matches"]
    assert "nodes" in body["data"]
    assert "edges" in body["data"]
    assert body["data"]["nodes"][0]["revision_token"]
    assert body["data"]["edges"][0]["revision_token"]
    assert rag.last_graph_call == {
        "node_label": "Tesla",
        "max_depth": 2,
        "max_nodes": 128,
    }


def test_delete_entity_route_exists_and_returns_expected_structure(graph_client):
    client, rag = graph_client

    response = client.request(
        "DELETE", "/graph/entity", json={"entity_name": "Tesla Motors"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["doc_id"] == ""
    assert body["message"] == "Deleted entity Tesla Motors"
    assert body["status_code"] == 200
    assert rag.last_deleted_entity == "Tesla Motors"


def test_delete_relation_route_exists_and_returns_expected_structure(graph_client):
    client, rag = graph_client

    response = client.request(
        "DELETE",
        "/graph/relation",
        json={"source_entity": "Elon Musk", "target_entity": "Tesla"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["doc_id"] == ""
    assert body["message"] == "Deleted relation Elon Musk->Tesla"
    assert body["status_code"] == 200
    assert rag.last_deleted_relation == ("Elon Musk", "Tesla")


def test_delete_entity_not_allowed_maps_to_http_403(monkeypatch):
    rag = _DummyRAG(entity_delete_status="not_allowed")
    with _build_graph_client(monkeypatch, rag) as client:
        response = client.request(
            "DELETE", "/graph/entity", json={"entity_name": "Tesla Motors"}
        )

    assert response.status_code == 403
    assert "Deleted entity Tesla Motors" in response.json()["detail"]


def test_delete_relation_not_allowed_maps_to_http_403(monkeypatch):
    rag = _DummyRAG(relation_delete_status="not_allowed")
    with _build_graph_client(monkeypatch, rag) as client:
        response = client.request(
            "DELETE",
            "/graph/relation",
            json={"source_entity": "Elon Musk", "target_entity": "Tesla"},
        )

    assert response.status_code == 403
    assert "Deleted relation Elon Musk->Tesla" in response.json()["detail"]


def test_entity_edit_with_stale_token_maps_to_http_409(monkeypatch):
    rag = _DummyRAG(stale_entity_edit=True)
    with _build_graph_client(monkeypatch, rag) as client:
        response = client.post(
            "/graph/entity/edit",
            json={
                "entity_name": "Tesla",
                "updated_data": {"description": "Updated description"},
                "expected_revision_token": "stale-entity-token",
            },
        )

    assert response.status_code == 409
    assert "revision token" in response.json()["detail"].lower()
    assert rag.last_entity_edit_request is not None
    assert (
        rag.last_entity_edit_request["expected_revision_token"]
        == "stale-entity-token"
    )


def test_relation_edit_with_stale_token_maps_to_http_409(monkeypatch):
    rag = _DummyRAG(stale_relation_edit=True)
    with _build_graph_client(monkeypatch, rag) as client:
        response = client.post(
            "/graph/relation/edit",
            json={
                "source_id": "Elon Musk",
                "target_id": "Tesla",
                "updated_data": {"description": "Updated relation"},
                "expected_revision_token": "stale-relation-token",
            },
        )

    assert response.status_code == 409
    assert "revision token" in response.json()["detail"].lower()
    assert rag.last_relation_edit_request is not None
    assert (
        rag.last_relation_edit_request["expected_revision_token"]
        == "stale-relation-token"
    )


def test_relation_delete_with_stale_token_maps_to_http_409(monkeypatch):
    rag = _DummyRAG(
        relation_delete_status="not_allowed",
        relation_delete_status_code=409,
        relation_delete_message="Stale relation revision token",
    )
    with _build_graph_client(monkeypatch, rag) as client:
        response = client.request(
            "DELETE",
            "/graph/relation",
            json={
                "source_entity": "Elon Musk",
                "target_entity": "Tesla",
                "expected_revision_token": "stale-delete-token",
            },
        )

    assert response.status_code == 409
    assert "revision token" in response.json()["detail"].lower()
    assert rag.last_deleted_relation_request is not None
    assert (
        rag.last_deleted_relation_request["expected_revision_token"]
        == "stale-delete-token"
    )


def test_merge_expected_revision_tokens_are_forwarded_to_rag(graph_client):
    client, rag = graph_client
    payload = {
        "entities_to_change": ["Tesla Motors", "Tesla Inc."],
        "entity_to_change_into": "Tesla",
        "expected_revision_tokens": {
            "Tesla Motors": "token-source",
            "Tesla": "token-target",
        },
    }

    response = client.post("/graph/entities/merge", json=payload)

    assert response.status_code == 200
    assert rag.last_merge_request is not None
    assert rag.last_merge_request["expected_revision_tokens"] == {
        "Tesla Motors": "token-source",
        "Tesla": "token-target",
    }


def test_merge_with_stale_tokens_maps_to_http_409(monkeypatch):
    rag = _DummyRAG(stale_merge=True)
    with _build_graph_client(monkeypatch, rag) as client:
        response = client.post(
            "/graph/entities/merge",
            json={
                "entities_to_change": ["Tesla Motors"],
                "entity_to_change_into": "Tesla",
                "expected_revision_tokens": {"Tesla Motors": "stale-token"},
            },
        )

    assert response.status_code == 409
    assert "revision token" in response.json()["detail"].lower()


def test_graph_query_rejects_unknown_extra_fields(graph_client):
    client, _ = graph_client

    response = client.post(
        "/graph/query",
        json={
            "scope": {"label": "Tesla"},
            "node_filters": {"entity_types": ["PERSON"], "unexpected_field": True},
        },
    )

    assert response.status_code == 422


def test_graph_query_highlight_only_is_explicitly_ignored_in_meta(graph_client):
    client, _ = graph_client

    response = client.post(
        "/graph/query",
        json={
            "scope": {"label": "Tesla", "max_depth": 2, "max_nodes": 128},
            "view_options": {"highlight_matches": True},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["meta"]["filtering_applied"] is False
    assert "view_options.highlight_matches" in body["meta"]["ignored_filter_groups"]


def test_merge_suggestions_route_exists_and_returns_candidate_structure(graph_client):
    client, rag = graph_client

    response = client.post(
        "/graph/merge/suggestions",
        json={
            "scope": {"label": "Tesla", "max_depth": 1, "max_nodes": 64},
            "limit": 5,
            "min_score": 0.3,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert "candidates" in body
    assert "meta" in body
    assert body["candidates"]
    candidate = body["candidates"][0]
    assert candidate["target_entity"] == "Tesla"
    assert candidate["source_entities"] == ["Tesla Inc.", "Tesla Motors"]
    assert candidate["score"] == pytest.approx(0.87)
    assert candidate["reasons"][0]["code"] == "name_similarity"
    assert rag.last_merge_suggestions_request is not None
    assert isinstance(rag.last_merge_suggestions_request, dict)
    assert rag.last_merge_suggestions_request["scope"]["label"] == "Tesla"


def test_merge_suggestions_returns_501_when_rag_not_supported(monkeypatch):
    rag = _DummyRAGNoMergeSupport()
    with _build_graph_client(monkeypatch, rag) as client:
        response = client.post(
            "/graph/merge/suggestions",
            json={"scope": {"label": "Tesla", "max_depth": 1, "max_nodes": 64}},
        )

    assert response.status_code == 501
    assert "not implemented" in response.json()["detail"].lower()


def test_merge_suggestions_rejects_unknown_extra_fields(graph_client):
    client, _ = graph_client

    response = client.post(
        "/graph/merge/suggestions",
        json={
            "scope": {"label": "Tesla"},
            "limit": 5,
            "unexpected": "forbidden",
        },
    )

    assert response.status_code == 422
