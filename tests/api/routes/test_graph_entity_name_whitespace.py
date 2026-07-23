"""Graph create/update/merge reject whitespace-only entity names like delete."""

import importlib
import sys

import pytest
from pydantic import ValidationError

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_gr = importlib.import_module("lightrag.api.routers.graph_routes")
sys.argv = _original_argv

EntityCreateRequest = _gr.EntityCreateRequest
EntityUpdateRequest = _gr.EntityUpdateRequest
EntityMergeRequest = _gr.EntityMergeRequest
RelationCreateRequest = _gr.RelationCreateRequest
RelationUpdateRequest = _gr.RelationUpdateRequest
DeleteEntityRequest = _gr.DeleteEntityRequest

pytestmark = pytest.mark.offline


@pytest.mark.parametrize("name", ["  ", "\t", "\n"])
def test_entity_create_rejects_whitespace_only_name(name):
    with pytest.raises(ValidationError):
        EntityCreateRequest(entity_name=name, entity_data={"description": "x"})


def test_entity_create_strips_surrounding_whitespace():
    req = EntityCreateRequest(entity_name="  Tesla  ", entity_data={"description": "x"})
    assert req.entity_name == "Tesla"


@pytest.mark.parametrize("name", ["  ", "\t"])
def test_entity_update_rejects_whitespace_only_name(name):
    with pytest.raises(ValidationError):
        EntityUpdateRequest(entity_name=name, updated_data={"description": "x"})


@pytest.mark.parametrize("name", ["  ", "\t", "\n"])
def test_entity_update_rejects_whitespace_only_rename_target(name):
    with pytest.raises(ValidationError):
        EntityUpdateRequest(
            entity_name="Tesla",
            updated_data={"entity_name": name},
            allow_rename=True,
        )


def test_entity_update_strips_rename_target_name():
    req = EntityUpdateRequest(
        entity_name="Tesla",
        updated_data={"entity_name": "  Elon Musk  "},
        allow_rename=True,
    )
    assert req.updated_data["entity_name"] == "Elon Musk"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"entities_to_change": ["  "], "entity_to_change_into": "Keep"},
        {"entities_to_change": ["Keep"], "entity_to_change_into": "  "},
        {"entities_to_change": ["A", "  "], "entity_to_change_into": "Keep"},
    ],
)
def test_entity_merge_rejects_whitespace_only_names(kwargs):
    with pytest.raises(ValidationError):
        EntityMergeRequest(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_entity": "  ", "target_entity": "B", "relation_data": {"weight": 1.0}},
        {"source_entity": "A", "target_entity": "  ", "relation_data": {"weight": 1.0}},
    ],
)
def test_relation_create_rejects_whitespace_only_endpoints(kwargs):
    with pytest.raises(ValidationError):
        RelationCreateRequest(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_id": "  ", "target_id": "B", "updated_data": {"weight": 1.0}},
        {"source_id": "A", "target_id": "\t", "updated_data": {"weight": 1.0}},
    ],
)
def test_relation_update_rejects_whitespace_only_endpoints(kwargs):
    with pytest.raises(ValidationError):
        RelationUpdateRequest(**kwargs)


def test_delete_entity_still_rejects_whitespace_only_name():
    with pytest.raises(ValidationError):
        DeleteEntityRequest(entity_name="  ")
