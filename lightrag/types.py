from __future__ import annotations

from functools import reduce
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, create_model


class GPTKeywordExtractionFormat(BaseModel):
    high_level_keywords: list[str]
    low_level_keywords: list[str]


class EntityExtraction(BaseModel):
    """Schema for a single extracted entity from text."""

    entity_name: str
    entity_type: str
    entity_description: str


class RelationshipExtraction(BaseModel):
    """Schema for a single extracted relationship between two entities."""

    source_entity: str
    target_entity: str
    relationship_keywords: str
    relationship_description: str


class GraphExtraction(BaseModel):
    """Schema for entity and relationship extraction output from LLM."""

    entities: list[EntityExtraction] = []
    relationships: list[RelationshipExtraction] = []


def create_graph_extraction_schema(
    entity_types: list[str],
) -> type[BaseModel]:
    """Create a dynamic GraphExtraction schema with schema-level entity_type enforcement.

    Builds a Pydantic model where entity_type is restricted to the allowed values via
    Literal, so the JSON schema sent to the LLM includes an enum constraint. Providers
    that support structured output (e.g. OpenAI response_format) will enforce this.

    Args:
        entity_types: Allowed entity type strings (e.g. ["Person", "Organization"]).
            "Other" is appended if not present, since prompts instruct the LLM to use
            it when no provided type applies.

    Returns:
        A dynamically created model class compatible with GraphExtraction structure.
    """
    allowed = list(entity_types) if entity_types else []
    if "Other" not in allowed:
        allowed.append("Other")

    def _make_literal_type(values: list[str]):
        if not values:
            return str
        if len(values) == 1:
            return Literal[values[0]]  # type: ignore[misc]
        return reduce(
            lambda a, b: Union[a, Literal[b]],  # type: ignore[misc]
            values[1:],
            Literal[values[0]],  # type: ignore[misc]
        )

    literal_type = _make_literal_type(allowed)
    EntityExtractionDynamic = create_model(
        "EntityExtraction",
        entity_name=(str, ...),
        entity_type=(literal_type, ...),
        entity_description=(str, ...),
        __base__=None,
    )
    return create_model(
        "GraphExtraction",
        entities=(list[EntityExtractionDynamic], []),  # type: ignore[valid-type]
        relationships=(list[RelationshipExtraction], []),
        __base__=None,
    )


class KnowledgeGraphNode(BaseModel):
    id: str
    labels: list[str]
    properties: dict[str, Any]  # anything else goes here


class KnowledgeGraphEdge(BaseModel):
    id: str
    type: Optional[str]
    source: str  # id of source node
    target: str  # id of target node
    properties: dict[str, Any]  # anything else goes here


class KnowledgeGraph(BaseModel):
    nodes: list[KnowledgeGraphNode] = []
    edges: list[KnowledgeGraphEdge] = []
    is_truncated: bool = False
