from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Optional


class GPTKeywordExtractionFormat(BaseModel):
    high_level_keywords: list[str]
    low_level_keywords: list[str]


class ExtractedEntity(BaseModel):
    """A single entity extracted from text by the LLM."""

    entity_name: str = Field(
        description="Name of the entity. Use title case for case-insensitive names."
    )
    entity_type: str = Field(
        description="Type/category of the entity."
    )
    entity_description: str = Field(
        description="Concise yet comprehensive description of the entity based on the input text."
    )


class ExtractedRelationship(BaseModel):
    """A single relationship between two entities extracted from text."""

    source_entity: str = Field(
        description="Name of the source entity in the relationship."
    )
    target_entity: str = Field(
        description="Name of the target entity in the relationship."
    )
    relationship_keywords: str = Field(
        description="Comma-separated high-level keywords summarizing the relationship."
    )
    relationship_description: str = Field(
        description="Concise explanation of the relationship between source and target entities."
    )


class EntityExtractionResult(BaseModel):
    """Structured output format for entity and relationship extraction from text."""

    entities: list[ExtractedEntity] = Field(
        default_factory=list,
        description="List of entities extracted from the input text.",
    )
    relationships: list[ExtractedRelationship] = Field(
        default_factory=list,
        description="List of relationships between entities extracted from the input text.",
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
