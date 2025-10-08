"""
Pydantic schemas for structured LLM outputs in LightRAG.
This module defines strict schemas for entity extraction, relationship extraction,
and keyword extraction to ensure consistent and validated outputs from LLMs.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, field_validator
import re


class Entity(BaseModel):
    """Schema for a single extracted entity."""
    
    entity_name: str = Field(
        ...,
        description="The name of the entity in title case",
        min_length=1,
        max_length=500
    )
    entity_type: str = Field(
        ...,
        description="The category/type of the entity",
        min_length=1,
        max_length=100
    )
    entity_description: str = Field(
        ...,
        description="A comprehensive description of the entity's attributes and activities",
        min_length=1,
        max_length=5000
    )
    
    @field_validator('entity_name', 'entity_type')
    @classmethod
    def validate_no_delimiters(cls, v: str) -> str:
        """Ensure no delimiter characters in critical fields."""
        if '<|' in v or '|>' in v:
            raise ValueError(f"Field contains delimiter characters: {v}")
        return v.strip()
    
    @field_validator('entity_type')
    @classmethod
    def normalize_entity_type(cls, v: str) -> str:
        """Normalize entity type: lowercase, no spaces."""
        return v.replace(" ", "").lower()


class Relationship(BaseModel):
    """Schema for a single extracted relationship between entities."""
    
    source_entity: str = Field(
        ...,
        description="The name of the source entity",
        min_length=1,
        max_length=500
    )
    target_entity: str = Field(
        ...,
        description="The name of the target entity",
        min_length=1,
        max_length=500
    )
    relationship_keywords: str = Field(
        ...,
        description="High-level keywords summarizing the relationship (comma-separated)",
        min_length=1,
        max_length=500
    )
    relationship_description: str = Field(
        ...,
        description="A clear explanation of the relationship between entities",
        min_length=1,
        max_length=5000
    )
    
    @field_validator('source_entity', 'target_entity', 'relationship_keywords')
    @classmethod
    def validate_no_delimiters(cls, v: str) -> str:
        """Ensure no delimiter characters in critical fields."""
        if '<|' in v or '|>' in v:
            raise ValueError(f"Field contains delimiter characters: {v}")
        return v.strip()


class ExtractionResult(BaseModel):
    """Complete extraction result containing entities and relationships."""
    
    entities: List[Entity] = Field(
        default_factory=list,
        description="List of extracted entities"
    )
    relationships: List[Relationship] = Field(
        default_factory=list,
        description="List of extracted relationships"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "entities": [
                    {
                        "entity_name": "Alex",
                        "entity_type": "person",
                        "entity_description": "Alex is a character who experiences frustration and is observant of dynamics."
                    }
                ],
                "relationships": [
                    {
                        "source_entity": "Alex",
                        "target_entity": "Taylor",
                        "relationship_keywords": "observation, tension",
                        "relationship_description": "Alex observes Taylor's authoritarian certainty with frustration."
                    }
                ]
            }
        }


class KeywordExtraction(BaseModel):
    """Schema for keyword extraction from queries."""
    
    high_level_keywords: List[str] = Field(
        ...,
        description="High-level concepts or themes",
        min_items=1,
        max_items=10
    )
    low_level_keywords: List[str] = Field(
        ...,
        description="Specific entities or details",
        min_items=1,
        max_items=20
    )
    
    @field_validator('high_level_keywords', 'low_level_keywords')
    @classmethod
    def validate_keywords(cls, v: List[str]) -> List[str]:
        """Ensure keywords are non-empty strings."""
        return [k.strip() for k in v if k.strip()]
    
    class Config:
        json_schema_extra = {
            "example": {
                "high_level_keywords": ["machine learning", "artificial intelligence"],
                "low_level_keywords": ["neural networks", "GPT", "transformer architecture"]
            }
        }