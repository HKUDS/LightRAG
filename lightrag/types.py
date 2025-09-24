from __future__ import annotations

from pydantic import BaseModel, Field, validator
from typing import Any, Optional, List, Union, Dict


class GPTKeywordExtractionFormat(BaseModel):
    high_level_keywords: list[str]
    low_level_keywords: list[str]


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


class MetadataFilter(BaseModel):
    """
    Represents a logical expression for metadata filtering.

    Args:
        operator: "AND", "OR", or "NOT"
        operands: List of either simple key-value pairs or nested MetadataFilter objects
    """
    operator: str = Field(..., description="Logical operator: AND, OR, or NOT")
    operands: List[Union[Dict[str, Any], 'MetadataFilter']] = Field(default_factory=list, description="List of operands for filtering")

    @validator('operator')
    def validate_operator(cls, v):
        if v not in ["AND", "OR", "NOT"]:
            raise ValueError('operator must be one of: "AND", "OR", "NOT"')
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "operator": self.operator,
            "operands": [
                operand.dict() if isinstance(operand, MetadataFilter) else operand
                for operand in self.operands
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetadataFilter':
        """Create from dictionary representation."""
        operands = []
        for operand in data.get("operands", []):
            if isinstance(operand, dict) and "operator" in operand:
                operands.append(cls.from_dict(operand))
            else:
                operands.append(operand)
        return cls(operator=data.get("operator", "AND"), operands=operands)

    class Config:
        """Pydantic configuration."""
        validate_assignment = True
