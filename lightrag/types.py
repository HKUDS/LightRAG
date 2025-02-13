from pydantic import BaseModel
from typing import List, Dict, Any


class GPTKeywordExtractionFormat(BaseModel):
    high_level_keywords: List[str]
    low_level_keywords: List[str]


class KnowledgeGraphNode(BaseModel):
    id: str
    labels: List[str]
    properties: Dict[str, Any]  # anything else goes here


class KnowledgeGraphEdge(BaseModel):
    id: str
    type: str
    source: str  # id of source node
    target: str  # id of target node
    properties: Dict[str, Any]  # anything else goes here


class KnowledgeGraph(BaseModel):
    nodes: List[KnowledgeGraphNode] = []
    edges: List[KnowledgeGraphEdge] = []
