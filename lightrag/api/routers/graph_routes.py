"""
This module contains all graph-related routes for the LightRAG API.
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from ..utils_api import get_combined_auth_dependency

router = APIRouter(tags=["graph"])

# Pydantic models for graph routes
class GraphLabelsResponse(BaseModel):
    """Response model: List of graph labels"""
    labels: List[str] = Field(description="List of graph labels")

class KnowledgeGraphNode(BaseModel):
    """Model for a node in the knowledge graph"""
    id: str = Field(description="Unique identifier of the node")
    label: str = Field(description="Label of the node")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Properties of the node")

class KnowledgeGraphEdge(BaseModel):
    """Model for an edge in the knowledge graph"""
    source: str = Field(description="Source node ID")
    target: str = Field(description="Target node ID")
    type: str = Field(description="Type of the relationship")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Properties of the edge")

class KnowledgeGraphResponse(BaseModel):
    """Response model: Knowledge graph data"""
    nodes: List[KnowledgeGraphNode] = Field(description="List of nodes in the graph")
    edges: List[KnowledgeGraphEdge] = Field(description="List of edges in the graph")


def create_graph_routes(rag, api_key: Optional[str] = None):
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get("/graph/label/list", 
                dependencies=[Depends(combined_auth)],
                response_model=GraphLabelsResponse)
    async def get_graph_labels():
        """
        Get all graph labels

        Returns:
            GraphLabelsResponse: List of graph labels
        """
        labels = await rag.get_graph_labels()
        return GraphLabelsResponse(labels=labels)

    @router.get("/graphs", 
                dependencies=[Depends(combined_auth)],
                response_model=KnowledgeGraphResponse)
    async def get_knowledge_graph(
        label: str = Query(..., description="Label to get knowledge graph for"),
        max_depth: int = Query(3, description="Maximum depth of graph", ge=1),
        max_nodes: int = Query(1000, description="Maxiumu nodes to return", ge=1),
    ):
        """
        Retrieve a connected subgraph of nodes where the label includes the specified label.
        When reducing the number of nodes, the prioritization criteria are as follows:
            1. Hops(path) to the staring node take precedence
            2. Followed by the degree of the nodes

        Args:
            label (str): Label of the starting node
            max_depth (int, optional): Maximum depth of the subgraph,Defaults to 3
            max_nodes: Maxiumu nodes to return

        Returns:
            KnowledgeGraphResponse: Knowledge graph containing nodes and edges
        """
        graph_data = await rag.get_knowledge_graph(
            node_label=label,
            max_depth=max_depth,
            max_nodes=max_nodes,
        )
        
        # Convert the returned dictionary to our response model format
        # Assuming the returned dictionary has 'nodes' and 'edges' keys
        return KnowledgeGraphResponse(
            nodes=graph_data.get("nodes", []),
            edges=graph_data.get("edges", [])
        )

    return router
