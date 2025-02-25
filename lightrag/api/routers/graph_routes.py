"""
This module contains all graph-related routes for the LightRAG API.
"""

from typing import Optional

from fastapi import APIRouter, Depends

from ..utils_api import get_api_key_dependency

router = APIRouter(tags=["graph"])


def create_graph_routes(rag, api_key: Optional[str] = None):
    optional_api_key = get_api_key_dependency(api_key)

    @router.get("/graph/label/list", dependencies=[Depends(optional_api_key)])
    async def get_graph_labels():
        """Get all graph labels"""
        return await rag.get_graph_labels()

    @router.get("/graphs", dependencies=[Depends(optional_api_key)])
    async def get_knowledge_graph(label: str, max_depth: int = 3):
        """Get knowledge graph for a specific label"""
        return await rag.get_knowledge_graph(node_label=label, max_depth=max_depth)

    return router
