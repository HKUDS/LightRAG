"""
This module contains all graph-related routes for the LightRAG API.
"""

from typing import Optional
from fastapi import APIRouter, Depends

from ..utils_api import get_api_key_dependency, get_auth_dependency

router = APIRouter(tags=["graph"], dependencies=[Depends(get_auth_dependency())])


def create_graph_routes(rag, api_key: Optional[str] = None):
    optional_api_key = get_api_key_dependency(api_key)

    @router.get("/graph/label/list", dependencies=[Depends(optional_api_key)])
    async def get_graph_labels():
        """
        Get all graph labels

        Returns:
            List[str]: List of graph labels
        """
        return await rag.get_graph_labels()

    @router.get("/graphs", dependencies=[Depends(optional_api_key)])
    async def get_knowledge_graph(
        label: str, max_depth: int = 3, min_degree: int = 0, inclusive: bool = False
    ):
        """
        Retrieve a connected subgraph of nodes where the label includes the specified label.
        Maximum number of nodes is constrained by the environment variable `MAX_GRAPH_NODES` (default: 1000).
        When reducing the number of nodes, the prioritization criteria are as follows:
            1. min_degree does not affect nodes directly connected to the matching nodes
            2. Label matching nodes take precedence
            3. Followed by nodes directly connected to the matching nodes
            4. Finally, the degree of the nodes
        Maximum number of nodes is limited to env MAX_GRAPH_NODES(default: 1000)

        Args:
            label (str): Label to get knowledge graph for
            max_depth (int, optional): Maximum depth of graph. Defaults to 3.
            inclusive_search (bool, optional): If True, search for nodes that include the label. Defaults to False.
            min_degree (int, optional): Minimum degree of nodes. Defaults to 0.

        Returns:
            Dict[str, List[str]]: Knowledge graph for label
        """
        return await rag.get_knowledge_graph(
            node_label=label,
            max_depth=max_depth,
            inclusive=inclusive,
            min_degree=min_degree,
        )

    return router
