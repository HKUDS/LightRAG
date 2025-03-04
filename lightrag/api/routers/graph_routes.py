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
        """
        Get all graph labels

        Returns:
            List[str]: List of graph labels
        """
        return await rag.get_graph_labels()

    @router.get("/graphs", dependencies=[Depends(optional_api_key)])
    async def get_knowledge_graph(label: str, max_depth: int = 3):
        """
        Retrieve a connected subgraph of nodes where the label includes the specified label.
        Maximum number of nodes is constrained by the environment variable `MAX_GRAPH_NODES` (default: 1000).
        When reducing the number of nodes, the prioritization criteria are as follows:
            1. Label matching nodes take precedence
            2. Followed by nodes directly connected to the matching nodes
            3. Finally, the degree of the nodes
        Maximum number of nodes is limited to env MAX_GRAPH_NODES(default: 1000)
        Control search mode by label content:
            1. only label-name :  exact search with the label name (selecting from the label list return previously)
            2. label-name follow by '>n' : exact search of nodes with degree more than n
            3. label-name follow by* :  inclusive search of nodes with degree more than n
            4. label-name follow by '>n*' : inclusive search 

        Args:
            label (str): Label to get knowledge graph for
            max_depth (int, optional): Maximum depth of graph. Defaults to 3.

        Returns:
            Dict[str, List[str]]: Knowledge graph for label
        """
        # Parse label to extract search mode and min degree if specified
        search_mode = "exact"  # Default search mode
        min_degree = 0  # Default minimum degree
        original_label = label
        
        # First check if label ends with *
        if label.endswith("*"):
            search_mode = "inclusive"  # Always set to inclusive if ends with *
            label = label[:-1].strip()  # Remove trailing *
            
            # Try to parse >n if it exists
            if ">" in label:
                try:
                    degree_pos = label.rfind(">")
                    degree_str = label[degree_pos + 1:].strip()
                    min_degree = int(degree_str) + 1
                    label = label[:degree_pos].strip()
                except ValueError:
                    # If degree parsing fails, just remove * and keep the rest as label
                    label = original_label[:-1].strip()
        # If no *, check for >n pattern
        elif ">" in label:
            try:
                degree_pos = label.rfind(">")
                degree_str = label[degree_pos + 1:].strip()
                min_degree = int(degree_str) + 1
                label = label[:degree_pos].strip()
            except ValueError:
                # If degree parsing fails, treat the whole string as label
                label = original_label
                    
        return await rag.get_knowledge_graph(node_label=label, max_depth=max_depth, search_mode=search_mode, min_degree=min_degree)

    return router
