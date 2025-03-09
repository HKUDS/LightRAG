"""
This module contains all graph-related routes for the LightRAG API.
"""

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel


from ..utils_api import get_api_key_dependency, get_auth_dependency

class DataResponse(BaseModel):
    status: str
    message: str
    data: Any

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


    # Knowledge Graph - Entity Addition
    @router.post(
        "/graph/entity",
        response_model=DataResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def create_entity(
        entity_name: str, entity_data: dict, 
    ):
        try:
            print("Updating entity:", entity_name)
            print("Entity data:", entity_data)
            entity_type = entity_data.get("entity_type", None)
            description = entity_data.get("description", None)
            source_id = entity_data.get("source_id", None)
            # Prepare node data
            node_data = {
                "entity_type": entity_type,
                "description": description,
                "source_id": source_id,
            }
            await rag.acreate_entity(entity_name, node_data)

            data = {"id": entity_name, "label": entity_name, **node_data}
            return DataResponse(status="success", message="ok", data=data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Knowledge Graph - Entity Modification
    @router.put(
        "/graph/entity/{entity_name}",
        response_model=DataResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def update_entity(
        entity_name: str, entity_data: dict, 
    ):
        try:
            print("Updating entity:", entity_name)
            print("Entity data:", entity_data)
            new_entity_name = entity_data.get("entity_name", None)
            entity_type = entity_data.get("entity_type", None)
            description = entity_data.get("description", None)
            source_id = entity_data.get("source_id", None)
            # Prepare node data
            node_data = {
                "entity_type": entity_type,
                "description": description,
                "source_id": source_id,
            }
            # If new_entity_name exists, add it to node_data.
            if new_entity_name:
                node_data["entity_name"] = new_entity_name

            await rag.aedit_entity(entity_name, node_data)
            new_entity_name = entity_name
            data = {"id": new_entity_name, "label": new_entity_name, **node_data}
            return DataResponse(status="success", message="ok", data=data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Knowledge Graph - Entity Deletion
    @router.delete(
        "/graph/entity/{entity_name}",
        response_model=DataResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def delete_entity(entity_name: str, ):
        print(f"Deleting entity {entity_name}")
        try:
            await rag.adelete_by_entity(entity_name)
            return DataResponse(status="success", message="ok", data=None)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Knowledge graph - Relationship addition - Query relationships by start node and end node
    @router.post(
        "/graph/relation/by_nodes",
        response_model=DataResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def create_relation_by_nodes(
        src_entity_name: str,
        tgt_entity_name: str,
        relation_data: dict
    ):
        try:
            src_id = src_entity_name
            tgt_id = tgt_entity_name
            weight = relation_data.get("weight", None)
            keywords = relation_data.get("keywords", None)
            description = relation_data.get("description", None)
            source_id = relation_data.get("source_id", None)

            edge_data = dict(
                weight=weight,
                description=description,
                keywords=keywords,
                source_id=source_id,
            )

            print("Edge Data:", edge_data)  # Debugging print
            # Insert node data into the knowledge graph
            await rag.acreate_relation(src_id, tgt_id, edge_data)
            data = {
                "id": src_id + "-" + tgt_id,
                "source": src_id,
                "target": tgt_id,
                **edge_data,
            }
            return DataResponse(status="success", message="ok", data=data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Knowledge Graph - Relationship Modification - Query Relationships by Start and End Nodes
    @router.put(
        "/graph/relation/by_nodes",
        response_model=DataResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def update_relation_by_nodes(
        src_entity_name: str,
        tgt_entity_name: str,
        relation_data: dict
    ):
        try:
            src_id = src_entity_name
            tgt_id = tgt_entity_name
            weight = relation_data.get("weight", None)
            keywords = relation_data.get("keywords", None)
            description = relation_data.get("description", None)
            source_id = relation_data.get("source_id", None)

            edge_data = dict(
                weight=weight,
                description=description,
                keywords=keywords,
                source_id=source_id,
            )

            print("Edge Data:", edge_data)  # Debugging print
            await rag.aedit_relation(src_id, tgt_id, edge_data)
            data = {
                "id": src_id + "-" + tgt_id,
                "source": src_id,
                "target": tgt_id,
                **edge_data,
            }
            return DataResponse(status="success", message="ok", data=data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Knowledge Graph - Relationship Deletion - Query relationships through start and end nodes
    @router.delete(
        "/graph/relation/by_nodes",
        response_model=DataResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def delete_relation_by_nodes(
        src_entity_name: str, tgt_entity_name: str, 
    ):
        try:
            await rag.adelete_by_relation(src_entity_name, tgt_entity_name)
            return DataResponse(status="success", message="ok", data="")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Knowledge Graph - Entity Query
    @router.get(
        "/graph/entity/{entity_name}",
        response_model=DataResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def get_node(entity_name: str, ):
        try:
            node = await rag.chunk_entity_relation_graph.get_node(entity_name)
            return DataResponse(status="success", message="ok", data=node)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Knowledge Graph - Relationship Query - Query Relationships by Start and End Nodes
    @router.get(
        "/graph/relation/by_nodes",
        response_model=DataResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def get_relation_by_nodes(
        src_entity_name: str, tgt_entity_name: str, 
    ):
        try:
            relation = await rag.chunk_entity_relation_graph.get_edge(
                src_entity_name, tgt_entity_name
            )
            return DataResponse(status="success", message="ok", data=relation)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Knowledge Graph - Relationship Query - Query Relationships by Node ID
    @router.get(
        "/graph/relation/node/{node_id}",
        response_model=DataResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def get_relation_by_node(node_id: str, ):
        try:
            relations = await rag.chunk_entity_relation_graph.get_node_edges(node_id)
            return DataResponse(status="success", message="ok", data=relations)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Knowledge Graph - Entity List - Query
    @router.get(
        "/graph/entity",
        response_model=DataResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def get_graph_entity_list():
        try:
            entities = await rag.chunk_entity_relation_graph.query_all()
            return DataResponse(
                status="success",
                message="ok",
                data=entities,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router