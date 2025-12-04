"""
This module contains all graph-related routes for the LightRAG API.
"""

from typing import Optional, Dict, Any
import traceback
from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel

from lightrag import LightRAG
from lightrag.utils import logger
from lightrag.models.tenant import TenantContext
from lightrag.tenant_rag_manager import TenantRAGManager
from ..utils_api import get_combined_auth_dependency
from ..dependencies import get_tenant_context_optional

router = APIRouter(tags=["graph"])


class EntityUpdateRequest(BaseModel):
    entity_name: str
    updated_data: Dict[str, Any]
    allow_rename: bool = False


class RelationUpdateRequest(BaseModel):
    source_id: str
    target_id: str
    updated_data: Dict[str, Any]


def create_graph_routes(rag, api_key: Optional[str] = None, rag_manager: Optional[TenantRAGManager] = None):
    combined_auth = get_combined_auth_dependency(api_key)
    
    async def get_tenant_rag(tenant_context: Optional[TenantContext] = Depends(get_tenant_context_optional)) -> LightRAG:
        """Dependency to get tenant-specific RAG instance for graph operations"""
        if rag_manager and tenant_context and tenant_context.tenant_id and tenant_context.kb_id:
            return await rag_manager.get_rag_instance(
                tenant_context.tenant_id, 
                tenant_context.kb_id,
                tenant_context.user_id  # Pass user_id for security validation
            )
        return rag

    @router.get("/graph/label/list", dependencies=[Depends(combined_auth)])
    async def get_graph_labels(tenant_rag: LightRAG = Depends(get_tenant_rag)):
        """
        Get all graph labels (tenant-scoped)

        Returns:
            List[str]: List of graph labels for the selected tenant/KB
        """
        try:
            return await tenant_rag.get_graph_labels()
        except Exception as e:
            logger.error(f"Error getting graph labels: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Error getting graph labels: {str(e)}"
            )

    @router.get("/graph/label/popular", dependencies=[Depends(combined_auth)])
    async def get_popular_labels(
        limit: int = Query(
            300, description="Maximum number of popular labels to return", ge=1, le=1000
        ),
        tenant_rag: LightRAG = Depends(get_tenant_rag),
    ):
        """
        Get popular labels by node degree (tenant-scoped)

        Args:
            limit (int): Maximum number of labels to return (default: 300, max: 1000)

        Returns:
            List[str]: List of popular labels sorted by degree (highest first) for the selected tenant/KB
        """
        try:
            return await tenant_rag.chunk_entity_relation_graph.get_popular_labels(limit)
        except Exception as e:
            logger.error(f"Error getting popular labels: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Error getting popular labels: {str(e)}"
            )

    @router.get("/graph/label/search", dependencies=[Depends(combined_auth)])
    async def search_labels(
        q: str = Query(..., description="Search query string"),
        limit: int = Query(
            50, description="Maximum number of search results to return", ge=1, le=100
        ),
        tenant_rag: LightRAG = Depends(get_tenant_rag),
    ):
        """
        Search labels with fuzzy matching (tenant-scoped)

        Args:
            q (str): Search query string
            limit (int): Maximum number of results to return (default: 50, max: 100)

        Returns:
            List[str]: List of matching labels sorted by relevance for the selected tenant/KB
        """
        try:
            return await tenant_rag.chunk_entity_relation_graph.search_labels(q, limit)
        except Exception as e:
            logger.error(f"Error searching labels with query '{q}': {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Error searching labels: {str(e)}"
            )

    @router.get("/graphs", dependencies=[Depends(combined_auth)])
    async def get_knowledge_graph(
        label: str = Query(..., description="Label to get knowledge graph for"),
        max_depth: int = Query(3, description="Maximum depth of graph", ge=1),
        max_nodes: int = Query(1000, description="Maximum nodes to return", ge=1),
        tenant_rag: LightRAG = Depends(get_tenant_rag),
    ):
        """
        Retrieve a connected subgraph of nodes (tenant-scoped).
        When reducing the number of nodes, the prioritization criteria are as follows:
            1. Hops(path) to the staring node take precedence
            2. Followed by the degree of the nodes

        Args:
            label (str): Label of the starting node
            max_depth (int, optional): Maximum depth of the subgraph,Defaults to 3
            max_nodes: Maxiumu nodes to return

        Returns:
            Dict[str, List[str]]: Knowledge graph for label from the selected tenant/KB
        """
        try:
            # Log the label parameter to check for leading spaces
            logger.debug(
                f"get_knowledge_graph called with label: '{label}' (length: {len(label)}, repr: {repr(label)})"
            )

            return await tenant_rag.get_knowledge_graph(
                node_label=label,
                max_depth=max_depth,
                max_nodes=max_nodes,
            )
        except Exception as e:
            logger.error(f"Error getting knowledge graph for label '{label}': {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Error getting knowledge graph: {str(e)}"
            )

    @router.get("/graph/entity/exists", dependencies=[Depends(combined_auth)])
    async def check_entity_exists(
        name: str = Query(..., description="Entity name to check"),
        tenant_rag: LightRAG = Depends(get_tenant_rag),
    ):
        """
        Check if an entity with the given name exists in the knowledge graph (tenant-scoped)

        Args:
            name (str): Name of the entity to check

        Returns:
            Dict[str, bool]: Dictionary with 'exists' key indicating if entity exists in the selected tenant/KB
        """
        try:
            exists = await tenant_rag.chunk_entity_relation_graph.has_node(name)
            return {"exists": exists}
        except Exception as e:
            logger.error(f"Error checking entity existence for '{name}': {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Error checking entity existence: {str(e)}"
            )

    @router.post("/graph/entity/edit", dependencies=[Depends(combined_auth)])
    async def update_entity(
        request: EntityUpdateRequest,
        tenant_rag: LightRAG = Depends(get_tenant_rag),
    ):
        """
        Update an entity's properties in the knowledge graph (tenant-scoped)

        Args:
            request (EntityUpdateRequest): Request containing entity name, updated data, and rename flag

        Returns:
            Dict: Updated entity information
        """
        try:
            result = await tenant_rag.aedit_entity(
                entity_name=request.entity_name,
                updated_data=request.updated_data,
                allow_rename=request.allow_rename,
            )
            return {
                "status": "success",
                "message": "Entity updated successfully",
                "data": result,
            }
        except ValueError as ve:
            logger.error(
                f"Validation error updating entity '{request.entity_name}': {str(ve)}"
            )
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(f"Error updating entity '{request.entity_name}': {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Error updating entity: {str(e)}"
            )

    @router.post("/graph/relation/edit", dependencies=[Depends(combined_auth)])
    async def update_relation(
        request: RelationUpdateRequest,
        tenant_rag: LightRAG = Depends(get_tenant_rag),
    ):
        """Update a relation's properties in the knowledge graph (tenant-scoped)

        Args:
            request (RelationUpdateRequest): Request containing source ID, target ID and updated data

        Returns:
            Dict: Updated relation information
        """
        try:
            result = await tenant_rag.aedit_relation(
                source_entity=request.source_id,
                target_entity=request.target_id,
                updated_data=request.updated_data,
            )
            return {
                "status": "success",
                "message": "Relation updated successfully",
                "data": result,
            }
        except ValueError as ve:
            logger.error(
                f"Validation error updating relation between '{request.source_id}' and '{request.target_id}': {str(ve)}"
            )
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(
                f"Error updating relation between '{request.source_id}' and '{request.target_id}': {str(e)}"
            )
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Error updating relation: {str(e)}"
            )

    return router
