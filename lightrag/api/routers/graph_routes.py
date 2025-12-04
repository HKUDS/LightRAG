"""
This module contains all graph-related routes for the LightRAG API.
"""

from typing import Optional, Dict, Any
import traceback
from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel, Field

from lightrag.utils import logger
from ..utils_api import get_combined_auth_dependency

router = APIRouter(tags=["graph"])


class EntityUpdateRequest(BaseModel):
    entity_name: str
    updated_data: Dict[str, Any]
    allow_rename: bool = False


class RelationUpdateRequest(BaseModel):
    source_id: str
    target_id: str
    updated_data: Dict[str, Any]


class EntityMergeRequest(BaseModel):
    entities_to_change: list[str] = Field(
        ...,
        description="List of entity names to be merged and deleted. These are typically duplicate or misspelled entities.",
        min_length=1,
        examples=[["Elon Msk", "Ellon Musk"]],
    )
    entity_to_change_into: str = Field(
        ...,
        description="Target entity name that will receive all relationships from the source entities. This entity will be preserved.",
        min_length=1,
        examples=["Elon Musk"],
    )


class EntityCreateRequest(BaseModel):
    entity_name: str = Field(
        ...,
        description="Unique name for the new entity",
        min_length=1,
        examples=["Tesla"],
    )
    entity_data: Dict[str, Any] = Field(
        ...,
        description="Dictionary containing entity properties. Common fields include 'description' and 'entity_type'.",
        examples=[
            {
                "description": "Electric vehicle manufacturer",
                "entity_type": "ORGANIZATION",
            }
        ],
    )


class RelationCreateRequest(BaseModel):
    source_entity: str = Field(
        ...,
        description="Name of the source entity. This entity must already exist in the knowledge graph.",
        min_length=1,
        examples=["Elon Musk"],
    )
    target_entity: str = Field(
        ...,
        description="Name of the target entity. This entity must already exist in the knowledge graph.",
        min_length=1,
        examples=["Tesla"],
    )
    relation_data: Dict[str, Any] = Field(
        ...,
        description="Dictionary containing relationship properties. Common fields include 'description', 'keywords', and 'weight'.",
        examples=[
            {
                "description": "Elon Musk is the CEO of Tesla",
                "keywords": "CEO, founder",
                "weight": 1.0,
            }
        ],
    )


def create_graph_routes(rag, api_key: Optional[str] = None):
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get("/graph/label/list", dependencies=[Depends(combined_auth)])
    async def get_graph_labels():
        """
        Get all graph labels

        Returns:
            List[str]: List of graph labels
        """
        try:
            return await rag.get_graph_labels()
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
    ):
        """
        Get popular labels by node degree (most connected entities)

        Args:
            limit (int): Maximum number of labels to return (default: 300, max: 1000)

        Returns:
            List[str]: List of popular labels sorted by degree (highest first)
        """
        try:
            return await rag.chunk_entity_relation_graph.get_popular_labels(limit)
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
    ):
        """
        Search labels with fuzzy matching

        Args:
            q (str): Search query string
            limit (int): Maximum number of results to return (default: 50, max: 100)

        Returns:
            List[str]: List of matching labels sorted by relevance
        """
        try:
            return await rag.chunk_entity_relation_graph.search_labels(q, limit)
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
            Dict[str, List[str]]: Knowledge graph for label
        """
        try:
            # Log the label parameter to check for leading spaces
            logger.debug(
                f"get_knowledge_graph called with label: '{label}' (length: {len(label)}, repr: {repr(label)})"
            )

            return await rag.get_knowledge_graph(
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
    ):
        """
        Check if an entity with the given name exists in the knowledge graph

        Args:
            name (str): Name of the entity to check

        Returns:
            Dict[str, bool]: Dictionary with 'exists' key indicating if entity exists
        """
        try:
            exists = await rag.chunk_entity_relation_graph.has_node(name)
            return {"exists": exists}
        except Exception as e:
            logger.error(f"Error checking entity existence for '{name}': {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Error checking entity existence: {str(e)}"
            )

    @router.post("/graph/entity/edit", dependencies=[Depends(combined_auth)])
    async def update_entity(request: EntityUpdateRequest):
        """
        Update an entity's properties in the knowledge graph

        Args:
            request (EntityUpdateRequest): Request containing entity name, updated data, and rename flag

        Returns:
            Dict: Updated entity information
        """
        try:
            result = await rag.aedit_entity(
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
    async def update_relation(request: RelationUpdateRequest):
        """Update a relation's properties in the knowledge graph

        Args:
            request (RelationUpdateRequest): Request containing source ID, target ID and updated data

        Returns:
            Dict: Updated relation information
        """
        try:
            result = await rag.aedit_relation(
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

    @router.post("/graph/entity/create", dependencies=[Depends(combined_auth)])
    async def create_entity(request: EntityCreateRequest):
        """
        Create a new entity in the knowledge graph

        This endpoint creates a new entity node in the knowledge graph with the specified
        properties. The system automatically generates vector embeddings for the entity
        to enable semantic search and retrieval.

        Request Body:
            entity_name (str): Unique name identifier for the entity
            entity_data (dict): Entity properties including:
                - description (str): Textual description of the entity
                - entity_type (str): Category/type of the entity (e.g., PERSON, ORGANIZATION, LOCATION)
                - Additional custom properties as needed

        Response Schema:
            {
                "status": "success",
                "message": "Entity 'Tesla' created successfully",
                "data": {
                    "entity_name": "Tesla",
                    "description": "Electric vehicle manufacturer",
                    "entity_type": "ORGANIZATION",
                    ... (other entity properties)
                }
            }

        HTTP Status Codes:
            200: Entity created successfully
            400: Invalid request (e.g., missing required fields, duplicate entity)
            500: Internal server error

        Example Request:
            POST /graph/entity/create
            {
                "entity_name": "Tesla",
                "entity_data": {
                    "description": "Electric vehicle manufacturer",
                    "entity_type": "ORGANIZATION"
                }
            }
        """
        try:
            # Check if entity already exists
            exists = await rag.chunk_entity_relation_graph.has_node(request.entity_name)
            if exists:
                raise ValueError(f"Entity '{request.entity_name}' already exists")

            # Prepare entity data
            entity_data = request.entity_data.copy()
            entity_data["entity_id"] = request.entity_name

            # Create the entity
            await rag.chunk_entity_relation_graph.upsert_node(
                request.entity_name, entity_data
            )

            return {
                "status": "success",
                "message": f"Entity '{request.entity_name}' created successfully",
                "data": entity_data,
            }
        except ValueError as ve:
            logger.error(
                f"Validation error creating entity '{request.entity_name}': {str(ve)}"
            )
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(f"Error creating entity '{request.entity_name}': {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Error creating entity: {str(e)}"
            )

    @router.post("/graph/relation/create", dependencies=[Depends(combined_auth)])
    async def create_relation(request: RelationCreateRequest):
        """
        Create a new relationship between two entities in the knowledge graph

        This endpoint establishes a directed relationship between two existing entities.
        Both the source and target entities must already exist in the knowledge graph.
        The system automatically generates vector embeddings for the relationship to
        enable semantic search and graph traversal.

        Prerequisites:
            - Both source_entity and target_entity must exist in the knowledge graph
            - Use /graph/entity/create to create entities first if they don't exist

        Request Body:
            source_entity (str): Name of the source entity (relationship origin)
            target_entity (str): Name of the target entity (relationship destination)
            relation_data (dict): Relationship properties including:
                - description (str): Textual description of the relationship
                - keywords (str): Comma-separated keywords describing the relationship type
                - weight (float): Relationship strength/importance (default: 1.0)
                - Additional custom properties as needed

        Response Schema:
            {
                "status": "success",
                "message": "Relation created successfully between 'Elon Musk' and 'Tesla'",
                "data": {
                    "src_id": "Elon Musk",
                    "tgt_id": "Tesla",
                    "description": "Elon Musk is the CEO of Tesla",
                    "keywords": "CEO, founder",
                    "weight": 1.0,
                    ... (other relationship properties)
                }
            }

        HTTP Status Codes:
            200: Relationship created successfully
            400: Invalid request (e.g., missing entities, invalid data, duplicate relationship)
            500: Internal server error

        Example Request:
            POST /graph/relation/create
            {
                "source_entity": "Elon Musk",
                "target_entity": "Tesla",
                "relation_data": {
                    "description": "Elon Musk is the CEO of Tesla",
                    "keywords": "CEO, founder",
                    "weight": 1.0
                }
            }
        """
        try:
            # Check if both entities exist
            source_exists = await rag.chunk_entity_relation_graph.has_node(
                request.source_entity
            )
            target_exists = await rag.chunk_entity_relation_graph.has_node(
                request.target_entity
            )

            if not source_exists:
                raise ValueError(
                    f"Source entity '{request.source_entity}' does not exist"
                )
            if not target_exists:
                raise ValueError(
                    f"Target entity '{request.target_entity}' does not exist"
                )

            # Create the relationship
            await rag.chunk_entity_relation_graph.upsert_edge(
                request.source_entity, request.target_entity, request.relation_data
            )

            return {
                "status": "success",
                "message": f"Relation created successfully between '{request.source_entity}' and '{request.target_entity}'",
                "data": {
                    "source": request.source_entity,
                    "target": request.target_entity,
                    **request.relation_data,
                },
            }
        except ValueError as ve:
            logger.error(
                f"Validation error creating relation between '{request.source_entity}' and '{request.target_entity}': {str(ve)}"
            )
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(
                f"Error creating relation between '{request.source_entity}' and '{request.target_entity}': {str(e)}"
            )
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Error creating relation: {str(e)}"
            )

    @router.post("/graph/entities/merge", dependencies=[Depends(combined_auth)])
    async def merge_entities(request: EntityMergeRequest):
        """
        Merge multiple entities into a single entity, preserving all relationships

        This endpoint consolidates duplicate or misspelled entities while preserving the entire
        graph structure. It's particularly useful for cleaning up knowledge graphs after document
        processing or correcting entity name variations.

        What the Merge Operation Does:
            1. Deletes the specified source entities from the knowledge graph
            2. Transfers all relationships from source entities to the target entity
            3. Intelligently merges duplicate relationships (if multiple sources have the same relationship)
            4. Updates vector embeddings for accurate retrieval and search
            5. Preserves the complete graph structure and connectivity
            6. Maintains relationship properties and metadata

        Use Cases:
            - Fixing spelling errors in entity names (e.g., "Elon Msk" -> "Elon Musk")
            - Consolidating duplicate entities discovered after document processing
            - Merging name variations (e.g., "NY", "New York", "New York City")
            - Cleaning up the knowledge graph for better query performance
            - Standardizing entity names across the knowledge base

        Request Body:
            entities_to_change (list[str]): List of entity names to be merged and deleted
            entity_to_change_into (str): Target entity that will receive all relationships

        Response Schema:
            {
                "status": "success",
                "message": "Successfully merged 2 entities into 'Elon Musk'",
                "data": {
                    "merged_entity": "Elon Musk",
                    "deleted_entities": ["Elon Msk", "Ellon Musk"],
                    "relationships_transferred": 15,
                    ... (merge operation details)
                }
            }

        HTTP Status Codes:
            200: Entities merged successfully
            400: Invalid request (e.g., empty entity list, target entity doesn't exist)
            500: Internal server error

        Example Request:
            POST /graph/entities/merge
            {
                "entities_to_change": ["Elon Msk", "Ellon Musk"],
                "entity_to_change_into": "Elon Musk"
            }

        Note:
            - The target entity (entity_to_change_into) must exist in the knowledge graph
            - Source entities will be permanently deleted after the merge
            - This operation cannot be undone, so verify entity names before merging
        """
        try:
            result = await rag.amerge_entities(
                source_entities=request.entities_to_change,
                target_entity=request.entity_to_change_into,
            )
            return {
                "status": "success",
                "message": f"Successfully merged {len(request.entities_to_change)} entities into '{request.entity_to_change_into}'",
                "data": result,
            }
        except ValueError as ve:
            logger.error(
                f"Validation error merging entities {request.entities_to_change} into '{request.entity_to_change_into}': {str(ve)}"
            )
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            logger.error(
                f"Error merging entities {request.entities_to_change} into '{request.entity_to_change_into}': {str(e)}"
            )
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Error merging entities: {str(e)}"
            )

    return router
