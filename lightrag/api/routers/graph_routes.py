"""
This module contains all graph-related routes for the LightRAG API.
"""

import traceback
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from lightrag.api.utils_api import get_combined_auth_dependency
from lightrag.utils import logger

router = APIRouter(tags=['graph'])


class EntityUpdateRequest(BaseModel):
    entity_name: str
    updated_data: dict[str, Any]
    allow_rename: bool = False
    allow_merge: bool = False


class RelationUpdateRequest(BaseModel):
    source_id: str
    target_id: str
    updated_data: dict[str, Any]


class EntityMergeRequest(BaseModel):
    entities_to_change: list[str] = Field(
        ...,
        description='List of entity names to be merged and deleted. These are typically duplicate or misspelled entities.',
        min_length=1,
        examples=[['Elon Msk', 'Ellon Musk']],
    )
    entity_to_change_into: str = Field(
        ...,
        description='Target entity name that will receive all relationships from the source entities. This entity will be preserved.',
        min_length=1,
        examples=['Elon Musk'],
    )


class OrphanConnectionRequest(BaseModel):
    max_candidates: int = Field(
        default=3,
        description='Maximum number of candidate connections to evaluate per orphan',
        ge=1,
        le=10,
    )
    similarity_threshold: float | None = Field(
        default=None,
        description='Vector similarity threshold for candidates (0.0-1.0). Uses server config if not provided.',
        ge=0.0,
        le=1.0,
    )
    confidence_threshold: float | None = Field(
        default=None,
        description='LLM confidence threshold for creating connections (0.0-1.0). Uses server config if not provided.',
        ge=0.0,
        le=1.0,
    )
    cross_connect: bool | None = Field(
        default=None,
        description='Allow orphans to connect to other orphans. Uses server config if not provided.',
    )


class OrphanConnectionStatusResponse(BaseModel):
    """Response model for orphan connection pipeline status."""

    busy: bool = Field(description='Whether the orphan connection pipeline is currently running')
    job_name: str = Field(description='Name of the current or last job')
    job_start: str | None = Field(description='ISO timestamp when the job started')
    total_orphans: int = Field(description='Total number of orphan entities found')
    processed_orphans: int = Field(description='Number of orphans processed so far')
    connections_made: int = Field(description='Number of connections created so far')
    request_pending: bool = Field(description='Whether another request is pending')
    cancellation_requested: bool = Field(description='Whether cancellation has been requested')
    latest_message: str = Field(description='Most recent status message')
    history_messages: list[str] = Field(description='History of status messages')


class EntityCreateRequest(BaseModel):
    entity_name: str = Field(
        ...,
        description='Unique name for the new entity',
        min_length=1,
        examples=['Tesla'],
    )
    entity_data: dict[str, Any] = Field(
        ...,
        description="Dictionary containing entity properties. Common fields include 'description' and 'entity_type'.",
        examples=[
            {
                'description': 'Electric vehicle manufacturer',
                'entity_type': 'ORGANIZATION',
            }
        ],
    )


class RelationCreateRequest(BaseModel):
    source_entity: str = Field(
        ...,
        description='Name of the source entity. This entity must already exist in the knowledge graph.',
        min_length=1,
        examples=['Elon Musk'],
    )
    target_entity: str = Field(
        ...,
        description='Name of the target entity. This entity must already exist in the knowledge graph.',
        min_length=1,
        examples=['Tesla'],
    )
    relation_data: dict[str, Any] = Field(
        ...,
        description="Dictionary containing relationship properties. Common fields include 'description', 'keywords', and 'weight'.",
        examples=[
            {
                'description': 'Elon Musk is the CEO of Tesla',
                'keywords': 'CEO, founder',
                'weight': 1.0,
            }
        ],
    )


def create_graph_routes(rag, api_key: str | None = None):
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get('/graph/label/list', dependencies=[Depends(combined_auth)])
    async def get_graph_labels():
        """
        Get all graph labels

        Returns:
            List[str]: List of graph labels
        """
        try:
            return await rag.get_graph_labels()
        except Exception as e:
            logger.error(f'Error getting graph labels: {e!s}')
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f'Error getting graph labels: {e!s}') from e

    @router.get('/graph/label/popular', dependencies=[Depends(combined_auth)])
    async def get_popular_labels(
        limit: int = Query(300, description='Maximum number of popular labels to return', ge=1, le=1000),
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
            logger.error(f'Error getting popular labels: {e!s}')
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f'Error getting popular labels: {e!s}') from e

    @router.get('/graph/label/search', dependencies=[Depends(combined_auth)])
    async def search_labels(
        q: str = Query(..., description='Search query string'),
        limit: int = Query(50, description='Maximum number of search results to return', ge=1, le=100),
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
            logger.error(f"Error searching labels with query '{q}': {e!s}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f'Error searching labels: {e!s}') from e

    @router.get('/graphs', dependencies=[Depends(combined_auth)])
    async def get_knowledge_graph(
        label: str = Query(..., description='Label to get knowledge graph for'),
        max_depth: int = Query(3, description='Maximum depth of graph', ge=1),
        max_nodes: int = Query(1000, description='Maximum nodes to return', ge=1),
        min_degree: int = Query(
            0,
            description='Minimum degree (connections) required for nodes to be included. 0=all nodes, 1=exclude orphans, 2+=only well-connected nodes',
            ge=0,
            le=10,
        ),
        include_orphans: bool = Query(
            False,
            description='Include orphan nodes (degree=0) even when min_degree > 0. Useful for reviewing disconnected entities.',
        ),
    ):
        """
        Retrieve a connected subgraph of nodes where the label includes the specified label.
        When reducing the number of nodes, the prioritization criteria are as follows:
            1. Hops(path) to the staring node take precedence
            2. Followed by the degree of the nodes

        Args:
            label (str): Label of the starting node, use '*' for all nodes
            max_depth (int, optional): Maximum depth of the subgraph, Defaults to 3
            max_nodes (int): Maximum nodes to return
            min_degree (int): Minimum connections required (0=all, 1=exclude orphans, 2+=well-connected only)
            include_orphans (bool): Also include orphan nodes when min_degree > 0

        Returns:
            Dict[str, List[str]]: Knowledge graph for label
        """
        try:
            # Log the label parameter to check for leading spaces
            logger.debug(f"get_knowledge_graph called with label: '{label}' (length: {len(label)}, repr: {label!r})")

            return await rag.get_knowledge_graph(
                node_label=label,
                max_depth=max_depth,
                max_nodes=max_nodes,
                min_degree=min_degree,
                include_orphans=include_orphans,
            )
        except Exception as e:
            logger.error(f"Error getting knowledge graph for label '{label}': {e!s}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f'Error getting knowledge graph: {e!s}') from e

    @router.get('/graph/entity/exists', dependencies=[Depends(combined_auth)])
    async def check_entity_exists(
        name: str = Query(..., description='Entity name to check'),
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
            return {'exists': exists}
        except Exception as e:
            logger.error(f"Error checking entity existence for '{name}': {e!s}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f'Error checking entity existence: {e!s}') from e

    @router.post('/graph/entity/edit', dependencies=[Depends(combined_auth)])
    async def update_entity(request: EntityUpdateRequest):
        """
        Update an entity's properties in the knowledge graph

        This endpoint allows updating entity properties, including renaming entities.
        When renaming to an existing entity name, the behavior depends on allow_merge:

        Args:
            request (EntityUpdateRequest): Request containing:
                - entity_name (str): Name of the entity to update
                - updated_data (Dict[str, Any]): Dictionary of properties to update
                - allow_rename (bool): Whether to allow entity renaming (default: False)
                - allow_merge (bool): Whether to merge into existing entity when renaming
                                     causes name conflict (default: False)

        Returns:
            Dict with the following structure:
            {
                "status": "success",
                "message": "Entity updated successfully" | "Entity merged successfully into 'target_name'",
                "data": {
                    "entity_name": str,        # Final entity name
                    "description": str,        # Entity description
                    "entity_type": str,        # Entity type
                    "source_id": str,         # Source chunk IDs
                    ...                       # Other entity properties
                },
                "operation_summary": {
                    "merged": bool,           # Whether entity was merged into another
                    "merge_status": str,      # "success" | "failed" | "not_attempted"
                    "merge_error": str | None, # Error message if merge failed
                    "operation_status": str,  # "success" | "partial_success" | "failure"
                    "target_entity": str | None, # Target entity name if renaming/merging
                    "final_entity": str,      # Final entity name after operation
                    "renamed": bool           # Whether entity was renamed
                }
            }

        operation_status values explained:
            - "success": All operations completed successfully
                * For simple updates: entity properties updated
                * For renames: entity renamed successfully
                * For merges: non-name updates applied AND merge completed

            - "partial_success": Update succeeded but merge failed
                * Non-name property updates were applied successfully
                * Merge operation failed (entity not merged)
                * Original entity still exists with updated properties
                * Use merge_error for failure details

            - "failure": Operation failed completely
                * If merge_status == "failed": Merge attempted but both update and merge failed
                * If merge_status == "not_attempted": Regular update failed
                * No changes were applied to the entity

        merge_status values explained:
            - "success": Entity successfully merged into target entity
            - "failed": Merge operation was attempted but failed
            - "not_attempted": No merge was attempted (normal update/rename)

        Behavior when renaming to an existing entity:
            - If allow_merge=False: Raises ValueError with 400 status (default behavior)
            - If allow_merge=True: Automatically merges the source entity into the existing target entity,
                                  preserving all relationships and applying non-name updates first

        Example Request (simple update):
            POST /graph/entity/edit
            {
                "entity_name": "Tesla",
                "updated_data": {"description": "Updated description"},
                "allow_rename": false,
                "allow_merge": false
            }

        Example Response (simple update success):
            {
                "status": "success",
                "message": "Entity updated successfully",
                "data": { ... },
                "operation_summary": {
                    "merged": false,
                    "merge_status": "not_attempted",
                    "merge_error": null,
                    "operation_status": "success",
                    "target_entity": null,
                    "final_entity": "Tesla",
                    "renamed": false
                }
            }

        Example Request (rename with auto-merge):
            POST /graph/entity/edit
            {
                "entity_name": "Elon Msk",
                "updated_data": {
                    "entity_name": "Elon Musk",
                    "description": "Corrected description"
                },
                "allow_rename": true,
                "allow_merge": true
            }

        Example Response (merge success):
            {
                "status": "success",
                "message": "Entity merged successfully into 'Elon Musk'",
                "data": { ... },
                "operation_summary": {
                    "merged": true,
                    "merge_status": "success",
                    "merge_error": null,
                    "operation_status": "success",
                    "target_entity": "Elon Musk",
                    "final_entity": "Elon Musk",
                    "renamed": true
                }
            }

        Example Response (partial success - update succeeded but merge failed):
            {
                "status": "success",
                "message": "Entity updated successfully",
                "data": { ... },  # Data reflects updated "Elon Msk" entity
                "operation_summary": {
                    "merged": false,
                    "merge_status": "failed",
                    "merge_error": "Target entity locked by another operation",
                    "operation_status": "partial_success",
                    "target_entity": "Elon Musk",
                    "final_entity": "Elon Msk",  # Original entity still exists
                    "renamed": true
                }
            }
        """
        try:
            result = await rag.aedit_entity(
                entity_name=request.entity_name,
                updated_data=request.updated_data,
                allow_rename=request.allow_rename,
                allow_merge=request.allow_merge,
            )

            # Extract operation_summary from result, with fallback for backward compatibility
            operation_summary = result.get(
                'operation_summary',
                {
                    'merged': False,
                    'merge_status': 'not_attempted',
                    'merge_error': None,
                    'operation_status': 'success',
                    'target_entity': None,
                    'final_entity': request.updated_data.get('entity_name', request.entity_name),
                    'renamed': request.updated_data.get('entity_name', request.entity_name) != request.entity_name,
                },
            )

            # Separate entity data from operation_summary for clean response
            entity_data = dict(result)
            entity_data.pop('operation_summary', None)

            # Generate appropriate response message based on merge status
            response_message = (
                f"Entity merged successfully into '{operation_summary['final_entity']}'"
                if operation_summary.get('merged')
                else 'Entity updated successfully'
            )
            return {
                'status': 'success',
                'message': response_message,
                'data': entity_data,
                'operation_summary': operation_summary,
            }
        except ValueError as ve:
            logger.error(f"Validation error updating entity '{request.entity_name}': {ve!s}")
            raise HTTPException(status_code=400, detail=str(ve)) from ve
        except Exception as e:
            logger.error(f"Error updating entity '{request.entity_name}': {e!s}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f'Error updating entity: {e!s}') from e

    @router.post('/graph/relation/edit', dependencies=[Depends(combined_auth)])
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
                'status': 'success',
                'message': 'Relation updated successfully',
                'data': result,
            }
        except ValueError as ve:
            logger.error(
                f"Validation error updating relation between '{request.source_id}' and '{request.target_id}': {ve!s}"
            )
            raise HTTPException(status_code=400, detail=str(ve)) from ve
        except Exception as e:
            logger.error(f"Error updating relation between '{request.source_id}' and '{request.target_id}': {e!s}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f'Error updating relation: {e!s}') from e

    @router.post('/graph/entity/create', dependencies=[Depends(combined_auth)])
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
                - source_id (str): Related chunk_id from which the description originates
                - Additional custom properties as needed

        Response Schema:
            {
                "status": "success",
                "message": "Entity 'Tesla' created successfully",
                "data": {
                    "entity_name": "Tesla",
                    "description": "Electric vehicle manufacturer",
                    "entity_type": "ORGANIZATION",
                    "source_id": "chunk-123<SEP>chunk-456"
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
            # Use the proper acreate_entity method which handles:
            # - Graph lock for concurrency
            # - Vector embedding creation in entities_vdb
            # - Metadata population and defaults
            # - Index consistency via _edit_entity_done
            result = await rag.acreate_entity(
                entity_name=request.entity_name,
                entity_data=request.entity_data,
            )

            return {
                'status': 'success',
                'message': f"Entity '{request.entity_name}' created successfully",
                'data': result,
            }
        except ValueError as ve:
            logger.error(f"Validation error creating entity '{request.entity_name}': {ve!s}")
            raise HTTPException(status_code=400, detail=str(ve)) from ve
        except Exception as e:
            logger.error(f"Error creating entity '{request.entity_name}': {e!s}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f'Error creating entity: {e!s}') from e

    @router.post('/graph/relation/create', dependencies=[Depends(combined_auth)])
    async def create_relation(request: RelationCreateRequest):
        """
        Create a new relationship between two entities in the knowledge graph

        This endpoint establishes an undirected relationship between two existing entities.
        The provided source/target order is accepted for convenience, but the backend
        stored edge is undirected and may be returned with the entities swapped.
        Both entities must already exist in the knowledge graph. The system automatically
        generates vector embeddings for the relationship to enable semantic search and graph traversal.

        Prerequisites:
            - Both source_entity and target_entity must exist in the knowledge graph
            - Use /graph/entity/create to create entities first if they don't exist

        Request Body:
            source_entity (str): Name of the source entity (relationship origin)
            target_entity (str): Name of the target entity (relationship destination)
            relation_data (dict): Relationship properties including:
                - description (str): Textual description of the relationship
                - keywords (str): Comma-separated keywords describing the relationship type
                - source_id (str): Related chunk_id from which the description originates
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
                    "source_id": "chunk-123<SEP>chunk-456"
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
            # Use the proper acreate_relation method which handles:
            # - Graph lock for concurrency
            # - Entity existence validation
            # - Duplicate relation checks
            # - Vector embedding creation in relationships_vdb
            # - Index consistency via _edit_relation_done
            result = await rag.acreate_relation(
                source_entity=request.source_entity,
                target_entity=request.target_entity,
                relation_data=request.relation_data,
            )

            return {
                'status': 'success',
                'message': f"Relation created successfully between '{request.source_entity}' and '{request.target_entity}'",
                'data': result,
            }
        except ValueError as ve:
            logger.error(
                f"Validation error creating relation between '{request.source_entity}' and '{request.target_entity}': {ve!s}"
            )
            raise HTTPException(status_code=400, detail=str(ve)) from ve
        except Exception as e:
            logger.error(
                f"Error creating relation between '{request.source_entity}' and '{request.target_entity}': {e!s}"
            )
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f'Error creating relation: {e!s}') from e

    @router.post('/graph/entities/merge', dependencies=[Depends(combined_auth)])
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
                'status': 'success',
                'message': f"Successfully merged {len(request.entities_to_change)} entities into '{request.entity_to_change_into}'",
                'data': result,
            }
        except ValueError as ve:
            logger.error(
                f"Validation error merging entities {request.entities_to_change} into '{request.entity_to_change_into}': {ve!s}"
            )
            raise HTTPException(status_code=400, detail=str(ve)) from ve
        except Exception as e:
            logger.error(
                f"Error merging entities {request.entities_to_change} into '{request.entity_to_change_into}': {e!s}"
            )
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f'Error merging entities: {e!s}') from e

    @router.post('/graph/orphans/connect', dependencies=[Depends(combined_auth)])
    async def connect_orphan_entities(request: OrphanConnectionRequest):
        """
        Connect orphan entities (entities with no relationships) to the knowledge graph

        This endpoint identifies entities that have no connections (orphans) and attempts
        to find meaningful relationships using vector similarity and LLM validation.
        This helps improve graph connectivity and retrieval quality.

        The process:
            1. Identifies all orphan entities (entities with zero relationships)
            2. For each orphan, finds candidate connections using vector similarity
            3. Validates each candidate with LLM to ensure meaningful relationships
            4. Creates connections only for validated relationships above confidence threshold

        Request Body:
            max_candidates (int): Maximum candidates to evaluate per orphan (default: 3)
            similarity_threshold (float): Vector similarity threshold (0.0-1.0)
            confidence_threshold (float): LLM confidence required (0.0-1.0)
            cross_connect (bool): Allow orphan-to-orphan connections

        Response Schema:
            {
                "status": "success",
                "message": "Connected 15 out of 72 orphan entities",
                "data": {
                    "orphans_found": 72,
                    "connections_made": 15,
                    "connections": [
                        {
                            "orphan": "Amazon",
                            "connected_to": "E-Commerce",
                            "relationship_type": "categorical",
                            "keywords": "technology, retail",
                            "confidence": 0.85,
                            "similarity": 0.72
                        },
                        ...
                    ],
                    "errors": []
                }
            }

        HTTP Status Codes:
            200: Operation completed (check connections_made for results)
            500: Internal server error

        Note:
            - Requires PostgreSQL vector storage (PGVectorStorage)
            - LLM calls are made for each candidate, so cost scales with orphans × candidates
            - Only one connection is made per orphan (to the first valid candidate)
        """
        try:
            result = await rag.aconnect_orphan_entities(
                max_candidates=request.max_candidates,
                similarity_threshold=request.similarity_threshold,
                confidence_threshold=request.confidence_threshold,
                cross_connect=request.cross_connect,
            )

            return {
                'status': 'success',
                'message': f'Connected {result["connections_made"]} out of {result["orphans_found"]} orphan entities',
                'data': result,
            }
        except Exception as e:
            logger.error(f'Error connecting orphan entities: {e!s}')
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f'Error connecting orphan entities: {e!s}') from e

    @router.get(
        '/graph/orphans/status',
        response_model=OrphanConnectionStatusResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def get_orphan_connection_status():
        """
        Get current orphan connection pipeline status.

        Returns the real-time status of the orphan connection background pipeline,
        including progress, messages, and whether cancellation has been requested.

        This endpoint can be polled to monitor the progress of a running orphan
        connection job.

        Response Schema:
            {
                "busy": true,
                "job_name": "Connecting orphan entities",
                "job_start": "2024-01-15T10:30:00",
                "total_orphans": 100,
                "processed_orphans": 45,
                "connections_made": 12,
                "request_pending": false,
                "cancellation_requested": false,
                "latest_message": "[10:35:22] Processing orphan 46/100...",
                "history_messages": ["[10:30:00] Starting...", ...]
            }
        """
        try:
            from lightrag.kg.shared_storage import get_namespace_data

            status = await get_namespace_data('orphan_connection_status', workspace=rag.workspace)

            return OrphanConnectionStatusResponse(
                busy=status.get('busy', False),
                job_name=status.get('job_name', ''),
                job_start=status.get('job_start'),
                total_orphans=status.get('total_orphans', 0),
                processed_orphans=status.get('processed_orphans', 0),
                connections_made=status.get('connections_made', 0),
                request_pending=status.get('request_pending', False),
                cancellation_requested=status.get('cancellation_requested', False),
                latest_message=status.get('latest_message', ''),
                history_messages=list(status.get('history_messages', []))[-1000:],
            )
        except Exception as e:
            logger.error(f'Error getting orphan connection status: {e!s}')
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f'Error getting orphan connection status: {e!s}',
            ) from e

    @router.post('/graph/orphans/start', dependencies=[Depends(combined_auth)])
    async def start_orphan_connection_background(
        background_tasks: BackgroundTasks,
        max_candidates: int = Query(
            default=3,
            description='Maximum candidates to evaluate per entity',
            ge=1,
            le=10,
        ),
        max_degree: int = Query(
            default=0,
            description='Maximum connection degree to target. 0=orphans only, 1=include leaf nodes, 2+=include sparse nodes',
            ge=0,
            le=5,
        ),
    ):
        """
        Start orphan/sparse entity connection as a background job.

        This endpoint starts the connection process as a background task
        that runs independently from the document processing pipeline. Progress
        can be monitored via the /graph/orphans/status endpoint.

        The job will:
            1. Find all target entities (based on max_degree setting)
            2. Process each entity to find connection candidates
            3. Validate candidates with LLM
            4. Create connections for validated relationships
            5. Update progress in real-time

        Query Parameters:
            max_candidates (int): Maximum candidates per entity (default: 3)
            max_degree (int): Maximum connection degree to target (default: 0)
                - 0: True orphans only (entities with no connections)
                - 1: Orphans + leaf nodes (entities with 0-1 connections)
                - 2+: Include sparsely connected nodes

        Response:
            {"status": "started"} - Job was started
            {"status": "already_running"} - A job is already in progress

        Note:
            - Poll /graph/orphans/status to monitor progress
            - Use /graph/orphans/cancel to request cancellation
        """
        try:
            from lightrag.kg.shared_storage import get_namespace_data

            # Check if already running
            status = await get_namespace_data('orphan_connection_status', workspace=rag.workspace)
            if status.get('busy'):
                return {'status': 'already_running'}

            # Start background task
            background_tasks.add_task(
                rag.aprocess_orphan_connections_background,
                max_candidates=max_candidates,
                max_degree=max_degree,
            )

            return {'status': 'started'}
        except Exception as e:
            logger.error(f'Error starting orphan connection: {e!s}')
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f'Error starting orphan connection: {e!s}',
            ) from e

    @router.post('/graph/orphans/cancel', dependencies=[Depends(combined_auth)])
    async def cancel_orphan_connection():
        """
        Request cancellation of a running orphan connection job.

        This endpoint sets a flag that the background job checks periodically.
        Cancellation is graceful - the job will stop at the next checkpoint
        (after completing the current orphan).

        Response:
            {"status": "cancellation_requested"} - Flag was set
            {"status": "not_running"} - No job is currently running
        """
        try:
            from lightrag.kg.shared_storage import get_namespace_data, get_namespace_lock

            status = await get_namespace_data('orphan_connection_status', workspace=rag.workspace)
            lock = get_namespace_lock('orphan_connection_status', workspace=rag.workspace)

            async with lock:
                if not status.get('busy'):
                    return {'status': 'not_running'}
                status['cancellation_requested'] = True

            return {'status': 'cancellation_requested'}
        except Exception as e:
            logger.error(f'Error cancelling orphan connection: {e!s}')
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f'Error cancelling orphan connection: {e!s}',
            ) from e

    return router
