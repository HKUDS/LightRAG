"""
This module contains all graph-related routes for the LightRAG API.
"""

from typing import Optional, Dict, Any, Literal
import traceback
from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from lightrag.api.graph_workbench import (
    get_legacy_graph_payload,
    query_graph_workbench,
)
from lightrag.utils import logger
from ..utils_api import get_combined_auth_dependency

router = APIRouter(tags=["graph"])


class EntityUpdateRequest(BaseModel):
    entity_name: str
    updated_data: Dict[str, Any]
    allow_rename: bool = False
    allow_merge: bool = False


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


class GraphQueryScope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(default="*", min_length=1)
    max_depth: int = Field(default=3, ge=1)
    max_nodes: int = Field(default=1000, ge=1)
    only_matched_neighborhood: bool = False

    @field_validator("label", mode="after")
    @classmethod
    def validate_label(cls, label: str) -> str:
        normalized = label.strip()
        if not normalized:
            raise ValueError("label cannot be empty")
        return normalized


class GraphNodeFiltersV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_types: list[str] = Field(default_factory=list)
    name_query: str = ""
    description_query: str = ""
    degree_min: Optional[int] = Field(default=None, ge=0)
    degree_max: Optional[int] = Field(default=None, ge=0)
    isolated_only: bool = False


class GraphEdgeFiltersV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relation_types: list[str] = Field(default_factory=list)
    keyword_query: str = ""
    weight_min: Optional[float] = Field(default=None, ge=0)
    weight_max: Optional[float] = Field(default=None, ge=0)
    source_entity_types: list[str] = Field(default_factory=list)
    target_entity_types: list[str] = Field(default_factory=list)


class GraphSourceFiltersV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id_query: str = ""
    file_paths: list[str] = Field(default_factory=list)
    time_from: Optional[str] = None
    time_to: Optional[str] = None


class GraphViewOptionsV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    show_nodes_only: bool = False
    show_edges_only: bool = False
    hide_low_weight_edges: bool = False
    hide_empty_description: bool = False
    highlight_matches: bool = False


class GraphQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: GraphQueryScope
    node_filters: GraphNodeFiltersV1 = Field(default_factory=GraphNodeFiltersV1)
    edge_filters: GraphEdgeFiltersV1 = Field(default_factory=GraphEdgeFiltersV1)
    source_filters: GraphSourceFiltersV1 = Field(default_factory=GraphSourceFiltersV1)
    view_options: GraphViewOptionsV1 = Field(default_factory=GraphViewOptionsV1)


class GraphQueryData(BaseModel):
    model_config = ConfigDict(extra="allow")

    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)
    is_truncated: bool = False


class GraphQueryTruncation(BaseModel):
    requested_max_nodes: int
    effective_max_nodes: int
    was_truncated_before_filtering: bool
    was_truncated_after_filtering: bool


class GraphQueryFilterSemantics(BaseModel):
    group_operator: Literal["AND"] = "AND"
    field_operator: Literal["AND"] = "AND"
    array_operator: Literal["OR"] = "OR"
    version: Literal["v1"] = "v1"


class GraphQueryMeta(BaseModel):
    filter_semantics: GraphQueryFilterSemantics = Field(
        default_factory=GraphQueryFilterSemantics
    )
    execution_mode: Literal["base_graph_only_placeholder"] = (
        "base_graph_only_placeholder"
    )
    filtering_applied: bool = False
    ignored_filter_groups: list[str] = Field(
        default_factory=lambda: [
            "node_filters",
            "edge_filters",
            "source_filters",
            "view_options",
        ]
    )


class GraphQueryResponse(BaseModel):
    data: GraphQueryData
    truncation: GraphQueryTruncation
    meta: GraphQueryMeta = Field(default_factory=GraphQueryMeta)


class GraphDeleteEntityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_name: str = Field(..., min_length=1)

    @field_validator("entity_name", mode="after")
    @classmethod
    def validate_entity_name(cls, entity_name: str) -> str:
        normalized = entity_name.strip()
        if not normalized:
            raise ValueError("entity_name cannot be empty")
        return normalized


class GraphDeleteRelationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_entity: str = Field(..., min_length=1)
    target_entity: str = Field(..., min_length=1)

    @field_validator("source_entity", "target_entity", mode="after")
    @classmethod
    def validate_relation_entity_name(cls, entity_name: str) -> str:
        normalized = entity_name.strip()
        if not normalized:
            raise ValueError("entity_name cannot be empty")
        return normalized


class GraphDeletionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    status: Literal["success", "not_found", "not_allowed", "fail"]
    doc_id: str
    message: str
    status_code: int = 200
    file_path: Optional[str] = None


class GraphMergeSuggestionReason(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    score: float


class GraphMergeSuggestionCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_entity: str
    source_entities: list[str] = Field(min_length=1)
    score: float
    reasons: list[GraphMergeSuggestionReason] = Field(default_factory=list)


class GraphMergeSuggestionsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: GraphQueryScope
    limit: int = Field(default=20, ge=1, le=200)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)


class GraphMergeSuggestionsMeta(BaseModel):
    strategy: str = "placeholder_v1"
    requested_limit: int
    min_score: float
    returned_candidates: int


class GraphMergeSuggestionsResponse(BaseModel):
    candidates: list[GraphMergeSuggestionCandidate] = Field(default_factory=list)
    meta: GraphMergeSuggestionsMeta


def _normalize_deletion_response(raw_result: Any) -> GraphDeletionResponse:
    deletion = GraphDeletionResponse.model_validate(raw_result, from_attributes=True)
    deletion.doc_id = ""
    return deletion


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
            normalized_label = label.strip()
            if not normalized_label:
                raise HTTPException(status_code=422, detail="label cannot be empty")

            # Log the label parameter to check for leading spaces
            logger.debug(
                f"get_knowledge_graph called with label: '{label}' (length: {len(label)}, repr: {repr(label)})"
            )

            return await get_legacy_graph_payload(
                rag=rag,
                label=normalized_label,
                max_depth=max_depth,
                max_nodes=max_nodes,
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting knowledge graph for label '{label}': {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Error getting knowledge graph: {str(e)}"
            )

    @router.post(
        "/graph/query",
        response_model=GraphQueryResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def query_graph(request: GraphQueryRequest):
        """Structured graph query contract (v1)."""
        try:
            response_payload = await query_graph_workbench(
                rag=rag,
                request=request.model_dump(),
            )
            return GraphQueryResponse.model_validate(response_payload)
        except Exception as e:
            logger.error(f"Error querying graph with structured request: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Error querying graph: {str(e)}"
            )

    @router.delete(
        "/graph/entity",
        response_model=GraphDeletionResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def delete_graph_entity(request: GraphDeleteEntityRequest):
        try:
            raw_result = await rag.adelete_by_entity(entity_name=request.entity_name)
            result = _normalize_deletion_response(raw_result)
            if result.status == "not_found":
                raise HTTPException(status_code=404, detail=result.message)
            if result.status == "not_allowed":
                raise HTTPException(status_code=403, detail=result.message)
            if result.status == "fail":
                raise HTTPException(status_code=500, detail=result.message)
            return result
        except HTTPException:
            raise
        except Exception as e:
            error_msg = f"Error deleting graph entity '{request.entity_name}': {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=error_msg)

    @router.delete(
        "/graph/relation",
        response_model=GraphDeletionResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def delete_graph_relation(request: GraphDeleteRelationRequest):
        try:
            raw_result = await rag.adelete_by_relation(
                source_entity=request.source_entity,
                target_entity=request.target_entity,
            )
            result = _normalize_deletion_response(raw_result)
            if result.status == "not_found":
                raise HTTPException(status_code=404, detail=result.message)
            if result.status == "not_allowed":
                raise HTTPException(status_code=403, detail=result.message)
            if result.status == "fail":
                raise HTTPException(status_code=500, detail=result.message)
            return result
        except HTTPException:
            raise
        except Exception as e:
            error_msg = (
                f"Error deleting graph relation from '{request.source_entity}' to "
                f"'{request.target_entity}': {str(e)}"
            )
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=error_msg)

    @router.post(
        "/graph/merge/suggestions",
        response_model=GraphMergeSuggestionsResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def graph_merge_suggestions(request: GraphMergeSuggestionsRequest):
        try:
            if not hasattr(rag, "aget_merge_suggestions"):
                raise HTTPException(
                    status_code=501,
                    detail="Merge suggestions not implemented by current backend",
                )

            request_payload = request.model_dump()
            raw_candidates: list[dict[str, Any]] = await rag.aget_merge_suggestions(
                request_payload
            )

            candidates = [
                GraphMergeSuggestionCandidate.model_validate(candidate)
                for candidate in raw_candidates
            ]
            filtered_candidates = [
                candidate
                for candidate in candidates
                if candidate.score >= request.min_score
            ][: request.limit]

            return GraphMergeSuggestionsResponse(
                candidates=filtered_candidates,
                meta=GraphMergeSuggestionsMeta(
                    requested_limit=request.limit,
                    min_score=request.min_score,
                    returned_candidates=len(filtered_candidates),
                ),
            )
        except NotImplementedError as e:
            raise HTTPException(
                status_code=501,
                detail=f"Merge suggestions not implemented: {str(e)}",
            ) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error generating merge suggestions: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500, detail=f"Error generating merge suggestions: {str(e)}"
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
                "operation_summary",
                {
                    "merged": False,
                    "merge_status": "not_attempted",
                    "merge_error": None,
                    "operation_status": "success",
                    "target_entity": None,
                    "final_entity": request.updated_data.get(
                        "entity_name", request.entity_name
                    ),
                    "renamed": request.updated_data.get(
                        "entity_name", request.entity_name
                    )
                    != request.entity_name,
                },
            )

            # Separate entity data from operation_summary for clean response
            entity_data = dict(result)
            entity_data.pop("operation_summary", None)

            # Generate appropriate response message based on merge status
            response_message = (
                f"Entity merged successfully into '{operation_summary['final_entity']}'"
                if operation_summary.get("merged")
                else "Entity updated successfully"
            )
            return {
                "status": "success",
                "message": response_message,
                "data": entity_data,
                "operation_summary": operation_summary,
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
                "status": "success",
                "message": f"Entity '{request.entity_name}' created successfully",
                "data": result,
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
                "status": "success",
                "message": f"Relation created successfully between '{request.source_entity}' and '{request.target_entity}'",
                "data": result,
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
