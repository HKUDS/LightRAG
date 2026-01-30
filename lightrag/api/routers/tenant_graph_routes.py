from typing import Any, Dict, List, Optional
import traceback
from fastapi import APIRouter, Depends, Query, HTTPException
from lightrag import LightRAG
from lightrag.utils import logger
from lightrag.api.routers.graph_routes import (
    EntityUpdateRequest, RelationUpdateRequest, 
    EntityMergeRequest, EntityCreateRequest, RelationCreateRequest
)
from ..dependencies import get_current_rag

router = APIRouter(tags=["graph"])

@router.get("/graph/label/list")
async def get_graph_labels(rag: LightRAG = Depends(get_current_rag)):
    try:
        return await rag.get_graph_labels()
    except Exception as e:
        logger.error(f"Error getting graph labels: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/graph/label/popular")
async def get_popular_labels(
    limit: int = Query(300, ge=1, le=1000),
    rag: LightRAG = Depends(get_current_rag)
):
    try:
        return await rag.chunk_entity_relation_graph.get_popular_labels(limit)
    except Exception as e:
        logger.error(f"Error getting popular labels: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/graph/label/search")
async def search_labels(
    q: str = Query(...),
    limit: int = Query(50, ge=1, le=100),
    rag: LightRAG = Depends(get_current_rag)
):
    try:
        return await rag.chunk_entity_relation_graph.search_labels(q, limit)
    except Exception as e:
        logger.error(f"Error searching labels: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/graphs")
async def get_knowledge_graph(
    label: str = Query(...),
    max_depth: int = Query(3, ge=1),
    max_nodes: int = Query(1000, ge=1),
    rag: LightRAG = Depends(get_current_rag)
):
    try:
        return await rag.get_knowledge_graph(
            node_label=label,
            max_depth=max_depth,
            max_nodes=max_nodes
        )
    except Exception as e:
        logger.error(f"Error getting knowledge graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/graph/entity/exists")
async def check_entity_exists(
    name: str = Query(...),
    rag: LightRAG = Depends(get_current_rag)
):
    try:
        exists = await rag.chunk_entity_relation_graph.has_node(name)
        return {"exists": exists}
    except Exception as e:
        logger.error(f"Error checking entity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/graph/entity/edit")
async def update_entity(
    request: EntityUpdateRequest,
    rag: LightRAG = Depends(get_current_rag)
):
    try:
        result = await rag.aedit_entity(
            entity_name=request.entity_name,
            updated_data=request.updated_data,
            allow_rename=request.allow_rename,
            allow_merge=request.allow_merge
        )
        # Assuming simplified response or mirroring full logic?
        # Mirroring minimal necessary for success, as UI likely depends on structure
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Error updating entity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/graph/relation/edit")
async def update_relation(
    request: RelationUpdateRequest,
    rag: LightRAG = Depends(get_current_rag)
):
    try:
        result = await rag.aedit_relation(
            source_entity=request.source_id,
            target_entity=request.target_id,
            updated_data=request.updated_data
        )
        return {"status": "success", "data": result}
    except Exception as e:
         logger.error(f"Error updating relation: {e}")
         raise HTTPException(status_code=500, detail=str(e))

@router.post("/graph/entity/create")
async def create_entity(
    request: EntityCreateRequest,
    rag: LightRAG = Depends(get_current_rag)
):
    try:
        result = await rag.acreate_entity(
            entity_name=request.entity_name,
            entity_data=request.entity_data
        )
        return {"status": "success", "data": result}
    except Exception as e:
         logger.error(f"Error creating entity: {e}")
         raise HTTPException(status_code=500, detail=str(e))

@router.post("/graph/relation/create")
async def create_relation(
    request: RelationCreateRequest,
    rag: LightRAG = Depends(get_current_rag)
):
    try:
        result = await rag.acreate_relation(
            source_entity=request.source_entity,
            target_entity=request.target_entity,
            relation_data=request.relation_data
        )
        return {"status": "success", "data": result}
    except Exception as e:
         logger.error(f"Error creating relation: {e}")
         raise HTTPException(status_code=500, detail=str(e))

@router.post("/graph/entities/merge")
async def merge_entities(
    request: EntityMergeRequest,
    rag: LightRAG = Depends(get_current_rag)
):
    try:
        result = await rag.amerge_entities(
            source_entities=request.entities_to_change,
            target_entity=request.entity_to_change_into
        )
        return {"status": "success", "data": result}
    except Exception as e:
         logger.error(f"Error merging entities: {e}")
         raise HTTPException(status_code=500, detail=str(e))
