from fastapi import APIRouter
import os
from ..rag_manager import rag_manager

router = APIRouter(tags=["health"])

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "working_directory": os.getcwd(),
        "input_directory": "",
        "configuration": {
             "llm_binding": "multi-tenant",
             "llm_model": "multi-tenant",
             "embedding_binding": "multi-tenant",
             "embedding_model": "multi-tenant",
             
             # Essential fields for frontend type safety
             "llm_binding_host": "",
             "embedding_binding_host": "",
             "kv_storage": "sqlite",
             "doc_status_storage": "sqlite",
             "graph_storage": "neo4j/networkx",
             "vector_storage": "nano",
             "summary_language": "en",
             "force_llm_summary_on_merge": False,
             "max_parallel_insert": 1,
             "max_async": 4,
             "embedding_func_max_async": 4,
             "embedding_batch_num": 32,
             "cosine_threshold": 0.75,
             "min_rerank_score": 0.35,
             "related_chunk_number": 5
        },
        "description": "Multi-Tenant LightRAG",
        "pipeline_busy": False
    }
