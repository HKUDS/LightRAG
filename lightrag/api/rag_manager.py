import os
import asyncio
from typing import Dict, Optional
from lightrag import LightRAG
from lightrag.utils import logger
from .config import global_args
from .llm_factory import (
    create_llm_model_func, 
    create_llm_model_kwargs, 
    create_optimized_embedding_function, 
    create_server_rerank_func,
    LLMConfigCache
)
from lightrag.utils import get_env_value

# Use a Singleton pattern for the Manager

class RAGManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGManager, cls).__new__(cls)
            cls._instance.instances: Dict[str, LightRAG] = {}
            cls._instance.config_cache = LLMConfigCache(global_args)
            cls._instance.lock = asyncio.Lock()
        return cls._instance

    async def get_rag(self, workspace: str) -> LightRAG:
        """
        Get or create a LightRAG instance for the specific workspace (Org).
        """
        if not workspace:
            workspace = "default"
            
        async with self.lock:
            if workspace in self.instances:
                return self.instances[workspace]
            
            logger.info(f"Initializing LightRAG instance for workspace: {workspace}")
            
            # Re-use logic from create_app in lightrag_server.py
            args = global_args
            
            # Helper logic from server (replicated here or refactored)
            # We reference the global config but override workspace
            
            # 1. LLM Model Func
            llm_binding = args.llm_binding
            llm_timeout = get_env_value("LLM_TIMEOUT", 60, int) # Default fallback
            embedding_timeout = get_env_value("EMBEDDING_TIMEOUT", 60, int)

            # Note: We need to import the creator functions. 
            # Ideally lightrag_server should expose them cleanly.
            # I will assume we can import them from .lightrag_server as done in imports
            
            config_cache = self.config_cache
            
            # Create Embedding Func
            embedding_func = create_optimized_embedding_function(
                config_cache=config_cache,
                binding=args.embedding_binding,
                model=args.embedding_model,
                host=args.embedding_binding_host,
                api_key=args.embedding_binding_api_key,
                args=args,
            )
            
            # Send dimensions logic (replicated from server)
            import inspect
            sig = inspect.signature(embedding_func.func)
            has_embedding_dim_param = "embedding_dim" in sig.parameters
            embedding_send_dim = args.embedding_send_dim
            
            if args.embedding_binding in ["jina", "gemini"]:
                 embedding_func.send_dimensions = has_embedding_dim_param
            else:
                 embedding_func.send_dimensions = embedding_send_dim and has_embedding_dim_param

            # Rerank Func
            # We need to recreate the rerank function logic or extract it.
            # For simplicity, we assume create_optimized_rerank_func exists or we replicate it.
            # Wait, I didn't see `create_optimized_rerank_func` in `lightrag_server.py` in previous `view_file`.
            # Use query in lightrag_server.py again if needed, or better, implement the logic here
            
            # Re-implementing rerank logic briefly to avoid import issues if function not exposed
            rerank_model_func = None
            if args.rerank_binding != "null":
                try:
                    rerank_model_func = create_server_rerank_func(args)
                except Exception as e:
                     logger.warning(f"Failed to create rerank function: {e}") 

            # Ollama Info
            from lightrag.api.config import OllamaServerInfos
            ollama_server_infos = OllamaServerInfos(
                name=args.simulated_model_name, tag=args.simulated_model_tag
            )

            try:
                rag = LightRAG(
                    working_dir=args.working_dir, # This is base dir
                    workspace=workspace,          # THIS IS THE KEY CHANGE
                    llm_model_func=create_llm_model_func(
                        args.llm_binding, args, config_cache, llm_timeout
                    ),
                    llm_model_name=args.llm_model,
                    llm_model_max_async=args.max_async,
                    summary_max_tokens=args.summary_max_tokens,
                    summary_context_size=args.summary_context_size,
                    chunk_token_size=int(args.chunk_size),
                    chunk_overlap_token_size=int(args.chunk_overlap_size),
                    llm_model_kwargs=create_llm_model_kwargs(
                        args.llm_binding, args, llm_timeout
                    ),
                    embedding_func=embedding_func,
                    default_llm_timeout=llm_timeout,
                    default_embedding_timeout=embedding_timeout,
                    kv_storage=args.kv_storage,
                    graph_storage=args.graph_storage,
                    vector_storage=args.vector_storage,
                    doc_status_storage=args.doc_status_storage,
                    vector_db_storage_cls_kwargs={
                        "cosine_better_than_threshold": args.cosine_threshold
                    },
                    enable_llm_cache_for_entity_extract=args.enable_llm_cache_for_extract,
                    enable_llm_cache=args.enable_llm_cache,
                    rerank_model_func=rerank_model_func,
                    max_parallel_insert=args.max_parallel_insert,
                    max_graph_nodes=args.max_graph_nodes,
                    addon_params={
                        "language": args.summary_language,
                        "entity_types": args.entity_types,
                    },
                    ollama_server_infos=ollama_server_infos,
                )
                
                # Initialize Storages
                await rag.initialize_storages()
                
                self.instances[workspace] = rag
                return rag
                
            except Exception as e:
                logger.error(f"Failed to initialize LightRAG for workspace {workspace}: {e}")
                raise

rag_manager = RAGManager()
