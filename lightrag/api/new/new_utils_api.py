import argparse
import base64
import logging
from urllib.parse import unquote
import os
from pathlib import Path
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from lightrag.api.new.new_kg.new_lightrag import NewLightRAG
from lightrag.api.utils_api import (
    DefaultRAGStorageConfig,
    OllamaServerInfos,
    get_default_host,
    get_env_value,
)
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.types import GPTKeywordExtractionFormat
from lightrag.utils import EmbeddingFunc

ollama_server_infos = OllamaServerInfos()


def parse_args(is_uvicorn_mode: bool = False) -> argparse.Namespace:
    """
    Parse command line arguments with environment variable fallback

    Args:
        is_uvicorn_mode: Whether running under uvicorn mode

    Returns:
        argparse.Namespace: Parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="LightRAG FastAPI Server with separate working and input directories"
    )

    # Server configuration
    parser.add_argument(
        "--host",
        default=get_env_value("HOST", "0.0.0.0"),
        help="Server host (default: from env or 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=get_env_value("PORT", 9621, int),
        help="Server port (default: from env or 9621)",
    )

    # Directory configuration
    parser.add_argument(
        "--working-dir",
        default=get_env_value("WORKING_DIR", "./rag_storage"),
        help="Working directory for RAG storage (default: from env or ./rag_storage)",
    )
    parser.add_argument(
        "--input-dir",
        default=get_env_value("INPUT_DIR", "./inputs"),
        help="Directory containing input documents (default: from env or ./inputs)",
    )

    def timeout_type(value):
        if value is None:
            return 150
        if value is None or value == "None":
            return None
        return int(value)

    parser.add_argument(
        "--timeout",
        default=get_env_value("TIMEOUT", None, timeout_type),
        type=timeout_type,
        help="Timeout in seconds (useful when using slow AI). Use None for infinite timeout",
    )

    # RAG configuration
    parser.add_argument(
        "--max-async",
        type=int,
        default=get_env_value("MAX_ASYNC", 4, int),
        help="Maximum async operations (default: from env or 4)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=get_env_value("MAX_TOKENS", 32768, int),
        help="Maximum token size (default: from env or 32768)",
    )

    # Logging configuration
    parser.add_argument(
        "--log-level",
        default=get_env_value("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: from env or INFO)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=get_env_value("VERBOSE", False, bool),
        help="Enable verbose debug output(only valid for DEBUG log-level)",
    )

    parser.add_argument(
        "--key",
        type=str,
        default=get_env_value("LIGHTRAG_API_KEY", None),
        help="API key for authentication. This protects lightrag server against unauthorized access",
    )

    # Optional https parameters
    parser.add_argument(
        "--ssl",
        action="store_true",
        default=get_env_value("SSL", False, bool),
        help="Enable HTTPS (default: from env or False)",
    )
    parser.add_argument(
        "--ssl-certfile",
        default=get_env_value("SSL_CERTFILE", None),
        help="Path to SSL certificate file (required if --ssl is enabled)",
    )
    parser.add_argument(
        "--ssl-keyfile",
        default=get_env_value("SSL_KEYFILE", None),
        help="Path to SSL private key file (required if --ssl is enabled)",
    )

    parser.add_argument(
        "--history-turns",
        type=int,
        default=get_env_value("HISTORY_TURNS", 3, int),
        help="Number of conversation history turns to include (default: from env or 3)",
    )

    # Search parameters
    parser.add_argument(
        "--top-k",
        type=int,
        default=get_env_value("TOP_K", 60, int),
        help="Number of most similar results to return (default: from env or 60)",
    )
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=get_env_value("COSINE_THRESHOLD", 0.2, float),
        help="Cosine similarity threshold (default: from env or 0.4)",
    )

    # Ollama model name
    parser.add_argument(
        "--simulated-model-name",
        type=str,
        default=get_env_value(
            "SIMULATED_MODEL_NAME", ollama_server_infos.LIGHTRAG_MODEL
        ),
        help="Number of conversation history turns to include (default: from env or 3)",
    )

    # Namespace
    parser.add_argument(
        "--namespace-prefix",
        type=str,
        default=get_env_value("NAMESPACE_PREFIX", ""),
        help="Prefix of the namespace",
    )

    parser.add_argument(
        "--auto-scan-at-startup",
        action="store_true",
        default=False,
        help="Enable automatic scanning when the program starts",
    )

    # LLM and embedding bindings
    parser.add_argument(
        "--llm-binding",
        type=str,
        default=get_env_value("LLM_BINDING", "ollama"),
        choices=["lollms", "ollama", "openai", "openai-ollama", "azure_openai"],
        help="LLM binding type (default: from env or ollama)",
    )
    parser.add_argument(
        "--embedding-binding",
        type=str,
        default=get_env_value("EMBEDDING_BINDING", "ollama"),
        choices=["lollms", "ollama", "openai", "azure_openai"],
        help="Embedding binding type (default: from env or ollama)",
    )

    # Inject server configuration
    parser.add_argument(
        "--workers",
        default=get_env_value("SERVER_BIND_WORKERS", 1, int),
        type=int,
        help="Number of worker processes to use for inference (default: from env or 1)",
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        default=get_env_value("SERVER_BIND_RELOAD", False, bool),
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()
    # If in uvicorn mode and workers > 1, force it to 1 and log warning
    if is_uvicorn_mode and args.workers > 1:
        original_workers = args.workers
        args.workers = 1
        # Log warning directly here
        logging.warning(
            f"In uvicorn mode, workers parameter was set to {original_workers}. Forcing workers=1"
        )
    # convert relative path to absolute path
    args.working_dir = os.path.abspath(args.working_dir)
    args.input_dir = os.path.abspath(args.input_dir)

    # Inject storage configuration from environment variables
    args.kv_storage = get_env_value(
        "LIGHTRAG_KV_STORAGE", DefaultRAGStorageConfig.KV_STORAGE
    )
    args.doc_status_storage = get_env_value(
        "LIGHTRAG_DOC_STATUS_STORAGE", DefaultRAGStorageConfig.DOC_STATUS_STORAGE
    )
    args.graph_storage = get_env_value(
        "LIGHTRAG_GRAPH_STORAGE", DefaultRAGStorageConfig.GRAPH_STORAGE
    )
    args.vector_storage = get_env_value(
        "LIGHTRAG_VECTOR_STORAGE", DefaultRAGStorageConfig.VECTOR_STORAGE
    )

    # Handle openai-ollama special case
    if args.llm_binding == "openai-ollama":
        args.llm_binding = "openai"
        args.embedding_binding = "ollama"

    args.llm_binding_host = get_env_value(
        "LLM_BINDING_HOST", get_default_host(args.llm_binding)
    )
    args.embedding_binding_host = get_env_value(
        "EMBEDDING_BINDING_HOST", get_default_host(args.embedding_binding)
    )
    args.llm_binding_api_key = get_env_value("LLM_BINDING_API_KEY", None)
    args.embedding_binding_api_key = get_env_value("EMBEDDING_BINDING_API_KEY", "")

    # Inject model configuration
    args.llm_model = get_env_value("LLM_MODEL", "mistral-nemo:latest")
    args.embedding_model = get_env_value("EMBEDDING_MODEL", "bge-m3:latest")
    args.embedding_dim = get_env_value("EMBEDDING_DIM", 1024, int)
    args.max_embed_tokens = get_env_value("MAX_EMBED_TOKENS", 8192, int)

    # Inject chunk configuration
    args.chunk_size = get_env_value("CHUNK_SIZE", 1200, int)
    args.chunk_overlap_size = get_env_value("CHUNK_OVERLAP_SIZE", 100, int)

    ollama_server_infos.LIGHTRAG_MODEL = args.simulated_model_name

    # addon params
    args.example_number = int(os.getenv("ADDON_PARAMS_EXAMPLE_NUMBER", 1))
    args.language = os.getenv("ADDON_PARAMS_LANGUAGE", "Simplfied Chinese")
    args.enable_llm_cache_for_entity_extract = os.getenv(
        "ENABLE_LLM_CACHE_FOR_ENTITY_EXTRACT", False
    )
    args.enable_llm_cache = bool(os.getenv("ENABLE_LLM_CACHE", True))

    return args


async def get_rag(args, workspace_path):
    if not workspace_path:
        return None
    if not args:
        return None
    if args.llm_binding == "lollms" or args.embedding_binding == "lollms":
        from lightrag.llm.lollms import lollms_model_complete, lollms_embed
    if args.llm_binding == "ollama" or args.embedding_binding == "ollama":
        from lightrag.llm.ollama import ollama_model_complete, ollama_embed
    if args.llm_binding == "openai" or args.embedding_binding == "openai":
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    if args.llm_binding == "azure_openai" or args.embedding_binding == "azure_openai":
        from lightrag.llm.azure_openai import (
            azure_openai_complete_if_cache,
            azure_openai_embed,
        )
    if args.llm_binding_host == "openai-ollama" or args.embedding_binding == "ollama":
        from lightrag.llm.openai import openai_complete_if_cache
        from lightrag.llm.ollama import ollama_embed

    async def openai_alike_model_complete(
        prompt,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ) -> str:
        keyword_extraction = kwargs.pop("keyword_extraction", None)
        if keyword_extraction:
            kwargs["response_format"] = GPTKeywordExtractionFormat
        if history_messages is None:
            history_messages = []
        return await openai_complete_if_cache(
            args.llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=args.llm_binding_host,
            api_key=args.llm_binding_api_key,
            **kwargs,
        )

    async def azure_openai_model_complete(
        prompt,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ) -> str:
        keyword_extraction = kwargs.pop("keyword_extraction", None)
        if keyword_extraction:
            kwargs["response_format"] = GPTKeywordExtractionFormat
        if history_messages is None:
            history_messages = []
        return await azure_openai_complete_if_cache(
            args.llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=args.llm_binding_host,
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
            **kwargs,
        )

    embedding_func = EmbeddingFunc(
        embedding_dim=args.embedding_dim,
        max_token_size=args.max_embed_tokens,
        func=lambda texts: (
            lollms_embed(
                texts,
                embed_model=args.embedding_model,
                host=args.embedding_binding_host,
                api_key=args.embedding_binding_api_key,
            )
            if args.embedding_binding == "lollms"
            else (
                ollama_embed(
                    texts,
                    embed_model=args.embedding_model,
                    host=args.embedding_binding_host,
                    api_key=args.embedding_binding_api_key,
                )
                if args.embedding_binding == "ollama"
                else (
                    azure_openai_embed(
                        texts,
                        model=args.embedding_model,  # no host is used for openai,
                        api_key=args.embedding_binding_api_key,
                    )
                    if args.embedding_binding == "azure_openai"
                    else openai_embed(
                        texts,
                        model=args.embedding_model,
                        base_url=args.embedding_binding_host,
                        api_key=args.embedding_binding_api_key,
                    )
                )
            )
        ),
    )
    # Initialize RAG
    if args.llm_binding in ["lollms", "ollama", "openai-ollama"]:
        rag = NewLightRAG(
            working_dir=workspace_path,
            llm_model_func=(
                lollms_model_complete
                if args.llm_binding == "lollms"
                else (
                    ollama_model_complete
                    if args.llm_binding == "ollama"
                    else openai_alike_model_complete
                )
            ),
            llm_model_name=args.llm_model,
            llm_model_max_async=args.max_async,
            llm_model_max_token_size=args.max_tokens,
            chunk_token_size=int(args.chunk_size),
            chunk_overlap_token_size=int(args.chunk_overlap_size),
            llm_model_kwargs=(
                {
                    "host": args.llm_binding_host,
                    "timeout": args.timeout,
                    "options": {"num_ctx": args.max_tokens},
                    "api_key": args.llm_binding_api_key,
                }
                if args.llm_binding == "lollms" or args.llm_binding == "ollama"
                else {}
            ),
            embedding_func=embedding_func,
            kv_storage=args.kv_storage,
            graph_storage=args.graph_storage,
            vector_storage=args.vector_storage,
            doc_status_storage=args.doc_status_storage,
            vector_db_storage_cls_kwargs={
                "cosine_better_than_threshold": args.cosine_threshold
            },
            enable_llm_cache_for_entity_extract=args.enable_llm_cache_for_entity_extract,  # set to True for debuging to reduce llm fee
            embedding_cache_config={
                "enabled": True,
                "similarity_threshold": 0.95,
                "use_llm_check": False,
            },
            log_level=args.log_level,
            namespace_prefix=args.namespace_prefix,
            addon_params={
                "example_number": args.example_number,
                "language": args.language,
            },
            enable_llm_cache=args.enable_llm_cache,
            auto_manage_storages_states=False,
        )
    else:
        rag = NewLightRAG(
            working_dir=workspace_path,
            llm_model_func=azure_openai_model_complete,
            chunk_token_size=int(args.chunk_size),
            chunk_overlap_token_size=int(args.chunk_overlap_size),
            llm_model_kwargs={
                "timeout": args.timeout,
            },
            llm_model_name=args.llm_model,
            llm_model_max_async=args.max_async,
            llm_model_max_token_size=args.max_tokens,
            embedding_func=embedding_func,
            kv_storage=args.kv_storage,
            graph_storage=args.graph_storage,
            vector_storage=args.vector_storage,
            doc_status_storage=args.doc_status_storage,
            vector_db_storage_cls_kwargs={
                "cosine_better_than_threshold": args.cosine_threshold
            },
            enable_llm_cache_for_entity_extract=args.enable_llm_cache_for_entity_extract,  # set to True for debuging to reduce llm fee
            embedding_cache_config={
                "enabled": True,
                "similarity_threshold": 0.95,
                "use_llm_check": False,
            },
            log_level=args.log_level,
            namespace_prefix=args.namespace_prefix,
            addon_params={
                "example_number": args.example_number,
                "language": args.language,
            },
            enable_llm_cache=args.enable_llm_cache,
            auto_manage_storages_states=False,
        )
    
        await rag.initialize_storages()
        await initialize_pipeline_status()
    return rag


def get_working_dir_dependency(args):
    working_dir = args.working_dir
    working_dir_header = APIKeyHeader(name="X-Workspace", auto_error=False)

    async def working_dir_auth(
        working_dir_header_value: str | None = Security(working_dir_header),
    ):
        if not working_dir_header_value:
            rag = await get_rag(args, working_dir)
            return rag
        else:
            working_dir_header_value = unquote(working_dir_header_value)
            workspace_path = Path(
                working_dir,
                base64.urlsafe_b64encode(
                    working_dir_header_value.encode("utf-8")
                ).decode("utf-8"),
            )
            # 如果不存在该工作目录，返回错误信息
            if not Path(workspace_path).exists() or not Path(workspace_path).is_dir():
                raise HTTPException(status_code=404, detail="Workspace not found")
            rag = await get_rag(args, workspace_path)
            return rag

    return working_dir_auth
