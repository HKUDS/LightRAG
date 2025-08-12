"""
Configs for the LightRAG API.
"""

import os
import argparse
import logging
from dotenv import load_dotenv
from lightrag.utils import get_env_value
from lightrag.llm.binding_options import OllamaEmbeddingOptions, OllamaLLMOptions
import sys

from lightrag.constants import (
    DEFAULT_WOKERS,
    DEFAULT_TIMEOUT,
    DEFAULT_TOP_K,
    DEFAULT_CHUNK_TOP_K,
    DEFAULT_HISTORY_TURNS,
    DEFAULT_MAX_ENTITY_TOKENS,
    DEFAULT_MAX_RELATION_TOKENS,
    DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_COSINE_THRESHOLD,
    DEFAULT_RELATED_CHUNK_NUMBER,
    DEFAULT_MIN_RERANK_SCORE,
    DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE,
    DEFAULT_MAX_ASYNC,
    DEFAULT_SUMMARY_MAX_TOKENS,
    DEFAULT_SUMMARY_LANGUAGE,
    DEFAULT_EMBEDDING_FUNC_MAX_ASYNC,
    DEFAULT_EMBEDDING_BATCH_NUM,
    DEFAULT_OLLAMA_MODEL_NAME,
    DEFAULT_OLLAMA_MODEL_TAG,
    DEFAULT_OLLAMA_MODEL_SIZE,
    DEFAULT_OLLAMA_CREATED_AT,
    DEFAULT_OLLAMA_DIGEST,
)

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


class OllamaServerInfos:
    def __init__(self, name=None, tag=None):
        self._lightrag_name = name or os.getenv(
            "OLLAMA_EMULATING_MODEL_NAME", DEFAULT_OLLAMA_MODEL_NAME
        )
        self._lightrag_tag = tag or os.getenv(
            "OLLAMA_EMULATING_MODEL_TAG", DEFAULT_OLLAMA_MODEL_TAG
        )
        self.LIGHTRAG_SIZE = DEFAULT_OLLAMA_MODEL_SIZE
        self.LIGHTRAG_CREATED_AT = DEFAULT_OLLAMA_CREATED_AT
        self.LIGHTRAG_DIGEST = DEFAULT_OLLAMA_DIGEST

    @property
    def LIGHTRAG_NAME(self):
        return self._lightrag_name

    @LIGHTRAG_NAME.setter
    def LIGHTRAG_NAME(self, value):
        self._lightrag_name = value

    @property
    def LIGHTRAG_TAG(self):
        return self._lightrag_tag

    @LIGHTRAG_TAG.setter
    def LIGHTRAG_TAG(self, value):
        self._lightrag_tag = value

    @property
    def LIGHTRAG_MODEL(self):
        return f"{self._lightrag_name}:{self._lightrag_tag}"


ollama_server_infos = OllamaServerInfos()


class DefaultRAGStorageConfig:
    KV_STORAGE = "JsonKVStorage"
    VECTOR_STORAGE = "NanoVectorDBStorage"
    GRAPH_STORAGE = "NetworkXStorage"
    DOC_STATUS_STORAGE = "JsonDocStatusStorage"


def get_ollama_host() -> str:
    """Get Ollama host with proper environment variable priority and fallback."""
    # Priority order: OLLAMA_HOST -> EMBEDDING_BINDING_HOST -> EMBEDDING_BASE_URL -> LLM_BINDING_HOST -> default
    return (
        os.getenv("OLLAMA_HOST")
        or os.getenv("EMBEDDING_BINDING_HOST")
        or os.getenv("EMBEDDING_BASE_URL", "").rstrip(
            "/"
        )  # Remove trailing slash if present
        or os.getenv("LLM_BINDING_HOST")
        or "http://localhost:11434"
    )


def get_default_host(binding_type: str) -> str:
    """Get default host URL for different binding types with proper environment variable priority."""
    default_hosts = {
        "ollama": get_ollama_host(),
        "lollms": os.getenv("LLM_BINDING_HOST", "http://localhost:9600"),
        "azure_openai": os.getenv("AZURE_OPENAI_ENDPOINT", "https://api.openai.com/v1"),
        "openai": os.getenv("LLM_BINDING_HOST", "https://api.openai.com/v1"),
        "xai": os.getenv("XAI_API_BASE", "https://api.x.ai/v1"),
    }
    return default_hosts.get(
        binding_type, os.getenv("LLM_BINDING_HOST", "http://localhost:11434")
    )  # fallback to ollama if unknown


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments with environment variable fallback

    Args:
        is_uvicorn_mode: Whether running under uvicorn mode

    Returns:
        argparse.Namespace: Parsed arguments
    """

    # For production deployment, if any unrecognized arguments are present,
    # skip argument parsing and use environment variables only
    production_indicators = [
        "--bind",
        "--worker-class",
        "--workers",
        "--timeout",
        "--max-requests",
        "--preload",
        "--access-logfile",
        "--error-logfile",
    ]

    if any(indicator in " ".join(sys.argv) for indicator in production_indicators):
        # Running in production mode (likely gunicorn), skip argument parsing
        return None

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

    parser.add_argument(
        "--timeout",
        default=get_env_value("TIMEOUT", DEFAULT_TIMEOUT, int, special_none=True),
        type=int,
        help="Timeout in seconds (useful when using slow AI). Use None for infinite timeout",
    )

    # RAG configuration
    parser.add_argument(
        "--max-async",
        type=int,
        default=get_env_value("MAX_ASYNC", DEFAULT_MAX_ASYNC, int),
        help=f"Maximum async operations (default: from env or {DEFAULT_MAX_ASYNC})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=get_env_value("MAX_TOKENS", DEFAULT_SUMMARY_MAX_TOKENS, int),
        help=f"Maximum token size (default: from env or {DEFAULT_SUMMARY_MAX_TOKENS})",
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

    # Ollama model configuration
    parser.add_argument(
        "--simulated-model-name",
        type=str,
        default=get_env_value("OLLAMA_EMULATING_MODEL_NAME", DEFAULT_OLLAMA_MODEL_NAME),
        help="Name for the simulated Ollama model (default: from env or lightrag)",
    )

    parser.add_argument(
        "--simulated-model-tag",
        type=str,
        default=get_env_value("OLLAMA_EMULATING_MODEL_TAG", DEFAULT_OLLAMA_MODEL_TAG),
        help="Tag for the simulated Ollama model (default: from env or latest)",
    )

    # Namespace
    parser.add_argument(
        "--workspace",
        type=str,
        default=get_env_value("WORKSPACE", ""),
        help="Default workspace for all storage",
    )

    parser.add_argument(
        "--auto-scan-at-startup",
        action="store_true",
        default=False,
        help="Enable automatic scanning when the program starts",
    )

    # Server workers configuration
    parser.add_argument(
        "--workers",
        type=int,
        default=get_env_value("WORKERS", DEFAULT_WOKERS, int),
        help="Number of worker processes (default: from env or 1)",
    )

    # LLM and embedding bindings
    parser.add_argument(
        "--llm-binding",
        type=str,
        default=get_env_value("LLM_BINDING", "ollama"),
        choices=["lollms", "ollama", "openai", "openai-ollama", "azure_openai", "xai"],
        help="LLM binding type (default: from env or ollama)",
    )
    parser.add_argument(
        "--embedding-binding",
        type=str,
        default=get_env_value("EMBEDDING_BINDING", "ollama"),
        choices=["lollms", "ollama", "openai", "azure_openai", "xai"],
        help="Embedding binding type (default: from env or ollama)",
    )

    # Conditionally add binding options defined in binding_options module
    # This will add command line arguments for all binding options (e.g., --ollama-embedding-num_ctx)
    # and corresponding environment variables (e.g., OLLAMA_EMBEDDING_NUM_CTX)
    if "--llm-binding" in sys.argv:
        try:
            idx = sys.argv.index("--llm-binding")
            if idx + 1 < len(sys.argv) and sys.argv[idx + 1] == "ollama":
                OllamaLLMOptions.add_args(parser)
        except IndexError:
            pass
    elif os.environ.get("LLM_BINDING") == "ollama":
        OllamaLLMOptions.add_args(parser)

    if "--embedding-binding" in sys.argv:
        try:
            idx = sys.argv.index("--embedding-binding")
            if idx + 1 < len(sys.argv) and sys.argv[idx + 1] == "ollama":
                OllamaEmbeddingOptions.add_args(parser)
        except IndexError:
            pass
    elif os.environ.get("EMBEDDING_BINDING") == "ollama":
        OllamaEmbeddingOptions.add_args(parser)

    args = parser.parse_args()

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

    # Get MAX_PARALLEL_INSERT from environment
    args.max_parallel_insert = get_env_value("MAX_PARALLEL_INSERT", 2, int)

    # Get MAX_GRAPH_NODES from environment
    args.max_graph_nodes = get_env_value("MAX_GRAPH_NODES", 1000, int)

    # Handle openai-ollama special case
    if args.llm_binding == "openai-ollama":
        args.llm_binding = "openai"
        args.embedding_binding = "ollama"

    # Ollama ctx_num
    args.ollama_num_ctx = get_env_value("OLLAMA_NUM_CTX", 32768, int)

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

    # Inject chunk configuration
    args.chunk_size = get_env_value("CHUNK_SIZE", 1200, int)
    args.chunk_overlap_size = get_env_value("CHUNK_OVERLAP_SIZE", 100, int)

    # Inject LLM cache configuration
    args.enable_llm_cache_for_extract = get_env_value(
        "ENABLE_LLM_CACHE_FOR_EXTRACT", True, bool
    )
    args.enable_llm_cache = get_env_value("ENABLE_LLM_CACHE", True, bool)

    # Inject LLM temperature configuration
    args.temperature = get_env_value("TEMPERATURE", 0.5, float)

    # Document loading engine configuration
    args.document_loading_engine = get_env_value("DOCUMENT_LOADING_ENGINE", "DEFAULT")

    # Enhanced Docling configuration options
    args.docling_export_format = get_env_value("DOCLING_EXPORT_FORMAT", "markdown")
    args.docling_max_workers = get_env_value("DOCLING_MAX_WORKERS", 2, int)
    args.docling_enable_ocr = get_env_value("DOCLING_ENABLE_OCR", True, bool)
    args.docling_enable_table_structure = get_env_value(
        "DOCLING_ENABLE_TABLE_STRUCTURE", True, bool
    )
    args.docling_enable_figures = get_env_value("DOCLING_ENABLE_FIGURES", True, bool)

    # Docling model selection
    args.docling_layout_model = get_env_value("DOCLING_LAYOUT_MODEL", "auto")
    args.docling_ocr_model = get_env_value("DOCLING_OCR_MODEL", "auto")
    args.docling_table_model = get_env_value("DOCLING_TABLE_MODEL", "auto")

    # Docling content processing options
    args.docling_include_page_numbers = get_env_value(
        "DOCLING_INCLUDE_PAGE_NUMBERS", True, bool
    )
    args.docling_include_headings = get_env_value(
        "DOCLING_INCLUDE_HEADINGS", True, bool
    )
    args.docling_extract_metadata = get_env_value(
        "DOCLING_EXTRACT_METADATA", True, bool
    )
    args.docling_process_images = get_env_value("DOCLING_PROCESS_IMAGES", True, bool)

    # Docling quality settings
    args.docling_image_dpi = get_env_value("DOCLING_IMAGE_DPI", 300, int)
    args.docling_ocr_confidence = get_env_value("DOCLING_OCR_CONFIDENCE", 0.7, float)
    args.docling_table_confidence = get_env_value(
        "DOCLING_TABLE_CONFIDENCE", 0.8, float
    )

    # Docling cache settings
    args.docling_enable_cache = get_env_value("DOCLING_ENABLE_CACHE", True, bool)
    args.docling_cache_dir = get_env_value("DOCLING_CACHE_DIR", "./docling_cache")
    args.docling_cache_ttl_hours = get_env_value("DOCLING_CACHE_TTL_HOURS", 168, int)

    # Add environment variables that were previously read directly
    args.cors_origins = get_env_value("CORS_ORIGINS", "*")
    args.summary_language = get_env_value("SUMMARY_LANGUAGE", DEFAULT_SUMMARY_LANGUAGE)
    args.whitelist_paths = get_env_value("WHITELIST_PATHS", "/health,/api/*")

    # For JWT Auth
    args.auth_accounts = get_env_value("AUTH_ACCOUNTS", "")
    args.token_secret = get_env_value("TOKEN_SECRET", "lightrag-jwt-default-secret")
    args.token_expire_hours = get_env_value("TOKEN_EXPIRE_HOURS", 48, int)
    args.guest_token_expire_hours = get_env_value("GUEST_TOKEN_EXPIRE_HOURS", 24, int)
    args.jwt_algorithm = get_env_value("JWT_ALGORITHM", "HS256")

    # Rerank model configuration
    args.rerank_model = get_env_value("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    args.rerank_binding_host = get_env_value("RERANK_BINDING_HOST", None)
    args.rerank_binding_api_key = get_env_value("RERANK_BINDING_API_KEY", None)

    # Min rerank score configuration
    args.min_rerank_score = get_env_value(
        "MIN_RERANK_SCORE", DEFAULT_MIN_RERANK_SCORE, float
    )

    # Query configuration
    args.history_turns = get_env_value("HISTORY_TURNS", DEFAULT_HISTORY_TURNS, int)
    args.top_k = get_env_value("TOP_K", DEFAULT_TOP_K, int)
    args.chunk_top_k = get_env_value("CHUNK_TOP_K", DEFAULT_CHUNK_TOP_K, int)
    args.max_entity_tokens = get_env_value(
        "MAX_ENTITY_TOKENS", DEFAULT_MAX_ENTITY_TOKENS, int
    )
    args.max_relation_tokens = get_env_value(
        "MAX_RELATION_TOKENS", DEFAULT_MAX_RELATION_TOKENS, int
    )
    args.max_total_tokens = get_env_value(
        "MAX_TOTAL_TOKENS", DEFAULT_MAX_TOTAL_TOKENS, int
    )
    args.cosine_threshold = get_env_value(
        "COSINE_THRESHOLD", DEFAULT_COSINE_THRESHOLD, float
    )
    args.related_chunk_number = get_env_value(
        "RELATED_CHUNK_NUMBER", DEFAULT_RELATED_CHUNK_NUMBER, int
    )

    # Add missing environment variables for health endpoint
    args.force_llm_summary_on_merge = get_env_value(
        "FORCE_LLM_SUMMARY_ON_MERGE", DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE, int
    )
    args.embedding_func_max_async = get_env_value(
        "EMBEDDING_FUNC_MAX_ASYNC", DEFAULT_EMBEDDING_FUNC_MAX_ASYNC, int
    )
    args.embedding_batch_num = get_env_value(
        "EMBEDDING_BATCH_NUM", DEFAULT_EMBEDDING_BATCH_NUM, int
    )

    ollama_server_infos.LIGHTRAG_NAME = args.simulated_model_name
    ollama_server_infos.LIGHTRAG_TAG = args.simulated_model_tag

    return args


def update_uvicorn_mode_config():
    # If in uvicorn mode and workers > 1, force it to 1 and log warning
    args = get_global_args()
    if args.workers > 1:
        original_workers = args.workers
        args.workers = 1
        # Log warning directly here
        logging.warning(
            f"In uvicorn mode, workers parameter was set to {original_workers}. Forcing workers=1"
        )


# Initialize global_args as None - will be parsed on first access
global_args = None


# For backward compatibility, provide global_args as an alias
def __getattr__(name):
    if name == "global_args":
        return get_global_args()
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def get_global_args():
    """Get global arguments, parsing them if not already done."""
    global global_args
    if global_args is not None:
        return global_args

    # Try to parse arguments, fall back to environment-only configuration if parsing fails
    try:
        global_args = parse_args()
        if global_args is None:
            raise ValueError("Gunicorn detected, using environment variables")
    except (SystemExit, argparse.ArgumentError, Exception):
        # Argument parsing failed (likely due to gunicorn arguments or test environment)
        # Create args from environment variables only
        global_args = argparse.Namespace()

    # Set all required attributes with defaults from environment
    global_args.host = get_env_value("HOST", "0.0.0.0")
    global_args.port = get_env_value("PORT", 9621, int)
    global_args.working_dir = get_env_value("WORKING_DIR", "./rag_storage")
    global_args.input_dir = get_env_value("INPUT_DIR", "./inputs")
    global_args.timeout = get_env_value(
        "TIMEOUT", DEFAULT_TIMEOUT, int, special_none=True
    )
    global_args.max_async = get_env_value("MAX_ASYNC", DEFAULT_MAX_ASYNC, int)
    global_args.max_tokens = get_env_value(
        "MAX_TOKENS", DEFAULT_SUMMARY_MAX_TOKENS, int
    )
    global_args.log_level = get_env_value("LOG_LEVEL", "INFO")
    global_args.verbose = get_env_value("VERBOSE", False, bool)
    global_args.key = get_env_value("LIGHTRAG_API_KEY", None)
    global_args.ssl = get_env_value("SSL", False, bool)
    global_args.ssl_certfile = get_env_value("SSL_CERTFILE", None)
    global_args.ssl_keyfile = get_env_value("SSL_KEYFILE", None)
    global_args.workers = get_env_value("WORKERS", DEFAULT_WOKERS, int)
    global_args.cors_origins = get_env_value("CORS_ORIGINS", "*")
    global_args.temperature = get_env_value("TEMPERATURE", 0, float)
    global_args.llm_binding = get_env_value("LLM_BINDING", "openai")
    global_args.llm_model = get_env_value("LLM_MODEL", "gpt-4o")
    # Use proper Ollama host resolution for LLM binding
    if get_env_value("LLM_BINDING", "openai") == "ollama":
        global_args.llm_binding_host = get_env_value(
            "LLM_BINDING_HOST", get_ollama_host()
        )
    else:
        global_args.llm_binding_host = get_env_value("LLM_BINDING_HOST", None)
    global_args.llm_binding_api_key = get_env_value("LLM_BINDING_API_KEY", None)
    global_args.embedding_binding = get_env_value("EMBEDDING_BINDING", "openai")
    global_args.embedding_model = get_env_value(
        "EMBEDDING_MODEL", "text-embedding-3-small"
    )
    global_args.embedding_dim = get_env_value("EMBEDDING_DIM", 1536, int)
    # Use proper Ollama host resolution for embedding binding
    if get_env_value("EMBEDDING_BINDING", "openai") == "ollama":
        global_args.embedding_binding_host = get_env_value(
            "EMBEDDING_BINDING_HOST", get_ollama_host()
        )
    else:
        global_args.embedding_binding_host = get_env_value(
            "EMBEDDING_BINDING_HOST", None
        )
    global_args.embedding_binding_api_key = get_env_value(
        "EMBEDDING_BINDING_API_KEY", None
    )
    global_args.kv_storage = get_env_value(
        "LIGHTRAG_KV_STORAGE", DefaultRAGStorageConfig.KV_STORAGE
    )
    global_args.vector_storage = get_env_value(
        "LIGHTRAG_VECTOR_STORAGE", DefaultRAGStorageConfig.VECTOR_STORAGE
    )
    global_args.graph_storage = get_env_value(
        "LIGHTRAG_GRAPH_STORAGE", DefaultRAGStorageConfig.GRAPH_STORAGE
    )
    global_args.doc_status_storage = get_env_value(
        "LIGHTRAG_DOC_STATUS_STORAGE", DefaultRAGStorageConfig.DOC_STATUS_STORAGE
    )
    global_args.workspace = get_env_value("WORKSPACE", None)
    global_args.chunk_size = get_env_value("CHUNK_SIZE", 1200, int)
    global_args.chunk_overlap_size = get_env_value("CHUNK_OVERLAP_SIZE", 100, int)
    global_args.max_parallel_insert = get_env_value("MAX_PARALLEL_INSERT", 2, int)
    global_args.max_graph_nodes = get_env_value("MAX_GRAPH_NODES", 1000, int)
    global_args.summary_language = get_env_value(
        "SUMMARY_LANGUAGE", DEFAULT_SUMMARY_LANGUAGE
    )
    global_args.enable_llm_cache = get_env_value("ENABLE_LLM_CACHE", True, bool)
    global_args.enable_llm_cache_for_extract = get_env_value(
        "ENABLE_LLM_CACHE_FOR_EXTRACT", True, bool
    )
    global_args.auth_accounts = get_env_value("AUTH_ACCOUNTS", "")
    global_args.token_secret = get_env_value(
        "TOKEN_SECRET", "lightrag-jwt-default-secret"
    )
    global_args.token_expire_hours = get_env_value("TOKEN_EXPIRE_HOURS", 48, int)
    global_args.guest_token_expire_hours = get_env_value(
        "GUEST_TOKEN_EXPIRE_HOURS", 24, int
    )
    global_args.jwt_algorithm = get_env_value("JWT_ALGORITHM", "HS256")
    global_args.whitelist_paths = get_env_value("WHITELIST_PATHS", "/health,/api/*")
    global_args.auto_scan_at_startup = get_env_value(
        "AUTO_SCAN_AT_STARTUP", False, bool
    )
    global_args.simulated_model_name = get_env_value(
        "OLLAMA_EMULATING_MODEL_NAME", DEFAULT_OLLAMA_MODEL_NAME
    )
    global_args.simulated_model_tag = get_env_value(
        "OLLAMA_EMULATING_MODEL_TAG", DEFAULT_OLLAMA_MODEL_TAG
    )
    global_args.rerank_model = get_env_value("RERANK_MODEL", None)
    global_args.rerank_binding_host = get_env_value("RERANK_BINDING_HOST", None)
    global_args.rerank_binding_api_key = get_env_value("RERANK_BINDING_API_KEY", None)
    global_args.min_rerank_score = get_env_value(
        "MIN_RERANK_SCORE", DEFAULT_MIN_RERANK_SCORE, float
    )
    global_args.history_turns = get_env_value(
        "HISTORY_TURNS", DEFAULT_HISTORY_TURNS, int
    )
    global_args.top_k = get_env_value("TOP_K", DEFAULT_TOP_K, int)
    global_args.chunk_top_k = get_env_value("CHUNK_TOP_K", DEFAULT_CHUNK_TOP_K, int)
    global_args.max_entity_tokens = get_env_value(
        "MAX_ENTITY_TOKENS", DEFAULT_MAX_ENTITY_TOKENS, int
    )
    global_args.max_relation_tokens = get_env_value(
        "MAX_RELATION_TOKENS", DEFAULT_MAX_RELATION_TOKENS, int
    )
    global_args.max_total_tokens = get_env_value(
        "MAX_TOTAL_TOKENS", DEFAULT_MAX_TOTAL_TOKENS, int
    )
    global_args.cosine_threshold = get_env_value(
        "COSINE_THRESHOLD", DEFAULT_COSINE_THRESHOLD, float
    )
    global_args.related_chunk_number = get_env_value(
        "RELATED_CHUNK_NUMBER", DEFAULT_RELATED_CHUNK_NUMBER, int
    )
    global_args.force_llm_summary_on_merge = get_env_value(
        "FORCE_LLM_SUMMARY_ON_MERGE", DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE, int
    )
    global_args.embedding_func_max_async = get_env_value(
        "EMBEDDING_FUNC_MAX_ASYNC", DEFAULT_EMBEDDING_FUNC_MAX_ASYNC, int
    )
    global_args.embedding_batch_num = get_env_value(
        "EMBEDDING_BATCH_NUM", DEFAULT_EMBEDDING_BATCH_NUM, int
    )

    # Document loading engine configuration
    global_args.document_loading_engine = get_env_value(
        "DOCUMENT_LOADING_ENGINE", "DEFAULT"
    )

    # Set ollama server infos
    ollama_server_infos.LIGHTRAG_NAME = global_args.simulated_model_name
    ollama_server_infos.LIGHTRAG_TAG = global_args.simulated_model_tag

    return global_args
