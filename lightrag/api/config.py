"""
Configs for the LightRAG API.
"""

import os
import argparse
import logging
from dotenv import load_dotenv
from lightrag.utils import get_env_value
from lightrag.llm.binding_options import (
    GeminiEmbeddingOptions,
    GeminiLLMOptions,
    OllamaEmbeddingOptions,
    OllamaLLMOptions,
    OpenAILLMOptions,
)
from lightrag.base import OllamaServerInfos
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
    DEFAULT_SUMMARY_LENGTH_RECOMMENDED,
    DEFAULT_SUMMARY_CONTEXT_SIZE,
    DEFAULT_SUMMARY_LANGUAGE,
    DEFAULT_EMBEDDING_FUNC_MAX_ASYNC,
    DEFAULT_EMBEDDING_BATCH_NUM,
    DEFAULT_OLLAMA_MODEL_NAME,
    DEFAULT_OLLAMA_MODEL_TAG,
    DEFAULT_RERANK_BINDING,
    DEFAULT_ENTITY_TYPES,
)

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)


ollama_server_infos = OllamaServerInfos()


class DefaultRAGStorageConfig:
    KV_STORAGE = "JsonKVStorage"
    VECTOR_STORAGE = "NanoVectorDBStorage"
    GRAPH_STORAGE = "NetworkXStorage"
    DOC_STATUS_STORAGE = "JsonDocStatusStorage"


def get_default_host(binding_type: str) -> str:
    default_hosts = {
        "ollama": os.getenv("LLM_BINDING_HOST", "http://localhost:11434"),
        "lollms": os.getenv("LLM_BINDING_HOST", "http://localhost:9600"),
        "azure_openai": os.getenv("AZURE_OPENAI_ENDPOINT", "https://api.openai.com/v1"),
        "openai": os.getenv("LLM_BINDING_HOST", "https://api.openai.com/v1"),
        "gemini": os.getenv(
            "LLM_BINDING_HOST", "https://generativelanguage.googleapis.com"
        ),
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

    parser = argparse.ArgumentParser(description="LightRAG API Server")

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
        "--summary-max-tokens",
        type=int,
        default=get_env_value("SUMMARY_MAX_TOKENS", DEFAULT_SUMMARY_MAX_TOKENS, int),
        help=f"Maximum token size for entity/relation summary(default: from env or {DEFAULT_SUMMARY_MAX_TOKENS})",
    )
    parser.add_argument(
        "--summary-context-size",
        type=int,
        default=get_env_value(
            "SUMMARY_CONTEXT_SIZE", DEFAULT_SUMMARY_CONTEXT_SIZE, int
        ),
        help=f"LLM Summary Context size (default: from env or {DEFAULT_SUMMARY_CONTEXT_SIZE})",
    )
    parser.add_argument(
        "--summary-length-recommended",
        type=int,
        default=get_env_value(
            "SUMMARY_LENGTH_RECOMMENDED", DEFAULT_SUMMARY_LENGTH_RECOMMENDED, int
        ),
        help=f"LLM Summary Context size (default: from env or {DEFAULT_SUMMARY_LENGTH_RECOMMENDED})",
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
        choices=[
            "lollms",
            "ollama",
            "openai",
            "openai-ollama",
            "azure_openai",
            "aws_bedrock",
            "gemini",
        ],
        help="LLM binding type (default: from env or ollama)",
    )
    parser.add_argument(
        "--embedding-binding",
        type=str,
        default=get_env_value("EMBEDDING_BINDING", "ollama"),
        choices=[
            "lollms",
            "ollama",
            "openai",
            "azure_openai",
            "aws_bedrock",
            "jina",
            "gemini",
            "voyageai",
        ],
        help="Embedding binding type (default: from env or ollama)",
    )
    parser.add_argument(
        "--rerank-binding",
        type=str,
        default=get_env_value("RERANK_BINDING", DEFAULT_RERANK_BINDING),
        choices=["null", "cohere", "jina", "aliyun"],
        help=f"Rerank binding type (default: from env or {DEFAULT_RERANK_BINDING})",
    )

    # Document loading engine configuration
    parser.add_argument(
        "--docling",
        action="store_true",
        default=False,
        help="Enable DOCLING document loading engine (default: from env or DEFAULT)",
    )

    # Conditionally add binding-specific options (Ollama, OpenAI, Azure OpenAI, Gemini)
    # This registers command line arguments (e.g., --openai-llm-temperature)
    # and reads corresponding environment variables (e.g., OPENAI_LLM_TEMPERATURE)

    # Determine LLM binding value consistently from command line or environment
    llm_binding_value = None
    if "--llm-binding" in sys.argv:
        try:
            idx = sys.argv.index("--llm-binding")
            if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("-"):
                llm_binding_value = sys.argv[idx + 1]
        except IndexError:
            pass

    # Fall back to environment variable using same function as argparse default
    if llm_binding_value is None:
        llm_binding_value = get_env_value("LLM_BINDING", "ollama")

    # Add LLM binding options based on determined value
    if llm_binding_value == "ollama":
        OllamaLLMOptions.add_args(parser)
    elif llm_binding_value in ["openai", "azure_openai"]:
        OpenAILLMOptions.add_args(parser)
    elif llm_binding_value == "gemini":
        GeminiLLMOptions.add_args(parser)

    # Determine embedding binding value consistently from command line or environment
    embedding_binding_value = None
    if "--embedding-binding" in sys.argv:
        try:
            idx = sys.argv.index("--embedding-binding")
            if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("-"):
                embedding_binding_value = sys.argv[idx + 1]
        except IndexError:
            pass

    # Fall back to environment variable using same function as argparse default
    if embedding_binding_value is None:
        embedding_binding_value = get_env_value("EMBEDDING_BINDING", "ollama")

    # Add embedding binding options based on determined value
    if embedding_binding_value == "ollama":
        OllamaEmbeddingOptions.add_args(parser)
    elif embedding_binding_value == "gemini":
        GeminiEmbeddingOptions.add_args(parser)

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
    # EMBEDDING_MODEL defaults to None - each binding will use its own default model
    # e.g., OpenAI uses "text-embedding-3-small", Jina uses "jina-embeddings-v4"
    args.embedding_model = get_env_value("EMBEDDING_MODEL", None, special_none=True)
    # EMBEDDING_DIM defaults to None - each binding will use its own default dimension
    # Value is inherited from provider defaults via wrap_embedding_func_with_attrs decorator
    args.embedding_dim = get_env_value("EMBEDDING_DIM", None, int, special_none=True)
    args.embedding_send_dim = get_env_value("EMBEDDING_SEND_DIM", False, bool)

    # Inject chunk configuration
    args.chunk_size = get_env_value("CHUNK_SIZE", 1200, int)
    args.chunk_overlap_size = get_env_value("CHUNK_OVERLAP_SIZE", 100, int)

    # Inject LLM cache configuration
    args.enable_llm_cache_for_extract = get_env_value(
        "ENABLE_LLM_CACHE_FOR_EXTRACT", True, bool
    )
    args.enable_llm_cache = get_env_value("ENABLE_LLM_CACHE", True, bool)

    # Set document_loading_engine from --docling flag
    if args.docling:
        args.document_loading_engine = "DOCLING"
    else:
        args.document_loading_engine = get_env_value(
            "DOCUMENT_LOADING_ENGINE", "DEFAULT"
        )

    # PDF decryption password
    args.pdf_decrypt_password = get_env_value("PDF_DECRYPT_PASSWORD", None)

    # Add environment variables that were previously read directly
    args.cors_origins = get_env_value("CORS_ORIGINS", "*")
    args.summary_language = get_env_value("SUMMARY_LANGUAGE", DEFAULT_SUMMARY_LANGUAGE)
    args.entity_types = get_env_value("ENTITY_TYPES", DEFAULT_ENTITY_TYPES, list)
    args.whitelist_paths = get_env_value("WHITELIST_PATHS", "/health,/api/*")

    # For JWT Auth
    args.auth_accounts = get_env_value("AUTH_ACCOUNTS", "")
    args.token_secret = get_env_value("TOKEN_SECRET", "lightrag-jwt-default-secret")
    args.token_expire_hours = get_env_value("TOKEN_EXPIRE_HOURS", 48, float)
    args.guest_token_expire_hours = get_env_value("GUEST_TOKEN_EXPIRE_HOURS", 24, float)
    args.jwt_algorithm = get_env_value("JWT_ALGORITHM", "HS256")

    # Token auto-renewal configuration (sliding window expiration)
    args.token_auto_renew = get_env_value("TOKEN_AUTO_RENEW", True, bool)
    args.token_renew_threshold = get_env_value("TOKEN_RENEW_THRESHOLD", 0.5, float)

    # Rerank model configuration
    args.rerank_model = get_env_value("RERANK_MODEL", None)
    args.rerank_binding_host = get_env_value("RERANK_BINDING_HOST", None)
    args.rerank_binding_api_key = get_env_value("RERANK_BINDING_API_KEY", None)
    # Note: rerank_binding is already set by argparse, no need to override from env

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

    # Embedding token limit configuration
    args.embedding_token_limit = get_env_value(
        "EMBEDDING_TOKEN_LIMIT", None, int, special_none=True
    )

    # File upload size limit (in bytes, None for unlimited)
    # Default: 100MB (104857600 bytes)
    args.max_upload_size = get_env_value(
        "MAX_UPLOAD_SIZE", 104857600, int, special_none=True
    )

    ollama_server_infos.LIGHTRAG_NAME = args.simulated_model_name
    ollama_server_infos.LIGHTRAG_TAG = args.simulated_model_tag

    return args


def update_uvicorn_mode_config():
    # If in uvicorn mode and workers > 1, force it to 1 and log warning
    if global_args.workers > 1:
        original_workers = global_args.workers
        global_args.workers = 1
        # Log warning directly here
        logging.warning(
            f">> Forcing workers=1 in uvicorn mode(Ignoring workers={original_workers})"
        )


# Global configuration with lazy initialization
_global_args = None
_initialized = False


def initialize_config(args=None, force=False):
    """Initialize global configuration

    This function allows explicit initialization of the configuration,
    which is useful for programmatic usage, testing, or embedding LightRAG
    in other applications.

    Args:
        args: Pre-parsed argparse.Namespace or None to parse from sys.argv
        force: Force re-initialization even if already initialized

    Returns:
        argparse.Namespace: The configured arguments

    Example:
        # Use parsed command line arguments (default)
        initialize_config()

        # Use custom configuration programmatically
        custom_args = argparse.Namespace(
            host='localhost',
            port=8080,
            working_dir='./custom_rag',
            # ... other config
        )
        initialize_config(custom_args)
    """
    global _global_args, _initialized

    if _initialized and not force:
        return _global_args

    _global_args = args if args is not None else parse_args()
    _initialized = True
    return _global_args


def get_config():
    """Get global configuration, auto-initializing if needed

    Returns:
        argparse.Namespace: The configured arguments
    """
    if not _initialized:
        initialize_config()
    return _global_args


class _GlobalArgsProxy:
    """Proxy object that auto-initializes configuration on first access

    This maintains backward compatibility with existing code while
    allowing programmatic control over initialization timing.

    The proxy fully delegates to the underlying argparse.Namespace,
    including support for vars() calls which is used by binding_options
    to extract provider-specific configuration options.
    """

    def __getattribute__(self, name):
        """Override attribute access to support vars() and regular attribute access.

        This method intercepts __dict__ access (used by vars()) and delegates
        to the underlying _global_args namespace, ensuring binding options
        can be properly extracted.
        """
        global _initialized, _global_args

        # Handle __dict__ access for vars() support
        if name == "__dict__":
            if not _initialized:
                initialize_config()
            return vars(_global_args)

        # Handle class-level attributes that should come from the proxy itself
        if name in ("__class__", "__repr__", "__getattribute__", "__setattr__"):
            return object.__getattribute__(self, name)

        # Delegate all other attribute access to the underlying namespace
        if not _initialized:
            initialize_config()
        return getattr(_global_args, name)

    def __setattr__(self, name, value):
        global _initialized, _global_args
        if not _initialized:
            initialize_config()
        setattr(_global_args, name, value)

    def __repr__(self):
        global _initialized, _global_args
        if not _initialized:
            return "<GlobalArgsProxy: Not initialized>"
        return repr(_global_args)


# Create proxy instance for backward compatibility
# Existing code like `from config import global_args` continues to work
# The proxy will auto-initialize on first attribute access
global_args = _GlobalArgsProxy()
