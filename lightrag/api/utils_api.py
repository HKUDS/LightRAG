"""
Utility functions for the LightRAG API.
"""

import os
import argparse
from typing import Optional
import sys
from ascii_colors import ASCIIColors
from lightrag.api import __api_version__
from fastapi import HTTPException, Security
from dotenv import load_dotenv
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN

# Load environment variables
load_dotenv(override=True)


class OllamaServerInfos:
    # Constants for emulated Ollama model information
    LIGHTRAG_NAME = "lightrag"
    LIGHTRAG_TAG = os.getenv("OLLAMA_EMULATING_MODEL_TAG", "latest")
    LIGHTRAG_MODEL = f"{LIGHTRAG_NAME}:{LIGHTRAG_TAG}"
    LIGHTRAG_SIZE = 7365960935  # it's a dummy value
    LIGHTRAG_CREATED_AT = "2024-01-15T00:00:00Z"
    LIGHTRAG_DIGEST = "sha256:lightrag"


ollama_server_infos = OllamaServerInfos()


def get_api_key_dependency(api_key: Optional[str]):
    """
    Create an API key dependency for route protection.

    Args:
        api_key (Optional[str]): The API key to validate against.
                                If None, no authentication is required.

    Returns:
        Callable: A dependency function that validates the API key.
    """
    if not api_key:
        # If no API key is configured, return a dummy dependency that always succeeds
        async def no_auth():
            return None

        return no_auth

    # If API key is configured, use proper authentication
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    async def api_key_auth(
        api_key_header_value: Optional[str] = Security(api_key_header),
    ):
        if not api_key_header_value:
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN, detail="API Key required"
            )
        if api_key_header_value != api_key:
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN, detail="Invalid API Key"
            )
        return api_key_header_value

    return api_key_auth


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
    }
    return default_hosts.get(
        binding_type, os.getenv("LLM_BINDING_HOST", "http://localhost:11434")
    )  # fallback to ollama if unknown


def get_env_value(env_key: str, default: any, value_type: type = str) -> any:
    """
    Get value from environment variable with type conversion

    Args:
        env_key (str): Environment variable key
        default (any): Default value if env variable is not set
        value_type (type): Type to convert the value to

    Returns:
        any: Converted value from environment or default
    """
    value = os.getenv(env_key)
    if value is None:
        return default

    if value_type is bool:
        return value.lower() in ("true", "1", "yes", "t", "on")
    try:
        return value_type(value)
    except ValueError:
        return default


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments with environment variable fallback

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

    return args


def display_splash_screen(args: argparse.Namespace) -> None:
    """
    Display a colorful splash screen showing LightRAG server configuration

    Args:
        args: Parsed command line arguments
    """
    # Banner
    ASCIIColors.cyan(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                   🚀 LightRAG Server v{__api_version__}                  ║
    ║          Fast, Lightweight RAG Server Implementation         ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Server Configuration
    ASCIIColors.magenta("\n📡 Server Configuration:")
    ASCIIColors.white("    ├─ Host: ", end="")
    ASCIIColors.yellow(f"{args.host}")
    ASCIIColors.white("    ├─ Port: ", end="")
    ASCIIColors.yellow(f"{args.port}")
    ASCIIColors.white("    ├─ CORS Origins: ", end="")
    ASCIIColors.yellow(f"{os.getenv('CORS_ORIGINS', '*')}")
    ASCIIColors.white("    ├─ SSL Enabled: ", end="")
    ASCIIColors.yellow(f"{args.ssl}")
    ASCIIColors.white("    └─ API Key: ", end="")
    ASCIIColors.yellow("Set" if args.key else "Not Set")
    if args.ssl:
        ASCIIColors.white("    ├─ SSL Cert: ", end="")
        ASCIIColors.yellow(f"{args.ssl_certfile}")
        ASCIIColors.white("    └─ SSL Key: ", end="")
        ASCIIColors.yellow(f"{args.ssl_keyfile}")

    # Directory Configuration
    ASCIIColors.magenta("\n📂 Directory Configuration:")
    ASCIIColors.white("    ├─ Working Directory: ", end="")
    ASCIIColors.yellow(f"{args.working_dir}")
    ASCIIColors.white("    └─ Input Directory: ", end="")
    ASCIIColors.yellow(f"{args.input_dir}")

    # LLM Configuration
    ASCIIColors.magenta("\n🤖 LLM Configuration:")
    ASCIIColors.white("    ├─ Binding: ", end="")
    ASCIIColors.yellow(f"{args.llm_binding}")
    ASCIIColors.white("    ├─ Host: ", end="")
    ASCIIColors.yellow(f"{args.llm_binding_host}")
    ASCIIColors.white("    └─ Model: ", end="")
    ASCIIColors.yellow(f"{args.llm_model}")

    # Embedding Configuration
    ASCIIColors.magenta("\n📊 Embedding Configuration:")
    ASCIIColors.white("    ├─ Binding: ", end="")
    ASCIIColors.yellow(f"{args.embedding_binding}")
    ASCIIColors.white("    ├─ Host: ", end="")
    ASCIIColors.yellow(f"{args.embedding_binding_host}")
    ASCIIColors.white("    ├─ Model: ", end="")
    ASCIIColors.yellow(f"{args.embedding_model}")
    ASCIIColors.white("    └─ Dimensions: ", end="")
    ASCIIColors.yellow(f"{args.embedding_dim}")

    # RAG Configuration
    ASCIIColors.magenta("\n⚙️ RAG Configuration:")
    ASCIIColors.white("    ├─ Max Async Operations: ", end="")
    ASCIIColors.yellow(f"{args.max_async}")
    ASCIIColors.white("    ├─ Max Tokens: ", end="")
    ASCIIColors.yellow(f"{args.max_tokens}")
    ASCIIColors.white("    ├─ Max Embed Tokens: ", end="")
    ASCIIColors.yellow(f"{args.max_embed_tokens}")
    ASCIIColors.white("    ├─ Chunk Size: ", end="")
    ASCIIColors.yellow(f"{args.chunk_size}")
    ASCIIColors.white("    ├─ Chunk Overlap Size: ", end="")
    ASCIIColors.yellow(f"{args.chunk_overlap_size}")
    ASCIIColors.white("    ├─ History Turns: ", end="")
    ASCIIColors.yellow(f"{args.history_turns}")
    ASCIIColors.white("    ├─ Cosine Threshold: ", end="")
    ASCIIColors.yellow(f"{args.cosine_threshold}")
    ASCIIColors.white("    └─ Top-K: ", end="")
    ASCIIColors.yellow(f"{args.top_k}")

    # System Configuration
    ASCIIColors.magenta("\n💾 Storage Configuration:")
    ASCIIColors.white("    ├─ KV Storage: ", end="")
    ASCIIColors.yellow(f"{args.kv_storage}")
    ASCIIColors.white("    ├─ Vector Storage: ", end="")
    ASCIIColors.yellow(f"{args.vector_storage}")
    ASCIIColors.white("    ├─ Graph Storage: ", end="")
    ASCIIColors.yellow(f"{args.graph_storage}")
    ASCIIColors.white("    └─ Document Status Storage: ", end="")
    ASCIIColors.yellow(f"{args.doc_status_storage}")

    ASCIIColors.magenta("\n🛠️ System Configuration:")
    ASCIIColors.white("    ├─ Ollama Emulating Model: ", end="")
    ASCIIColors.yellow(f"{ollama_server_infos.LIGHTRAG_MODEL}")
    ASCIIColors.white("    ├─ Log Level: ", end="")
    ASCIIColors.yellow(f"{args.log_level}")
    ASCIIColors.white("    ├─ Verbose Debug: ", end="")
    ASCIIColors.yellow(f"{args.verbose}")
    ASCIIColors.white("    └─ Timeout: ", end="")
    ASCIIColors.yellow(f"{args.timeout if args.timeout else 'None (infinite)'}")

    # Server Status
    ASCIIColors.green("\n✨ Server starting up...\n")

    # Server Access Information
    protocol = "https" if args.ssl else "http"
    if args.host == "0.0.0.0":
        ASCIIColors.magenta("\n🌐 Server Access Information:")
        ASCIIColors.white("    ├─ Local Access: ", end="")
        ASCIIColors.yellow(f"{protocol}://localhost:{args.port}")
        ASCIIColors.white("    ├─ Remote Access: ", end="")
        ASCIIColors.yellow(f"{protocol}://<your-ip-address>:{args.port}")
        ASCIIColors.white("    ├─ API Documentation (local): ", end="")
        ASCIIColors.yellow(f"{protocol}://localhost:{args.port}/docs")
        ASCIIColors.white("    ├─ Alternative Documentation (local): ", end="")
        ASCIIColors.yellow(f"{protocol}://localhost:{args.port}/redoc")
        ASCIIColors.white("    └─ WebUI (local): ", end="")
        ASCIIColors.yellow(f"{protocol}://localhost:{args.port}/webui")

        ASCIIColors.yellow("\n📝 Note:")
        ASCIIColors.white("""    Since the server is running on 0.0.0.0:
    - Use 'localhost' or '127.0.0.1' for local access
    - Use your machine's IP address for remote access
    - To find your IP address:
      • Windows: Run 'ipconfig' in terminal
      • Linux/Mac: Run 'ifconfig' or 'ip addr' in terminal
    """)
    else:
        base_url = f"{protocol}://{args.host}:{args.port}"
        ASCIIColors.magenta("\n🌐 Server Access Information:")
        ASCIIColors.white("    ├─ Base URL: ", end="")
        ASCIIColors.yellow(f"{base_url}")
        ASCIIColors.white("    ├─ API Documentation: ", end="")
        ASCIIColors.yellow(f"{base_url}/docs")
        ASCIIColors.white("    └─ Alternative Documentation: ", end="")
        ASCIIColors.yellow(f"{base_url}/redoc")

    # Usage Examples
    ASCIIColors.magenta("\n📚 Quick Start Guide:")
    ASCIIColors.cyan("""
    1. Access the Swagger UI:
       Open your browser and navigate to the API documentation URL above

    2. API Authentication:""")
    if args.key:
        ASCIIColors.cyan("""       Add the following header to your requests:
       X-API-Key: <your-api-key>
    """)
    else:
        ASCIIColors.cyan("       No authentication required\n")

    ASCIIColors.cyan("""    3. Basic Operations:
       - POST /upload_document: Upload new documents to RAG
       - POST /query: Query your document collection
       - GET /collections: List available collections

    4. Monitor the server:
       - Check server logs for detailed operation information
       - Use healthcheck endpoint: GET /health
    """)

    # Security Notice
    if args.key:
        ASCIIColors.yellow("\n⚠️  Security Notice:")
        ASCIIColors.white("""    API Key authentication is enabled.
    Make sure to include the X-API-Key header in all your requests.
    """)

    # Ensure splash output flush to system log
    sys.stdout.flush()
