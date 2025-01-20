from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Request
from pydantic import BaseModel
import logging
import argparse
import json
import time
import re
from typing import List, Dict, Any, Optional, Union
from lightrag import LightRAG, QueryParam
from lightrag.llm import lollms_model_complete, lollms_embed
from lightrag.llm import ollama_model_complete, ollama_embed
from lightrag.llm import openai_complete_if_cache, openai_embedding
from lightrag.llm import azure_openai_complete_if_cache, azure_openai_embedding
from lightrag.api import __api_version__

from lightrag.utils import EmbeddingFunc
from enum import Enum
from pathlib import Path
import shutil
import aiofiles
from ascii_colors import trace_exception, ASCIIColors
import os

from fastapi import Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from starlette.status import HTTP_403_FORBIDDEN
import pipmaster as pm

from dotenv import load_dotenv

load_dotenv()


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in text
    Chinese characters: approximately 1.5 tokens per character
    English characters: approximately 0.25 tokens per character
    """
    # Use regex to match Chinese and non-Chinese characters separately
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    non_chinese_chars = len(re.findall(r"[^\u4e00-\u9fff]", text))

    # Calculate estimated token count
    tokens = chinese_chars * 1.5 + non_chinese_chars * 0.25

    return int(tokens)


# Constants for emulated Ollama model information
LIGHTRAG_NAME = "lightrag"
LIGHTRAG_TAG = os.getenv("OLLAMA_EMULATING_MODEL_TAG", "latest")
LIGHTRAG_MODEL = f"{LIGHTRAG_NAME}:{LIGHTRAG_TAG}"
LIGHTRAG_SIZE = 7365960935  # it's a dummy value
LIGHTRAG_CREATED_AT = "2024-01-15T00:00:00Z"
LIGHTRAG_DIGEST = "sha256:lightrag"


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


def get_env_value(env_key: str, default: Any, value_type: type = str) -> Any:
    """
    Get value from environment variable with type conversion

    Args:
        env_key (str): Environment variable key
        default (Any): Default value if env variable is not set
        value_type (type): Type to convert the value to

    Returns:
        Any: Converted value from environment or default
    """
    value = os.getenv(env_key)
    if value is None:
        return default

    if isinstance(value_type, bool):
        return value.lower() in ("true", "1", "yes")
    try:
        return value_type(value)
    except ValueError:
        return default


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
    ASCIIColors.white("    ├─ SSL Enabled: ", end="")
    ASCIIColors.yellow(f"{args.ssl}")
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
    ASCIIColors.white("    └─ Max Embed Tokens: ", end="")
    ASCIIColors.yellow(f"{args.max_embed_tokens}")

    # System Configuration
    ASCIIColors.magenta("\n🛠️ System Configuration:")
    ASCIIColors.white("    ├─ Ollama Emulating Model: ", end="")
    ASCIIColors.yellow(f"{LIGHTRAG_MODEL}")
    ASCIIColors.white("    ├─ Log Level: ", end="")
    ASCIIColors.yellow(f"{args.log_level}")
    ASCIIColors.white("    ├─ Timeout: ", end="")
    ASCIIColors.yellow(f"{args.timeout if args.timeout else 'None (infinite)'}")
    ASCIIColors.white("    └─ API Key: ", end="")
    ASCIIColors.yellow("Set" if args.key else "Not Set")

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
        ASCIIColors.white("    └─ Alternative Documentation (local): ", end="")
        ASCIIColors.yellow(f"{protocol}://localhost:{args.port}/redoc")

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

    ASCIIColors.green("Server is ready to accept connections! 🚀\n")


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments with environment variable fallback

    Returns:
        argparse.Namespace: Parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="LightRAG FastAPI Server with separate working and input directories"
    )

    # Bindings (with env var support)
    parser.add_argument(
        "--llm-binding",
        default=get_env_value("LLM_BINDING", "ollama"),
        help="LLM binding to be used. Supported: lollms, ollama, openai (default: from env or ollama)",
    )
    parser.add_argument(
        "--embedding-binding",
        default=get_env_value("EMBEDDING_BINDING", "ollama"),
        help="Embedding binding to be used. Supported: lollms, ollama, openai (default: from env or ollama)",
    )

    # Parse temporary args for host defaults
    temp_args, _ = parser.parse_known_args()

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

    # LLM Model configuration
    default_llm_host = get_env_value(
        "LLM_BINDING_HOST", get_default_host(temp_args.llm_binding)
    )
    parser.add_argument(
        "--llm-binding-host",
        default=default_llm_host,
        help=f"llm server host URL (default: from env or {default_llm_host})",
    )

    default_llm_api_key = get_env_value("LLM_BINDING_API_KEY", None)

    parser.add_argument(
        "--llm-binding-api-key",
        default=default_llm_api_key,
        help="llm server API key (default: from env or empty string)",
    )

    parser.add_argument(
        "--llm-model",
        default=get_env_value("LLM_MODEL", "mistral-nemo:latest"),
        help="LLM model name (default: from env or mistral-nemo:latest)",
    )

    # Embedding model configuration
    default_embedding_host = get_env_value(
        "EMBEDDING_BINDING_HOST", get_default_host(temp_args.embedding_binding)
    )
    parser.add_argument(
        "--embedding-binding-host",
        default=default_embedding_host,
        help=f"embedding server host URL (default: from env or {default_embedding_host})",
    )

    default_embedding_api_key = get_env_value("EMBEDDING_BINDING_API_KEY", "")
    parser.add_argument(
        "--embedding-binding-api-key",
        default=default_embedding_api_key,
        help="embedding server API key (default: from env or empty string)",
    )

    parser.add_argument(
        "--embedding-model",
        default=get_env_value("EMBEDDING_MODEL", "bge-m3:latest"),
        help="Embedding model name (default: from env or bge-m3:latest)",
    )

    def timeout_type(value):
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
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=get_env_value("EMBEDDING_DIM", 1024, int),
        help="Embedding dimensions (default: from env or 1024)",
    )
    parser.add_argument(
        "--max-embed-tokens",
        type=int,
        default=get_env_value("MAX_EMBED_TOKENS", 8192, int),
        help="Maximum embedding token size (default: from env or 8192)",
    )

    # Logging configuration
    parser.add_argument(
        "--log-level",
        default=get_env_value("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: from env or INFO)",
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

    args = parser.parse_args()
    display_splash_screen(args)

    return args


class DocumentManager:
    """Handles document operations and tracking"""

    def __init__(
        self,
        input_dir: str,
        supported_extensions: tuple = (".txt", ".md", ".pdf", ".docx", ".pptx"),
    ):
        self.input_dir = Path(input_dir)
        self.supported_extensions = supported_extensions
        self.indexed_files = set()

        # Create input directory if it doesn't exist
        self.input_dir.mkdir(parents=True, exist_ok=True)

    def scan_directory(self) -> List[Path]:
        """Scan input directory for new files"""
        new_files = []
        for ext in self.supported_extensions:
            for file_path in self.input_dir.rglob(f"*{ext}"):
                if file_path not in self.indexed_files:
                    new_files.append(file_path)
        return new_files

    def mark_as_indexed(self, file_path: Path):
        """Mark a file as indexed"""
        self.indexed_files.add(file_path)

    def is_supported_file(self, filename: str) -> bool:
        """Check if file type is supported"""
        return any(filename.lower().endswith(ext) for ext in self.supported_extensions)


# Pydantic models
class SearchMode(str, Enum):
    naive = "naive"
    local = "local"
    global_ = "global"
    hybrid = "hybrid"
    mix = "mix"


class OllamaMessage(BaseModel):
    role: str
    content: str
    images: Optional[List[str]] = None


class OllamaChatRequest(BaseModel):
    model: str = LIGHTRAG_MODEL
    messages: List[OllamaMessage]
    stream: bool = True  # Default to streaming mode
    options: Optional[Dict[str, Any]] = None


class OllamaChatResponse(BaseModel):
    model: str
    created_at: str
    message: OllamaMessage
    done: bool


class OllamaVersionResponse(BaseModel):
    version: str


class OllamaModelDetails(BaseModel):
    parent_model: str
    format: str
    family: str
    families: List[str]
    parameter_size: str
    quantization_level: str


class OllamaModel(BaseModel):
    name: str
    model: str
    size: int
    digest: str
    modified_at: str
    details: OllamaModelDetails


class OllamaTagResponse(BaseModel):
    models: List[OllamaModel]


class QueryRequest(BaseModel):
    query: str
    mode: SearchMode = SearchMode.hybrid
    stream: bool = False
    only_need_context: bool = False


class QueryResponse(BaseModel):
    response: str


class InsertTextRequest(BaseModel):
    text: str
    description: Optional[str] = None


class InsertResponse(BaseModel):
    status: str
    message: str
    document_count: int


def get_api_key_dependency(api_key: Optional[str]):
    if not api_key:
        # If no API key is configured, return a dummy dependency that always succeeds
        async def no_auth():
            return None

        return no_auth

    # If API key is configured, use proper authentication
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    async def api_key_auth(api_key_header_value: str | None = Security(api_key_header)):
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


def create_app(args):
    # Verify that bindings arer correctly setup
    if args.llm_binding not in ["lollms", "ollama", "openai"]:
        raise Exception("llm binding not supported")

    if args.embedding_binding not in ["lollms", "ollama", "openai"]:
        raise Exception("embedding binding not supported")

    # Add SSL validation
    if args.ssl:
        if not args.ssl_certfile or not args.ssl_keyfile:
            raise Exception(
                "SSL certificate and key files must be provided when SSL is enabled"
            )
        if not os.path.exists(args.ssl_certfile):
            raise Exception(f"SSL certificate file not found: {args.ssl_certfile}")
        if not os.path.exists(args.ssl_keyfile):
            raise Exception(f"SSL key file not found: {args.ssl_keyfile}")

    # Setup logging
    logging.basicConfig(
        format="%(levelname)s:%(message)s", level=getattr(logging, args.log_level)
    )

    # Check if API key is provided either through env var or args
    api_key = os.getenv("LIGHTRAG_API_KEY") or args.key

    # Initialize document manager
    doc_manager = DocumentManager(args.input_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for startup and shutdown events"""
        # Startup logic
        try:
            new_files = doc_manager.scan_directory()
            for file_path in new_files:
                try:
                    await index_file(file_path)
                except Exception as e:
                    trace_exception(e)
                    logging.error(f"Error indexing file {file_path}: {str(e)}")

            logging.info(f"Indexed {len(new_files)} documents from {args.input_dir}")
        except Exception as e:
            logging.error(f"Error during startup indexing: {str(e)}")
        yield
        # Cleanup logic (if needed)
        pass

    # Initialize FastAPI
    app = FastAPI(
        title="LightRAG API",
        description="API for querying text using LightRAG with separate storage and input directories"
        + "(With authentication)"
        if api_key
        else "",
        version=__api_version__,
        openapi_tags=[{"name": "api"}],
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Create the optional API key dependency
    optional_api_key = get_api_key_dependency(api_key)

    # Create working directory if it doesn't exist
    Path(args.working_dir).mkdir(parents=True, exist_ok=True)

    async def openai_alike_model_complete(
        prompt,
        system_prompt=None,
        history_messages=[],
        keyword_extraction=False,
        **kwargs,
    ) -> str:
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
        history_messages=[],
        keyword_extraction=False,
        **kwargs,
    ) -> str:
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
        func=lambda texts: lollms_embed(
            texts,
            embed_model=args.embedding_model,
            host=args.embedding_binding_host,
            api_key=args.embedding_binding_api_key,
        )
        if args.embedding_binding == "lollms"
        else ollama_embed(
            texts,
            embed_model=args.embedding_model,
            host=args.embedding_binding_host,
            api_key=args.embedding_binding_api_key,
        )
        if args.embedding_binding == "ollama"
        else azure_openai_embedding(
            texts,
            model=args.embedding_model,  # no host is used for openai,
            api_key=args.embedding_binding_api_key,
        )
        if args.embedding_binding == "azure_openai"
        else openai_embedding(
            texts,
            model=args.embedding_model,  # no host is used for openai,
            api_key=args.embedding_binding_api_key,
        ),
    )

    # Initialize RAG
    if args.llm_binding in ["lollms", "ollama"]:
        rag = LightRAG(
            working_dir=args.working_dir,
            llm_model_func=lollms_model_complete
            if args.llm_binding == "lollms"
            else ollama_model_complete,
            llm_model_name=args.llm_model,
            llm_model_max_async=args.max_async,
            llm_model_max_token_size=args.max_tokens,
            llm_model_kwargs={
                "host": args.llm_binding_host,
                "timeout": args.timeout,
                "options": {"num_ctx": args.max_tokens},
                "api_key": args.llm_binding_api_key,
            },
            embedding_func=embedding_func,
        )
    else:
        rag = LightRAG(
            working_dir=args.working_dir,
            llm_model_func=azure_openai_model_complete
            if args.llm_binding == "azure_openai"
            else openai_alike_model_complete,
            embedding_func=embedding_func,
        )

    async def index_file(file_path: Union[str, Path]) -> None:
        """Index all files inside the folder with support for multiple file formats

        Args:
            file_path: Path to the file to be indexed (str or Path object)

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        if not pm.is_installed("aiofiles"):
            pm.install("aiofiles")

        # Convert to Path object if string
        file_path = Path(file_path)

        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = ""
        # Get file extension in lowercase
        ext = file_path.suffix.lower()

        match ext:
            case ".txt" | ".md":
                # Text files handling
                async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                    content = await f.read()

            case ".pdf":
                if not pm.is_installed("pypdf2"):
                    pm.install("pypdf2")
                from PyPDF2 import PdfReader

                # PDF handling
                reader = PdfReader(str(file_path))
                content = ""
                for page in reader.pages:
                    content += page.extract_text() + "\n"

            case ".docx":
                if not pm.is_installed("python-docx"):
                    pm.install("python-docx")
                from docx import Document

                # Word document handling
                doc = Document(file_path)
                content = "\n".join([paragraph.text for paragraph in doc.paragraphs])

            case ".pptx":
                if not pm.is_installed("pptx"):
                    pm.install("pptx")
                from pptx import Presentation

                # PowerPoint handling
                prs = Presentation(file_path)
                content = ""
                for slide in prs.slides:
                    for shape in slide.shapes:
                        if hasattr(shape, "text"):
                            content += shape.text + "\n"

            case _:
                raise ValueError(f"Unsupported file format: {ext}")

        # Insert content into RAG system
        if content:
            await rag.ainsert(content)
            doc_manager.mark_as_indexed(file_path)
            logging.info(f"Successfully indexed file: {file_path}")
        else:
            logging.warning(f"No content extracted from file: {file_path}")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for startup and shutdown events"""
        # Startup logic
        try:
            new_files = doc_manager.scan_directory()
            for file_path in new_files:
                try:
                    await index_file(file_path)
                except Exception as e:
                    trace_exception(e)
                    logging.error(f"Error indexing file {file_path}: {str(e)}")

            logging.info(f"Indexed {len(new_files)} documents from {args.input_dir}")
        except Exception as e:
            logging.error(f"Error during startup indexing: {str(e)}")

    @app.post("/documents/scan", dependencies=[Depends(optional_api_key)])
    async def scan_for_new_documents():
        """Manually trigger scanning for new documents"""
        try:
            new_files = doc_manager.scan_directory()
            indexed_count = 0

            for file_path in new_files:
                try:
                    await index_file(file_path)
                    indexed_count += 1
                except Exception as e:
                    logging.error(f"Error indexing file {file_path}: {str(e)}")

            return {
                "status": "success",
                "indexed_count": indexed_count,
                "total_documents": len(doc_manager.indexed_files),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/documents/upload", dependencies=[Depends(optional_api_key)])
    async def upload_to_input_dir(file: UploadFile = File(...)):
        """Upload a file to the input directory"""
        try:
            if not doc_manager.is_supported_file(file.filename):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type. Supported types: {doc_manager.supported_extensions}",
                )

            file_path = doc_manager.input_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Immediately index the uploaded file
            await index_file(file_path)

            return {
                "status": "success",
                "message": f"File uploaded and indexed: {file.filename}",
                "total_documents": len(doc_manager.indexed_files),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/query", response_model=QueryResponse, dependencies=[Depends(optional_api_key)]
    )
    async def query_text(request: QueryRequest):
        try:
            response = await rag.aquery(
                request.query,
                param=QueryParam(
                    mode=request.mode,
                    stream=request.stream,
                    only_need_context=request.only_need_context,
                ),
            )

            # If response is a string (e.g. cache hit), return directly
            if isinstance(response, str):
                return QueryResponse(response=response)

            # If it's an async generator, decide whether to stream based on stream parameter
            if request.stream:
                result = ""
                async for chunk in response:
                    result += chunk
                return QueryResponse(response=result)
            else:
                result = ""
                async for chunk in response:
                    result += chunk
                return QueryResponse(response=result)
        except Exception as e:
            trace_exception(e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/query/stream", dependencies=[Depends(optional_api_key)])
    async def query_text_stream(request: QueryRequest):
        try:
            response = await rag.aquery(  # Use aquery instead of query, and add await
                request.query,
                param=QueryParam(
                    mode=request.mode,
                    stream=True,
                    only_need_context=request.only_need_context,
                ),
            )

            from fastapi.responses import StreamingResponse

            async def stream_generator():
                if isinstance(response, str):
                    # If it's a string, send it all at once
                    yield f"{json.dumps({'response': response})}\n"
                else:
                    # If it's an async generator, send chunks one by one
                    try:
                        async for chunk in response:
                            if chunk:  # Only send non-empty content
                                yield f"{json.dumps({'response': chunk})}\n"
                    except Exception as e:
                        logging.error(f"Streaming error: {str(e)}")
                        yield f"{json.dumps({'error': str(e)})}\n"

            return StreamingResponse(
                stream_generator(),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "application/x-ndjson",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "POST, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type",
                    "X-Accel-Buffering": "no",  # Disable Nginx buffering
                },
            )
        except Exception as e:
            trace_exception(e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/documents/text",
        response_model=InsertResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def insert_text(request: InsertTextRequest):
        try:
            await rag.ainsert(request.text)
            return InsertResponse(
                status="success",
                message="Text successfully inserted",
                document_count=1,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/documents/file",
        response_model=InsertResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def insert_file(file: UploadFile = File(...), description: str = Form(None)):
        """Insert a file directly into the RAG system

        Args:
            file: Uploaded file
            description: Optional description of the file

        Returns:
            InsertResponse: Status of the insertion operation

        Raises:
            HTTPException: For unsupported file types or processing errors
        """
        try:
            content = ""
            # Get file extension in lowercase
            ext = Path(file.filename).suffix.lower()

            match ext:
                case ".txt" | ".md":
                    # Text files handling
                    text_content = await file.read()
                    content = text_content.decode("utf-8")

                case ".pdf":
                    if not pm.is_installed("pypdf2"):
                        pm.install("pypdf2")
                    from PyPDF2 import PdfReader
                    from io import BytesIO

                    # Read PDF from memory
                    pdf_content = await file.read()
                    pdf_file = BytesIO(pdf_content)
                    reader = PdfReader(pdf_file)
                    content = ""
                    for page in reader.pages:
                        content += page.extract_text() + "\n"

                case ".docx":
                    if not pm.is_installed("python-docx"):
                        pm.install("python-docx")
                    from docx import Document
                    from io import BytesIO

                    # Read DOCX from memory
                    docx_content = await file.read()
                    docx_file = BytesIO(docx_content)
                    doc = Document(docx_file)
                    content = "\n".join(
                        [paragraph.text for paragraph in doc.paragraphs]
                    )

                case ".pptx":
                    if not pm.is_installed("pptx"):
                        pm.install("pptx")
                    from pptx import Presentation
                    from io import BytesIO

                    # Read PPTX from memory
                    pptx_content = await file.read()
                    pptx_file = BytesIO(pptx_content)
                    prs = Presentation(pptx_file)
                    content = ""
                    for slide in prs.slides:
                        for shape in slide.shapes:
                            if hasattr(shape, "text"):
                                content += shape.text + "\n"

                case _:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported file type. Supported types: {doc_manager.supported_extensions}",
                    )

            # Insert content into RAG system
            if content:
                # Add description if provided
                if description:
                    content = f"{description}\n\n{content}"

                await rag.ainsert(content)
                logging.info(f"Successfully indexed file: {file.filename}")

                return InsertResponse(
                    status="success",
                    message=f"File '{file.filename}' successfully inserted",
                    document_count=1,
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No content could be extracted from the file",
                )

        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File encoding not supported")
        except Exception as e:
            logging.error(f"Error processing file {file.filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/documents/batch",
        response_model=InsertResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def insert_batch(files: List[UploadFile] = File(...)):
        """Process multiple files in batch mode

        Args:
            files: List of files to process

        Returns:
            InsertResponse: Status of the batch insertion operation

        Raises:
            HTTPException: For processing errors
        """
        try:
            inserted_count = 0
            failed_files = []

            for file in files:
                try:
                    content = ""
                    ext = Path(file.filename).suffix.lower()

                    match ext:
                        case ".txt" | ".md":
                            text_content = await file.read()
                            content = text_content.decode("utf-8")

                        case ".pdf":
                            if not pm.is_installed("pypdf2"):
                                pm.install("pypdf2")
                            from PyPDF2 import PdfReader
                            from io import BytesIO

                            pdf_content = await file.read()
                            pdf_file = BytesIO(pdf_content)
                            reader = PdfReader(pdf_file)
                            for page in reader.pages:
                                content += page.extract_text() + "\n"

                        case ".docx":
                            if not pm.is_installed("docx"):
                                pm.install("docx")
                            from docx import Document
                            from io import BytesIO

                            docx_content = await file.read()
                            docx_file = BytesIO(docx_content)
                            doc = Document(docx_file)
                            content = "\n".join(
                                [paragraph.text for paragraph in doc.paragraphs]
                            )

                        case ".pptx":
                            if not pm.is_installed("pptx"):
                                pm.install("pptx")
                            from pptx import Presentation
                            from io import BytesIO

                            pptx_content = await file.read()
                            pptx_file = BytesIO(pptx_content)
                            prs = Presentation(pptx_file)
                            for slide in prs.slides:
                                for shape in slide.shapes:
                                    if hasattr(shape, "text"):
                                        content += shape.text + "\n"

                        case _:
                            failed_files.append(f"{file.filename} (unsupported type)")
                            continue

                    if content:
                        await rag.ainsert(content)
                        inserted_count += 1
                        logging.info(f"Successfully indexed file: {file.filename}")
                    else:
                        failed_files.append(f"{file.filename} (no content extracted)")

                except UnicodeDecodeError:
                    failed_files.append(f"{file.filename} (encoding error)")
                except Exception as e:
                    failed_files.append(f"{file.filename} ({str(e)})")
                    logging.error(f"Error processing file {file.filename}: {str(e)}")

            # Prepare status message
            if inserted_count == len(files):
                status = "success"
                status_message = f"Successfully inserted all {inserted_count} documents"
            elif inserted_count > 0:
                status = "partial_success"
                status_message = f"Successfully inserted {inserted_count} out of {len(files)} documents"
                if failed_files:
                    status_message += f". Failed files: {', '.join(failed_files)}"
            else:
                status = "failure"
                status_message = "No documents were successfully inserted"
                if failed_files:
                    status_message += f". Failed files: {', '.join(failed_files)}"

            return InsertResponse(
                status=status,
                message=status_message,
                document_count=inserted_count,
            )

        except Exception as e:
            logging.error(f"Batch processing error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete(
        "/documents",
        response_model=InsertResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def clear_documents():
        try:
            rag.text_chunks = []
            rag.entities_vdb = None
            rag.relationships_vdb = None
            return InsertResponse(
                status="success",
                message="All documents cleared successfully",
                document_count=0,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Ollama compatible API endpoints
    @app.get("/api/version")
    async def get_version():
        """Get Ollama version information"""
        return OllamaVersionResponse(version="0.5.4")

    @app.get("/api/tags")
    async def get_tags():
        """Get available models"""
        return OllamaTagResponse(
            models=[
                {
                    "name": LIGHTRAG_MODEL,
                    "model": LIGHTRAG_MODEL,
                    "size": LIGHTRAG_SIZE,
                    "digest": LIGHTRAG_DIGEST,
                    "modified_at": LIGHTRAG_CREATED_AT,
                    "details": {
                        "parent_model": "",
                        "format": "gguf",
                        "family": LIGHTRAG_NAME,
                        "families": [LIGHTRAG_NAME],
                        "parameter_size": "13B",
                        "quantization_level": "Q4_0",
                    },
                }
            ]
        )

    def parse_query_mode(query: str) -> tuple[str, SearchMode]:
        """Parse query prefix to determine search mode
        Returns tuple of (cleaned_query, search_mode)
        """
        mode_map = {
            "/local ": SearchMode.local,
            "/global ": SearchMode.global_,  # global_ is used because 'global' is a Python keyword
            "/naive ": SearchMode.naive,
            "/hybrid ": SearchMode.hybrid,
            "/mix ": SearchMode.mix,
        }

        for prefix, mode in mode_map.items():
            if query.startswith(prefix):
                # After removing prefix an leading spaces
                cleaned_query = query[len(prefix) :].lstrip()
                return cleaned_query, mode

        return query, SearchMode.hybrid

    @app.post("/api/chat")
    async def chat(raw_request: Request, request: OllamaChatRequest):
        """Handle chat completion requests"""
        try:
            # Get all messages
            messages = request.messages
            if not messages:
                raise HTTPException(status_code=400, detail="No messages provided")

            # Get the last message as query
            query = messages[-1].content

            # 解析查询模式
            cleaned_query, mode = parse_query_mode(query)

            # 开始计时
            start_time = time.time_ns()

            # 计算输入token数量
            prompt_tokens = estimate_tokens(cleaned_query)

            # 调用RAG进行查询
            query_param = QueryParam(
                mode=mode, stream=request.stream, only_need_context=False
            )

            if request.stream:
                from fastapi.responses import StreamingResponse

                response = await rag.aquery(  # Need await to get async generator
                    cleaned_query, param=query_param
                )

                async def stream_generator():
                    try:
                        first_chunk_time = None
                        last_chunk_time = None
                        total_response = ""

                        # Ensure response is an async generator
                        if isinstance(response, str):
                            # If it's a string, send in two parts
                            first_chunk_time = time.time_ns()
                            last_chunk_time = first_chunk_time
                            total_response = response

                            data = {
                                "model": LIGHTRAG_MODEL,
                                "created_at": LIGHTRAG_CREATED_AT,
                                "message": {
                                    "role": "assistant",
                                    "content": response,
                                    "images": None,
                                },
                                "done": False,
                            }
                            yield f"{json.dumps(data, ensure_ascii=False)}\n"

                            completion_tokens = estimate_tokens(total_response)
                            total_time = last_chunk_time - start_time
                            prompt_eval_time = first_chunk_time - start_time
                            eval_time = last_chunk_time - first_chunk_time

                            data = {
                                "model": LIGHTRAG_MODEL,
                                "created_at": LIGHTRAG_CREATED_AT,
                                "done": True,
                                "total_duration": total_time,
                                "load_duration": 0,
                                "prompt_eval_count": prompt_tokens,
                                "prompt_eval_duration": prompt_eval_time,
                                "eval_count": completion_tokens,
                                "eval_duration": eval_time,
                            }
                            yield f"{json.dumps(data, ensure_ascii=False)}\n"
                        else:
                            async for chunk in response:
                                if chunk:
                                    if first_chunk_time is None:
                                        first_chunk_time = time.time_ns()

                                    last_chunk_time = time.time_ns()

                                    total_response += chunk
                                    data = {
                                        "model": LIGHTRAG_MODEL,
                                        "created_at": LIGHTRAG_CREATED_AT,
                                        "message": {
                                            "role": "assistant",
                                            "content": chunk,
                                            "images": None,
                                        },
                                        "done": False,
                                    }
                                    yield f"{json.dumps(data, ensure_ascii=False)}\n"

                            completion_tokens = estimate_tokens(total_response)
                            total_time = last_chunk_time - start_time
                            prompt_eval_time = first_chunk_time - start_time
                            eval_time = last_chunk_time - first_chunk_time

                            data = {
                                "model": LIGHTRAG_MODEL,
                                "created_at": LIGHTRAG_CREATED_AT,
                                "done": True,
                                "total_duration": total_time,
                                "load_duration": 0,
                                "prompt_eval_count": prompt_tokens,
                                "prompt_eval_duration": prompt_eval_time,
                                "eval_count": completion_tokens,
                                "eval_duration": eval_time,
                            }
                            yield f"{json.dumps(data, ensure_ascii=False)}\n"
                            return  # Ensure the generator ends immediately after sending the completion marker
                    except Exception as e:
                        logging.error(f"Error in stream_generator: {str(e)}")
                        raise

                return StreamingResponse(
                    stream_generator(),
                    media_type="application/x-ndjson",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Content-Type": "application/x-ndjson",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Methods": "POST, OPTIONS",
                        "Access-Control-Allow-Headers": "Content-Type",
                    },
                )
            else:
                first_chunk_time = time.time_ns()
                response_text = await rag.aquery(cleaned_query, param=query_param)
                last_chunk_time = time.time_ns()

                if not response_text:
                    response_text = "No response generated"

                completion_tokens = estimate_tokens(str(response_text))
                total_time = last_chunk_time - start_time
                prompt_eval_time = first_chunk_time - start_time
                eval_time = last_chunk_time - first_chunk_time

                return {
                    "model": LIGHTRAG_MODEL,
                    "created_at": LIGHTRAG_CREATED_AT,
                    "message": {
                        "role": "assistant",
                        "content": str(response_text),
                        "images": None,
                    },
                    "done": True,
                    "total_duration": total_time,
                    "load_duration": 0,
                    "prompt_eval_count": prompt_tokens,
                    "prompt_eval_duration": prompt_eval_time,
                    "eval_count": completion_tokens,
                    "eval_duration": eval_time,
                }
        except Exception as e:
            trace_exception(e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health", dependencies=[Depends(optional_api_key)])
    async def get_status():
        """Get current system status"""
        return {
            "status": "healthy",
            "working_directory": str(args.working_dir),
            "input_directory": str(args.input_dir),
            "indexed_files": len(doc_manager.indexed_files),
            "configuration": {
                # LLM configuration binding/host address (if applicable)/model (if applicable)
                "llm_binding": args.llm_binding,
                "llm_binding_host": args.llm_binding_host,
                "llm_model": args.llm_model,
                # embedding model configuration binding/host address (if applicable)/model (if applicable)
                "embedding_binding": args.embedding_binding,
                "embedding_binding_host": args.embedding_binding_host,
                "embedding_model": args.embedding_model,
                "max_tokens": args.max_tokens,
            },
        }

    return app


def main():
    args = parse_args()
    import uvicorn

    app = create_app(args)
    uvicorn_config = {
        "app": app,
        "host": args.host,
        "port": args.port,
    }
    if args.ssl:
        uvicorn_config.update(
            {
                "ssl_certfile": args.ssl_certfile,
                "ssl_keyfile": args.ssl_keyfile,
            }
        )
    uvicorn.run(**uvicorn_config)


if __name__ == "__main__":
    main()
