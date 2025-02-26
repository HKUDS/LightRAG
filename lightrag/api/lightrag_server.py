"""
LightRAG FastAPI Server
"""

from fastapi import (
    FastAPI,
    Depends,
)
from fastapi.responses import FileResponse
import asyncio
import os
import json
import logging
import logging.config
import uvicorn
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import configparser
from ascii_colors import ASCIIColors
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from .utils_api import (
    get_api_key_dependency,
    parse_args,
    get_default_host,
    display_splash_screen,
)
from lightrag import LightRAG
from lightrag.types import GPTKeywordExtractionFormat
from lightrag.api import __api_version__
from lightrag.utils import EmbeddingFunc
from .routers.document_routes import (
    DocumentManager,
    create_document_routes,
    run_scanning_process,
)
from .routers.query_routes import create_query_routes
from .routers.graph_routes import create_graph_routes
from .routers.ollama_api import OllamaAPI

from lightrag.utils import logger as utils_logger

# Load environment variables
try:
    load_dotenv(override=True)
except Exception as e:
    utils_logger.warning(f"Failed to load .env file: {e}")

# Initialize config parser
config = configparser.ConfigParser()
config.read("config.ini")


class LightragPathFilter(logging.Filter):
    """Filter for lightrag logger to filter out frequent path access logs"""
    def __init__(self):
        super().__init__()
        # Define paths to be filtered
        self.filtered_paths = ["/documents", "/health", "/webui/"]
    
    def filter(self, record):
        try:
            # Check if record has the required attributes for an access log
            if not hasattr(record, "args") or not isinstance(record.args, tuple):
                return True
            if len(record.args) < 5:
                return True

            # Extract method, path and status from the record args
            method = record.args[1]
            path = record.args[2]
            status = record.args[4]

            # Filter out successful GET requests to filtered paths
            if (
                method == "GET"
                and (status == 200 or status == 304)
                and path in self.filtered_paths
            ):
                return False

            return True
        except Exception:
            # In case of any error, let the message through
            return True


def create_app(args):
    # Initialize verbose debug setting
    # Can not use the logger at the top of this module when workers > 1
    from lightrag.utils import set_verbose_debug, logger
    # Setup logging
    logger.setLevel(getattr(logging, args.log_level))
    set_verbose_debug(args.verbose)

    from lightrag.kg.shared_storage import is_multiprocess
    logger.info(f"==== Multi-processor mode: {is_multiprocess} ====")

    # Verify that bindings are correctly setup
    if args.llm_binding not in [
        "lollms",
        "ollama",
        "openai",
        "openai-ollama",
        "azure_openai",
    ]:
        raise Exception("llm binding not supported")

    if args.embedding_binding not in ["lollms", "ollama", "openai", "azure_openai"]:
        raise Exception("embedding binding not supported")

    # Set default hosts if not provided
    if args.llm_binding_host is None:
        args.llm_binding_host = get_default_host(args.llm_binding)

    if args.embedding_binding_host is None:
        args.embedding_binding_host = get_default_host(args.embedding_binding)

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

    # Check if API key is provided either through env var or args
    api_key = os.getenv("LIGHTRAG_API_KEY") or args.key

    # Initialize document manager
    doc_manager = DocumentManager(args.input_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for startup and shutdown events"""
        # Store background tasks
        app.state.background_tasks = set()

        try:
            # Initialize database connections
            await rag.initialize_storages()

            # Auto scan documents if enabled
            if args.auto_scan_at_startup:
                # Create background task
                task = asyncio.create_task(
                    run_scanning_process(rag, doc_manager)
                )
                app.state.background_tasks.add(task)
                task.add_done_callback(app.state.background_tasks.discard)

            ASCIIColors.green("\nServer is ready to accept connections! 🚀\n")

            yield

        finally:
            # Clean up database connections
            await rag.finalize_storages()

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

    def get_cors_origins():
        """Get allowed origins from environment variable
        Returns a list of allowed origins, defaults to ["*"] if not set
        """
        origins_str = os.getenv("CORS_ORIGINS", "*")
        if origins_str == "*":
            return ["*"]
        return [origin.strip() for origin in origins_str.split(",")]

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Create the optional API key dependency
    optional_api_key = get_api_key_dependency(api_key)

    # Create working directory if it doesn't exist
    Path(args.working_dir).mkdir(parents=True, exist_ok=True)
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
        else azure_openai_embed(
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
        ),
    )

    # Initialize RAG
    if args.llm_binding in ["lollms", "ollama", "openai"]:
        rag = LightRAG(
            working_dir=args.working_dir,
            llm_model_func=lollms_model_complete
            if args.llm_binding == "lollms"
            else ollama_model_complete
            if args.llm_binding == "ollama"
            else openai_alike_model_complete,
            llm_model_name=args.llm_model,
            llm_model_max_async=args.max_async,
            llm_model_max_token_size=args.max_tokens,
            chunk_token_size=int(args.chunk_size),
            chunk_overlap_token_size=int(args.chunk_overlap_size),
            llm_model_kwargs={
                "host": args.llm_binding_host,
                "timeout": args.timeout,
                "options": {"num_ctx": args.max_tokens},
                "api_key": args.llm_binding_api_key,
            }
            if args.llm_binding == "lollms" or args.llm_binding == "ollama"
            else {},
            embedding_func=embedding_func,
            kv_storage=args.kv_storage,
            graph_storage=args.graph_storage,
            vector_storage=args.vector_storage,
            doc_status_storage=args.doc_status_storage,
            vector_db_storage_cls_kwargs={
                "cosine_better_than_threshold": args.cosine_threshold
            },
            enable_llm_cache_for_entity_extract=False,  # set to True for debuging to reduce llm fee
            embedding_cache_config={
                "enabled": True,
                "similarity_threshold": 0.95,
                "use_llm_check": False,
            },
            log_level=args.log_level,
            namespace_prefix=args.namespace_prefix,
            auto_manage_storages_states=False,
        )
    else:  # azure_openai
        rag = LightRAG(
            working_dir=args.working_dir,
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
            enable_llm_cache_for_entity_extract=False,  # set to True for debuging to reduce llm fee
            embedding_cache_config={
                "enabled": True,
                "similarity_threshold": 0.95,
                "use_llm_check": False,
            },
            log_level=args.log_level,
            namespace_prefix=args.namespace_prefix,
            auto_manage_storages_states=False,
        )

    # Add routes
    app.include_router(create_document_routes(rag, doc_manager, api_key))
    app.include_router(create_query_routes(rag, api_key, args.top_k))
    app.include_router(create_graph_routes(rag, api_key))

    # Add Ollama API routes
    ollama_api = OllamaAPI(rag, top_k=args.top_k)
    app.include_router(ollama_api.router, prefix="/api")

    @app.get("/health", dependencies=[Depends(optional_api_key)])
    async def get_status():
        """Get current system status"""
        return {
            "status": "healthy",
            "working_directory": str(args.working_dir),
            "input_directory": str(args.input_dir),
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
                "kv_storage": args.kv_storage,
                "doc_status_storage": args.doc_status_storage,
                "graph_storage": args.graph_storage,
                "vector_storage": args.vector_storage,
            },
        }

    # Webui mount webui/index.html
    static_dir = Path(__file__).parent / "webui"
    static_dir.mkdir(exist_ok=True)
    app.mount(
        "/webui",
        StaticFiles(directory=static_dir, html=True, check_dir=True),
        name="webui",
    )

    @app.get("/webui/")
    async def webui_root():
        return FileResponse(static_dir / "index.html")

    return app


def get_application():
    """Factory function for creating the FastAPI application"""
    # Configure logging for this worker process
    configure_logging()
    
    # Get args from environment variable
    args_json = os.environ.get('LIGHTRAG_ARGS')
    if not args_json:
        args = parse_args()  # Fallback to parsing args if env var not set
    else:
        import types
        args = types.SimpleNamespace(**json.loads(args_json))
    
    if args.workers > 1:
        from lightrag.kg.shared_storage import initialize_share_data
        initialize_share_data()

    return create_app(args)


def configure_logging():
    """Configure logging for both uvicorn and lightrag"""
    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn.access", "lightrag"]:
        logger = logging.getLogger(logger_name)
        logger.handlers = []
        logger.filters = []
    
    # Configure basic logging
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(levelname)s: %(message)s",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
        },
        "loggers": {
            "uvicorn.access": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False,
                "filters": ["path_filter"],
            },
            "lightrag": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False,
                "filters": ["path_filter"],
            },
        },
        "filters": {
            "path_filter": {
                "()": "lightrag.api.lightrag_server.LightragPathFilter",
            },
        },
    })

def main():
    from multiprocessing import freeze_support
    freeze_support()
    
    args = parse_args()
    # Save args to environment variable for child processes
    os.environ['LIGHTRAG_ARGS'] = json.dumps(vars(args))

    # Configure logging before starting uvicorn
    configure_logging()

    display_splash_screen(args)


    uvicorn_config = {
        "app": "lightrag.api.lightrag_server:get_application",
        "factory": True,
        "host": args.host,
        "port": args.port,
        "workers": args.workers,
        "log_config": None,  # Disable default config
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
