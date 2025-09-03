"""
LightRAG FastAPI Server
"""
from __future__ import annotations
from fastapi import FastAPI, Depends, HTTPException
import asyncio
from dataclasses import dataclass
import os
import inspect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from lightrag.api.utils_api import get_combined_auth_dependency
from .config import (
    global_args,
    get_default_host,
)
from lightrag.utils import get_env_value
from lightrag import LightRAG, __version__ as core_version
from lightrag.api import __api_version__
from lightrag.types import GPTKeywordExtractionFormat
from lightrag.utils import EmbeddingFunc
from lightrag.constants import (
    DEFAULT_LLM_TIMEOUT,
    DEFAULT_EMBEDDING_TIMEOUT,
)
from lightrag.api.routers.document_routes import (
    DocumentManager,
    create_document_routes,
)
from lightrag.api.routers.query_routes import create_query_routes
from lightrag.api.routers.graph_routes import create_graph_routes
from lightrag.api.routers.ollama_api import OllamaAPI

from lightrag.utils import logger, set_verbose_debug
from lightrag.kg.shared_storage import (
    get_namespace_data,
    initialize_pipeline_status,
    cleanup_keyed_lock,
    finalize_share_data,
)
from fastapi.security import OAuth2PasswordRequestForm
from lightrag.api.auth import auth_handler

# use the .env that is inside the current folder
# allows to use different .env file for each lightrag instance
# the OS environment variables take precedence over the .env file
load_dotenv(dotenv_path=".env", override=False)

webui_title = os.getenv("WEBUI_TITLE")
webui_description = os.getenv("WEBUI_DESCRIPTION")

# Global authentication configuration
auth_configured = bool(auth_handler.accounts)

from cachetools import LRUCache
import hashlib, base64, unicodedata
from typing import Any

def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip()

def make_workspace_key(user_id: str, workspace: str, *, length: int = 26) -> str:
    a = _norm(user_id).encode("utf-8")
    b = _norm(workspace).encode("utf-8")
    payload = len(a).to_bytes(4, "big") + a + b"|" + len(b).to_bytes(4, "big") + b
    digest = hashlib.sha256(payload).digest()
    b32 = base64.b32encode(digest).decode("ascii").rstrip("=")
    return b32[:length]

@dataclass
class InstanceBundle:
    rag: LightRAG
    doc_manager: DocumentManager

class InstanceManager:
    """
    Multi-user, multi-workspace LightRAG instance manager.
    - LRU capped via cachetools
    - Async-safe creation
    - Proper cleanup on eviction and shutdown
    """

    def __init__(
            self,
            max_instances: int = 100,
            root_dir: str = "./rag_storage",
            input_dir: str = "./input",
            **lightrag_kwargs: Any,
    ):
        self._cache: LRUCache[str, InstanceBundle] = LRUCache(maxsize=max_instances)
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._input_dir = Path(input_dir)
        self._input_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._lightrag_kwargs = lightrag_kwargs
    
    async def get_instance(self, user_id: str, workspace: str) -> tuple[LightRAG, DocumentManager]:
        key = make_workspace_key(user_id, workspace)

        bundle = self._cache.get(key)
        if bundle is not None:
            return bundle.rag, bundle.doc_manager
        
        new_bundle = await self._create_bundle(user_id, workspace)

        async with self._lock:
            existing = self._cache.get(key)
            if existing is not None:
                await self._finalize_safely(new_bundle)
                return existing.rag, existing.doc_manager
            
            self._evict_one_if_needed()
            self._cache[key] = new_bundle
            return new_bundle.rag, new_bundle.doc_manager
    
    async def _create_bundle(self, user_id: str, workspace: str) -> InstanceBundle:
        working_dir = self._root / _norm(user_id)
        working_dir.mkdir(parents=True, exist_ok=True)

        rag = LightRAG(
            working_dir=str(working_dir),
            workspace=_norm(workspace),
            **self._lightrag_kwargs,
        )
        await rag.initialize_storages()
        await initialize_pipeline_status()

        doc_manager = DocumentManager(
            input_dir = str(self._input_dir),
            workspace = _norm(workspace)
        )

        return InstanceBundle(rag=rag, doc_manager=doc_manager)
    
    def _evict_one_if_needed(self) -> None:
        if len(self._cache) >= self._cache.maxsize:
            _, victim = self._cache.popitem()
            asyncio.create_task(self._finalize_safely(victim))

    async def _finalize_safely(self, bundle: InstanceBundle) -> None:
        try:
            await bundle.rag.finalize_storages()
        except Exception:
            pass
    
    async def aclose(self) -> None:
        # Close everything gracefully on shutdown
        async with self._lock:
            items = list(self._cache.items())
            self._cache.clear()
        await asyncio.gather(*(self._finalize_safely(b) for _, b in items))

def create_multi_workspace_app(args):
    logger.setLevel(args.log_level)
    set_verbose_debug(args.verbose)

    if args.llm_binding not in [
        "lollms",
        "ollama",
        "openai",
        "azure_openai",
        "aws_bedrock",
    ]:
        raise Exception("llm binding not supported")
    
    if args.embedding_binding not in [
        "lollms",
        "ollama",
        "openai",
        "azure_openai",
        "aws_bedrock",
        "jina",
    ]:
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

    def create_llm_model_func(binding: str):
        """
        Create LLM model function based on binding type.
        Uses lazy import to avoid unnecessary dependencies.
        """
        try:
            if binding == "lollms":
                from lightrag.llm.lollms import lollms_model_complete

                return lollms_model_complete
            elif binding == "ollama":
                from lightrag.llm.ollama import ollama_model_complete

                return ollama_model_complete
            elif binding == "aws_bedrock":
                return bedrock_model_complete  # Already defined locally
            elif binding == "azure_openai":
                return azure_openai_model_complete  # Already defined locally
            else:  # openai and compatible
                return openai_alike_model_complete  # Already defined locally
        except ImportError as e:
            raise Exception(f"Failed to import {binding} LLM binding: {e}")

    def create_llm_model_kwargs(binding: str, args, llm_timeout: int) -> dict:
        """
        Create LLM model kwargs based on binding type.
        Uses lazy import for binding-specific options.
        """
        if binding in ["lollms", "ollama"]:
            try:
                from lightrag.llm.binding_options import OllamaLLMOptions

                return {
                    "host": args.llm_binding_host,
                    "timeout": llm_timeout,
                    "options": OllamaLLMOptions.options_dict(args),
                    "api_key": args.llm_binding_api_key,
                }
            except ImportError as e:
                raise Exception(f"Failed to import {binding} options: {e}")
        return {}

    def create_embedding_function_with_lazy_import(
        binding, model, host, api_key, dimensions, args
    ):
        """
        Create embedding function with lazy imports for all bindings.
        Replaces the current create_embedding_function with full lazy import support.
        """

        async def embedding_function(texts):
            try:
                if binding == "lollms":
                    from lightrag.llm.lollms import lollms_embed

                    return await lollms_embed(
                        texts, embed_model=model, host=host, api_key=api_key
                    )
                elif binding == "ollama":
                    from lightrag.llm.binding_options import OllamaEmbeddingOptions
                    from lightrag.llm.ollama import ollama_embed

                    ollama_options = OllamaEmbeddingOptions.options_dict(args)
                    return await ollama_embed(
                        texts,
                        embed_model=model,
                        host=host,
                        api_key=api_key,
                        options=ollama_options,
                    )
                elif binding == "azure_openai":
                    from lightrag.llm.azure_openai import azure_openai_embed

                    return await azure_openai_embed(texts, model=model, api_key=api_key)
                elif binding == "aws_bedrock":
                    from lightrag.llm.bedrock import bedrock_embed

                    return await bedrock_embed(texts, model=model)
                elif binding == "jina":
                    from lightrag.llm.jina import jina_embed

                    return await jina_embed(
                        texts, dimensions=dimensions, base_url=host, api_key=api_key
                    )
                else:  # openai and compatible
                    from lightrag.llm.openai import openai_embed

                    return await openai_embed(
                        texts, model=model, base_url=host, api_key=api_key
                    )
            except ImportError as e:
                raise Exception(f"Failed to import {binding} embedding: {e}")

        return embedding_function

    llm_timeout = get_env_value("LLM_TIMEOUT", DEFAULT_LLM_TIMEOUT, int)
    embedding_timeout = get_env_value(
        "EMBEDDING_TIMEOUT", DEFAULT_EMBEDDING_TIMEOUT, int
    )

    async def openai_alike_model_complete(
        prompt,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ) -> str:
        # Lazy import
        from lightrag.llm.openai import openai_complete_if_cache
        from lightrag.llm.binding_options import OpenAILLMOptions

        keyword_extraction = kwargs.pop("keyword_extraction", None)
        if keyword_extraction:
            kwargs["response_format"] = GPTKeywordExtractionFormat
        if history_messages is None:
            history_messages = []

        # Use OpenAI LLM options if available
        openai_options = OpenAILLMOptions.options_dict(args)
        kwargs["timeout"] = llm_timeout
        kwargs.update(openai_options)

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
        # Lazy import
        from lightrag.llm.azure_openai import azure_openai_complete_if_cache
        from lightrag.llm.binding_options import OpenAILLMOptions

        keyword_extraction = kwargs.pop("keyword_extraction", None)
        if keyword_extraction:
            kwargs["response_format"] = GPTKeywordExtractionFormat
        if history_messages is None:
            history_messages = []

        # Use OpenAI LLM options
        openai_options = OpenAILLMOptions.options_dict(args)
        kwargs["timeout"] = llm_timeout
        kwargs.update(openai_options)

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

    async def bedrock_model_complete(
        prompt,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        **kwargs,
    ) -> str:
        # Lazy import
        from lightrag.llm.bedrock import bedrock_complete_if_cache

        keyword_extraction = kwargs.pop("keyword_extraction", None)
        if keyword_extraction:
            kwargs["response_format"] = GPTKeywordExtractionFormat
        if history_messages is None:
            history_messages = []

        # Use global temperature for Bedrock
        kwargs["temperature"] = get_env_value("BEDROCK_LLM_TEMPERATURE", 1.0, float)

        return await bedrock_complete_if_cache(
            args.llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )

    # Create embedding function with lazy imports
    embedding_func = EmbeddingFunc(
        embedding_dim=args.embedding_dim,
        func=create_embedding_function_with_lazy_import(
            binding=args.embedding_binding,
            model=args.embedding_model,
            host=args.embedding_binding_host,
            api_key=args.embedding_binding_api_key,
            dimensions=args.embedding_dim,
            args=args,  # Pass args object for dynamic option generation
        ),
    )

    # Configure rerank function based on args.rerank_bindingparameter
    rerank_model_func = None
    if args.rerank_binding != "null":
        from lightrag.rerank import cohere_rerank, jina_rerank, ali_rerank

        # Map rerank binding to corresponding function
        rerank_functions = {
            "cohere": cohere_rerank,
            "jina": jina_rerank,
            "aliyun": ali_rerank,
        }

        # Select the appropriate rerank function based on binding
        selected_rerank_func = rerank_functions.get(args.rerank_binding)
        if not selected_rerank_func:
            logger.error(f"Unsupported rerank binding: {args.rerank_binding}")
            raise ValueError(f"Unsupported rerank binding: {args.rerank_binding}")

        # Get default values from selected_rerank_func if args values are None
        if args.rerank_model is None or args.rerank_binding_host is None:
            sig = inspect.signature(selected_rerank_func)

            # Set default model if args.rerank_model is None
            if args.rerank_model is None and "model" in sig.parameters:
                default_model = sig.parameters["model"].default
                if default_model != inspect.Parameter.empty:
                    args.rerank_model = default_model

            # Set default base_url if args.rerank_binding_host is None
            if args.rerank_binding_host is None and "base_url" in sig.parameters:
                default_base_url = sig.parameters["base_url"].default
                if default_base_url != inspect.Parameter.empty:
                    args.rerank_binding_host = default_base_url

        async def server_rerank_func(
            query: str, documents: list, top_n: int = None, extra_body: dict = None
        ):
            """Server rerank function with configuration from environment variables"""
            return await selected_rerank_func(
                query=query,
                documents=documents,
                top_n=top_n,
                api_key=args.rerank_binding_api_key,
                model=args.rerank_model,
                base_url=args.rerank_binding_host,
                extra_body=extra_body,
            )

        rerank_model_func = server_rerank_func
        logger.info(
            f"Reranking is enabled: {args.rerank_model or 'default model'} using {args.rerank_binding} provider"
        )
    else:
        logger.info("Reranking is disabled")

    # Create ollama_server_infos from command line arguments
    from lightrag.api.config import OllamaServerInfos

    ollama_server_infos = OllamaServerInfos(
        name=args.simulated_model_name, tag=args.simulated_model_tag
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for startup and shutdown events"""

        try:
            # Instance Manager for multi-workspace and multi-user lightrag-server
            app.state.instance_manager = InstanceManager(
                max_instances=100,
                root_dir="rag_storage",
                input_dir="input",
                llm_model_func=create_llm_model_func(args.llm_binding),
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
        
            yield

        finally:
            # Clean up database connections for all open instances
            await app.state.instance_manager.aclose()

            # Clean up shared data
            finalize_share_data()

    # Initialize FastAPI
    app_kwargs = {
        "title": "EDUMind Server API",
        "description": (
            "Providing API for LightRAG core and EDUMind app"
            + "(With authentication)"
            if api_key
            else ""
        ),
        "version": __api_version__,
        "openapi_url": "/openapi.json",  # Explicitly set OpenAPI schema URL
        "docs_url": "/docs",  # Explicitly set docs URL
        "redoc_url": "/redoc",  # Explicitly set redoc URL
        "lifespan": lifespan,
    }

    # Configure Swagger UI parameters
    # Enable persistAuthorization and tryItOutEnabled for better user experience
    app_kwargs["swagger_ui_parameters"] = {
        "persistAuthorization": True,
        "tryItOutEnabled": True,
    }

    app = FastAPI(**app_kwargs)

    def get_cors_origins():
        """Get allowed origins from global_args
        Returns a list of allowed origins, defaults to ["*"] if not set
        """
        origins_str = global_args.cors_origins
        if origins_str == "*":
            return ["*"]
        return [origin.strip() for origin in origins_str.split(",")]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Create combined auth dependency for all endpoints
    combined_auth = get_combined_auth_dependency(api_key)

    # Create working directory if it doesn't exist
    Path(args.working_dir).mkdir(parents=True, exist_ok=True)

    # Add routes
    app.include_router(create_document_routes(api_key))
    app.include_router(create_query_routes(api_key, args.top_k))
    app.include_router(create_graph_routes(api_key))

    # Add Ollama API routes
    # ollama_api = OllamaAPI(top_k=args.top_k, api_key=api_key)
    # app.include_router(ollama_api.router, prefix="/api")

    @app.get("/")
    async def redirect_to_webui():
        """Redirect root path to /webui"""
        return RedirectResponse(url="/webui")
    
    @app.get("/auth-status")
    async def get_auth_status():
        """Get authentication status and guest token if auth is not configured"""

        if not auth_handler.accounts:
            # Authentication not configured, return guest token
            guest_token = auth_handler.create_token(
                username="guest", role="guest", metadata={"auth_mode": "disabled"}
            )
            return {
                "auth_configured": False,
                "access_token": guest_token,
                "token_type": "bearer",
                "auth_mode": "disabled",
                "message": "Authentication is disabled. Using guest access.",
                "core_version": core_version,
                "api_version": __api_version__,
                "webui_title": webui_title,
                "webui_description": webui_description,
            }

        return {
            "auth_configured": True,
            "auth_mode": "enabled",
            "core_version": core_version,
            "api_version": __api_version__,
            "webui_title": webui_title,
            "webui_description": webui_description,
        }

    @app.post("/login")
    async def login(form_data: OAuth2PasswordRequestForm = Depends()):
        if not auth_handler.accounts:
            # Authentication not configured, return guest token
            guest_token = auth_handler.create_token(
                username="guest", role="guest", metadata={"auth_mode": "disabled"}
            )
            return {
                "access_token": guest_token,
                "token_type": "bearer",
                "auth_mode": "disabled",
                "message": "Authentication is disabled. Using guest access.",
                "core_version": core_version,
                "api_version": __api_version__,
                "webui_title": webui_title,
                "webui_description": webui_description,
            }
        username = form_data.username
        if auth_handler.accounts.get(username) != form_data.password:
            raise HTTPException(status_code=401, detail="Incorrect credentials")

        # Regular user login
        user_token = auth_handler.create_token(
            username=username, role="user", metadata={"auth_mode": "enabled"}
        )
        return {
            "access_token": user_token,
            "token_type": "bearer",
            "auth_mode": "enabled",
            "core_version": core_version,
            "api_version": __api_version__,
            "webui_title": webui_title,
            "webui_description": webui_description,
        }

    @app.get("/health", dependencies=[Depends(combined_auth)])
    async def get_status():
        """Get current system status"""
        try:
            pipeline_status = await get_namespace_data("pipeline_status")

            if not auth_configured:
                auth_mode = "disabled"
            else:
                auth_mode = "enabled"

            # Cleanup expired keyed locks and get status
            keyed_lock_info = cleanup_keyed_lock()

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
                    "summary_max_tokens": args.summary_max_tokens,
                    "summary_context_size": args.summary_context_size,
                    "kv_storage": args.kv_storage,
                    "doc_status_storage": args.doc_status_storage,
                    "graph_storage": args.graph_storage,
                    "vector_storage": args.vector_storage,
                    "enable_llm_cache_for_extract": args.enable_llm_cache_for_extract,
                    "enable_llm_cache": args.enable_llm_cache,
                    "workspace": args.workspace,
                    "max_graph_nodes": args.max_graph_nodes,
                    # Rerank configuration
                    "enable_rerank": rerank_model_func is not None,
                    "rerank_binding": args.rerank_binding,
                    "rerank_model": args.rerank_model if rerank_model_func else None,
                    "rerank_binding_host": args.rerank_binding_host
                    if rerank_model_func
                    else None,
                    # Environment variable status (requested configuration)
                    "summary_language": args.summary_language,
                    "force_llm_summary_on_merge": args.force_llm_summary_on_merge,
                    "max_parallel_insert": args.max_parallel_insert,
                    "cosine_threshold": args.cosine_threshold,
                    "min_rerank_score": args.min_rerank_score,
                    "related_chunk_number": args.related_chunk_number,
                    "max_async": args.max_async,
                    "embedding_func_max_async": args.embedding_func_max_async,
                    "embedding_batch_num": args.embedding_batch_num,
                },
                "auth_mode": auth_mode,
                "pipeline_busy": pipeline_status.get("busy", False),
                "keyed_locks": keyed_lock_info,
                "core_version": core_version,
                "api_version": __api_version__,
                "webui_title": webui_title,
                "webui_description": webui_description,
            }
        except Exception as e:
            logger.error(f"Error getting health status: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    # Custom StaticFiles class for smart caching
    class SmartStaticFiles(StaticFiles):  # Renamed from NoCacheStaticFiles
        async def get_response(self, path: str, scope):
            response = await super().get_response(path, scope)

            if path.endswith(".html"):
                response.headers["Cache-Control"] = (
                    "no-cache, no-store, must-revalidate"
                )
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
            elif (
                "/assets/" in path
            ):  # Assets (JS, CSS, images, fonts) generated by Vite with hash in filename
                response.headers["Cache-Control"] = (
                    "public, max-age=31536000, immutable"
                )
            # Add other rules here if needed for non-HTML, non-asset files

            # Ensure correct Content-Type
            if path.endswith(".js"):
                response.headers["Content-Type"] = "application/javascript"
            elif path.endswith(".css"):
                response.headers["Content-Type"] = "text/css"

            return response

    # Webui mount webui/index.html
    static_dir = Path(__file__).parent / "webui"
    static_dir.mkdir(exist_ok=True)
    app.mount(
        "/webui",
        SmartStaticFiles(
            directory=static_dir, html=True, check_dir=True
        ),  # Use SmartStaticFiles
        name="webui",
    )

    return app
        

