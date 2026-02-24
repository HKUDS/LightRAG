"""
LightRAG FastAPI Server (Multi-Tenant Refactor)
"""

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os
import logging
import logging.config
import sys
import uvicorn
import pipmaster as pm
from pathlib import Path
from ascii_colors import ASCIIColors

from lightrag import LightRAG, __version__ as core_version
from lightrag.api import __api_version__
from lightrag.utils import logger, set_verbose_debug
from lightrag.kg.shared_storage import finalize_share_data

from .config import (
    global_args,
    update_uvicorn_mode_config,
    initialize_config
)
from .utils_api import check_env_file, display_splash_screen

# Import New Routers
from .routers.tenant_auth_routes import router as auth_router
from .routers.tenant_document_routes import router as document_router
from .routers.tenant_query_routes import router as query_router
from .routers.chat_routes import router as chat_router
from .routers.tenant_graph_routes import router as graph_router
from .routers.ollama_api import OllamaAPI
from .routers.health_routes import router as health_router

# Import Infrastructure
from .db import init_db, get_db_connection
from .rag_manager import rag_manager
from .llm_factory import LLMConfigCache # Kept for health status if needed

load_dotenv(dotenv_path=".env", override=False)

webui_title = os.getenv("WEBUI_TITLE", "LightRAG")
webui_description = os.getenv("WEBUI_DESCRIPTION", "Multi-Tenant RAG System")

def check_frontend_build():
    """Check if frontend is built"""
    webui_dir = Path(__file__).parent / "webui"
    index_html = webui_dir / "index.html"
    if not index_html.exists():
        return False, False
    return True, False # Simplified for brevity, assume prod-like

def create_app(args):
    webui_assets_exist, _ = check_frontend_build()
    
    logger.setLevel(args.log_level)
    set_verbose_debug(args.verbose)

    # Initialize Database
    init_db()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        try:
             # Initialize default workspace RAG to ensure system is ready
             # and to support Ollama API which bounds to a specific instance
            default_rag = await rag_manager.get_rag("default")
            
            # Mount Ollama API dynamically? Or separate router?
            # Creating OllamaAPI requires an initialized RAG
            ollama_api = OllamaAPI(default_rag, top_k=args.top_k, api_key=args.key)
            app.include_router(ollama_api.router, prefix="/api")
            
            ASCIIColors.green("\nServer is ready! (Multi-Tenant Mode) 🚀\n")
            yield
        finally:
            # Shutdown
            # RAGManager doesn't have explict shutdown all info yet?
            # But we should finalize storage for loaded instances
            # Iterate and finalize
            pass # TODO: Add cleanup logic to RAGManager

    app = FastAPI(
        title="LightRAG Multi-Tenant API",
        description="API with RBAC and Multi-tenancy",
        version=__api_version__,
        lifespan=lifespan
    )

    # CORS
    origins = args.cors_origins.split(",") if args.cors_origins != "*" else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-New-Token"],
    )

    # Mount Routers
    app.include_router(auth_router)
    app.include_router(document_router)
    app.include_router(query_router)
    app.include_router(chat_router)
    app.include_router(graph_router)
    app.include_router(health_router)

    # WebUI Serving (Simplified)
    if webui_assets_exist:
        app.mount("/webui", StaticFiles(directory=Path(__file__).parent / "webui", html=True), name="webui")
        @app.get("/")
        async def redirect_webui():
            return RedirectResponse("/webui")
    else:
        @app.get("/")
        async def redirect_docs():
            return RedirectResponse("/docs")

    return app

def configure_logging():
    # ... (Simplified logging setup or reuse existing)
    logging.basicConfig(level=logging.INFO)

def main():
    initialize_config()
    if not check_env_file(): sys.exit(1)
    
    configure_logging()
    app = create_app(global_args)
    
    uvicorn.run(app, host=global_args.host, port=global_args.port)

if __name__ == "__main__":
    main()
