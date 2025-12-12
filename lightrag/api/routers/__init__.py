"""
This module contains all the routers for the LightRAG API.
"""

from .document_routes import router as document_router
from .graph_routes import router as graph_router
from .ollama_api import OllamaAPI
from .query_routes import router as query_router
from .s3_routes import create_s3_routes
from .search_routes import create_search_routes
from .upload_routes import create_upload_routes

__all__ = [
    'OllamaAPI',
    'create_s3_routes',
    'create_search_routes',
    'create_upload_routes',
    'document_router',
    'graph_router',
    'query_router',
]
