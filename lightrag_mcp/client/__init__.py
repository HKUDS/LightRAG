"""
LightRAG client interfaces for MCP server.

This module provides client interfaces for communicating with LightRAG,
supporting both REST API and direct library access modes.
"""

from .api_client import LightRAGAPIClient
from .direct_client import LightRAGDirectClient, DirectClientError

__all__ = ["LightRAGAPIClient", "LightRAGDirectClient", "DirectClientError"]
