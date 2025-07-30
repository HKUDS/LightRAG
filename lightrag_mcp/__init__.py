"""
LightRAG MCP Server - Model Context Protocol integration for LightRAG

This package provides MCP (Model Context Protocol) tools and resources
for accessing LightRAG's advanced RAG and knowledge graph capabilities
through Claude CLI and other MCP-compatible clients.

Key Features:
- Complete RAG query capabilities with 6 different modes
- Document management and processing tools
- Knowledge graph exploration and manipulation
- System health monitoring and statistics
- Streaming query support
- Resource access for documents and graph data

Usage:
    python -m lightrag_mcp.server

Environment Variables:
    LIGHTRAG_API_URL: LightRAG API endpoint (default: http://localhost:9621)
    LIGHTRAG_API_KEY: Optional API key for authentication
    MCP_SERVER_NAME: MCP server name (default: lightrag-mcp)
"""

__version__ = "1.0.0"
__author__ = "LightRAG MCP Integration Team"
__description__ = "Model Context Protocol server for LightRAG"

# Package metadata
__all__ = ["server", "tools", "resources", "client", "config", "utils"]
