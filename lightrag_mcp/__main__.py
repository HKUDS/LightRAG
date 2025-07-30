#!/usr/bin/env python3
"""
LightRAG MCP Server - Module entry point

Allows running the server as a Python module:
    python -m lightrag_mcp
"""

from .server import main

if __name__ == "__main__":
    main()