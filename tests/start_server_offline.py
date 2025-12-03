#!/usr/bin/env python3
"""
Start LightRAG server for integration testing with offline-compatible tokenizer.

This script initializes the LightRAG server with a simple tokenizer that doesn't
require internet access, making it suitable for integration testing in restricted
network environments.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import from tests
sys.path.insert(0, str(Path(__file__).parent))


def start_server():
    """Start LightRAG server with offline-compatible configuration."""
    # Import here after setting up the path
    from lightrag.api.lightrag_server import main

    # Override the tokenizer in global args before server starts
    # This will be used when creating the LightRAG instance
    os.environ["LIGHTRAG_OFFLINE_TOKENIZER"] = "true"

    # Start the server
    main()


if __name__ == "__main__":
    start_server()
