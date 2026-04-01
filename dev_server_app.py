"""Development ASGI entrypoint for Uvicorn reload mode.

This module avoids LightRAG's config parser from consuming Uvicorn CLI
arguments during import time. Use it with:

    uvicorn dev_server_app:app --reload --host 0.0.0.0 --port 9621
"""

from __future__ import annotations

import sys

from lightrag.api.config import initialize_config, parse_args


def _load_env_backed_args():
    """Parse LightRAG args from .env only, ignoring Uvicorn CLI args."""
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        return parse_args()
    finally:
        sys.argv = original_argv


args = _load_env_backed_args()
initialize_config(args, force=True)

from lightrag.api.lightrag_server import create_app  # noqa: E402


app = create_app(args)
