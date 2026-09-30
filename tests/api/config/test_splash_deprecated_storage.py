"""The startup banner flags storage backends scheduled for removal.

The core logs the same notice when LightRAG is constructed, but that line
lands after the banner and is buried by startup logs; the banner is where an
operator reads the storage configuration.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.offline

_BLOCK_MARKER = "Deprecated Storage"


def _splash(monkeypatch, *, kv_storage: str, doc_status_storage: str) -> str:
    """Render the real banner over real parsed args and return its output.

    ASCIIColors writes through its own stream rather than the stdout capsys
    replaces, so the collector below captures the arguments it is handed."""
    # setenv, not delenv -- load_dotenv(override=False) would re-populate a
    # deleted one from a developer-local .env.
    monkeypatch.setenv("LIGHTRAG_KV_STORAGE", kv_storage)
    monkeypatch.setenv("LIGHTRAG_DOC_STATUS_STORAGE", doc_status_storage)

    original_argv = sys.argv.copy()
    try:
        sys.argv = ["lightrag-server"]
        config = importlib.import_module("lightrag.api.config")
        utils_api = importlib.import_module("lightrag.api.utils_api")
        args = config.parse_args()
        colors = MagicMock()
        monkeypatch.setattr(utils_api, "ASCIIColors", colors)
        utils_api.display_splash_screen(args)
    finally:
        sys.argv = original_argv

    return "\n".join(str(arg) for call in colors.mock_calls for arg in call.args)


def test_redis_storages_are_flagged(monkeypatch):
    output = _splash(
        monkeypatch,
        kv_storage="RedisKVStorage",
        doc_status_storage="RedisDocStatusStorage",
    )

    assert _BLOCK_MARKER in output
    assert "(deprecated)" in output
    assert "PGKVStorage" in output
    assert "PGDocStatusStorage" in output


def test_default_storages_are_not_flagged(monkeypatch):
    output = _splash(
        monkeypatch,
        kv_storage="JsonKVStorage",
        doc_status_storage="JsonDocStatusStorage",
    )

    assert _BLOCK_MARKER not in output
    assert "(deprecated)" not in output
