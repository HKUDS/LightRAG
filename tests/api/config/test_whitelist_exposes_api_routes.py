"""Offline tests for whitelist_exposes_api_routes (GHSA-mmg5-8x8q-v934 banner).

The startup banner uses this helper to warn when WHITELIST_PATHS leaves the
Ollama-compatible /api/* routes unauthenticated on a network-exposed bind. It
must mirror the prefix/exact matching in get_combined_auth_dependency so a
catch-all entry like "/*" is detected, not just literal "/api..." entries.
"""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.offline


@pytest.fixture
def whitelist_exposes_api_routes(monkeypatch):
    """Import utils_api with a stub global_args (consumed at import time)."""
    config = importlib.import_module("lightrag.api.config")
    monkeypatch.setattr(
        config,
        "global_args",
        SimpleNamespace(
            token_secret=None,  # -> AuthHandler falls back to DEFAULT_TOKEN_SECRET
            jwt_algorithm="HS256",
            token_expire_hours=48,
            guest_token_expire_hours=24,
            auth_accounts="",
            whitelist_paths="/health",  # consumed by utils_api at import time
            token_auto_renew=False,
        ),
    )
    sys.modules.pop("lightrag.api.auth", None)
    importlib.reload(importlib.import_module("lightrag.api.auth"))
    sys.modules.pop("lightrag.api.utils_api", None)
    utils_api = importlib.import_module("lightrag.api.utils_api")
    try:
        yield utils_api.whitelist_exposes_api_routes
    finally:
        sys.modules.pop("lightrag.api.utils_api", None)


@pytest.mark.parametrize(
    "whitelist",
    [
        "/health,/api/*",  # default whitelist
        "/api/*",
        "/api",
        "/api/chat",  # exact Ollama route
        "/*",  # catch-all: empty prefix matches every path
        " /* ",  # catch-all with surrounding whitespace
        "/health,/*",
        "/a/*",  # prefix that /api also starts with
    ],
)
def test_exposes_api_routes(whitelist_exposes_api_routes, whitelist: str) -> None:
    assert whitelist_exposes_api_routes(whitelist) is True


@pytest.mark.parametrize(
    "whitelist",
    [
        "/health",
        "/health,/docs",
        "/health/*",  # unrelated prefix
        "/apiary/*",  # prefix under /api... but not /api/ — exempts only /apiary
        "/apidocs",  # exact path that merely shares the /api substring
        "",
        "   ",
    ],
)
def test_does_not_expose_api_routes(
    whitelist_exposes_api_routes, whitelist: str
) -> None:
    assert whitelist_exposes_api_routes(whitelist) is False
