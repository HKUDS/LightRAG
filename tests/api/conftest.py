"""Shared isolation for the API test package.

``lightrag.api.config`` / ``auth`` / ``utils_api`` compute module-level state at
IMPORT time (``auth_handler.accounts``, ``whitelist_patterns``,
``auth_configured``) and the router modules capture
``get_combined_auth_dependency`` by value — so its closure reads whatever module
object was live when the router was imported, for the rest of the session.

Several tests here deliberately re-import those modules under a mocked config to
exercise a different auth profile. That leaves a DIFFERENT object in
``sys.modules`` than the one the routers actually read, and the next test that
tries to open up auth by patching ``sys.modules["lightrag.api.utils_api"]``
patches a module nobody consults: with ``AUTH_ACCOUNTS`` set in a developer
``.env``, the stale ``auth_configured=True`` then answers its requests with 401
(seen as order-dependent failures in ``tests/api/routes/test_query_stream_routes.py``).
"""

import sys

import pytest

_API_MODULE_PREFIX = "lightrag.api"


@pytest.fixture(autouse=True)
def _restore_lightrag_api_module_identity():
    """Restore ``lightrag.api.*`` entries a test replaced or removed.

    Only REPLACEMENTS and REMOVALS are healed (identity comparison): a module
    imported for the first time during a test is left in place, so this never
    creates a fresh duplicate of its own. Parent-package attributes are re-pointed
    too, because the re-import helpers ``delattr`` them to dodge the shadowing.
    """
    saved = {
        name: module
        for name, module in sys.modules.items()
        if name == _API_MODULE_PREFIX or name.startswith(_API_MODULE_PREFIX + ".")
    }
    yield
    for name, module in saved.items():
        if sys.modules.get(name) is module:
            continue
        sys.modules[name] = module
        parent_name, _, child = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None and child and getattr(parent, child, None) is not module:
            setattr(parent, child, module)
