"""Storage backend class factory.

Resolves a storage backend name (e.g. ``"JsonKVStorage"``) to its concrete
implementation class. The four default backends are imported directly so
they always work without depending on the ``STORAGES`` registry; everything
else is resolved lazily through the registry.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

from lightrag.kg import STORAGES


def get_storage_class(storage_name: str) -> Callable[..., Any]:
    """Return the storage backend class for ``storage_name``."""
    if storage_name == "JsonKVStorage":
        from lightrag.kg.json_kv_impl import JsonKVStorage

        return JsonKVStorage
    if storage_name == "NanoVectorDBStorage":
        from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage

        return NanoVectorDBStorage
    if storage_name == "NetworkXStorage":
        from lightrag.kg.networkx_impl import NetworkXStorage

        return NetworkXStorage
    if storage_name == "JsonDocStatusStorage":
        from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage

        return JsonDocStatusStorage

    # Fallback to dynamic import for other storage implementations.
    # STORAGES values are relative paths (e.g. ".kg.postgres_impl") authored
    # against the top-level ``lightrag`` package, so anchor the import there
    # rather than letting it resolve against this module's own package.
    import_path = STORAGES[storage_name]
    module = importlib.import_module(import_path, package="lightrag")
    return getattr(module, storage_name)
