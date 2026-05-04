"""Storage backend class factory.

Resolves a storage backend name (e.g. ``"JsonKVStorage"``) to its concrete
implementation class. The four default backends are imported directly so
they always work without depending on the ``STORAGES`` registry; everything
else is resolved lazily through the registry.
"""

from __future__ import annotations

from typing import Any, Callable

from lightrag.kg import STORAGES
from lightrag.utils import lazy_external_import


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

    # Fallback to dynamic import for other storage implementations
    import_path = STORAGES[storage_name]
    return lazy_external_import(import_path, storage_name)
