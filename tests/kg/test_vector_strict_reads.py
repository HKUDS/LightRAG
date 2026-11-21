"""Every bundled vector storage implements the maintenance read contract.

``lightrag-rebuild-vdb`` reports a target ``inconclusive`` when its storage
lacks ``get_by_ids_strict`` or ``get_exact_count``, so a bundled backend
inheriting either base default would silently make its check useless.
"""

import importlib

import pytest

from lightrag.base import BaseVectorStorage
from lightrag.kg import STORAGE_IMPLEMENTATIONS, STORAGES

pytestmark = pytest.mark.offline

_VECTOR_STORAGES = STORAGE_IMPLEMENTATIONS["VECTOR_STORAGE"]["implementations"]


def _load(name):
    module_path = STORAGES[name]
    try:
        module = importlib.import_module(module_path, package="lightrag")
    except ImportError as exc:  # an optional client library is not installed
        pytest.skip(f"{name}: {exc}")
    return getattr(module, name)


@pytest.mark.parametrize("name", _VECTOR_STORAGES)
def test_bundled_vector_storage_overrides_the_strict_read(name):
    cls = _load(name)
    assert cls.get_by_ids_strict is not BaseVectorStorage.get_by_ids_strict


@pytest.mark.parametrize(
    "name", [name for name in _VECTOR_STORAGES if name != "NoopVectorDBStorage"]
)
def test_bundled_persistent_vector_storage_overrides_the_exact_count(name):
    cls = _load(name)
    assert cls.get_exact_count is not BaseVectorStorage.get_exact_count
