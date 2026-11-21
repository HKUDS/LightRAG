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


@pytest.mark.parametrize(
    "name", STORAGE_IMPLEMENTATIONS["VECTOR_STORAGE"]["implementations"]
)
def test_bundled_vector_storage_implements_the_maintenance_reads(name):
    try:
        module = importlib.import_module(STORAGES[name], package="lightrag")
    except ImportError as exc:  # an optional client library is not installed
        pytest.skip(f"{name}: {exc}")
    cls = getattr(module, name)

    assert cls.get_by_ids_strict is not BaseVectorStorage.get_by_ids_strict
    # Noop persists nothing, so it has nothing to count: the check reports it
    # not_applicable before asking.
    if name != "NoopVectorDBStorage":
        assert cls.get_exact_count is not BaseVectorStorage.get_exact_count
