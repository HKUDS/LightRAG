"""Registry and factory wiring for the Hologres backends.

The API server selects backends purely by name through environment variables
(``LIGHTRAG_KV_STORAGE=HologresKVStorage`` etc.), so this file pins the whole
resolution chain: type compatibility (``verify_storage_implementation``),
required connection variables (``check_storage_env_vars``), and class
resolution (``get_storage_class``).
"""

import pytest

from lightrag.kg import STORAGE_ENV_REQUIREMENTS, verify_storage_implementation
from lightrag.kg.factory import get_storage_class
from lightrag.utils import check_storage_env_vars

_BACKENDS = {
    "KV_STORAGE": "HologresKVStorage",
    "VECTOR_STORAGE": "HologresVectorStorage",
    "GRAPH_STORAGE": "HologresGraphStorage",
    "DOC_STATUS_STORAGE": "HologresDocStatusStorage",
}

_REQUIRED_ENV_VARS = [
    "HOLOGRES_HOST",
    "HOLOGRES_USER",
    "HOLOGRES_PASSWORD",
    "HOLOGRES_DATABASE",
]


@pytest.mark.parametrize(
    ("storage_type", "storage_name"), sorted(_BACKENDS.items())
)
def test_backend_is_registered_for_its_storage_type(storage_type, storage_name):
    verify_storage_implementation(storage_type, storage_name)


@pytest.mark.parametrize(
    ("storage_type", "storage_name"),
    [
        ("KV_STORAGE", "HologresGraphStorage"),
        ("VECTOR_STORAGE", "HologresKVStorage"),
        ("GRAPH_STORAGE", "HologresDocStatusStorage"),
        ("DOC_STATUS_STORAGE", "HologresVectorStorage"),
    ],
)
def test_backend_is_rejected_for_a_mismatched_storage_type(
    storage_type, storage_name
):
    with pytest.raises(ValueError, match="not compatible"):
        verify_storage_implementation(storage_type, storage_name)


@pytest.mark.parametrize("storage_name", sorted(_BACKENDS.values()))
def test_backend_requires_the_connection_env_vars(storage_name, monkeypatch):
    assert STORAGE_ENV_REQUIREMENTS[storage_name] == _REQUIRED_ENV_VARS

    for var in _REQUIRED_ENV_VARS:
        monkeypatch.setenv(var, "configured")
    check_storage_env_vars(storage_name)

    monkeypatch.delenv("HOLOGRES_PASSWORD")
    with pytest.raises(ValueError, match="HOLOGRES_PASSWORD"):
        check_storage_env_vars(storage_name)


@pytest.mark.parametrize(
    ("storage_name", "module_path"),
    [
        ("HologresKVStorage", "lightrag.kg.hologres.kv"),
        ("HologresVectorStorage", "lightrag.kg.hologres.vector"),
        ("HologresGraphStorage", "lightrag.kg.hologres.graph"),
        ("HologresDocStatusStorage", "lightrag.kg.hologres.doc_status"),
    ],
)
def test_factory_resolves_backend_name_to_its_class(storage_name, module_path):
    cls = get_storage_class(storage_name)
    assert cls.__name__ == storage_name
    assert cls.__module__ == module_path
