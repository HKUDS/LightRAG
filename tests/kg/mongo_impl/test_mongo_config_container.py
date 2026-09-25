"""``MongoKVStorage`` on the ``config`` namespace: the fixed container, and
the batching limits the fixed-container branch must not skip.

The container is named in CODE -- one collection, no workspace prefix to
choose and no ``MONGODB_WORKSPACE`` remap -- which its ``__post_init__``
implements by branching on the namespace and returning early. That early
return once skipped ``_resolve_upsert_batch_limits()``, leaving
``_max_upsert_payload_bytes`` and ``_max_upsert_records_per_batch``
undefined; a Mongo-backed configuration storage then raised ``AttributeError``
inside ``upsert`` while claiming its first absent embedding baseline, which is
during startup. See docs/design/ConfigurationStorageContract.md.
"""

from __future__ import annotations

import pytest

from lightrag.kg.mongo_impl import MongoKVStorage
from lightrag.namespace import CONFIG_CONTAINER_TAG

pytestmark = pytest.mark.offline


def _storage(namespace: str, workspace: str = "tenant") -> MongoKVStorage:
    return MongoKVStorage(
        namespace=namespace,
        global_config={},
        embedding_func=None,
        workspace=workspace,
    )


def test_the_config_container_has_its_batch_limits():
    """The regression: ``upsert`` reads both on the first baseline claim."""
    storage = _storage("config")
    assert isinstance(storage._max_upsert_payload_bytes, int)
    assert storage._max_upsert_payload_bytes > 0
    assert isinstance(storage._max_upsert_records_per_batch, int)
    assert storage._max_upsert_records_per_batch > 0


def test_the_config_container_gets_the_same_limits_as_a_tenant():
    """Nothing about the configuration container makes its batching special;
    it only makes its NAME special."""
    config = _storage("config")
    tenant = _storage("text_chunks")
    assert (
        config._max_upsert_payload_bytes,
        config._max_upsert_records_per_batch,
    ) == (
        tenant._max_upsert_payload_bytes,
        tenant._max_upsert_records_per_batch,
    )


@pytest.mark.parametrize("workspace", ["tenant", "other", ""])
def test_the_collection_is_fixed_whatever_the_caller_named(workspace):
    storage = _storage("config", workspace)
    assert storage.workspace == CONFIG_CONTAINER_TAG
    assert storage._collection_name == f"{CONFIG_CONTAINER_TAG}_config"


def test_mongodb_workspace_does_not_remap_the_container(monkeypatch):
    monkeypatch.setenv("MONGODB_WORKSPACE", "prod")
    assert _storage("config")._collection_name == f"{CONFIG_CONTAINER_TAG}_config"
    assert _storage("text_chunks")._collection_name == "prod_text_chunks"
