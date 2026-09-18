"""The ``_lightrag*`` workspace-name reservation and its one door.

``validate_workspace`` refuses the family; only the configuration-storage
factory's private grant lets ``_lightrag_config`` through, for one
construction; and no ``*_WORKSPACE`` environment variable may move a reserved
workspace onto a tenant's. See *The internal factory* in
docs/design/ConfigurationStorage.md.
"""

from __future__ import annotations

import pytest

from lightrag.namespace import CONFIG_WORKSPACE, RESERVED_WORKSPACE_PREFIX
from lightrag.utils import (
    _grant_reserved_workspace,
    is_reserved_workspace,
    validate_workspace,
)

pytestmark = pytest.mark.offline


@pytest.mark.parametrize("name", ["_lightrag_config", "_lightrag_server", "_lightragx"])
def test_the_reserved_family_is_refused(name):
    assert is_reserved_workspace(name)
    with pytest.raises(ValueError, match="reserved"):
        validate_workspace(name)


@pytest.mark.parametrize("name", ["", "lightrag", "my_lightrag", "v1.0", "_other"])
def test_ordinary_names_still_pass(name):
    assert not is_reserved_workspace(name)
    assert validate_workspace(name) == name


def test_the_grant_admits_exactly_one_name_and_only_while_held():
    with _grant_reserved_workspace(CONFIG_WORKSPACE):
        assert validate_workspace(CONFIG_WORKSPACE) == CONFIG_WORKSPACE
        with pytest.raises(ValueError):
            validate_workspace(RESERVED_WORKSPACE_PREFIX + "_other")
    with pytest.raises(ValueError):
        validate_workspace(CONFIG_WORKSPACE)


def test_the_grant_is_only_for_reserved_names():
    with pytest.raises(ValueError):
        _grant_reserved_workspace("tenant")


def test_a_public_lightrag_instance_cannot_take_a_reserved_workspace(tmp_path):
    """Scenario 12, the refusing half: refused at construction, by the rule,
    before any storage is built."""
    from lightrag import LightRAG

    async def _llm(prompt, **kwargs):  # pragma: no cover - never called
        return ""

    with pytest.raises(ValueError, match="reserved"):
        LightRAG(
            working_dir=str(tmp_path),
            workspace=CONFIG_WORKSPACE,
            llm_model_func=_llm,
        )


class TestEnvironmentRemapIsIgnoredForReservedNames:
    """``PG_WORKSPACE`` / ``REDIS_WORKSPACE`` / ``MONGODB_WORKSPACE`` /
    ``OPENSEARCH_WORKSPACE`` move TENANT data; the configuration container's
    workspace is fixed, not configured."""

    def test_opensearch(self, monkeypatch):
        from lightrag.kg.opensearch_impl import _resolve_workspace

        monkeypatch.setenv("OPENSEARCH_WORKSPACE", "prod")
        assert _resolve_workspace("tenant", "config") == "prod"
        assert _resolve_workspace(CONFIG_WORKSPACE, "config") == CONFIG_WORKSPACE

    def test_redis(self, monkeypatch):
        from unittest.mock import MagicMock

        from lightrag.kg.redis_impl import RedisKVStorage

        monkeypatch.setenv("REDIS_WORKSPACE", "prod")
        monkeypatch.setattr(
            "lightrag.kg.redis_impl.RedisConnectionManager.get_pool",
            lambda redis_url: MagicMock(name="pool"),
        )
        monkeypatch.setattr(
            "lightrag.kg.redis_impl.Redis",
            lambda connection_pool=None, **_: MagicMock(),
        )
        with _grant_reserved_workspace(CONFIG_WORKSPACE):
            storage = RedisKVStorage(
                namespace="config",
                workspace=CONFIG_WORKSPACE,
                global_config={},
                embedding_func=None,
            )
        assert storage.final_namespace == f"{CONFIG_WORKSPACE}_config"
        tenant = RedisKVStorage(
            namespace="config",
            workspace="tenant",
            global_config={},
            embedding_func=None,
        )
        assert tenant.final_namespace == "prod_config"

    def test_mongodb(self, monkeypatch):
        from lightrag.kg.mongo_impl import MongoKVStorage

        monkeypatch.setenv("MONGODB_WORKSPACE", "prod")
        with _grant_reserved_workspace(CONFIG_WORKSPACE):
            storage = MongoKVStorage(
                namespace="config",
                global_config={},
                embedding_func=None,
                workspace=CONFIG_WORKSPACE,
            )
        assert storage.workspace == CONFIG_WORKSPACE
        assert storage.final_namespace == f"{CONFIG_WORKSPACE}_config"
        tenant = MongoKVStorage(
            namespace="config",
            global_config={},
            embedding_func=None,
            workspace="tenant",
        )
        assert tenant.workspace == "prod"

    async def test_postgresql(self):
        from types import SimpleNamespace

        from lightrag.kg.postgres_impl import PGKVStorage
        from lightrag.kg.shared_storage import (
            finalize_share_data,
            initialize_share_data,
        )

        initialize_share_data()
        try:
            db = SimpleNamespace(workspace="prod")
            storage = PGKVStorage.__new__(PGKVStorage)
            storage.namespace = "config"
            storage.global_config = {}
            storage.db = db
            with _grant_reserved_workspace(CONFIG_WORKSPACE):
                storage.workspace = CONFIG_WORKSPACE
                storage.__post_init__()
            await storage.initialize()
            assert storage.workspace == CONFIG_WORKSPACE

            tenant = PGKVStorage.__new__(PGKVStorage)
            tenant.namespace = "config"
            tenant.global_config = {}
            tenant.db = db
            tenant.workspace = "tenant"
            tenant.__post_init__()
            await tenant.initialize()
            assert tenant.workspace == "prod"
        finally:
            finalize_share_data()
