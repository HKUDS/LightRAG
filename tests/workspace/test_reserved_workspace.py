"""The ``_lightrag*`` workspace-name reservation and its one door.

``validate_workspace`` refuses the family; only the configuration-storage
factory's private grant lets ``_lightrag_config`` through, for one
construction; and no ``*_WORKSPACE`` environment variable may move a reserved
workspace onto a tenant's. See *The internal factory* in
docs/design/ConfigurationStorage.md.
"""

from __future__ import annotations

import pytest
from types import SimpleNamespace

from lightrag.namespace import CONFIG_WORKSPACE, RESERVED_WORKSPACE_PREFIX
from lightrag.utils import (
    _grant_reserved_workspace,
    is_reserved_workspace,
    validate_workspace,
    validate_workspace_override,
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


class TestEnvironmentRemapCannotNameAReservedWorkspace:
    """The mirror of the class above. The override is applied AFTER
    ``validate_workspace()`` passed the constructor argument, so without a
    check of its own ``REDIS_WORKSPACE=_lightrag_config`` would bind tenant
    data into the reserved family through the front door the reservation
    exists to close."""

    def test_the_validator_refuses_the_family_and_strips_the_rest(self):
        assert validate_workspace_override("X_WORKSPACE", " prod ") == "prod"
        assert validate_workspace_override("X_WORKSPACE", None) is None
        assert validate_workspace_override("X_WORKSPACE", "") == ""
        with pytest.raises(ValueError, match="X_WORKSPACE.*reserved"):
            validate_workspace_override("X_WORKSPACE", CONFIG_WORKSPACE)

    def test_opensearch(self, monkeypatch):
        from lightrag.kg.opensearch_impl import _resolve_workspace

        monkeypatch.setenv("OPENSEARCH_WORKSPACE", CONFIG_WORKSPACE)
        with pytest.raises(ValueError, match="reserved"):
            _resolve_workspace("tenant", "text_chunks")

    def test_redis(self, monkeypatch):
        from unittest.mock import MagicMock

        from lightrag.kg.redis_impl import RedisDocStatusStorage, RedisKVStorage

        monkeypatch.setenv("REDIS_WORKSPACE", CONFIG_WORKSPACE)
        monkeypatch.setattr(
            "lightrag.kg.redis_impl.RedisConnectionManager.get_pool",
            lambda redis_url: MagicMock(name="pool"),
        )
        monkeypatch.setattr(
            "lightrag.kg.redis_impl.Redis",
            lambda connection_pool=None, **_: MagicMock(),
        )
        for cls in (RedisKVStorage, RedisDocStatusStorage):
            with pytest.raises(ValueError, match="reserved"):
                cls(
                    namespace="text_chunks",
                    workspace="tenant",
                    global_config={},
                    embedding_func=None,
                )

    def test_mongodb(self, monkeypatch):
        from lightrag.kg.mongo_impl import MongoDocStatusStorage, MongoKVStorage

        monkeypatch.setenv("MONGODB_WORKSPACE", CONFIG_WORKSPACE)
        for cls in (MongoKVStorage, MongoDocStatusStorage):
            with pytest.raises(ValueError, match="reserved"):
                cls(
                    namespace="text_chunks",
                    global_config={},
                    embedding_func=None,
                    workspace="tenant",
                )

    def test_postgresql_client(self):
        """One check for every PostgreSQL storage: they all take the override
        from the shared client, which reads it once."""
        from lightrag.kg.postgres_impl import PostgreSQLDB

        with pytest.raises(ValueError, match="POSTGRES_WORKSPACE.*reserved"):
            PostgreSQLDB(
                {
                    "host": "localhost",
                    "port": 5432,
                    "user": "u",
                    "password": "p",
                    "database": "d",
                    "workspace": CONFIG_WORKSPACE,
                    "max_connections": 1,
                    "connection_retry_attempts": 1,
                    "connection_retry_backoff": 0.1,
                    "connection_retry_backoff_max": 0.1,
                    "pool_close_timeout": 1,
                }
            )

    def test_milvus_and_qdrant(self, monkeypatch):
        from lightrag.kg.milvus_impl import MilvusVectorDBStorage
        from lightrag.kg.qdrant_impl import QdrantVectorDBStorage

        embedding = SimpleNamespace(embedding_dim=8, model_name="m")
        monkeypatch.setenv("MILVUS_WORKSPACE", CONFIG_WORKSPACE)
        monkeypatch.setenv("QDRANT_WORKSPACE", CONFIG_WORKSPACE)
        for cls in (MilvusVectorDBStorage, QdrantVectorDBStorage):
            storage = cls.__new__(cls)
            storage.namespace = "entities"
            storage.workspace = "tenant"
            storage.global_config = {"vector_db_storage_cls_kwargs": {}}
            storage.embedding_func = embedding
            storage.meta_fields = set()
            with pytest.raises(ValueError, match="reserved"):
                storage.__post_init__()


def test_a_refused_override_leaks_no_configuration_pool_reference(
    tmp_path, monkeypatch
):
    """``RedisKVStorage`` takes its shared-pool reference in its constructor,
    and ``LightRAG.__post_init__`` has no async teardown. Had the
    configuration storage been constructed FIRST, the refusal in the first
    ordinary storage's constructor would have left its reference behind,
    growing on every failed construction. It is constructed last, so a refused
    override acquires nothing."""
    pytest.importorskip("redis")
    from unittest.mock import MagicMock

    from lightrag import LightRAG
    from lightrag.kg.redis_impl import RedisConnectionManager
    from lightrag.utils import Tokenizer, TokenizerInterface

    class _StubTokenizer(TokenizerInterface):
        def encode(self, content: str) -> list[int]:
            return [ord(c) for c in content]

        def decode(self, tokens: list[int]) -> str:
            return "".join(chr(t) for t in tokens)

    url = f"redis://leak-check-{tmp_path.name}:6379"
    monkeypatch.setenv("REDIS_URI", url)
    monkeypatch.setenv("REDIS_WORKSPACE", CONFIG_WORKSPACE)
    monkeypatch.setattr(
        "lightrag.kg.redis_impl.ConnectionPool.from_url",
        lambda *args, **kwargs: MagicMock(name="pool"),
    )
    monkeypatch.setattr(
        "lightrag.kg.redis_impl.Redis", lambda connection_pool=None, **_: MagicMock()
    )

    async def _llm(prompt, **kwargs):  # pragma: no cover - never called
        return ""

    with pytest.raises(ValueError, match="reserved"):
        LightRAG(
            working_dir=str(tmp_path),
            workspace="tenant",
            llm_model_func=_llm,
            kv_storage="RedisKVStorage",
            # The default tokenizer downloads tiktoken data; not this test's
            # subject, and not available offline.
            tokenizer=Tokenizer("stub", _StubTokenizer()),
        )

    assert url not in RedisConnectionManager._pools
    assert RedisConnectionManager._pool_refs.get(url, 0) == 0
