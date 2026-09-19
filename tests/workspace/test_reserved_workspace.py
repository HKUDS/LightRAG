"""The ``_lightrag*`` workspace-name reservation and its one door.

``validate_workspace`` refuses the family; only the configuration-storage
factory's private grant lets ``_lightrag_config`` through, for one
construction; and no ``*_WORKSPACE`` environment variable may move a reserved
workspace onto a tenant's. See *The internal factory* in
docs/design/ConfigurationStorage.md.
"""

from __future__ import annotations

import logging

import pytest
from types import SimpleNamespace

from lightrag.namespace import CONFIG_WORKSPACE, RESERVED_WORKSPACE_PREFIX
from lightrag.utils import (
    WORKSPACE_OVERRIDE_SOURCES,
    _grant_reserved_workspace,
    is_reserved_workspace,
    validate_workspace,
    validate_workspace_override,
)

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    "name",
    [
        "_lightrag_config",
        "_lightrag_server",
        "_lightragx",
        # Case variants: OpenSearch lowercases index names, so these would
        # land on the internal container's index if they were tenant-safe.
        "_LightRAG_config",
        "_LIGHTRAG_CONFIG",
        "_Lightrag_other",
    ],
)
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
            # One SPELLING, not one name folded: the grant is exact.
            validate_workspace(CONFIG_WORKSPACE.upper())
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
        with pytest.raises(ValueError, match="reserved"):
            validate_workspace_override("X_WORKSPACE", "_LightRAG_config")

    def test_opensearch(self, monkeypatch):
        from lightrag.kg.opensearch_impl import _resolve_workspace

        monkeypatch.setenv("OPENSEARCH_WORKSPACE", CONFIG_WORKSPACE)
        with pytest.raises(ValueError, match="reserved"):
            _resolve_workspace("tenant", "text_chunks")

    def test_opensearch_refuses_the_case_variant_its_index_names_fold(
        self, monkeypatch
    ):
        """OpenSearch lowercases index names, so ``_LightRAG_config`` would
        share the internal container's index. The reservation is
        case-insensitive for exactly this reason."""
        from lightrag.kg.opensearch_impl import _resolve_workspace

        monkeypatch.setenv("OPENSEARCH_WORKSPACE", "_LightRAG_config")
        with pytest.raises(ValueError, match="reserved"):
            _resolve_workspace("tenant", "config")
        with pytest.raises(ValueError, match="reserved"):
            validate_workspace("_LightRAG_config")

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


@pytest.mark.parametrize(
    "bad_kwargs",
    [
        pytest.param({"llm_model_func": None}, id="missing-llm"),
        pytest.param({"role_llm_configs": 42}, id="bad-role-config"),
        pytest.param({"role_llm_configs": {"no_such_role": {}}}, id="unknown-role"),
    ],
)
def test_the_configuration_storage_is_never_built_before_a_refusal(
    tmp_path, monkeypatch, bad_kwargs
):
    """The validations that follow the storage constructors can refuse too,
    and they run in the same synchronous ``__post_init__`` with no teardown.
    The configuration storage is the last thing built that can raise, so a
    construction refused ANYWHERE never built it -- which is what keeps a
    Redis-backed one from leaking its pool reference on every failed
    construction."""
    import numpy as np

    from lightrag import LightRAG
    import lightrag.lightrag as lightrag_module
    from lightrag.utils import EmbeddingFunc, Tokenizer, TokenizerInterface

    class _StubTokenizer(TokenizerInterface):
        def encode(self, content: str) -> list[int]:
            return [ord(c) for c in content]

        def decode(self, tokens: list[int]) -> str:
            return "".join(chr(t) for t in tokens)

    async def _embed(texts, **kwargs):  # pragma: no cover - never called
        return np.zeros((len(texts), 8), dtype=np.float32)

    built = []
    real_factory = lightrag_module.create_configuration_storage

    def _spy(*args, **kwargs):
        built.append(True)
        return real_factory(*args, **kwargs)

    monkeypatch.setattr(lightrag_module, "create_configuration_storage", _spy)

    async def _llm(prompt, **kwargs):  # pragma: no cover - never called
        return ""

    # Every storage constructor must SUCCEED here, so the refusal under test
    # is one of the validations that follow them, not a storage's own.
    kwargs = {"llm_model_func": _llm, **bad_kwargs}
    with pytest.raises((ValueError, TypeError)) as excinfo:
        LightRAG(
            working_dir=str(tmp_path),
            workspace="tenant",
            tokenizer=Tokenizer("stub", _StubTokenizer()),
            embedding_func=EmbeddingFunc(
                embedding_dim=8, max_token_size=1024, func=_embed, model_name="m"
            ),
            **kwargs,
        )
    assert "storage" not in str(excinfo.value).lower(), (
        "the refusal came from a storage constructor, not a later validation"
    )

    assert built == [], "the configuration storage was built before a refusal"


_OVERRIDE_ENV_VARS = [env for env, _ in WORKSPACE_OVERRIDE_SOURCES]


class TestWorkspaceOverridesAreDeprecatedAndAnnounced:
    """``*_WORKSPACE`` exists to keep legacy data reachable, and the storage
    layer applies it where nothing above can see it: the workspace a caller
    asked for and the container its data lands in can differ, and every record
    keyed by the caller's workspace -- the embedding baselines among them --
    stays under the name the caller gave.

    That is tolerable while the override never moves, and is not tolerable when
    one is set, changed or cleared to MOVE an existing deployment's data: the
    instance follows the override to another container while those records stay
    behind, and no later check can tell that apart from an ordinary start. The
    rule is therefore announced to the operator rather than enforced, which
    makes the announcement itself worth pinning -- including WHERE it is made,
    since a per-instance warning would be noise and a per-worker one would
    repeat itself as many times as the server has workers.
    """

    @pytest.fixture(autouse=True)
    def warnings_seen(self, monkeypatch, tmp_path):
        """LightRAG's logger does not propagate, so caplog never sees these --
        collect them off the logger itself."""
        import lightrag.utils as _utils

        monkeypatch.setattr(_utils, "_workspace_override_warning_emitted", False)
        for env_var, _section in WORKSPACE_OVERRIDE_SOURCES:
            monkeypatch.delenv(env_var, raising=False)
        # config.ini is read relative to the cwd; keep the test off any real one.
        monkeypatch.chdir(tmp_path)

        records: list[str] = []

        class _Collect(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        handler = _Collect(level=logging.WARNING)
        _utils.logger.addHandler(handler)
        try:
            yield records
        finally:
            _utils.logger.removeHandler(handler)

    def test_no_override_says_nothing(self, warnings_seen):
        from lightrag.utils import warn_about_workspace_overrides

        assert warn_about_workspace_overrides() == []
        assert warnings_seen == []

    @pytest.mark.parametrize("env_var", _OVERRIDE_ENV_VARS)
    def test_each_override_is_named_and_called_deprecated(
        self, env_var, monkeypatch, warnings_seen
    ):
        from lightrag.utils import warn_about_workspace_overrides

        monkeypatch.setenv(env_var, "legacy_container")
        assert warn_about_workspace_overrides() == [env_var]
        assert len(warnings_seen) == 1
        message = warnings_seen[0]
        assert env_var in message
        assert "deprecated" in message
        assert "embedding baselines" in message

    def test_a_config_ini_override_counts_too(self, tmp_path, warnings_seen):
        """PostgreSQL and Neo4j fall back to config.ini, so the environment
        alone is not where the answer lives."""
        from lightrag.utils import warn_about_workspace_overrides

        (tmp_path / "config.ini").write_text("[postgres]\nworkspace = legacy\n")
        assert warn_about_workspace_overrides() == ["POSTGRES_WORKSPACE"]
        assert len(warnings_seen) == 1

    def test_a_second_call_stays_quiet(self, monkeypatch, warnings_seen):
        """Belt and braces: the call sites already make it once per server
        start, and a stray second call must not double the output."""
        from lightrag.utils import warn_about_workspace_overrides

        monkeypatch.setenv("REDIS_WORKSPACE", "legacy_container")
        first = warn_about_workspace_overrides()
        second = warn_about_workspace_overrides()
        assert first == second == ["REDIS_WORKSPACE"]
        assert len(warnings_seen) == 1

    def test_the_lightrag_object_does_not_warn(self, monkeypatch):
        """It belongs to the application's startup, not to an instance: a
        library user constructing several LightRAGs, or a server with N
        workers, must not get the deprecation N times."""
        import lightrag.lightrag as _lightrag

        assert not hasattr(_lightrag, "warn_about_workspace_overrides"), (
            "the warning was moved out of LightRAG and into the launchers"
        )

    def test_both_launchers_warn_before_serving(self, monkeypatch):
        """uvicorn's single process and the Gunicorn MASTER (which runs
        on_starting before forking) are the two once-per-server-start points."""
        import inspect
        import sys

        # Both modules parse argv at import time; pytest's would fail them.
        monkeypatch.setattr(sys, "argv", ["lightrag-server"])
        from lightrag.api import gunicorn_config, lightrag_server

        uvicorn_main = inspect.getsource(lightrag_server.main)
        assert "warn_about_workspace_overrides()" in uvicorn_main
        # Ahead of the splash screen, so it is not buried under it.
        assert uvicorn_main.index(
            "warn_about_workspace_overrides()"
        ) < uvicorn_main.index("display_splash_screen")

        master_hook = inspect.getsource(gunicorn_config.on_starting)
        assert "warn_about_workspace_overrides()" in master_hook
        assert "forking workers" in master_hook, (
            "on_starting must still be the pre-fork hook for this to be once "
            "per server start"
        )
