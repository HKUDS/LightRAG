"""The configuration container is unreachable by naming a workspace.

Slice 1 paid for "configuration is not a tenant" with a reserved ``_lightrag*``
NAME, defended in ``validate_workspace``, in a context-variable grant, and in
``validate_workspace_override`` at six backends. The configuration storage is
now its own category on a container named in CODE, so none of that has to
exist: there is no name to collide with, and no override can redirect into a
container no workspace addresses. These tests pin that property -- and that
the ``*_WORKSPACE`` variables still only move TENANT data.

See docs/design/ConfigurationStorageContract.md.
"""

from __future__ import annotations

import logging

import pytest
from types import SimpleNamespace

from lightrag.namespace import CONFIG_CONTAINER_TAG
from lightrag.utils import (
    WORKSPACE_OVERRIDE_SOURCES,
    validate_workspace,
    validate_workspace_override,
)

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    "name",
    [
        "",
        "lightrag",
        "my_lightrag",
        "v1.0",
        "_other",
        # No longer reserved: there is nothing to reserve it from.
        "_lightrag_config",
        "_LightRAG_config",
        "_lightrag_server",
    ],
)
def test_every_legal_name_passes_including_the_old_reserved_family(name):
    assert validate_workspace(name) == name


@pytest.mark.parametrize("name", ["a/b", "a\\b", ".", ".."])
def test_path_traversal_is_still_refused(name):
    with pytest.raises(ValueError, match="path separators|relative path"):
        validate_workspace(name)


def test_the_reserved_machinery_is_gone():
    """A rule that no longer exists must not linger as dead code someone
    later re-wires: the grant was the thing that made a public
    ``allow_reserved`` flag tempting."""
    import lightrag.namespace as ns
    import lightrag.utils as utils

    for gone in ("RESERVED_WORKSPACE_PREFIX", "CONFIG_WORKSPACE"):
        assert not hasattr(ns, gone), gone
    for gone in ("is_reserved_workspace", "_grant_reserved_workspace"):
        assert not hasattr(utils, gone), gone


def test_a_tenant_named_after_the_container_shares_nothing_with_it(tmp_path):
    """The whole point of the category. A workspace may now legally be called
    ``_lightrag_config`` and still cannot reach the configuration container --
    on the JSON backend they are different FILES in the same directory."""
    from lightrag import config_store as cs
    from lightrag.kg.json_kv_impl import JsonKVStorage

    global_config = {"working_dir": str(tmp_path)}
    config = cs.create_configuration_storage(
        JsonKVStorage, global_config=global_config, embedding_func=None
    )
    for namespace in ("full_docs", "text_chunks", "llm_response_cache"):
        tenant = JsonKVStorage(
            namespace=namespace,
            workspace=CONFIG_CONTAINER_TAG,
            global_config=global_config,
            embedding_func=None,
        )
        assert tenant._file_name != config._file_name


def test_a_public_lightrag_instance_may_take_the_old_reserved_name(tmp_path):
    """It is an ordinary workspace now. Its data lands beside the
    configuration file, not in it."""
    import numpy as np

    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc, Tokenizer, TokenizerInterface

    class _StubTokenizer(TokenizerInterface):
        def encode(self, content: str) -> list[int]:
            return [ord(c) for c in content]

        def decode(self, tokens: list[int]) -> str:
            return "".join(chr(t) for t in tokens)

    async def _llm(prompt, **kwargs):  # pragma: no cover - never called
        return ""

    async def _embed(texts, **kwargs):  # pragma: no cover - never called
        return np.zeros((len(texts), 8), dtype=np.float32)

    rag = LightRAG(
        working_dir=str(tmp_path),
        workspace=CONFIG_CONTAINER_TAG,
        llm_model_func=_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=1024, func=_embed, model_name="m"
        ),
        # The default tokenizer downloads tiktoken data; not this test's
        # subject, and not available offline.
        tokenizer=Tokenizer("stub", _StubTokenizer()),
    )
    assert rag.workspace == CONFIG_CONTAINER_TAG
    assert rag.full_docs._file_name != rag.configuration_storage._file_name


class TestEnvironmentRemapDoesNotReachTheConfigurationContainer:
    """``POSTGRES_WORKSPACE`` / ``MONGODB_WORKSPACE`` /
    ``OPENSEARCH_WORKSPACE`` move TENANT data. The configuration container is
    named in code and does not consult them -- which is now enforced by the
    ``config`` NAMESPACE, not by a name."""

    def test_opensearch(self, monkeypatch):
        from lightrag.kg.opensearch_impl import _build_index_name, _resolve_workspace

        monkeypatch.setenv("OPENSEARCH_WORKSPACE", "prod")
        assert _resolve_workspace("tenant", "text_chunks") == "prod"
        assert _resolve_workspace("tenant", "config") == CONFIG_CONTAINER_TAG
        # Fixed, whatever the caller named.
        assert (
            _build_index_name("tenant", "config")[2]
            == _build_index_name("other", "config")[2]
        )

    def test_mongodb(self, monkeypatch):
        from lightrag.kg.mongo_impl import MongoKVStorage

        monkeypatch.setenv("MONGODB_WORKSPACE", "prod")
        config = MongoKVStorage(
            namespace="config",
            global_config={},
            embedding_func=None,
            workspace="tenant",
        )
        assert config.workspace == CONFIG_CONTAINER_TAG
        assert config._collection_name == f"{CONFIG_CONTAINER_TAG}_config"
        tenant = MongoKVStorage(
            namespace="text_chunks",
            global_config={},
            embedding_func=None,
            workspace="tenant",
        )
        assert tenant.workspace == "prod"

    async def test_postgresql(self):
        from lightrag.kg.postgres_impl import PGKVStorage
        from lightrag.kg.shared_storage import (
            finalize_share_data,
            initialize_share_data,
        )

        initialize_share_data()
        try:
            db = SimpleNamespace(workspace="prod")
            config = PGKVStorage.__new__(PGKVStorage)
            config.namespace = "config"
            config.global_config = {}
            config.db = db
            config.workspace = "tenant"
            config.__post_init__()
            await config.initialize()
            assert config.workspace == CONFIG_CONTAINER_TAG

            tenant = PGKVStorage.__new__(PGKVStorage)
            tenant.namespace = "text_chunks"
            tenant.global_config = {}
            tenant.db = db
            tenant.workspace = "tenant"
            tenant.__post_init__()
            await tenant.initialize()
            assert tenant.workspace == "prod"
        finally:
            finalize_share_data()

    def test_json(self, tmp_path):
        """The JSON backend honors no override at all; what it must not do is
        follow the workspace into ``working_dir/<workspace>``."""
        from lightrag.kg.json_kv_impl import JsonKVStorage

        storage = JsonKVStorage(
            namespace="config",
            workspace="tenant",
            global_config={"working_dir": str(tmp_path)},
            embedding_func=None,
        )
        assert storage._file_name == str(
            tmp_path / CONFIG_CONTAINER_TAG / "kv_server_config.json"
        )


class TestTheOverrideValidatorStillValidates:
    """``validate_workspace_override`` loses its reserved-name check and keeps
    the rest: the value a ``*_WORKSPACE`` variable supplies is applied INSIDE a
    constructor, after ``validate_workspace()`` already passed the constructor
    argument, so this is the only place it is ever checked."""

    def test_it_strips_and_passes_ordinary_values(self):
        assert validate_workspace_override("X_WORKSPACE", " prod ") == "prod"
        assert validate_workspace_override("X_WORKSPACE", None) is None
        assert validate_workspace_override("X_WORKSPACE", "") == ""
        assert validate_workspace_override("X_WORKSPACE", CONFIG_CONTAINER_TAG) == (
            CONFIG_CONTAINER_TAG
        )

    @pytest.mark.parametrize("value", ["../etc", "a/b", "..", "a\\b"])
    def test_a_traversing_override_is_refused_and_names_the_variable(self, value):
        with pytest.raises(ValueError, match="X_WORKSPACE"):
            validate_workspace_override("X_WORKSPACE", value)


def test_a_refused_override_leaks_no_configuration_pool_reference(
    tmp_path, monkeypatch
):
    """``RedisKVStorage`` takes its shared-pool reference in its constructor,
    and ``LightRAG.__post_init__`` has no async teardown. Had the
    configuration storage been constructed FIRST, the refusal in the first
    ordinary storage's constructor would have left its reference behind,
    growing on every failed construction. It is constructed last, so a refused
    override acquires nothing.

    ``config_storage`` is named explicitly because Redis is not in the
    configuration category: leaving it to follow ``kv_storage`` would refuse
    the construction before any storage is built, which is a different test."""
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
    monkeypatch.setenv("REDIS_WORKSPACE", "../escape")
    monkeypatch.setattr(
        "lightrag.kg.redis_impl.ConnectionPool.from_url",
        lambda *args, **kwargs: MagicMock(name="pool"),
    )
    monkeypatch.setattr(
        "lightrag.kg.redis_impl.Redis", lambda connection_pool=None, **_: MagicMock()
    )

    async def _llm(prompt, **kwargs):  # pragma: no cover - never called
        return ""

    with pytest.raises(ValueError, match="REDIS_WORKSPACE"):
        LightRAG(
            working_dir=str(tmp_path),
            workspace="tenant",
            llm_model_func=_llm,
            kv_storage="RedisKVStorage",
            config_storage="JsonKVStorage",
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
