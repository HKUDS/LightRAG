import re
from unittest.mock import AsyncMock, Mock, patch

import pytest

from lightrag.kg import (
    STORAGE_IMPLEMENTATIONS,
    STORAGE_ENV_REQUIREMENTS,
    STORAGES,
    verify_storage_implementation,
)
from lightrag.kg.nebula_impl import (
    _canonical_edge_pair,
    _normalize_space_name,
    _parse_nebula_hosts,
    _short_hash_suffix,
    NebulaGraphStorage,
)


def build_storage(workspace: str | None = "finance") -> NebulaGraphStorage:
    return NebulaGraphStorage(
        namespace="test",
        workspace=workspace,
        global_config={},
        embedding_func=lambda *args, **kwargs: None,
    )


def test_nebula_graph_storage_is_registered():
    assert (
        "NebulaGraphStorage"
        in STORAGE_IMPLEMENTATIONS["GRAPH_STORAGE"]["implementations"]
    )
    assert STORAGES["NebulaGraphStorage"] == ".kg.nebula_impl"


def test_nebula_graph_storage_env_requirements():
    assert STORAGE_ENV_REQUIREMENTS["NebulaGraphStorage"] == [
        "NEBULA_HOSTS",
        "NEBULA_USER",
        "NEBULA_PASSWORD",
    ]


def test_nebula_graph_storage_verify_compatibility():
    verify_storage_implementation("GRAPH_STORAGE", "NebulaGraphStorage")


def test_normalize_space_name_uses_prefix_and_workspace():
    assert _normalize_space_name("lightrag", "hr-prod") == "lightrag__hr_prod"


def test_normalize_space_name_uses_base_for_empty_workspace():
    assert _normalize_space_name("lightrag", "") == "lightrag__base"


def test_canonical_edge_pair_is_undirected():
    assert _canonical_edge_pair("B", "A") == ("A", "B")


def test_short_hash_suffix_rejects_non_positive_length():
    with pytest.raises(ValueError, match="positive"):
        _short_hash_suffix("abc", length=0)


def test_normalize_space_name_truncates_and_appends_hash_suffix():
    normalized = _normalize_space_name("lightrag", "w" * 180)
    assert len(normalized) == 127
    assert re.search(r"__[0-9a-f]{8}$", normalized)


def test_parse_nebula_hosts_supports_ipv4_and_bracket_ipv6():
    assert _parse_nebula_hosts("127.0.0.1:9669, [::1]:9779") == [
        ("127.0.0.1", 9669),
        ("::1", 9779),
    ]


def test_parse_nebula_hosts_rejects_out_of_range_port():
    with pytest.raises(ValueError, match="must be in 1..65535"):
        _parse_nebula_hosts("127.0.0.1:70000")


def test_parse_nebula_hosts_supports_unbracketed_ipv6_with_default_port():
    assert _parse_nebula_hosts("::1") == [("::1", 9669)]


def test_nebula_graph_storage_sets_initialized_as_instance_attr():
    storage = build_storage(workspace=None)
    assert "_initialized" in storage.__dict__
    assert storage._initialized is False


@pytest.mark.asyncio
async def test_initialize_creates_space_and_schema():
    storage = build_storage(workspace="finance")
    session = Mock()
    exec_mock = AsyncMock(
        side_effect=lambda sql, **_: (
            [[storage._space_name]]
            if "SHOW SPACES" in sql
            else [["job", "entity_entity_id_idx", "FINISHED"]]
            if "SHOW TAG INDEX STATUS" in sql
            else [["job", "relation_pair_idx", "FINISHED"]]
            if "SHOW EDGE INDEX STATUS" in sql
            else object()
        )
    )
    with (
        patch.object(storage, "_execute", exec_mock, create=True),
        patch.object(storage, "_acquire_session", AsyncMock(return_value=session), create=True),
        patch.object(storage, "_release_session", AsyncMock(), create=True),
    ):
        await storage._ensure_space_ready()

    sql_calls = [call.args[0] for call in exec_mock.await_args_list]
    assert any("CREATE SPACE IF NOT EXISTS" in sql for sql in sql_calls)
    assert any("SHOW SPACES" in sql for sql in sql_calls)
    assert any("USE " in sql for sql in sql_calls)
    assert any("CREATE TAG IF NOT EXISTS entity" in sql for sql in sql_calls)
    assert any("CREATE EDGE IF NOT EXISTS relation" in sql for sql in sql_calls)
    assert any("CREATE FULLTEXT TAG INDEX IF NOT EXISTS entity_name_ft_idx" in sql for sql in sql_calls)
    assert any("CREATE FULLTEXT EDGE INDEX IF NOT EXISTS relation_rel_ft_idx" in sql for sql in sql_calls)
    assert any("REBUILD TAG INDEX entity_entity_id_idx" in sql for sql in sql_calls)
    assert any("REBUILD EDGE INDEX relation_pair_idx" in sql for sql in sql_calls)
    assert any("SHOW TAG INDEX STATUS" in sql for sql in sql_calls)
    assert any("SHOW EDGE INDEX STATUS" in sql for sql in sql_calls)


@pytest.mark.asyncio
async def test_initialize_rejects_empty_required_env_values():
    storage = build_storage()
    storage._hosts = []
    storage._user = "root"
    storage._password = "nebula"

    with pytest.raises(ValueError, match="NEBULA_HOSTS"):
        await storage.initialize()


@pytest.mark.asyncio
async def test_initialize_is_idempotent():
    storage = build_storage()
    storage._hosts = [("127.0.0.1", 9669)]
    storage._user = "root"
    storage._password = "nebula"
    bootstrap_mock = AsyncMock()
    ensure_mock = AsyncMock()
    with (
        patch.object(storage, "_bootstrap_client", bootstrap_mock, create=True),
        patch.object(storage, "_ensure_space_ready", ensure_mock, create=True),
    ):
        await storage.initialize()
        await storage.initialize()

    assert storage._initialized is True
    assert bootstrap_mock.await_count == 1
    assert ensure_mock.await_count == 1


@pytest.mark.asyncio
async def test_bootstrap_client_initializes_connection_pool_only():
    storage = build_storage()
    storage._hosts = [("127.0.0.1", 9669)]
    storage._user = "root"
    storage._password = "nebula"
    config = Mock()
    connection_pool = Mock()
    connection_pool.init.return_value = True
    connection_pool_cls = Mock(return_value=connection_pool)

    with (
        patch(
            "lightrag.kg.nebula_impl._load_nebula_client_types",
            return_value=(Mock(return_value=config), connection_pool_cls),
        ),
        patch.object(storage, "_ensure_space_ready", AsyncMock()),
    ):
        await storage.initialize()

    connection_pool_cls.assert_called_once()
    connection_pool.init.assert_called_once()
    assert storage._connection_pool is connection_pool


@pytest.mark.asyncio
async def test_finalize_closes_client_resources():
    storage = build_storage()
    connection_pool = Mock()
    storage._connection_pool = connection_pool
    storage._initialized = True

    await storage.finalize()

    connection_pool.close.assert_called_once()
    assert storage._connection_pool is None
    assert storage._initialized is False


@pytest.mark.asyncio
async def test_wait_for_space_ready_polls_until_target_space_visible():
    storage = build_storage(workspace="finance")
    storage._schema_retry_times = 3
    storage._schema_retry_delay_ms = 0
    execute_mock = AsyncMock(
        side_effect=[
            [["other_space"]],
            [[storage._space_name]],
        ]
    )
    with patch.object(storage, "_execute", execute_mock, create=True):
        await storage._wait_for_space_ready()

    assert execute_mock.await_count == 2


@pytest.mark.asyncio
async def test_wait_for_space_ready_times_out_when_space_not_visible():
    storage = build_storage(workspace="finance")
    storage._schema_retry_times = 2
    storage._schema_retry_delay_ms = 0
    execute_mock = AsyncMock(side_effect=[[["other_space"]], [["still_other"]]])
    with patch.object(storage, "_execute", execute_mock, create=True):
        with pytest.raises(TimeoutError, match="space"):
            await storage._wait_for_space_ready()


@pytest.mark.asyncio
async def test_wait_for_index_ready_polls_until_status_finished():
    storage = build_storage(workspace="finance")
    storage._schema_retry_times = 3
    storage._schema_retry_delay_ms = 0
    execute_in_space_mock = AsyncMock(
        side_effect=[
            [["job", "entity_entity_id_idx", "RUNNING"]],
            [["job", "relation_pair_idx", "RUNNING"]],
            [["job", "entity_entity_id_idx", "FINISHED"]],
            [["job", "relation_pair_idx", "FINISHED"]],
        ]
    )
    with patch.object(storage, "_execute_in_space", execute_in_space_mock, create=True):
        await storage._wait_for_index_ready()

    assert execute_in_space_mock.await_count == 4


@pytest.mark.asyncio
async def test_wait_for_index_ready_times_out_when_still_running():
    storage = build_storage(workspace="finance")
    storage._schema_retry_times = 2
    storage._schema_retry_delay_ms = 0
    execute_in_space_mock = AsyncMock(
        side_effect=[
            [["job", "entity_entity_id_idx", "RUNNING"]],
            [["job", "relation_pair_idx", "RUNNING"]],
            [["job", "entity_entity_id_idx", "RUNNING"]],
            [["job", "relation_pair_idx", "RUNNING"]],
        ]
    )
    with patch.object(storage, "_execute_in_space", execute_in_space_mock, create=True):
        with pytest.raises(TimeoutError, match="indexes"):
            await storage._wait_for_index_ready()


@pytest.mark.asyncio
async def test_wait_for_schema_ready_times_out_when_schema_not_visible():
    storage = build_storage(workspace="finance")
    storage._schema_retry_times = 2
    storage._schema_retry_delay_ms = 0
    execute_in_space_mock = AsyncMock(side_effect=RuntimeError("not ready"))
    with patch.object(storage, "_execute_in_space", execute_in_space_mock, create=True):
        with pytest.raises(TimeoutError, match="schema"):
            await storage._wait_for_schema_ready()


@pytest.mark.asyncio
async def test_wait_for_index_ready_raises_immediately_on_failed_job():
    storage = build_storage(workspace="finance")
    storage._schema_retry_times = 3
    storage._schema_retry_delay_ms = 0
    execute_in_space_mock = AsyncMock(
        side_effect=[
            [["job", "entity_entity_id_idx", "FAILED"]],
            [["job", "relation_pair_idx", "FINISHED"]],
        ]
    )
    with patch.object(storage, "_execute_in_space", execute_in_space_mock, create=True):
        with pytest.raises(RuntimeError, match="index job failed"):
            await storage._wait_for_index_ready()

    assert execute_in_space_mock.await_count == 2


@pytest.mark.asyncio
async def test_execute_in_space_uses_same_session_for_use_and_query():
    storage = build_storage(workspace="finance")
    session = Mock()
    session.execute.side_effect = [object(), object()]
    session.release = Mock()
    connection_pool = Mock()
    connection_pool.get_session.return_value = session
    storage._connection_pool = connection_pool
    storage._user = "root"
    storage._password = "nebula"

    await storage._execute_in_space("SHOW TAG INDEX STATUS;")

    connection_pool.get_session.assert_called_once_with("root", "nebula")
    assert session.execute.call_args_list[0].args[0] == f"USE `{storage._space_name}`;"
    assert session.execute.call_args_list[1].args[0] == "SHOW TAG INDEX STATUS;"
    session.release.assert_called_once()
