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
    exec_mock = AsyncMock()
    with (
        patch.object(storage, "_execute", exec_mock, create=True),
        patch.object(storage, "_wait_for_schema_ready", AsyncMock(), create=True),
        patch.object(storage, "_wait_for_index_ready", AsyncMock(), create=True),
    ):
        await storage._ensure_space_ready()

    sql_calls = [call.args[0] for call in exec_mock.await_args_list]
    assert any("CREATE SPACE IF NOT EXISTS" in sql for sql in sql_calls)
    assert any("USE " in sql for sql in sql_calls)
    assert any("CREATE TAG IF NOT EXISTS entity" in sql for sql in sql_calls)
    assert any("CREATE EDGE IF NOT EXISTS relation" in sql for sql in sql_calls)


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
async def test_finalize_closes_client_resources():
    storage = build_storage()
    session_pool = Mock()
    connection_pool = Mock()
    storage._session_pool = session_pool
    storage._connection_pool = connection_pool
    storage._initialized = True

    await storage.finalize()

    session_pool.close.assert_called_once()
    connection_pool.close.assert_called_once()
    assert storage._session_pool is None
    assert storage._connection_pool is None
    assert storage._initialized is False
