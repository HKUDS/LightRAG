import re
import asyncio
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
    _ngql_escape_string,
    _normalize_space_name,
    _parse_nebula_hosts,
    _short_hash_suffix,
    NebulaIndexJobError,
    NebulaGraphStorage,
)
from lightrag.types import KnowledgeGraph


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


def test_ngql_escape_string_escapes_control_characters():
    raw = 'line1\nline2\tcell\rend "quote" \\ slash'
    escaped = _ngql_escape_string(raw)
    assert "\n" not in escaped
    assert "\r" not in escaped
    assert "\t" not in escaped
    assert "\\n" in escaped
    assert "\\r" in escaped
    assert "\\t" in escaped
    assert '\\"quote\\"' in escaped
    assert "\\\\" in escaped


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
        patch.object(storage, "_execute", exec_mock),
        patch.object(storage, "_acquire_session", AsyncMock(return_value=session)),
        patch.object(storage, "_release_session", AsyncMock()),
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
    describe_tag_idx = next(i for i, sql in enumerate(sql_calls) if "DESCRIBE TAG entity" in sql)
    create_tag_index_idx = next(
        i for i, sql in enumerate(sql_calls) if "CREATE TAG INDEX IF NOT EXISTS entity_entity_id_idx" in sql
    )
    assert describe_tag_idx < create_tag_index_idx


@pytest.mark.asyncio
async def test_initialize_rejects_empty_required_env_values():
    storage = build_storage()
    storage._hosts = []
    storage._user = "root"
    storage._password = "nebula"

    with pytest.raises(ValueError, match="NEBULA_HOSTS"):
        await storage.initialize()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field_name", "field_value", "expected_env"),
    [
        ("_user", None, "NEBULA_USER"),
        ("_user", "   ", "NEBULA_USER"),
        ("_password", None, "NEBULA_PASSWORD"),
        ("_password", "   ", "NEBULA_PASSWORD"),
    ],
)
async def test_initialize_rejects_blank_user_or_password(
    field_name: str, field_value: str | None, expected_env: str
):
    storage = build_storage()
    storage._hosts = [("127.0.0.1", 9669)]
    storage._user = "root"
    storage._password = "nebula"
    setattr(storage, field_name, field_value)

    with pytest.raises(ValueError, match=expected_env):
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
        patch.object(storage, "_bootstrap_client", bootstrap_mock),
        patch.object(storage, "_ensure_space_ready", ensure_mock),
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
async def test_bootstrap_client_uses_http2_env_flag_independently_from_ssl():
    with patch.dict("os.environ", {"NEBULA_USE_HTTP2": "0", "NEBULA_SSL": "1"}):
        storage = build_storage()
    storage._hosts = [("127.0.0.1", 9669)]
    storage._user = "root"
    storage._password = "nebula"
    config = Mock()
    connection_pool = Mock()
    connection_pool.init.return_value = True
    with patch(
        "lightrag.kg.nebula_impl._load_nebula_client_types",
        return_value=(Mock(return_value=config), Mock(return_value=connection_pool)),
    ):
        await storage._bootstrap_client()

    assert config.use_http2 is False


@pytest.mark.asyncio
async def test_initialize_closes_connection_pool_when_ensure_space_ready_fails():
    storage = build_storage()
    storage._hosts = [("127.0.0.1", 9669)]
    storage._user = "root"
    storage._password = "nebula"
    connection_pool = Mock()

    async def fake_bootstrap():
        storage._connection_pool = connection_pool

    with (
        patch.object(storage, "_bootstrap_client", AsyncMock(side_effect=fake_bootstrap)),
        patch.object(storage, "_ensure_space_ready", AsyncMock(side_effect=RuntimeError("boom"))),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            await storage.initialize()

    connection_pool.close.assert_called_once()
    assert storage._connection_pool is None
    assert storage._initialized is False


@pytest.mark.asyncio
async def test_initialize_lock_prevents_duplicate_bootstrap():
    storage = build_storage()
    storage._hosts = [("127.0.0.1", 9669)]
    storage._user = "root"
    storage._password = "nebula"
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = {"bootstrap": 0}

    async def fake_bootstrap():
        calls["bootstrap"] += 1
        entered.set()
        await release.wait()

    with (
        patch.object(storage, "_bootstrap_client", AsyncMock(side_effect=fake_bootstrap)),
        patch.object(storage, "_ensure_space_ready", AsyncMock()),
    ):
        task1 = asyncio.create_task(storage.initialize())
        await entered.wait()
        task2 = asyncio.create_task(storage.initialize())
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(task1, task2)

    assert calls["bootstrap"] == 1


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
    with patch.object(storage, "_execute", execute_mock):
        await storage._wait_for_space_ready()

    assert execute_mock.await_count == 2


@pytest.mark.asyncio
async def test_wait_for_space_ready_times_out_when_space_not_visible():
    storage = build_storage(workspace="finance")
    storage._schema_retry_times = 2
    storage._schema_retry_delay_ms = 0
    execute_mock = AsyncMock(side_effect=[[["other_space"]], [["still_other"]]])
    with patch.object(storage, "_execute", execute_mock):
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
    with patch.object(storage, "_execute_in_space", execute_in_space_mock):
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
    with patch.object(storage, "_execute_in_space", execute_in_space_mock):
        with pytest.raises(TimeoutError, match="indexes"):
            await storage._wait_for_index_ready()


@pytest.mark.asyncio
async def test_wait_for_schema_ready_times_out_when_schema_not_visible():
    storage = build_storage(workspace="finance")
    storage._schema_retry_times = 2
    storage._schema_retry_delay_ms = 0
    execute_in_space_mock = AsyncMock(side_effect=RuntimeError("not ready"))
    with patch.object(storage, "_execute_in_space", execute_in_space_mock):
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
    with patch.object(storage, "_execute_in_space", execute_in_space_mock):
        with pytest.raises(NebulaIndexJobError, match="index job failed"):
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


@pytest.mark.asyncio
async def test_nebula_upsert_and_get_node_roundtrip():
    storage = build_storage(workspace="finance")
    execute_in_space = AsyncMock(
        side_effect=[
            object(),
            [
                {
                    "entity_id": "A",
                    "entity_type": "TypeX",
                    "description": "desc",
                    "keywords": "k1,k2",
                    "source_id": "src-1",
                }
            ],
        ]
    )
    with patch.object(storage, "_execute_in_space", execute_in_space):
        await storage.upsert_node(
            "A",
            {
                "entity_id": "A",
                "entity_type": "TypeX",
                "description": "desc",
                "keywords": "k1,k2",
                "source_id": "src-1",
            },
        )
        node = await storage.get_node("A")

    assert node is not None
    assert node["entity_id"] == "A"
    assert node["entity_type"] == "TypeX"
    assert node["description"] == "desc"
    assert node["keywords"] == "k1,k2"
    assert node["source_id"] == "src-1"
    upsert_sql = execute_in_space.await_args_list[0].args[0]
    assert "INSERT VERTEX entity" in upsert_sql
    assert 'VALUES "A"' in upsert_sql
    get_sql = execute_in_space.await_args_list[1].args[0]
    assert "FETCH PROP ON entity" in get_sql
    assert '"A"' in get_sql


@pytest.mark.asyncio
async def test_nebula_edge_reads_are_undirected():
    storage = build_storage(workspace="finance")
    execute_in_space = AsyncMock(
        side_effect=[
            object(),
            [
                {
                    "source": "A",
                    "target": "B",
                    "source_id": "chunk-1<SEP>chunk-2",
                    "target_id": "meta-target",
                    "relationship": "rel",
                    "description": "d",
                    "weight": 1.0,
                }
            ],
            [
                {
                    "source": "A",
                    "target": "B",
                    "source_id": "chunk-1<SEP>chunk-2",
                    "target_id": "meta-target",
                    "relationship": "rel",
                    "description": "d",
                    "weight": 1.0,
                }
            ],
        ]
    )
    with patch.object(storage, "_execute_in_space", execute_in_space):
        await storage.upsert_edge(
            "B",
            "A",
            {
                "relationship": "rel",
                "description": "d",
                "weight": 1.0,
            },
        )
        forward = await storage.get_edge("A", "B")
        reverse = await storage.get_edge("B", "A")

    assert forward == reverse
    upsert_sql = execute_in_space.await_args_list[0].args[0]
    assert 'VALUES "A"->"B"' in upsert_sql
    fetch_sql_1 = execute_in_space.await_args_list[1].args[0]
    fetch_sql_2 = execute_in_space.await_args_list[2].args[0]
    assert '"A"->"B"' in fetch_sql_1
    assert '"A"->"B"' in fetch_sql_2


@pytest.mark.asyncio
async def test_nebula_upsert_edge_forces_canonical_source_target_properties():
    storage = build_storage(workspace="finance")
    execute_in_space = AsyncMock(
        side_effect=[
            object(),
            [
                {
                    "source": "A",
                    "target": "B",
                    "source_id": "chunk1<SEP>chunk2",
                    "target_id": "meta-target",
                    "relationship": "rel",
                    "description": "d",
                    "weight": 1.0,
                }
            ],
        ]
    )
    with patch.object(storage, "_execute_in_space", execute_in_space):
        await storage.upsert_edge(
            "B",
            "A",
            {
                "source_id": "chunk1<SEP>chunk2",
                "target_id": "meta-target",
                "relationship": "rel",
                "description": "d",
                "weight": 1.0,
            },
        )
        edge = await storage.get_edge("A", "B")

    assert edge is not None
    assert edge["source"] == "A"
    assert edge["target"] == "B"
    assert edge["source_id"] == "chunk1<SEP>chunk2"
    assert edge["target_id"] == "meta-target"
    upsert_sql = execute_in_space.await_args_list[0].args[0]
    assert 'VALUES "A"->"B":("chunk1<SEP>chunk2", "meta-target"' in upsert_sql
    fetch_sql = execute_in_space.await_args_list[1].args[0]
    assert "src(edge) AS source" in fetch_sql
    assert "dst(edge) AS target" in fetch_sql


@pytest.mark.asyncio
async def test_nebula_delete_node_executes_delete_vertex():
    storage = build_storage(workspace="finance")
    execute_in_space = AsyncMock(return_value=object())
    with patch.object(storage, "_execute_in_space", execute_in_space):
        await storage.delete_node("A")

    sql = execute_in_space.await_args_list[0].args[0]
    assert "DELETE VERTEX" in sql
    assert '"A"' in sql


@pytest.mark.asyncio
async def test_nebula_remove_edges_canonicalizes_pairs():
    storage = build_storage(workspace="finance")
    execute_in_space = AsyncMock(return_value=object())
    with patch.object(storage, "_execute_in_space", execute_in_space):
        await storage.remove_edges([("B", "A"), ("D", "C")])

    sql_calls = [call.args[0] for call in execute_in_space.await_args_list]
    assert any('"A"->"B"' in sql for sql in sql_calls)
    assert any('"C"->"D"' in sql for sql in sql_calls)
    assert not any('"B"->"A"' in sql for sql in sql_calls)
    assert not any('"D"->"C"' in sql for sql in sql_calls)


@pytest.mark.asyncio
async def test_nebula_get_nodes_batch_uses_single_lookup_query():
    storage = build_storage(workspace="finance")
    execute_in_space = AsyncMock(
        return_value=[
            {
                "entity_id": "A",
                "name": "A",
                "entity_type": "TypeA",
                "description": "node-a",
                "keywords": "k1",
                "source_id": "s1",
            },
            {
                "entity_id": "B",
                "name": "B",
                "entity_type": "TypeB",
                "description": "node-b",
                "keywords": "k2",
                "source_id": "s2",
            },
        ]
    )
    with patch.object(storage, "_execute_in_space", execute_in_space):
        nodes = await storage.get_nodes_batch(["A", "B", "Missing"])

    assert nodes == {
        "A": {
            "entity_id": "A",
            "name": "A",
            "entity_type": "TypeA",
            "description": "node-a",
            "keywords": "k1",
            "source_id": "s1",
        },
        "B": {
            "entity_id": "B",
            "name": "B",
            "entity_type": "TypeB",
            "description": "node-b",
            "keywords": "k2",
            "source_id": "s2",
        },
    }
    assert execute_in_space.await_count == 1
    sql = execute_in_space.await_args_list[0].args[0]
    assert "LOOKUP ON entity" in sql
    assert "entity.entity_id" in sql


@pytest.mark.asyncio
async def test_nebula_node_degrees_batch_aggregates_with_single_query():
    storage = build_storage(workspace="finance")
    execute_in_space = AsyncMock(
        return_value=[
            {"source": "A", "target": "B"},
            {"source": "C", "target": "A"},
            {"source": "B", "target": "C"},
        ]
    )
    with patch.object(storage, "_execute_in_space", execute_in_space):
        degrees = await storage.node_degrees_batch(["A", "B", "C", "X"])

    assert degrees == {"A": 2, "B": 2, "C": 2, "X": 0}
    assert execute_in_space.await_count == 1
    sql = execute_in_space.await_args_list[0].args[0]
    assert "LOOKUP ON relation" in sql
    assert "src(edge) AS source" in sql
    assert "dst(edge) AS target" in sql


@pytest.mark.asyncio
async def test_nebula_get_edges_batch_uses_canonical_pairs_and_preserves_keys():
    storage = build_storage(workspace="finance")
    execute_in_space = AsyncMock(
        return_value=[
            {
                "source": "A",
                "target": "B",
                "source_id": "A",
                "target_id": "B",
                "relationship": "rel-ab",
                "description": "A-B edge",
                "weight": 2.5,
            }
        ]
    )
    pairs = [
        {"src": "B", "tgt": "A"},
        {"src": "A", "tgt": "B"},
        {"src": "A", "tgt": "C"},
    ]
    with patch.object(storage, "_execute_in_space", execute_in_space):
        edges = await storage.get_edges_batch(pairs)

    assert edges == {
        ("B", "A"): {
            "source": "A",
            "target": "B",
            "source_id": "A",
            "target_id": "B",
            "relationship": "rel-ab",
            "description": "A-B edge",
            "weight": 2.5,
        },
        ("A", "B"): {
            "source": "A",
            "target": "B",
            "source_id": "A",
            "target_id": "B",
            "relationship": "rel-ab",
            "description": "A-B edge",
            "weight": 2.5,
        },
    }
    assert execute_in_space.await_count == 1
    sql = execute_in_space.await_args_list[0].args[0]
    assert "LOOKUP ON relation" in sql
    assert 'src(edge) == "A"' in sql
    assert 'dst(edge) == "B"' in sql
    assert 'src(edge) == "B" AND dst(edge) == "A"' not in sql


@pytest.mark.asyncio
async def test_nebula_get_nodes_edges_batch_returns_adjacency_mapping():
    storage = build_storage(workspace="finance")
    execute_in_space = AsyncMock(
        return_value=[
            {"source": "A", "target": "B"},
            {"source": "C", "target": "A"},
        ]
    )
    with patch.object(storage, "_execute_in_space", execute_in_space):
        nodes_edges = await storage.get_nodes_edges_batch(["A", "B", "X"])

    assert nodes_edges == {
        "A": [("A", "B"), ("C", "A")],
        "B": [("A", "B")],
        "X": [],
    }
    assert execute_in_space.await_count == 1
    sql = execute_in_space.await_args_list[0].args[0]
    assert "LOOKUP ON relation" in sql
    assert "src(edge) AS source" in sql
    assert "dst(edge) AS target" in sql


@pytest.mark.asyncio
async def test_nebula_get_node_edges_returns_none_when_node_missing():
    storage = build_storage(workspace="finance")
    with patch.object(storage, "get_node", AsyncMock(return_value=None)):
        edges = await storage.get_node_edges("missing")
    assert edges is None


@pytest.mark.asyncio
async def test_nebula_get_node_edges_returns_edges_for_existing_node():
    storage = build_storage(workspace="finance")
    with (
        patch.object(storage, "get_node", AsyncMock(return_value={"entity_id": "A"})),
        patch.object(
            storage,
            "get_nodes_edges_batch",
            AsyncMock(return_value={"A": [("A", "B"), ("C", "A")]}),
        ),
    ):
        edges = await storage.get_node_edges("A")
    assert edges == [("A", "B"), ("C", "A")]


@pytest.mark.asyncio
async def test_nebula_get_all_labels_returns_sorted_entity_ids():
    storage = build_storage(workspace="finance")
    execute_in_space = AsyncMock(
        return_value=[
            {"entity_id": "B"},
            {"entity_id": "A"},
            {"entity_id": "C"},
        ]
    )
    with patch.object(storage, "_execute_in_space", execute_in_space):
        labels = await storage.get_all_labels()

    assert labels == ["A", "B", "C"]
    assert execute_in_space.await_count == 1
    sql = execute_in_space.await_args_list[0].args[0]
    assert "LOOKUP ON entity" in sql
    assert "entity.entity_id" in sql


@pytest.mark.asyncio
async def test_nebula_get_all_nodes_returns_node_property_dicts():
    storage = build_storage(workspace="finance")
    execute_in_space = AsyncMock(
        return_value=[
            {
                "entity_id": "A",
                "name": "A",
                "entity_type": "TypeA",
                "description": "desc-a",
                "keywords": "k1",
                "source_id": "src-a",
            },
            {
                "entity_id": "B",
                "name": "B",
                "entity_type": "TypeB",
                "description": "desc-b",
                "keywords": "k2",
                "source_id": "src-b",
            },
        ]
    )
    with patch.object(storage, "_execute_in_space", execute_in_space):
        nodes = await storage.get_all_nodes()

    assert nodes == [
        {
            "entity_id": "A",
            "name": "A",
            "entity_type": "TypeA",
            "description": "desc-a",
            "keywords": "k1",
            "source_id": "src-a",
            "id": "A",
        },
        {
            "entity_id": "B",
            "name": "B",
            "entity_type": "TypeB",
            "description": "desc-b",
            "keywords": "k2",
            "source_id": "src-b",
            "id": "B",
        },
    ]
    assert execute_in_space.await_count == 1
    sql = execute_in_space.await_args_list[0].args[0]
    assert "LOOKUP ON entity" in sql


@pytest.mark.asyncio
async def test_nebula_get_all_edges_returns_relation_properties():
    storage = build_storage(workspace="finance")
    execute_in_space = AsyncMock(
        return_value=[
            {
                "source": "A",
                "target": "B",
                "source_id": "chunk1<SEP>chunk2",
                "target_id": "meta-target",
                "relationship": "rel-ab",
                "description": "desc",
                "weight": 3.0,
            }
        ]
    )
    with patch.object(storage, "_execute_in_space", execute_in_space):
        edges = await storage.get_all_edges()

    assert edges == [
        {
            "source_id": "chunk1<SEP>chunk2",
            "target_id": "meta-target",
            "relationship": "rel-ab",
            "description": "desc",
            "weight": 3.0,
            "source": "A",
            "target": "B",
        }
    ]
    assert execute_in_space.await_count == 1
    sql = execute_in_space.await_args_list[0].args[0]
    assert "LOOKUP ON relation" in sql


@pytest.mark.asyncio
async def test_nebula_get_popular_labels_orders_by_degree_desc():
    storage = build_storage(workspace="finance")
    execute_in_space = AsyncMock(
        return_value=[
            {"source": "A", "target": "B"},
            {"source": "A", "target": "C"},
            {"source": "B", "target": "C"},
            {"source": "B", "target": "D"},
        ]
    )
    with patch.object(storage, "_execute_in_space", execute_in_space):
        labels = await storage.get_popular_labels(limit=3)

    assert labels == ["B", "A", "C"]
    assert execute_in_space.await_count == 1
    sql = execute_in_space.await_args_list[0].args[0]
    assert "LOOKUP ON relation" in sql
    assert "src(edge) AS source" in sql
    assert "dst(edge) AS target" in sql


@pytest.mark.asyncio
async def test_nebula_get_knowledge_graph_wildcard_returns_truncated_graph():
    storage = build_storage(workspace="finance")
    all_nodes = [
        {"entity_id": "A", "name": "A", "description": "node-a"},
        {"entity_id": "B", "name": "B", "description": "node-b"},
        {"entity_id": "C", "name": "C", "description": "node-c"},
    ]
    all_edges = [
        {"source": "A", "target": "B", "relationship": "ab"},
        {"source": "A", "target": "C", "relationship": "ac"},
    ]
    with (
        patch.object(storage, "get_all_nodes", AsyncMock(return_value=all_nodes)),
        patch.object(storage, "get_all_edges", AsyncMock(return_value=all_edges)),
    ):
        graph = await storage.get_knowledge_graph("*", max_depth=2, max_nodes=2)

    assert isinstance(graph, KnowledgeGraph)
    assert graph.is_truncated is True
    assert len(graph.nodes) == 2
    node_ids = {node.id for node in graph.nodes}
    assert "A" in node_ids
    assert len(graph.edges) == 1
    assert graph.edges[0].source in node_ids
    assert graph.edges[0].target in node_ids


@pytest.mark.asyncio
async def test_nebula_get_knowledge_graph_entity_returns_bounded_subgraph():
    storage = build_storage(workspace="finance")
    nodes_by_id = {
        "A": {"entity_id": "A", "name": "A"},
        "B": {"entity_id": "B", "name": "B"},
        "C": {"entity_id": "C", "name": "C"},
    }
    adjacency = {
        "A": [("A", "B"), ("A", "C")],
        "B": [("A", "B")],
    }
    edges = {
        ("A", "B"): {"source": "A", "target": "B", "relationship": "ab"},
        ("A", "C"): {"source": "A", "target": "C", "relationship": "ac"},
    }
    with (
        patch.object(storage, "get_nodes_batch", AsyncMock(return_value=nodes_by_id)),
        patch.object(
            storage, "get_nodes_edges_batch", AsyncMock(return_value=adjacency)
        ),
        patch.object(storage, "get_edges_batch", AsyncMock(return_value=edges)),
    ):
        graph = await storage.get_knowledge_graph("A", max_depth=1, max_nodes=2)

    assert isinstance(graph, KnowledgeGraph)
    assert graph.is_truncated is True
    assert len(graph.nodes) == 2
    node_ids = {node.id for node in graph.nodes}
    assert "A" in node_ids
    assert len(graph.edges) == 1
    assert graph.edges[0].source in node_ids
    assert graph.edges[0].target in node_ids


@pytest.mark.asyncio
async def test_search_labels_uses_fulltext_path_when_available():
    storage = build_storage(workspace="finance")
    fulltext_mock = AsyncMock(return_value=["learn", "learning"])
    fallback_mock = AsyncMock(return_value=["learning"])
    with (
        patch.object(storage, "_search_labels_fulltext", fulltext_mock),
        patch.object(storage, "_search_labels_contains", fallback_mock),
    ):
        labels = await storage.search_labels("learn", limit=10)

    assert labels == ["learn", "learning"]
    fulltext_mock.assert_awaited_once_with("learn", limit=10)
    fallback_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_search_labels_falls_back_when_fulltext_unavailable():
    storage = build_storage(workspace="finance")
    fulltext_mock = AsyncMock(side_effect=RuntimeError("no ft"))
    fallback_mock = AsyncMock(return_value=["Machine Learning"])
    with (
        patch.object(storage, "_search_labels_fulltext", fulltext_mock),
        patch.object(storage, "_search_labels_contains", fallback_mock),
    ):
        labels = await storage.search_labels("learn", limit=10)

    assert labels == ["Machine Learning"]
    fulltext_mock.assert_awaited_once_with("learn", limit=10)
    fallback_mock.assert_awaited_once_with("learn", limit=10)
