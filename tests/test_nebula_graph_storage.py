import re
import asyncio
from pathlib import Path
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
    _result_to_rows,
    _short_hash_suffix,
    _unwrap_nebula_value,
    NebulaIndexJobError,
    NebulaGraphStorage,
)
from lightrag.types import KnowledgeGraph


REPO_ROOT = Path(__file__).resolve().parents[1]


def _extract_details_section(content: str, summary_text: str) -> str:
    start = content.find(summary_text)
    if start == -1:
        raise AssertionError(f"Section summary not found: {summary_text}")

    end = content.find("</details>", start)
    if end == -1:
        raise AssertionError(f"Section end not found: {summary_text}")

    return content[start:end]


def build_storage(workspace: str | None = "finance") -> NebulaGraphStorage:
    return NebulaGraphStorage(
        namespace="test",
        workspace=workspace,
        global_config={},
        embedding_func=lambda *args, **kwargs: None,
    )


def _normalize_sql_whitespace(sql: str) -> str:
    return " ".join(str(sql).split())


def _assert_bounded_nebula_query(
    sql: str,
    *,
    required_tokens: list[str],
    forbidden_patterns: list[str],
) -> None:
    normalized_sql = _normalize_sql_whitespace(sql)
    for token in required_tokens:
        assert token in normalized_sql, f"Expected token {token!r} in SQL: {normalized_sql}"
    for pattern in forbidden_patterns:
        normalized_pattern = _normalize_sql_whitespace(pattern)
        assert (
            normalized_pattern not in normalized_sql
        ), f"Unexpected unbounded pattern {normalized_pattern!r} in SQL: {normalized_sql}"


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


def test_nebula_env_example_documents_required_keys():
    content = (REPO_ROOT / "env.example").read_text(encoding="utf-8")

    assert "LIGHTRAG_GRAPH_STORAGE=NebulaGraphStorage" in content
    assert "NEBULA_HOSTS" in content
    assert "NEBULA_USER" in content
    assert "NEBULA_PASSWORD" in content
    assert "NEBULA_LISTENER_HOSTS" in content


def test_nebula_readme_documents_manual_configuration_flow():
    readme_en = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    readme_zh = (REPO_ROOT / "README-zh.md").read_text(encoding="utf-8")
    nebula_en = _extract_details_section(
        readme_en, "<summary> <b>Using NebulaGraph Storage</b> </summary>"
    )
    nebula_zh = _extract_details_section(
        readme_zh, "<summary> <b>使用 NebulaGraph 存储</b> </summary>"
    )

    for content in (nebula_en, nebula_zh):
        assert "NebulaGraphStorage" in content
        assert "NEBULA_HOSTS" in content
        assert "NEBULA_USER" in content
        assert "NEBULA_PASSWORD" in content
        assert "NEBULA_LISTENER_HOSTS" in content
        assert "search_labels" in content
        assert "Elasticsearch" in content
        assert "Listener" in content
        assert "empty string" in content or "空字符串" in content
        assert re.search(
            r"workspace.{0,160}space|space.{0,160}workspace",
            content,
            re.IGNORECASE | re.DOTALL,
        )


def test_normalize_space_name_uses_prefix_and_workspace():
    assert _normalize_space_name("lightrag", "hr-prod") == "lightrag__hr_prod"


def test_normalize_space_name_uses_base_for_empty_workspace():
    assert _normalize_space_name("lightrag", "") == "lightrag__base"


def test_canonical_edge_pair_is_undirected():
    assert _canonical_edge_pair("B", "A") == ("A", "B")


def test_schema_field_exists_error_accepts_nebula_existed_message():
    assert (
        NebulaGraphStorage._is_schema_field_exists_error(RuntimeError("Existed!"))
        is True
    )


def test_schema_field_exists_error_accepts_wrapped_nebula_existed_message():
    assert (
        NebulaGraphStorage._is_schema_field_exists_error(
            RuntimeError(
                "Nebula query failed: ALTER TAG entity ADD (file_path string); (Existed!)"
            )
        )
        is True
    )


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


def test_unwrap_nebula_value_supports_thrift_style_getters():
    class DummyValue:
        def get_sVal(self):
            return b"nebula-space"

    assert _unwrap_nebula_value(DummyValue()) == "nebula-space"


def test_result_to_rows_parses_thrift_style_rows():
    class DummyValue:
        def __init__(self, value):
            self._value = value

        def get_sVal(self):
            return self._value.encode("utf-8")

    class DummyRow:
        def __init__(self, *values):
            self.values = [DummyValue(v) for v in values]

    class DummyResult:
        def keys(self):
            return [b"Name"]

        rows = [DummyRow("space_a"), DummyRow("space_b")]

    assert _result_to_rows(DummyResult()) == [
        {"Name": "space_a"},
        {"Name": "space_b"},
    ]


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
            [["job", "entity_entity_id_idx", "FINISHED"]]
            if "SHOW TAG INDEX STATUS" in sql
            else [["job", "relation_pair_idx", "FINISHED"]]
            if "SHOW EDGE INDEX STATUS" in sql
            else object()
        )
    )
    use_space_mock = AsyncMock()
    with (
        patch.object(storage, "_execute", exec_mock),
        patch.object(storage, "_acquire_session", AsyncMock(return_value=session)),
        patch.object(storage, "_release_session", AsyncMock()),
        patch.object(storage, "_use_space", use_space_mock),
        patch.object(storage, "_ensure_fulltext_ready", AsyncMock()),
    ):
        await storage._ensure_space_ready()

    sql_calls = [call.args[0] for call in exec_mock.await_args_list]
    assert any("CREATE SPACE IF NOT EXISTS" in sql for sql in sql_calls)
    assert use_space_mock.await_count >= 1
    assert any("CREATE TAG IF NOT EXISTS entity" in sql for sql in sql_calls)
    assert any("file_path string" in sql for sql in sql_calls)
    assert any("created_at int" in sql for sql in sql_calls)
    assert any("truncate string" in sql for sql in sql_calls)
    assert any("ALTER TAG entity ADD (file_path string);" == sql for sql in sql_calls)
    assert any("ALTER TAG entity ADD (created_at int);" == sql for sql in sql_calls)
    assert any("ALTER TAG entity ADD (truncate string);" == sql for sql in sql_calls)
    assert any("CREATE EDGE IF NOT EXISTS relation" in sql for sql in sql_calls)
    assert any("keywords string" in sql for sql in sql_calls)
    assert any("file_path string" in sql for sql in sql_calls)
    assert any("ALTER EDGE relation ADD (keywords string);" == sql for sql in sql_calls)
    assert any("ALTER EDGE relation ADD (file_path string);" == sql for sql in sql_calls)
    assert any(
        f"CREATE FULLTEXT TAG INDEX IF NOT EXISTS {storage._fulltext_tag_index_name}"
        in sql
        for sql in sql_calls
    )
    assert any(
        f"CREATE FULLTEXT EDGE INDEX IF NOT EXISTS {storage._fulltext_edge_index_name}"
        in sql
        for sql in sql_calls
    )
    assert any("REBUILD FULLTEXT INDEX" in sql for sql in sql_calls)
    assert any("REBUILD TAG INDEX entity_entity_id_idx" in sql for sql in sql_calls)
    assert any("REBUILD EDGE INDEX relation_pair_idx" in sql for sql in sql_calls)
    assert any("SHOW TAG INDEX STATUS" in sql for sql in sql_calls)
    assert any("SHOW EDGE INDEX STATUS" in sql for sql in sql_calls)
    describe_tag_idx = next(
        i
        for i, sql in enumerate(sql_calls)
        if "MATCH (v:entity) RETURN count(v) AS vertex_count" in sql
    )
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
async def test_initialize_allows_empty_password():
    storage = build_storage()
    storage._hosts = [("127.0.0.1", 9669)]
    storage._user = "root"
    storage._password = ""
    with (
        patch.object(storage, "_bootstrap_client", AsyncMock()),
        patch.object(storage, "_ensure_space_ready", AsyncMock()),
    ):
        await storage.initialize()

    assert storage._initialized is True


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
    session = Mock()
    with (
        patch.object(storage, "_acquire_session", AsyncMock(return_value=session)),
        patch.object(storage, "_release_session", AsyncMock()),
        patch.object(storage, "_use_space", AsyncMock(side_effect=[RuntimeError("not ready"), None])),
    ):
        await storage._wait_for_space_ready()


@pytest.mark.asyncio
async def test_wait_for_space_ready_retries_until_space_can_be_used():
    storage = build_storage(workspace="finance")
    storage._schema_retry_times = 4
    storage._schema_retry_delay_ms = 0
    session = Mock()
    acquire_mock = AsyncMock(return_value=session)
    release_mock = AsyncMock()
    use_mock = AsyncMock(
        side_effect=[RuntimeError("not ready"), RuntimeError("not ready"), None]
    )
    with (
        patch.object(storage, "_acquire_session", acquire_mock),
        patch.object(storage, "_release_session", release_mock),
        patch.object(storage, "_use_space", use_mock),
    ):
        await storage._wait_for_space_ready()

    assert acquire_mock.await_count == 3
    assert use_mock.await_count == 3
    assert release_mock.await_count == 3


@pytest.mark.asyncio
async def test_wait_for_space_ready_default_retry_budget_handles_live_cluster_delay():
    with patch.dict("os.environ", {}, clear=True):
        storage = build_storage(workspace="finance")
    storage._schema_retry_delay_ms = 0
    session = Mock()
    use_mock = AsyncMock(side_effect=[RuntimeError("not ready")] * 35 + [None])
    with (
        patch.object(storage, "_acquire_session", AsyncMock(return_value=session)),
        patch.object(storage, "_release_session", AsyncMock()),
        patch.object(storage, "_use_space", use_mock),
    ):
        await storage._wait_for_space_ready()

    assert use_mock.await_count == 36


@pytest.mark.asyncio
async def test_wait_for_space_ready_times_out_when_space_not_visible():
    storage = build_storage(workspace="finance")
    storage._schema_retry_times = 2
    storage._schema_retry_delay_ms = 0
    with (
        patch.object(storage, "_acquire_session", AsyncMock(return_value=Mock())),
        patch.object(storage, "_release_session", AsyncMock()),
        patch.object(storage, "_use_space", AsyncMock(side_effect=RuntimeError("not ready"))),
    ):
        with pytest.raises(TimeoutError, match="space"):
            await storage._wait_for_space_ready()


@pytest.mark.asyncio
async def test_ensure_fulltext_ready_accepts_named_listener_status_columns():
    storage = build_storage(workspace="finance")
    execute_mock = AsyncMock(
        side_effect=[
            [{"client_type": "ELASTICSEARCH"}],  # SHOW TEXT SEARCH CLIENTS
            [{"Host Status": "ONLINE"}],         # initial SHOW LISTENER
            [{"Host Status": "ONLINE"}],         # polling SHOW LISTENER
        ]
    )
    with patch.object(storage, "_execute_in_space", execute_mock), patch.object(
        storage, "_execute", execute_mock
    ):
        await storage._ensure_fulltext_ready()


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
async def test_wait_for_schema_ready_default_retry_budget_handles_live_cluster_delay():
    with patch.dict("os.environ", {}, clear=True):
        storage = build_storage(workspace="finance")
    storage._schema_retry_delay_ms = 0
    execute_in_space_mock = AsyncMock(
        side_effect=[RuntimeError("not ready")] * 42 + [object(), object()]
    )
    with patch.object(storage, "_execute_in_space", execute_in_space_mock):
        await storage._wait_for_schema_ready()

    assert execute_in_space_mock.await_count == 44


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


def test_is_index_status_ready_accepts_empty_status_output():
    storage = build_storage(workspace="finance")
    assert storage._is_index_status_ready([]) is True


def test_rank_search_candidate_rows_prefers_entity_id_before_name_matches():
    rows = [
        {"entity_id": "KB-001", "name": "learn"},
        {"entity_id": "learning", "name": "zzz"},
        {"entity_id": "learn", "name": "other"},
    ]

    labels = NebulaGraphStorage._rank_search_candidate_rows(rows, "learn", limit=10)

    assert labels == ["learn", "learning", "KB-001"]


def test_rank_search_candidate_rows_returns_entity_id_for_name_only_match():
    rows = [{"entity_id": "KB-001", "name": "learn-node"}]

    labels = NebulaGraphStorage._rank_search_candidate_rows(rows, "learn", limit=10)

    assert labels == ["KB-001"]


def test_rank_search_candidate_rows_deduplicates_entity_id():
    rows = [
        {"entity_id": "learn", "name": "learn"},
        {"entity_id": "learn", "name": "learn-node"},
        {"entity_id": "learning", "name": "learn-ish"},
    ]

    labels = NebulaGraphStorage._rank_search_candidate_rows(rows, "learn", limit=10)

    assert labels == ["learn", "learning"]


@pytest.mark.asyncio
async def test_create_indexes_falls_back_when_fulltext_if_not_exists_is_unsupported():
    storage = build_storage(workspace="finance")
    execute_in_space = AsyncMock(
        side_effect=[
            object(),  # create tag index
            object(),  # create edge index
            object(),  # rebuild tag index
            object(),  # rebuild edge index
            RuntimeError("SyntaxError: syntax error near `IF'"),
            object(),  # create fulltext tag index without IF
            RuntimeError("SyntaxError: syntax error near `IF'"),
            object(),  # create fulltext edge index without IF
            object(),  # rebuild fulltext index
            [],        # fulltext query-ready probe
        ]
    )
    with (
        patch.object(storage, "_execute_in_space", execute_in_space),
        patch.object(storage, "_ensure_fulltext_ready", AsyncMock()),
    ):
        await storage._create_indexes_if_needed()

    sql_calls = [call.args[0] for call in execute_in_space.await_args_list]
    assert any(
        sql
        == f"CREATE FULLTEXT TAG INDEX {storage._fulltext_tag_index_name} ON entity(entity_id);"
        for sql in sql_calls
    )
    assert any(
        sql
        == f"CREATE FULLTEXT EDGE INDEX {storage._fulltext_edge_index_name} ON relation(relationship);"
        for sql in sql_calls
    )
    assert storage._fulltext_init_error is None


@pytest.mark.asyncio
async def test_create_indexes_records_service_not_found_for_fulltext():
    storage = build_storage(workspace="finance")
    execute_in_space = AsyncMock(
        side_effect=[
            object(),  # create tag index
            object(),  # create edge index
            object(),  # rebuild tag index
            object(),  # rebuild edge index
            RuntimeError("SyntaxError: syntax error near `IF'"),
            RuntimeError("Service not found!"),
        ]
    )
    with (
        patch.object(storage, "_execute_in_space", execute_in_space),
        patch.object(storage, "_ensure_fulltext_ready", AsyncMock()),
    ):
        await storage._create_indexes_if_needed()

    assert storage._fulltext_init_error is not None
    assert "Service not found" in storage._fulltext_init_error


@pytest.mark.asyncio
async def test_wait_for_fulltext_query_ready_retries_until_query_succeeds():
    storage = build_storage(workspace="finance")
    storage._schema_retry_times = 3
    storage._schema_retry_delay_ms = 0
    execute_in_space = AsyncMock(
        side_effect=[RuntimeError("Index not found"), RuntimeError("Index not found"), []]
    )
    with patch.object(storage, "_execute_in_space", execute_in_space):
        await storage._wait_for_fulltext_query_ready("nebula_entity_name_ft_demo")

    assert execute_in_space.await_count == 3


@pytest.mark.asyncio
async def test_wait_for_fulltext_query_ready_times_out():
    storage = build_storage(workspace="finance")
    storage._schema_retry_times = 2
    storage._schema_retry_delay_ms = 0
    execute_in_space = AsyncMock(side_effect=RuntimeError("Index not found"))
    with patch.object(storage, "_execute_in_space", execute_in_space):
        with pytest.raises(RuntimeError, match="query-ready"):
            await storage._wait_for_fulltext_query_ready("nebula_entity_name_ft_demo")


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
                    "name": "A",
                    "entity_type": "TypeX",
                    "description": "desc",
                    "keywords": "k1,k2",
                    "source_id": "src-1",
                    "file_path": "doc/a.md",
                    "created_at": 123,
                    "truncate": "FIFO 1/2",
                }
            ],
        ]
    )
    with patch.object(storage, "_execute_in_space", execute_in_space):
        await storage.upsert_node(
            "A",
            {
                "entity_id": "A",
                "name": "A",
                "entity_type": "TypeX",
                "description": "desc",
                "keywords": "k1,k2",
                "source_id": "src-1",
                "file_path": "doc/a.md",
                "created_at": 123,
                "truncate": "FIFO 1/2",
            },
        )
        node = await storage.get_node("A")

    assert node is not None
    assert node["entity_id"] == "A"
    assert node["name"] == "A"
    assert node["entity_type"] == "TypeX"
    assert node["description"] == "desc"
    assert node["keywords"] == "k1,k2"
    assert node["source_id"] == "src-1"
    assert node["file_path"] == "doc/a.md"
    assert node["created_at"] == "123"
    assert node["truncate"] == "FIFO 1/2"
    upsert_sql = execute_in_space.await_args_list[0].args[0]
    assert "file_path" in upsert_sql
    assert "created_at" in upsert_sql
    assert "truncate" in upsert_sql
    assert "INSERT VERTEX entity" in upsert_sql
    assert 'VALUES "A"' in upsert_sql
    get_sql = execute_in_space.await_args_list[1].args[0]
    assert "FETCH PROP ON entity" in get_sql
    assert '"A"' in get_sql


@pytest.mark.asyncio
async def test_nebula_upsert_node_coerces_numeric_created_at_string():
    storage = build_storage(workspace="finance")
    execute_in_space = AsyncMock(return_value=object())

    with patch.object(storage, "_execute_in_space", execute_in_space):
        await storage.upsert_node(
            "A",
            {
                "entity_id": "A",
                "name": "A",
                "entity_type": "TypeX",
                "description": "desc",
                "keywords": "k1,k2",
                "source_id": "src-1",
                "file_path": "doc/a.md",
                "created_at": "123",
                "truncate": "",
            },
        )

    upsert_sql = execute_in_space.await_args_list[0].args[0]
    assert "INSERT VERTEX entity" in upsert_sql
    assert ', 123, ""' in upsert_sql
    assert '"123"' not in upsert_sql


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
                    "keywords": "k1,k2",
                    "weight": 1.0,
                    "file_path": "doc/a.md",
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
                    "keywords": "k1,k2",
                    "weight": 1.0,
                    "file_path": "doc/a.md",
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
                "keywords": "k1,k2",
                "weight": 1.0,
                "file_path": "doc/a.md",
            },
        )
        forward = await storage.get_edge("A", "B")
        reverse = await storage.get_edge("B", "A")

    assert forward == reverse
    assert forward["keywords"] == "k1,k2"
    assert forward["file_path"] == "doc/a.md"
    upsert_sql = execute_in_space.await_args_list[0].args[0]
    assert "keywords" in upsert_sql
    assert "file_path" in upsert_sql
    assert 'VALUES "A"->"B"' in upsert_sql
    fetch_sql_1 = execute_in_space.await_args_list[1].args[0]
    fetch_sql_2 = execute_in_space.await_args_list[2].args[0]
    assert 'WHERE id(a) == "A" AND id(b) == "B"' in fetch_sql_1
    assert 'WHERE id(a) == "A" AND id(b) == "B"' in fetch_sql_2


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
                    "keywords": "k1,k2",
                    "weight": 1.0,
                    "file_path": "doc/a.md",
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
                "keywords": "k1,k2",
                "weight": 1.0,
                "file_path": "doc/a.md",
            },
        )
        edge = await storage.get_edge("A", "B")

    assert edge is not None
    assert edge["source"] == "A"
    assert edge["target"] == "B"
    assert edge["source_id"] == "chunk1<SEP>chunk2"
    assert edge["target_id"] == "meta-target"
    assert edge["keywords"] == "k1,k2"
    assert edge["file_path"] == "doc/a.md"
    upsert_sql = execute_in_space.await_args_list[0].args[0]
    assert 'VALUES "A"->"B":("chunk1<SEP>chunk2", "meta-target"' in upsert_sql
    assert "keywords" in upsert_sql
    assert "file_path" in upsert_sql
    fetch_sql = execute_in_space.await_args_list[1].args[0]
    assert "id(a) AS source" in fetch_sql
    assert "id(b) AS target" in fetch_sql


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
async def test_nebula_remove_nodes_deletes_each_vertex_with_edges():
    storage = build_storage(workspace="finance")
    execute_in_space = AsyncMock(return_value=object())
    with patch.object(storage, "_execute_in_space", execute_in_space):
        await storage.remove_nodes(["B", "A", "B"])

    sql_calls = [call.args[0] for call in execute_in_space.await_args_list]
    assert len(sql_calls) == 2
    assert any('DELETE VERTEX "A" WITH EDGE;' == sql for sql in sql_calls)
    assert any('DELETE VERTEX "B" WITH EDGE;' == sql for sql in sql_calls)


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
async def test_nebula_drop_drops_space_and_resets_state():
    storage = build_storage(workspace="finance")
    storage._initialized = True
    storage._connection_pool = object()
    execute = AsyncMock(return_value=object())

    async def close_pool():
        storage._connection_pool = None

    close_connection_pool = AsyncMock(side_effect=close_pool)
    with (
        patch.object(storage, "_execute", execute),
        patch.object(storage, "_close_connection_pool", close_connection_pool),
    ):
        result = await storage.drop()

    sql = execute.await_args_list[0].args[0]
    assert sql == f'DROP SPACE IF EXISTS `{storage._space_name}`;'
    assert close_connection_pool.await_count == 1
    assert storage._initialized is False
    assert storage._connection_pool is None
    assert result == {
        "status": "success",
        "message": f"workspace '{storage._space_name}' dropped",
    }


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
                "file_path": "doc/a.md",
                "created_at": 100,
                "truncate": "",
            },
            {
                "entity_id": "B",
                "name": "B",
                "entity_type": "TypeB",
                "description": "node-b",
                "keywords": "k2",
                "source_id": "s2",
                "file_path": "doc/b.md",
                "created_at": 200,
                "truncate": "KEEP 1/2",
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
            "file_path": "doc/a.md",
            "created_at": "100",
            "truncate": "",
        },
        "B": {
            "entity_id": "B",
            "name": "B",
            "entity_type": "TypeB",
            "description": "node-b",
            "keywords": "k2",
            "source_id": "s2",
            "file_path": "doc/b.md",
            "created_at": "200",
            "truncate": "KEEP 1/2",
        },
    }
    assert execute_in_space.await_count == 1
    sql = execute_in_space.await_args_list[0].args[0]
    _assert_bounded_nebula_query(
        sql,
        required_tokens=['"A"', '"B"'],
        forbidden_patterns=[
            "MATCH (v:entity) RETURN id(v) AS entity_id",
        ],
    )
    assert "v.entity.name AS name" in sql
    assert "v.entity.entity_type AS entity_type" in sql
    assert "v.entity.file_path AS file_path" in sql
    assert "v.entity.created_at AS created_at" in sql
    assert "v.entity.truncate AS truncate" in sql
    assert "id(v) AS entity_id" in sql


@pytest.mark.asyncio
async def test_nebula_get_nodes_batch_splits_large_id_filters():
    storage = build_storage(workspace="finance")
    node_ids = [f"node-{i:03d}" for i in range(600)]

    async def fake_execute(sql: str):
        normalized_sql = _normalize_sql_whitespace(sql)
        has_first = '"node-000"' in normalized_sql
        has_last = '"node-599"' in normalized_sql
        assert not (
            has_first and has_last
        ), "large node filter should be split across multiple Nebula queries"
        rows: list[dict[str, object]] = []
        if has_first:
            rows.append(
                {
                    "entity_id": "node-000",
                    "name": "Node 0",
                    "entity_type": "TypeA",
                    "description": "first node",
                    "keywords": "k0",
                    "source_id": "s0",
                    "file_path": "doc/0.md",
                    "created_at": 100,
                    "truncate": "",
                }
            )
        if has_last:
            rows.append(
                {
                    "entity_id": "node-599",
                    "name": "Node 599",
                    "entity_type": "TypeZ",
                    "description": "last node",
                    "keywords": "k599",
                    "source_id": "s599",
                    "file_path": "doc/599.md",
                    "created_at": 599,
                    "truncate": "",
                }
            )
        return rows

    execute_in_space = AsyncMock(side_effect=fake_execute)
    with patch.object(storage, "_execute_in_space", execute_in_space):
        nodes = await storage.get_nodes_batch(node_ids)

    assert nodes["node-000"]["name"] == "Node 0"
    assert nodes["node-599"]["name"] == "Node 599"
    assert execute_in_space.await_count > 1


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
    _assert_bounded_nebula_query(
        sql,
        required_tokens=['"A"', '"B"', '"C"', '"X"'],
        forbidden_patterns=[
            "MATCH (a:entity)-[e:relation]->(b:entity) RETURN id(a) AS source, id(b) AS target;",
        ],
    )
    assert "id(a) AS source" in sql
    assert "id(b) AS target" in sql


@pytest.mark.asyncio
async def test_nebula_node_degrees_batch_splits_large_endpoint_filters():
    storage = build_storage(workspace="finance")
    node_ids = [f"node-{i:03d}" for i in range(600)]

    async def fake_execute(sql: str):
        normalized_sql = _normalize_sql_whitespace(sql)
        has_first = '"node-000"' in normalized_sql
        has_last = '"node-599"' in normalized_sql
        assert not (
            has_first and has_last
        ), "large degree filter should be split across multiple Nebula queries"
        rows: list[dict[str, str]] = []
        if has_first:
            rows.append({"source": "node-000", "target": "neighbor-000"})
        if has_last:
            rows.append({"source": "neighbor-599", "target": "node-599"})
        return rows

    execute_in_space = AsyncMock(side_effect=fake_execute)
    with patch.object(storage, "_execute_in_space", execute_in_space):
        degrees = await storage.node_degrees_batch(node_ids)

    assert degrees["node-000"] == 1
    assert degrees["node-599"] == 1
    assert execute_in_space.await_count > 1


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
                "keywords": "k1,k2",
                "weight": 2.5,
                "file_path": "doc/a.md",
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
            "keywords": "k1,k2",
            "weight": 2.5,
            "file_path": "doc/a.md",
        },
        ("A", "B"): {
            "source": "A",
            "target": "B",
            "source_id": "A",
            "target_id": "B",
            "relationship": "rel-ab",
            "description": "A-B edge",
            "keywords": "k1,k2",
            "weight": 2.5,
            "file_path": "doc/a.md",
        },
    }
    assert execute_in_space.await_count == 1
    sql = execute_in_space.await_args_list[0].args[0]
    _assert_bounded_nebula_query(
        sql,
        required_tokens=['"A"', '"B"', '"C"'],
        forbidden_patterns=[
            "MATCH (a:entity)-[e:relation]->(b:entity) RETURN id(a) AS source, id(b) AS target, e.source_id AS source_id, e.target_id AS target_id, e.relationship AS relationship, e.description AS description, e.weight AS weight;",
        ],
    )
    assert "id(a) AS source" in sql
    assert "id(b) AS target" in sql
    assert "e.keywords AS keywords" in sql
    assert "e.file_path AS file_path" in sql


@pytest.mark.asyncio
async def test_nebula_get_edges_batch_splits_large_pair_filters():
    storage = build_storage(workspace="finance")
    pairs = [{"src": f"left-{i:03d}", "tgt": f"right-{i:03d}"} for i in range(600)]

    async def fake_execute(sql: str):
        normalized_sql = _normalize_sql_whitespace(sql)
        has_first = '"left-000"' in normalized_sql and '"right-000"' in normalized_sql
        has_last = '"left-599"' in normalized_sql and '"right-599"' in normalized_sql
        assert not (
            has_first and has_last
        ), "large edge pair filter should be split across multiple Nebula queries"
        rows: list[dict[str, object]] = []
        if has_first:
            rows.append(
                {
                    "source": "left-000",
                    "target": "right-000",
                    "source_id": "left-000",
                    "target_id": "right-000",
                    "relationship": "first-edge",
                    "description": "first chunk",
                    "keywords": "k0",
                    "weight": 1.0,
                    "file_path": "doc/0.md",
                }
            )
        if has_last:
            rows.append(
                {
                    "source": "left-599",
                    "target": "right-599",
                    "source_id": "left-599",
                    "target_id": "right-599",
                    "relationship": "last-edge",
                    "description": "last chunk",
                    "keywords": "k599",
                    "weight": 2.0,
                    "file_path": "doc/599.md",
                }
            )
        return rows

    execute_in_space = AsyncMock(side_effect=fake_execute)
    with patch.object(storage, "_execute_in_space", execute_in_space):
        edges = await storage.get_edges_batch(pairs)

    assert edges[("left-000", "right-000")]["relationship"] == "first-edge"
    assert edges[("left-599", "right-599")]["relationship"] == "last-edge"
    assert execute_in_space.await_count > 1


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
    _assert_bounded_nebula_query(
        sql,
        required_tokens=['"A"', '"B"', '"X"'],
        forbidden_patterns=[
            "MATCH (a:entity)-[e:relation]->(b:entity) RETURN id(a) AS source, id(b) AS target;",
        ],
    )
    assert "id(a) AS source" in sql
    assert "id(b) AS target" in sql


@pytest.mark.asyncio
async def test_nebula_get_nodes_edges_batch_splits_large_endpoint_filters():
    storage = build_storage(workspace="finance")
    node_ids = [f"node-{i:03d}" for i in range(600)]

    async def fake_execute(sql: str):
        normalized_sql = _normalize_sql_whitespace(sql)
        has_first = '"node-000"' in normalized_sql
        has_last = '"node-599"' in normalized_sql
        assert not (
            has_first and has_last
        ), "large endpoint filter should be split across multiple Nebula queries"
        rows: list[dict[str, str]] = []
        if has_first:
            rows.append({"source": "node-000", "target": "neighbor-000"})
        if has_last:
            rows.append({"source": "neighbor-599", "target": "node-599"})
        return rows

    execute_in_space = AsyncMock(side_effect=fake_execute)
    with patch.object(storage, "_execute_in_space", execute_in_space):
        nodes_edges = await storage.get_nodes_edges_batch(node_ids)

    assert nodes_edges["node-000"] == [("node-000", "neighbor-000")]
    assert nodes_edges["node-599"] == [("neighbor-599", "node-599")]
    assert execute_in_space.await_count > 1


@pytest.mark.asyncio
async def test_nebula_has_node_uses_lightweight_existence_probe():
    storage = build_storage(workspace="finance")
    execute_in_space = AsyncMock(return_value=[{"entity_id": "A"}])
    with (
        patch.object(storage, "_execute_in_space", execute_in_space),
        patch.object(
            storage,
            "get_node",
            AsyncMock(side_effect=AssertionError("has_node should not call get_node")),
        ),
    ):
        assert await storage.has_node("A") is True

    sql = execute_in_space.await_args_list[0].args[0]
    assert "LIMIT 1" in _normalize_sql_whitespace(sql)


@pytest.mark.asyncio
async def test_nebula_has_edge_uses_lightweight_existence_probe():
    storage = build_storage(workspace="finance")
    execute_in_space = AsyncMock(return_value=[{"source": "A", "target": "B"}])
    with (
        patch.object(storage, "_execute_in_space", execute_in_space),
        patch.object(
            storage,
            "get_edge",
            AsyncMock(side_effect=AssertionError("has_edge should not call get_edge")),
        ),
    ):
        assert await storage.has_edge("A", "B") is True

    sql = execute_in_space.await_args_list[0].args[0]
    assert "LIMIT 1" in _normalize_sql_whitespace(sql)


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
    assert "MATCH (v:entity)" in sql
    assert "id(v) AS entity_id" in sql


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
                "file_path": "doc/a.md",
                "created_at": 100,
                "truncate": "",
            },
            {
                "entity_id": "B",
                "name": "B",
                "entity_type": "TypeB",
                "description": "desc-b",
                "keywords": "k2",
                "source_id": "src-b",
                "file_path": "doc/b.md",
                "created_at": 200,
                "truncate": "KEEP 1/2",
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
            "file_path": "doc/a.md",
            "created_at": "100",
            "truncate": "",
            "id": "A",
        },
        {
            "entity_id": "B",
            "name": "B",
            "entity_type": "TypeB",
            "description": "desc-b",
            "keywords": "k2",
            "source_id": "src-b",
            "file_path": "doc/b.md",
            "created_at": "200",
            "truncate": "KEEP 1/2",
            "id": "B",
        },
    ]
    assert execute_in_space.await_count == 1
    sql = execute_in_space.await_args_list[0].args[0]
    assert "MATCH (v:entity)" in sql
    assert "v.entity.file_path AS file_path" in sql
    assert "v.entity.created_at AS created_at" in sql
    assert "v.entity.truncate AS truncate" in sql


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
                "keywords": "k1,k2",
                "weight": 3.0,
                "file_path": "doc/a.md",
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
            "keywords": "k1,k2",
            "weight": 3.0,
            "file_path": "doc/a.md",
            "source": "A",
            "target": "B",
        }
    ]
    assert execute_in_space.await_count == 1
    sql = execute_in_space.await_args_list[0].args[0]
    assert "MATCH (a:entity)-[e:relation]->(b:entity)" in sql
    assert "e.keywords AS keywords" in sql
    assert "e.file_path AS file_path" in sql


@pytest.mark.asyncio
async def test_nebula_get_popular_labels_orders_by_degree_desc():
    storage = build_storage(workspace="finance")
    execute_in_space = AsyncMock(
        return_value=[
            {"label": "B", "degree": 3},
            {"label": "A", "degree": 2},
            {"label": "C", "degree": 2},
        ]
    )
    with patch.object(storage, "_execute_in_space", execute_in_space):
        labels = await storage.get_popular_labels(limit=3)

    assert labels == ["B", "A", "C"]
    assert execute_in_space.await_count == 1
    sql = execute_in_space.await_args_list[0].args[0]
    assert "MATCH (a:entity)-[e:relation]->(b:entity)" in sql
    assert "UNWIND" in sql
    assert "count(*) AS degree" in sql
    assert "ORDER BY degree DESC, label ASC" in _normalize_sql_whitespace(sql)
    assert "LIMIT 3" in _normalize_sql_whitespace(sql)
    assert "RETURN id(a) AS source, id(b) AS target" not in _normalize_sql_whitespace(sql)


@pytest.mark.asyncio
async def test_nebula_get_knowledge_graph_wildcard_returns_truncated_graph():
    storage = build_storage(workspace="finance")
    node_payloads = {
        "A": {"entity_id": "A", "name": "A", "description": "node-a"},
        "B": {"entity_id": "B", "name": "B", "description": "node-b"},
    }
    adjacency = {
        "A": [("A", "B"), ("A", "C")],
        "B": [("A", "B")],
    }
    edge_payloads = {
        ("A", "B"): {"source": "A", "target": "B", "relationship": "ab"},
    }
    with (
        patch.object(storage, "get_popular_labels", AsyncMock(return_value=["A", "B", "C"])),
        patch.object(storage, "get_nodes_batch", AsyncMock(return_value=node_payloads)),
        patch.object(
            storage, "get_nodes_edges_batch", AsyncMock(return_value=adjacency)
        ),
        patch.object(storage, "get_edges_batch", AsyncMock(return_value=edge_payloads)),
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
async def test_nebula_get_knowledge_graph_wildcard_does_not_depend_on_get_all_lists():
    storage = build_storage(workspace="finance")
    with (
        patch.object(
            storage,
            "get_all_nodes",
            AsyncMock(side_effect=AssertionError("wildcard graph should not call get_all_nodes")),
        ),
        patch.object(
            storage,
            "get_all_edges",
            AsyncMock(side_effect=AssertionError("wildcard graph should not call get_all_edges")),
        ),
        patch.object(storage, "get_popular_labels", AsyncMock(return_value=["A", "B"])),
        patch.object(storage, "get_all_labels", AsyncMock(return_value=["A", "B"])),
        patch.object(
            storage,
            "get_nodes_batch",
            AsyncMock(
                return_value={
                    "A": {"entity_id": "A", "name": "A"},
                    "B": {"entity_id": "B", "name": "B"},
                }
            ),
        ),
        patch.object(
            storage,
            "get_nodes_edges_batch",
            AsyncMock(return_value={"A": [("A", "B")], "B": [("A", "B")]}),
        ),
        patch.object(
            storage,
            "get_edges_batch",
            AsyncMock(return_value={("A", "B"): {"source": "A", "target": "B", "relationship": "ab"}}),
        ),
    ):
        graph = await storage.get_knowledge_graph("*", max_depth=2, max_nodes=2)

    assert isinstance(graph, KnowledgeGraph)
    assert {node.id for node in graph.nodes} == {"A", "B"}
    assert len(graph.edges) == 1


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
async def test_search_labels_uses_fulltext_path_when_available_and_merges_name_matches():
    storage = build_storage(workspace="finance")
    fulltext_mock = AsyncMock(return_value=["learn", "learning"])
    name_matches_mock = AsyncMock(return_value=[])
    nodes_mock = AsyncMock(
        return_value={
            "learn": {"entity_id": "learn", "name": "learn"},
            "learning": {"entity_id": "learning", "name": "learning"},
        }
    )
    with (
        patch.object(storage, "_search_labels_fulltext", fulltext_mock),
        patch.object(storage, "_search_labels_name_matches", name_matches_mock, create=True),
        patch.object(storage, "get_nodes_batch", nodes_mock),
    ):
        labels = await storage.search_labels("learn", limit=10)

    assert labels == ["learn", "learning"]
    fulltext_mock.assert_awaited_once_with("learn", limit=10)
    name_matches_mock.assert_awaited_once_with("learn", limit=10)
    nodes_mock.assert_awaited_once()


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


@pytest.mark.asyncio
async def test_search_labels_name_matches_return_entity_id_even_without_entity_id_hit():
    storage = build_storage(workspace="finance")
    fulltext_mock = AsyncMock(return_value=[])
    name_matches_mock = AsyncMock(return_value=[{"entity_id": "KB-001", "name": "learn-node"}])
    nodes_mock = AsyncMock(return_value={"KB-001": {"entity_id": "KB-001", "name": "learn-node"}})
    with (
        patch.object(storage, "_search_labels_fulltext", fulltext_mock),
        patch.object(storage, "_search_labels_name_matches", name_matches_mock, create=True),
        patch.object(storage, "get_nodes_batch", nodes_mock),
    ):
        labels = await storage.search_labels("learn", limit=10)

    assert labels == ["KB-001"]
    fulltext_mock.assert_awaited_once_with("learn", limit=10)
    name_matches_mock.assert_awaited_once_with("learn", limit=10)
    nodes_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_search_labels_uses_fallback_when_fulltext_init_error_exists():
    storage = build_storage(workspace="finance")
    storage._fulltext_init_error = "fulltext unavailable"
    fulltext_mock = AsyncMock(return_value=["SHOULD_NOT_BE_USED"])
    fallback_mock = AsyncMock(return_value=["Machine Learning"])
    with (
        patch.object(storage, "_search_labels_fulltext", fulltext_mock),
        patch.object(storage, "_search_labels_contains", fallback_mock),
    ):
        labels = await storage.search_labels("learn", limit=10)

    assert labels == ["Machine Learning"]
    fulltext_mock.assert_not_awaited()
    fallback_mock.assert_awaited_once_with("learn", limit=10)


@pytest.mark.asyncio
async def test_search_labels_entity_id_hits_outrank_name_only_hits():
    storage = build_storage(workspace="finance")
    fulltext_mock = AsyncMock(return_value=["learning"])
    name_matches_mock = AsyncMock(return_value=[{"entity_id": "KB-001", "name": "learn"}])
    nodes_mock = AsyncMock(
        return_value={
            "learning": {"entity_id": "learning", "name": "zzz"},
            "KB-001": {"entity_id": "KB-001", "name": "learn"},
        }
    )
    with (
        patch.object(storage, "_search_labels_fulltext", fulltext_mock),
        patch.object(storage, "_search_labels_name_matches", name_matches_mock, create=True),
        patch.object(storage, "get_nodes_batch", nodes_mock),
    ):
        labels = await storage.search_labels("learn", limit=10)

    assert labels == ["learning", "KB-001"]


@pytest.mark.asyncio
async def test_search_labels_deduplicates_candidates_across_fulltext_and_name_matches():
    storage = build_storage(workspace="finance")
    fulltext_mock = AsyncMock(return_value=["learn"])
    name_matches_mock = AsyncMock(
        return_value=[
            {"entity_id": "learn", "name": "learn-node"},
            {"entity_id": "KB-001", "name": "learn"},
        ]
    )
    nodes_mock = AsyncMock(
        return_value={
            "learn": {"entity_id": "learn", "name": "learn"},
            "KB-001": {"entity_id": "KB-001", "name": "learn"},
        }
    )
    with (
        patch.object(storage, "_search_labels_fulltext", fulltext_mock),
        patch.object(storage, "_search_labels_name_matches", name_matches_mock, create=True),
        patch.object(storage, "get_nodes_batch", nodes_mock),
    ):
        labels = await storage.search_labels("learn", limit=10)

    assert labels == ["learn", "KB-001"]


@pytest.mark.asyncio
async def test_nebula_get_knowledge_graph_marks_truncated_when_max_depth_reached():
    storage = build_storage(workspace="finance")
    with (
        patch.object(
            storage,
            "get_nodes_batch",
            AsyncMock(return_value={"A": {"entity_id": "A", "name": "A"}}),
        ),
        patch.object(
            storage,
            "get_nodes_edges_batch",
            AsyncMock(return_value={"A": [("A", "B")]}),
        ),
    ):
        graph = await storage.get_knowledge_graph("A", max_depth=0, max_nodes=10)

    assert isinstance(graph, KnowledgeGraph)
    assert graph.is_truncated is True
    assert [node.id for node in graph.nodes] == ["A"]
