"""Regression tests for one-time AGE graph creation (issue #1866).

`configure_age` used to issue `select create_graph(...)` on every pooled
connection checkout, i.e. before every single AGE read and write.  PostgreSQL
logs the resulting error *before* the client can swallow it, so a busy server
filled the database log with `graph "..." already exists` ERROR/STATEMENT
pairs.

The contract pinned here:

- AGE loading and `SET search_path` run on every checkout (RESET ALL clears it).
- The graph is looked up in `ag_catalog.ag_graph`, and `create_graph()` only
  runs when it is genuinely absent.
- Once the graph is known to exist, no SQL is issued for it at all.
- A creation race (3F000 / 23505) is tolerated.
- A transient failure must NOT mark the graph as ensured.
"""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest

from lightrag.kg.postgres_impl import PGGraphStorage, PostgreSQLDB


pytestmark = pytest.mark.offline

AGE_SESSION_SQL = (
    "SET search_path = ag_catalog, \"$user\", public; SELECT 'null'::ag_catalog.agtype"
)
GRAPH_LOOKUP_SQL = "SELECT 1 FROM ag_catalog.ag_graph WHERE name = left($1, 63)::name"


def _make_db() -> PostgreSQLDB:
    """Build a PostgreSQLDB without touching __init__/connection setup."""
    db = PostgreSQLDB.__new__(PostgreSQLDB)
    db._ensured_age_graphs = set()
    db._age_graph_ensure_lock = asyncio.Lock()
    return db


class FakeConnection:
    """Minimal asyncpg.Connection stand-in that records issued statements."""

    def __init__(
        self, graph_exists: bool = False, create_error: Exception | None = None
    ):
        self.graph_exists = graph_exists
        self.create_error = create_error
        self.executed: list[str] = []
        self.fetched: list[tuple[str, tuple]] = []

    async def execute(self, sql: str, *args):
        self.executed.append(sql)
        if "create_graph" in sql and self.create_error is not None:
            raise self.create_error
        return "OK"

    async def fetchval(self, sql: str, *args):
        self.fetched.append((sql, args))
        return 1 if self.graph_exists else None

    @property
    def create_graph_statements(self) -> list[str]:
        return [sql for sql in self.executed if "create_graph" in sql]


@pytest.mark.asyncio
async def test_existing_graph_is_not_recreated():
    """The defect itself: an existing graph must not trigger create_graph()."""
    db = _make_db()
    conn = FakeConnection(graph_exists=True)

    await db.configure_age(conn, "space1_chunk_entity_relation")

    assert conn.create_graph_statements == []
    assert conn.fetched == [
        (GRAPH_LOOKUP_SQL, ("space1_chunk_entity_relation",)),
    ]
    assert "space1_chunk_entity_relation" in db._ensured_age_graphs


@pytest.mark.asyncio
async def test_age_is_loaded_on_every_checkout_with_an_ensured_graph():
    """An ensured graph does not imply AGE is loaded in another session."""
    db = _make_db()
    db._ensured_age_graphs.add("g")
    conn = FakeConnection(graph_exists=True)
    replacement = FakeConnection(graph_exists=True)

    await db.configure_age(conn, "g")
    await db.configure_age(conn, "g")
    await db.configure_age(replacement, "g")

    assert conn.executed == [AGE_SESSION_SQL] * 2
    assert replacement.executed == [AGE_SESSION_SQL]
    assert conn.fetched == replacement.fetched == []


@pytest.mark.asyncio
async def test_age_load_failure_does_not_ensure_graph():
    db = _make_db()
    conn = AsyncMock()
    conn.execute.side_effect = asyncpg.exceptions.UndefinedObjectError(
        'type "ag_catalog.agtype" does not exist'
    )

    with pytest.raises(asyncpg.exceptions.UndefinedObjectError):
        await db.configure_age(conn, "g")

    conn.fetchval.assert_not_awaited()
    assert db._ensured_age_graphs == set()

    retry_conn = FakeConnection(graph_exists=True)
    await db.configure_age(retry_conn, "g")
    assert retry_conn.executed == [AGE_SESSION_SQL]
    assert db._ensured_age_graphs == {"g"}


@pytest.mark.asyncio
async def test_missing_graph_is_created_once():
    db = _make_db()
    conn = FakeConnection(graph_exists=False)

    await db.configure_age(conn, "g")

    assert conn.create_graph_statements == ["select create_graph('g')"]
    assert db._ensured_age_graphs == {"g"}


@pytest.mark.asyncio
async def test_ensured_graph_issues_no_further_sql():
    """Steady state must not even probe ag_graph again."""
    db = _make_db()
    conn = FakeConnection(graph_exists=False)

    await db.configure_age(conn, "g")
    lookups_after_first = len(conn.fetched)

    for _ in range(20):
        await db.configure_age(conn, "g")

    assert len(conn.fetched) == lookups_after_first
    assert conn.create_graph_statements == ["select create_graph('g')"]


@pytest.mark.asyncio
async def test_graphs_are_tracked_independently():
    db = _make_db()
    conn = FakeConnection(graph_exists=False)

    await db.configure_age(conn, "graph_a")
    await db.configure_age(conn, "graph_b")
    await db.configure_age(conn, "graph_a")

    assert conn.create_graph_statements == [
        "select create_graph('graph_a')",
        "select create_graph('graph_b')",
    ]
    assert db._ensured_age_graphs == {"graph_a", "graph_b"}


@pytest.mark.parametrize(
    "race_error",
    [
        # AGE reports "graph already exists" with ERRCODE_UNDEFINED_SCHEMA (3F000).
        asyncpg.exceptions.InvalidSchemaNameError('graph "g" already exists'),
        # Loser of the ag_graph name index race (23505).
        asyncpg.exceptions.UniqueViolationError("duplicate key value"),
    ],
    ids=["invalid_schema_name_3F000", "unique_violation_23505"],
)
@pytest.mark.asyncio
async def test_creation_race_is_tolerated(race_error):
    """AGE takes no lock before its own existence check, so races happen."""
    db = _make_db()
    conn = FakeConnection(graph_exists=False, create_error=race_error)

    await db.configure_age(conn, "g")

    # Race lost, but the graph exists now — that is all the method promises.
    assert db._ensured_age_graphs == {"g"}


@pytest.mark.asyncio
async def test_transient_creation_failure_does_not_mark_graph_ensured():
    """Caching an unconfirmed graph would break the process permanently."""
    db = _make_db()
    boom = asyncpg.exceptions.ConnectionDoesNotExistError("connection lost")
    failing_conn = FakeConnection(graph_exists=False, create_error=boom)

    with pytest.raises(asyncpg.exceptions.ConnectionDoesNotExistError):
        await db.configure_age(failing_conn, "g")

    assert db._ensured_age_graphs == set()

    # The next attempt must retry creation rather than assume it happened.
    retry_conn = FakeConnection(graph_exists=False)
    await db.configure_age(retry_conn, "g")
    assert retry_conn.create_graph_statements == ["select create_graph('g')"]


@pytest.mark.asyncio
async def test_concurrent_first_use_creates_graph_once():
    """Startup fan-out must not turn into a create_graph stampede."""
    db = _make_db()
    conn = FakeConnection(graph_exists=False)

    await asyncio.gather(*(db.configure_age(conn, "g") for _ in range(10)))

    assert conn.create_graph_statements == ["select create_graph('g')"]
    assert len(conn.fetched) == 1


@pytest.mark.asyncio
async def test_run_with_retry_configures_age_before_the_operation():
    """The graph must be ensured before the AGE statement runs, not after."""
    db = _make_db()
    order: list[str] = []

    async def fake_configure_age(connection, graph_name):
        order.append(f"configure_age:{graph_name}")

    db.configure_age = AsyncMock(side_effect=fake_configure_age)
    db.pool = _FakePool()
    db._ensure_pool = AsyncMock()
    db.connection_retry_attempts = 1
    db.connection_retry_backoff = 0
    db.connection_retry_backoff_max = 0
    db._transient_exceptions = ()

    async def operation(connection):
        order.append("operation")
        return "result"

    result = await db._run_with_retry(
        operation, with_age=True, graph_name="space1_chunk_entity_relation"
    )

    assert result == "result"
    assert order == ["configure_age:space1_chunk_entity_relation", "operation"]


@pytest.mark.asyncio
async def test_run_with_retry_skips_age_setup_when_not_requested():
    db = _make_db()
    db.configure_age = AsyncMock()
    db.pool = _FakePool()
    db._ensure_pool = AsyncMock()
    db.connection_retry_attempts = 1
    db.connection_retry_backoff = 0
    db.connection_retry_backoff_max = 0
    db._transient_exceptions = ()

    async def operation(connection):
        return "result"

    assert await db._run_with_retry(operation) == "result"
    db.configure_age.assert_not_called()


class _FakeAcquire:
    def __init__(self, connection):
        self._connection = connection

    async def __aenter__(self):
        return self._connection

    async def __aexit__(self, *exc_info):
        return False


class _FakePool:
    def __init__(self):
        self.connection = FakeConnection(graph_exists=True)

    def acquire(self):
        return _FakeAcquire(self.connection)


# ---------------------------------------------------------------------------
# Release / reacquire through the real pool reset callback.
#
# One physical connection is checked out with AGE, released (which runs the
# reset callback initdb() registers), then checked out without AGE and with
# AGE again.  The AGE search_path must not leak into the non-AGE checkout, and
# the AGE checkout after it must restore it.
# ---------------------------------------------------------------------------

DEFAULT_SEARCH_PATH = '"$user", public'


class _SessionConnection:
    """Tracks the session state a pool release does and does not reset."""

    def __init__(self):
        self.search_path = DEFAULT_SEARCH_PATH
        self.age_loaded = False
        self.executed: list[str] = []

    def get_reset_query(self) -> str:
        return "RESET ALL"

    async def execute(self, sql: str, *args):
        self.executed.append(sql)
        for statement in (s.strip() for s in sql.split(";")):
            if statement.startswith("SET search_path = "):
                self.search_path = statement.removeprefix("SET search_path = ")
            elif statement == "RESET ALL":
                # GUCs revert; loaded libraries stay loaded.
                self.search_path = DEFAULT_SEARCH_PATH
            elif statement == "SELECT 'null'::ag_catalog.agtype":
                self.age_loaded = True
        return "OK"

    async def fetchval(self, sql: str, *args):
        return 1


class _ReleaseResetAcquire:
    def __init__(self, connection, reset):
        self._connection = connection
        self._reset = reset

    async def __aenter__(self):
        return self._connection

    async def __aexit__(self, *exc_info):
        await self._reset(self._connection)
        return False


class _ReleaseResetPool:
    """Hands out one connection and runs the pool reset callback on release."""

    def __init__(self, connection, reset):
        self._connection = connection
        self._reset = reset

    def acquire(self):
        return _ReleaseResetAcquire(self._connection, self._reset)


async def _capture_pool_reset_callback(db: PostgreSQLDB):
    for name, value in {
        "workspace": "space1",
        "user": "u",
        "password": "p",
        "database": "d",
        "host": "h",
        "port": 5432,
        "max": 2,
        "statement_cache_size": None,
        "ssl_mode": None,
        "server_settings": None,
        "enable_vector": False,
        "vector_index_type": None,
        "connection_retry_attempts": 1,
        "connection_retry_backoff": 0,
        "connection_retry_backoff_max": 0,
    }.items():
        setattr(db, name, value)
    db._transient_exceptions = (ConnectionError,)
    db._create_ssl_context = lambda: None
    captured: dict = {}

    async def fake_create_pool(**kwargs):
        captured.update(kwargs)
        return MagicMock()

    with patch(
        "lightrag.kg.postgres_impl.asyncpg.create_pool",
        side_effect=fake_create_pool,
    ):
        await db.initdb()
    return captured["reset"]


@pytest.mark.asyncio
async def test_age_search_path_does_not_leak_across_release_and_reacquire():
    db = _make_db()
    reset = await _capture_pool_reset_callback(db)
    conn = _SessionConnection()
    db.pool = _ReleaseResetPool(conn, reset)
    db._ensure_pool = AsyncMock()
    seen: list[tuple[str, bool]] = []

    async def operation(connection):
        seen.append((connection.search_path, connection.age_loaded))

    await db._run_with_retry(operation, with_age=True, graph_name="g")
    await db._run_with_retry(operation)
    await db._run_with_retry(operation, with_age=True, graph_name="g")

    age_path = 'ag_catalog, "$user", public'
    assert seen == [
        (age_path, True),
        (DEFAULT_SEARCH_PATH, True),
        (age_path, True),
    ]
    assert conn.executed.count(AGE_SESSION_SQL) == 2
    assert conn.search_path == DEFAULT_SEARCH_PATH


# ---------------------------------------------------------------------------
# PGGraphStorage.initialize(): the DDL list must be idempotent too.
#
# Before the fix, every startup replayed create_graph, both create_*label calls
# and 11 plain CREATE INDEX statements, so a server that had already been
# initialised logged 14 ERROR lines on each boot (issue #1866).
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _null_lock():
    yield


def _make_graph_storage(existing_labels: list[str]) -> tuple[PGGraphStorage, AsyncMock]:
    storage = PGGraphStorage.__new__(PGGraphStorage)
    storage.workspace = "space1"
    storage.namespace = "chunk_entity_relation"
    storage.global_config = {"vector_storage": "PGVectorStorage"}

    db = AsyncMock()
    db.workspace = None
    db.query = AsyncMock(return_value=[{"name": name} for name in existing_labels])
    db.execute = AsyncMock()
    db._run_with_retry = AsyncMock()
    storage.db = db
    return storage, db


async def _run_initialize(monkeypatch, existing_labels: list[str]) -> list[str]:
    """Run initialize() and return the DDL statements it issued."""
    monkeypatch.setattr(
        "lightrag.kg.postgres_impl.get_data_init_lock", lambda: _null_lock()
    )
    storage, db = _make_graph_storage(existing_labels)
    await storage.initialize()
    return [call.args[0] for call in db.execute.await_args_list]


@pytest.mark.asyncio
async def test_initialize_skips_labels_that_already_exist(monkeypatch):
    statements = await _run_initialize(monkeypatch, ["base", "DIRECTED"])

    assert not [s for s in statements if "create_vlabel" in s]
    assert not [s for s in statements if "create_elabel" in s]


@pytest.mark.asyncio
async def test_initialize_creates_only_the_missing_label(monkeypatch):
    statements = await _run_initialize(monkeypatch, ["base"])

    assert not [s for s in statements if "create_vlabel" in s]
    assert [s for s in statements if "create_elabel" in s] == [
        "SELECT create_elabel('space1_chunk_entity_relation', 'DIRECTED');"
    ]


@pytest.mark.asyncio
async def test_initialize_creates_both_labels_on_a_fresh_graph(monkeypatch):
    statements = await _run_initialize(monkeypatch, [])

    assert statements[0] == (
        "SELECT create_vlabel('space1_chunk_entity_relation', 'base');"
    )
    assert statements[1] == (
        "SELECT create_elabel('space1_chunk_entity_relation', 'DIRECTED');"
    )


@pytest.mark.asyncio
async def test_initialize_never_issues_create_graph(monkeypatch):
    """_ensure_age_graph owns graph creation; initialize must not duplicate it."""
    statements = await _run_initialize(monkeypatch, [])

    assert not [s for s in statements if "create_graph" in s]


@pytest.mark.asyncio
async def test_initialize_index_statements_are_idempotent(monkeypatch):
    statements = await _run_initialize(monkeypatch, ["base", "DIRECTED"])

    index_statements = [s for s in statements if "CREATE INDEX" in s]
    assert index_statements, "expected the graph indexes to still be created"
    for statement in index_statements:
        assert "CONCURRENTLY IF NOT EXISTS" in statement, statement


@pytest.mark.asyncio
async def test_initialize_label_lookup_is_scoped_to_this_graph(monkeypatch):
    """Labels are per-graph, so the lookup must filter on the graph name."""
    monkeypatch.setattr(
        "lightrag.kg.postgres_impl.get_data_init_lock", lambda: _null_lock()
    )
    storage, db = _make_graph_storage(["base", "DIRECTED"])

    await storage.initialize()

    db.query.assert_awaited_once()
    sql, params = db.query.await_args.args[0], db.query.await_args.args[1]
    assert "ag_catalog.ag_label" in sql
    assert "ag_catalog.ag_graph" in sql
    # $1::name so PostgreSQL truncates the parameter the same way AGE did when
    # it stored the graph name; name::text = $1 would never match a graph name
    # longer than 63 bytes.
    assert "g.name = left($1, 63)::name" in sql
    assert "name::text = $1" not in sql
    assert params == ["space1_chunk_entity_relation"]
    assert db.query.await_args.kwargs["with_age"] is True
    assert db.query.await_args.kwargs["graph_name"] == "space1_chunk_entity_relation"


# ---------------------------------------------------------------------------
# Long graph names.  PostgreSQL's `name` type truncates at 63 bytes and AGE
# stores graph names as `name`, so both catalog lookups must cast the
# *parameter* to `name` rather than the stored value to `text`.  Comparing an
# untruncated text parameter never matches, which would silently reinstate the
# per-process create_graph and per-startup create_*label errors this PR removes.
# ---------------------------------------------------------------------------

# A workspace long enough that "<workspace>_chunk_entity_relation" exceeds 63 bytes.
LONG_GRAPH_NAME = (
    "verylongworkspacename_that_exceeds_the_limit_abcdefghij_chunk_entity_relation"
)


def test_long_graph_name_is_actually_over_the_identifier_limit():
    """Guard the premise of the tests below."""
    assert len(LONG_GRAPH_NAME.encode("utf-8")) > 63


@pytest.mark.asyncio
async def test_graph_lookup_casts_the_parameter_not_the_column():
    db = _make_db()
    conn = FakeConnection(graph_exists=True)

    await db.configure_age(conn, LONG_GRAPH_NAME)

    ((sql, args),) = conn.fetched
    # left($1, 63), not $1::name: PostgreSQL raises 42622 "identifier too long"
    # when a *bind parameter* is cast to name, while the literal AGE was given
    # is clipped silently. The comparison has to clip the same way.
    assert "name = left($1, 63)::name" in sql
    assert "$1::name" not in sql
    assert "name::text" not in sql
    assert args == (LONG_GRAPH_NAME,)


@pytest.mark.asyncio
async def test_existing_long_named_graph_is_not_recreated():
    """The truncation bug's user-visible symptom: one create_graph per process."""
    db = _make_db()
    conn = FakeConnection(graph_exists=True)

    await db.configure_age(conn, LONG_GRAPH_NAME)

    assert conn.create_graph_statements == []


@pytest.mark.asyncio
async def test_initialize_label_lookup_casts_the_parameter(monkeypatch):
    monkeypatch.setattr(
        "lightrag.kg.postgres_impl.get_data_init_lock", lambda: _null_lock()
    )
    storage, db = _make_graph_storage(["base", "DIRECTED"])
    storage.workspace = "verylongworkspacename_that_exceeds_the_limit_abcdefghij"

    await storage.initialize()

    sql = db.query.await_args.args[0]
    assert "g.name = left($1, 63)::name" in sql
    assert db.query.await_args.args[1] == [LONG_GRAPH_NAME]
    # And with both labels reported present, neither is recreated.
    statements = [call.args[0] for call in db.execute.await_args_list]
    assert not [s for s in statements if "create_vlabel" in s or "create_elabel" in s]


def test_name_limit_constant_matches_postgres_namedatalen():
    """NAMEDATALEN - 1. Both catalog lookups clip to this."""
    from lightrag.kg.postgres_impl import _PG_NAME_MAX_BYTES

    assert _PG_NAME_MAX_BYTES == 63


def test_graph_names_are_ascii_so_bytes_equal_characters():
    """left() clips by character; that is only equivalent for single-byte names.

    _get_workspace_graph_name() guarantees it by reducing both the workspace and
    the namespace to [A-Za-z0-9_].
    """
    storage = PGGraphStorage.__new__(PGGraphStorage)
    storage.workspace = "工作区 with spaces/and-punct"
    storage.namespace = "chunk_entity_relation"

    name = storage._get_workspace_graph_name()

    assert name.isascii(), name
    assert len(name) == len(name.encode("utf-8"))
