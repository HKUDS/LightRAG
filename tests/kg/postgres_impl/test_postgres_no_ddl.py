"""No-DDL mode separates runtime access from administrative provisioning."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from lightrag.kg.postgres_impl import ClientManager, PostgreSQLDB

pytestmark = pytest.mark.offline


def test_auto_migrate_configuration_precedence(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("POSTGRES_AUTO_MIGRATE", raising=False)
    assert ClientManager.get_config()["auto_migrate"] is True
    (tmp_path / "config.ini").write_text("[postgres]\nauto_migrate=no\n")
    assert ClientManager.get_config()["auto_migrate"] is False
    monkeypatch.setenv("POSTGRES_AUTO_MIGRATE", " ON ")
    assert ClientManager.get_config()["auto_migrate"] is True
    monkeypatch.setenv("POSTGRES_AUTO_MIGRATE", "tru")
    with pytest.raises(ValueError, match="POSTGRES_AUTO_MIGRATE"):
        ClientManager.get_config()


@pytest.mark.asyncio
async def test_extension_validation_keeps_compatibility_checks():
    db = object.__new__(PostgreSQLDB)
    db.auto_migrate = False
    db.vector_index_type = "HNSW_HALFVEC"
    db._probe_iterative_scan_support = AsyncMock()
    connection = AsyncMock()
    connection.fetchrow.return_value = {"extversion": "0.6.2"}
    with pytest.raises(RuntimeError, match="pgvector >= 0.7.0"):
        await db.configure_vector_extension(connection)
    connection.execute.assert_not_awaited()
    connection.fetchrow.return_value = {"extversion": "0.8.7"}
    await db.configure_vector_extension(connection)
    connection.execute.assert_not_awaited()
    db._probe_iterative_scan_support.assert_awaited_once_with(connection)


@pytest.mark.parametrize("version", [None, ""])
@pytest.mark.asyncio
async def test_missing_vector_extension_is_actionable(version):
    db = object.__new__(PostgreSQLDB)
    db.auto_migrate = False
    db.vector_index_type = "HNSW"
    connection = AsyncMock()
    connection.fetchrow.return_value = (
        None if version is None else {"extversion": version}
    )
    with pytest.raises(RuntimeError, match="administrator install pgvector"):
        await db.configure_vector_extension(connection)
    connection.execute.assert_not_awaited()


@pytest.mark.parametrize(
    "failure", [RuntimeError("schema rejected"), asyncio.CancelledError()]
)
@pytest.mark.asyncio
async def test_failed_shared_client_is_closed_and_not_published(monkeypatch, failure):
    db = MagicMock()
    db.initdb = AsyncMock()
    db.check_tables = AsyncMock(side_effect=failure)
    pool = db.pool
    pool.close = AsyncMock()
    db.pool_close_timeout = 1
    initial = {"db": None, "ref_count": 0, "vector_signature": None}
    monkeypatch.setattr(ClientManager, "_instances", dict(initial))
    monkeypatch.setattr("lightrag.kg.postgres_impl.PostgreSQLDB", lambda config: db)
    with pytest.raises(type(failure)):
        await ClientManager.get_client()
    pool.close.assert_awaited_once()
    assert db.pool is None
    assert ClientManager._instances == initial


@pytest.mark.asyncio
async def test_shared_client_rejects_migration_policy_change(monkeypatch):
    monkeypatch.setattr(
        ClientManager,
        "_instances",
        {"db": None, "ref_count": 0, "vector_signature": None},
    )
    db = MagicMock()
    db.initdb = AsyncMock()
    db.check_tables = AsyncMock()
    monkeypatch.setattr("lightrag.kg.postgres_impl.PostgreSQLDB", lambda config: db)
    monkeypatch.setenv("POSTGRES_AUTO_MIGRATE", "true")
    assert await ClientManager.get_client() is db
    monkeypatch.setenv("POSTGRES_AUTO_MIGRATE", "false")
    with pytest.raises(RuntimeError, match="process-wide"):
        await ClientManager.get_client()
    assert ClientManager._instances["ref_count"] == 1
    db.initdb.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_ddl_common_validation_skips_all_migrations(monkeypatch):
    from lightrag.kg import postgres_schema

    validate = AsyncMock()
    monkeypatch.setattr(postgres_schema, "validate_table", validate)
    db = object.__new__(PostgreSQLDB)
    db.auto_migrate = False
    db.query = AsyncMock(return_value=None)
    db.execute = AsyncMock()
    migrations = {}
    for name in dir(PostgreSQLDB):
        if name.startswith(("_migrate_", "_create_")):
            migrations[name] = AsyncMock()
            monkeypatch.setattr(db, name, migrations[name])
    await db.check_tables()
    assert [c.args[1] for c in validate.await_args_list] == list(
        postgres_schema.COMMON_TABLES
    )
    assert [n for n, m in migrations.items() if m.await_count] == []
    db.execute.assert_not_awaited()
    # A read failure cannot be misreported as a compatible cache.
    db.query.side_effect = RuntimeError("catalog unavailable")
    with pytest.raises(RuntimeError, match="catalog unavailable"):
        await db.check_tables()


@pytest.mark.asyncio
async def test_no_ddl_reconnect_registers_codec_without_extension_ddl(monkeypatch):
    from tests.kg.postgres_impl.test_postgres_pool_reconnect import make_db
    from pgvector.asyncpg import register_vector

    db = make_db()
    db.auto_migrate = False
    db.enable_vector = True
    db.vector_index_type = "HNSW"
    bootstrap = AsyncMock()
    bootstrap.fetchrow.return_value = {"extversion": "0.8.7"}
    monkeypatch.setattr(
        "lightrag.kg.postgres_impl.asyncpg.connect", AsyncMock(return_value=bootstrap)
    )
    register = AsyncMock(spec=register_vector)
    monkeypatch.setattr("pgvector.asyncpg.register_vector", register)
    physical = AsyncMock()
    physical.get_reset_query = lambda: "RESET ALL"
    pools = []

    async def create_pool(**kwargs):
        await kwargs["init"](physical)
        await kwargs["reset"](physical)
        pool = MagicMock(close=AsyncMock())
        pools.append(pool)
        return pool

    monkeypatch.setattr("lightrag.kg.postgres_impl.asyncpg.create_pool", create_pool)
    await db.initdb()
    await db._reset_pool()
    await db._ensure_pool()
    assert len(pools) == 2
    assert register.await_count == 2
    assert db.iterative_scan_supported is True
    bootstrap.execute.assert_not_awaited()
    assert [c.args[0] for c in physical.execute.await_args_list] == [
        "RESET ALL",
        "RESET ALL",
    ]
    assert bootstrap.close.await_count == 2


@pytest.mark.asyncio
async def test_age_no_ddl_is_rejected_before_opening_client(monkeypatch):
    from lightrag.kg.postgres_impl import PGGraphStorage

    storage = object.__new__(PGGraphStorage)
    storage.db = None
    storage.global_config = {}
    monkeypatch.setenv("POSTGRES_AUTO_MIGRATE", "false")
    get_client = AsyncMock()
    monkeypatch.setattr(ClientManager, "get_client", get_client)
    with pytest.raises(ValueError, match="PGTableGraphStorage"):
        await storage.initialize()
    get_client.assert_not_awaited()


def _vector_snapshot():
    return (
        731,
        {"content_vector": {"typname": "vector", "vector_type": True, "atttypmod": 8}},
        [
            {
                "usable": True,
                "amname": "hnsw",
                "predicate": None,
                "columns": ["content_vector"],
                "opclasses": ["vector_cosine_ops"],
            }
        ],
    )


@pytest.mark.parametrize(
    ("change", "value"),
    [
        ("usable", False),
        ("amname", "btree"),
        ("opclasses", ["vector_l2_ops"]),
        ("predicate", "workspace = 'other'"),
    ],
)
@pytest.mark.asyncio
async def test_vector_index_definition_is_validated(monkeypatch, change, value):
    from lightrag.kg import postgres_schema
    from lightrag.kg.postgres_impl import PGVectorStorage

    snapshot = _vector_snapshot()
    snapshot[2][0][change] = value
    monkeypatch.setattr(
        postgres_schema, "validate_table", AsyncMock(return_value=snapshot)
    )
    db = MagicMock(auto_migrate=False, vector_index_type="HNSW")
    db.execute = AsyncMock()
    with pytest.raises(RuntimeError, match="usable nonpartial hnsw"):
        await PGVectorStorage.setup_table(
            db, "model_8d", "tenant", 8, "legacy", "LIGHTRAG_VDB_CHUNKS"
        )
    db.execute.assert_not_awaited()


@pytest.mark.parametrize(
    ("source", "target", "marker", "refuses"),
    [
        (True, False, False, True),
        (True, True, True, True),
        (True, True, False, False),
        (False, False, True, False),
    ],
)
@pytest.mark.asyncio
async def test_no_ddl_legacy_migration_respects_accepted_residues(
    monkeypatch, source, target, marker, refuses
):
    from lightrag.kg import postgres_schema
    from lightrag.kg.postgres_impl import PGVectorStorage

    monkeypatch.setattr(
        postgres_schema, "validate_table", AsyncMock(return_value=_vector_snapshot())
    )
    migration = AsyncMock()
    monkeypatch.setattr(PGVectorStorage, "_pg_migrate_workspace_data", migration)
    monkeypatch.setattr(
        "lightrag.kg.postgres_impl._pg_vector_sibling_tables",
        AsyncMock(return_value=[]),
    )
    db = MagicMock(auto_migrate=False, vector_index_type="HNSW")
    db.check_table_exists = AsyncMock(return_value=True)
    responses = [{"relkind": "r", "permitted": True}, {"populated": source}]
    if source:
        responses.append({"populated": target})
        if target:
            responses.append({"in_progress": marker})
    db.query = AsyncMock(side_effect=responses)
    db.execute = AsyncMock()
    call = PGVectorStorage.setup_table(
        db, "model_8d", "tenant", 8, "legacy", "LIGHTRAG_VDB_CHUNKS"
    )
    if refuses:
        with pytest.raises(RuntimeError, match="legacy vector migration.*incomplete"):
            await call
    else:
        await call
    assert db.query.await_count == len(responses)
    migration.assert_not_awaited()
    db.execute.assert_not_awaited()


@pytest.mark.parametrize("installed", [False, True])
@pytest.mark.asyncio
async def test_vchord_extension_is_required_without_bypassing_checks(installed):
    db = object.__new__(PostgreSQLDB)
    db.auto_migrate = False
    db.vector_index_type = "VCHORDRQ"
    db._probe_iterative_scan_support = AsyncMock()
    connection = AsyncMock()
    connection.fetchrow.side_effect = [
        {"extversion": "0.8.7"},
        {"extversion": "0.4.3"} if installed else None,
    ]
    if installed:
        await db.configure_vector_extension(connection)
        db._probe_iterative_scan_support.assert_awaited_once_with(connection)
    else:
        with pytest.raises(RuntimeError, match="administrator install VectorChord"):
            await db.configure_vector_extension(connection)
        db._probe_iterative_scan_support.assert_not_awaited()
    assert connection.fetchrow.await_count == 2
    connection.execute.assert_not_awaited()


@pytest.mark.parametrize(
    ("configured", "method"), [("IVFFLAT", "ivfflat"), ("VCHORDRQ", "vchordrq")]
)
@pytest.mark.asyncio
async def test_no_ddl_accepts_other_configured_ann_methods(
    monkeypatch, configured, method
):
    from lightrag.kg import postgres_schema
    from lightrag.kg.postgres_impl import PGVectorStorage

    snapshot = _vector_snapshot()
    snapshot[2][0]["amname"] = method
    validate = AsyncMock(return_value=snapshot)
    monkeypatch.setattr(postgres_schema, "validate_table", validate)
    db = MagicMock(auto_migrate=False, vector_index_type=configured)
    db.check_table_exists = AsyncMock(return_value=False)
    db.query = AsyncMock(return_value=[])
    db.execute = AsyncMock()
    await PGVectorStorage.setup_table(
        db, "model_8d", "tenant", 8, "legacy", "LIGHTRAG_VDB_CHUNKS"
    )
    validate.assert_awaited_once()
    db.check_table_exists.assert_awaited_once_with("legacy")
    db.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_physical_vector_schema_mismatch_is_not_a_recoverable_refusal(
    monkeypatch,
):
    from lightrag.exceptions import VectorSpaceMismatchError
    from lightrag.kg import postgres_schema
    from lightrag.kg.postgres_impl import PGVectorStorage

    snapshot = _vector_snapshot()
    snapshot[1]["content_vector"]["atttypmod"] = 7
    monkeypatch.setattr(
        postgres_schema, "validate_table", AsyncMock(return_value=snapshot)
    )
    db = MagicMock(auto_migrate=False, vector_index_type="HNSW")
    db.execute = AsyncMock()
    with pytest.raises(RuntimeError, match="content_vector.*administrator") as error:
        await PGVectorStorage.setup_table(
            db, "model_8d", "tenant", 8, "legacy", "LIGHTRAG_VDB_CHUNKS"
        )
    # The rebuild tool tolerates exactly this type before offering drop + retry.
    assert not isinstance(error.value, VectorSpaceMismatchError)
    db.execute.assert_not_awaited()
