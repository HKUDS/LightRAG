"""Live-PostgreSQL tests for the lexical (hybrid retrieval) schema lifecycle.

Pins the schema lifecycle on a real pgvector server: idempotent migration,
reentrant batched backfill, CREATE INDEX CONCURRENTLY that does not lock
writes, invalid-index detection with drop/retry, concurrent workers racing
the migration, and upsert-maintained tsvector consistency for CJK, latin and
mixed text. A SIGKILL mid-backfill proof is a manual QA run, not here; the
in-process fault injection below covers the same reentrancy seam at the
db.query boundary.

Markers and the env guard follow the live-DB conventions of
``test_postgres_anchor_migration_smoke.py``: skipped unless POSTGRES_HOST and
POSTGRES_PASSWORD are set, and skipped when POSTGRES_WORKSPACE would remap
the per-test workspaces. Every test uses its own workspace on shared
model-suffixed test tables; an autouse fixture empties those tables after
each test so a failure cannot leak rows into the next case.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from uuid import uuid4

import pytest

pytestmark = [pytest.mark.requires_db, pytest.mark.integration]

if not (os.environ.get("POSTGRES_HOST") and os.environ.get("POSTGRES_PASSWORD")):
    pytest.skip(
        "POSTGRES_HOST/POSTGRES_PASSWORD not set — skipping live lexical tests",
        allow_module_level=True,
    )

import asyncpg  # noqa: E402
import numpy as np  # noqa: E402

from lightrag.kg import shared_storage  # noqa: E402
from lightrag.kg.lexical_tokenizer import tokenize_for_index  # noqa: E402
from lightrag.kg.postgres_impl import (  # noqa: E402
    LEXICAL_TSV_COLUMN,
    ClientManager,
    PGVectorStorage,
    _safe_index_name,
)
from lightrag.kg.shared_storage import initialize_share_data  # noqa: E402
from lightrag.namespace import NameSpace  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402

_MODEL_NAME = "lexschema"
_DIM = 8
_BASE_TABLES = {
    NameSpace.VECTOR_STORE_CHUNKS: "LIGHTRAG_VDB_CHUNKS",
    NameSpace.VECTOR_STORE_ENTITIES: "LIGHTRAG_VDB_ENTITY",
    NameSpace.VECTOR_STORE_RELATIONSHIPS: "LIGHTRAG_VDB_RELATION",
}


async def _embed(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), _DIM), dtype=np.float32)


def _make_storage(namespace: str, workspace: str) -> PGVectorStorage:
    return PGVectorStorage(
        namespace=namespace,
        workspace=workspace,
        global_config={
            "embedding_batch_num": 16,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=_DIM,
            max_token_size=8192,
            func=_embed,
            model_name=_MODEL_NAME,
        ),
    )


def _raw_conn_kwargs() -> dict:
    config = ClientManager.get_config()
    return {
        "host": config["host"],
        "port": int(config["port"]),
        "user": config["user"],
        "password": config["password"],
        "database": config["database"],
    }


def _index_name(table_name: str) -> str:
    return _safe_index_name(table_name, f"{LEXICAL_TSV_COLUMN}_gin")


async def _wait_until(predicate, description: str, timeout: float = 30.0):
    """Bounded condition wait on server state; raises TimeoutError, never hangs."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        result = await predicate()
        if result:
            return result
        if loop.time() >= deadline:
            raise TimeoutError(f"timed out waiting for {description}")
        await asyncio.sleep(0.05)


async def _seed_rows(db, namespace: str, table: str, workspace: str, n: int) -> None:
    """Insert n rows with NULL tsv via one parameterized statement."""
    if namespace == NameSpace.VECTOR_STORE_CHUNKS:
        sql = (
            f"INSERT INTO {table} (workspace, id, content, tokens, "
            "chunk_order_index, full_doc_id, file_path) "
            "SELECT $1, 'seed-' || g, '乐高机器人混合检索 ' || g, 1, 0, "
            "'doc-seed', 'seed.txt' "
            "FROM generate_series(1, $2) AS g ON CONFLICT DO NOTHING"
        )
    elif namespace == NameSpace.VECTOR_STORE_ENTITIES:
        sql = (
            f"INSERT INTO {table} (workspace, id, entity_name, content) "
            "SELECT $1, 'ent-' || g, '实体名称' || g, 'entity description ' || g "
            "FROM generate_series(1, $2) AS g ON CONFLICT DO NOTHING"
        )
    else:
        sql = (
            f"INSERT INTO {table} (workspace, id, source_id, target_id, content) "
            "SELECT $1, 'rel-' || g, 'src-' || g, 'tgt-' || g, '关系描述 ' || g "
            "FROM generate_series(1, $2) AS g ON CONFLICT DO NOTHING"
        )
    await db.execute(sql, {"workspace": workspace, "n": n})


async def _covered(db, table: str, workspace: str) -> int:
    row = await db.query(
        f"SELECT COUNT(*) AS n FROM {table} "
        f"WHERE workspace=$1 AND {LEXICAL_TSV_COLUMN} IS NOT NULL",
        [workspace],
    )
    return row["n"]


async def _pending(db, table: str, workspace: str) -> int:
    row = await db.query(
        f"SELECT COUNT(*) AS n FROM {table} "
        f"WHERE workspace=$1 AND {LEXICAL_TSV_COLUMN} IS NULL",
        [workspace],
    )
    return row["n"]


async def _row_count(db, table: str, workspace: str) -> int:
    row = await db.query(
        f"SELECT COUNT(*) AS n FROM {table} WHERE workspace=$1", [workspace]
    )
    return row["n"]


async def _schema_snapshot(db, table: str) -> dict:
    columns = await db.query(
        "SELECT attname, format_type(atttypid, atttypmod) AS coltype "
        "FROM pg_attribute WHERE attrelid = to_regclass($1) "
        "AND attnum > 0 AND NOT attisdropped ORDER BY attname",
        [table.lower()],
        multirows=True,
    )
    indexes = await db.query(
        "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = $1 "
        "ORDER BY indexname",
        [table.lower()],
        multirows=True,
    )
    return {"columns": columns, "indexes": indexes}


async def _index_is_valid(db, table: str) -> bool:
    row = await db.query(
        "SELECT i.indisvalid FROM pg_index i WHERE i.indexrelid = to_regclass($1)",
        [_index_name(table)],
    )
    return bool(row and row["indisvalid"])


def _upsert_item(namespace: str, text: str) -> dict:
    if namespace == NameSpace.VECTOR_STORE_CHUNKS:
        return {
            "tokens": 1,
            "chunk_order_index": 0,
            "full_doc_id": "doc-live",
            "content": text,
            "file_path": "live.txt",
        }
    if namespace == NameSpace.VECTOR_STORE_ENTITIES:
        return {
            "entity_name": text,
            "content": "unrelated description zzz",
            "source_id": "chunk-live",
            "file_path": "live.txt",
        }
    return {
        "src_id": "Alpha",
        "tgt_id": "Beta",
        "content": text,
        "source_id": "chunk-live",
        "file_path": "live.txt",
    }


def _lexical_source(namespace: str, item: dict) -> str:
    if namespace == NameSpace.VECTOR_STORE_ENTITIES:
        return item["entity_name"]
    return item["content"]


@pytest.fixture(autouse=True)
async def _lexical_test_env():
    """Shared-data init, workspace-remap guard, and post-test table cleanup."""
    if ClientManager.get_config()["workspace"]:
        pytest.skip(
            "POSTGRES_WORKSPACE / [postgres] workspace would remap the test "
            "workspaces onto an existing one"
        )
    initialize_share_data(workers=1)
    # asyncio.Lock binds to the loop of its first CONTENDED acquire. pytest
    # gives every test a fresh loop, while these two primitives are
    # process-wide; the concurrency cases contend them, so without a reset
    # the second contending test dies on "bound to a different event loop".
    # Nothing holds either lock at fixture setup (prior tests finalized).
    shared_storage._data_init_lock = asyncio.Lock()
    ClientManager._lock = asyncio.Lock()
    yield
    conn = await asyncpg.connect(**_raw_conn_kwargs())
    try:
        for base in _BASE_TABLES.values():
            table = f"{base}_{_MODEL_NAME}_{_DIM}d".lower()
            if await conn.fetchval("SELECT to_regclass($1) IS NOT NULL", table):
                await conn.execute(f"DELETE FROM {table}")
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# 1. Idempotency: migration twice -> identical schema and row counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("namespace", list(_BASE_TABLES))
async def test_migration_is_idempotent(namespace, monkeypatch):
    monkeypatch.setenv("POSTGRES_LEXICAL_BACKFILL_BATCH_SIZE", "1000")
    workspace = f"lexidem_{uuid4().hex[:8]}"
    storage = _make_storage(namespace, workspace)
    await storage.initialize()
    table = storage.table_name
    try:
        await _seed_rows(storage.db, namespace, table, workspace, 25)
        # Re-run the readiness step on the SAME instance: it must spawn the
        # backfill for the newly seeded rows without touching the schema.
        await storage._ensure_lexical_readiness()
        assert storage._lexical_backfill_task is not None
        await asyncio.wait_for(storage._lexical_backfill_task, timeout=60)
        assert await _covered(storage.db, table, workspace) == 25

        snapshot_before = await _schema_snapshot(storage.db, table)
        rows_before = await _row_count(storage.db, table, workspace)
        assert await _index_is_valid(storage.db, table)
    finally:
        await storage.finalize()

    restarted = _make_storage(namespace, workspace)
    await restarted.initialize()
    try:
        snapshot_after = await _schema_snapshot(restarted.db, table)
        assert snapshot_after == snapshot_before
        assert await _row_count(restarted.db, table, workspace) == rows_before == 25
        assert await _covered(restarted.db, table, workspace) == 25
        assert await _pending(restarted.db, table, workspace) == 0
        # Nothing pending: the restart must not spawn a backfill task.
        assert restarted._lexical_backfill_task is None
        assert await _index_is_valid(restarted.db, table)
    finally:
        await restarted.finalize()


# ---------------------------------------------------------------------------
# 2. Reentrancy: fault mid-batch, restart completes, counts consistent
# ---------------------------------------------------------------------------


async def test_backfill_fault_mid_batch_resumes_on_restart(
    monkeypatch, lightrag_log_records
):
    monkeypatch.setenv("POSTGRES_LEXICAL_BACKFILL_BATCH_SIZE", "1000")
    workspace = f"lexfault_{uuid4().hex[:8]}"
    storage = _make_storage(NameSpace.VECTOR_STORE_CHUNKS, workspace)
    await storage.initialize()
    table = storage.table_name
    db = storage.db
    original_query = db.query
    fault_calls = {"backfill_selects": 0}

    async def faulting_query(sql, params=None, *args, **kwargs):
        if "lexical_source" in sql:
            fault_calls["backfill_selects"] += 1
            if fault_calls["backfill_selects"] == 2:
                raise RuntimeError("injected mid-batch failure")
        return await original_query(sql, params, *args, **kwargs)

    try:
        await _seed_rows(db, NameSpace.VECTOR_STORE_CHUNKS, table, workspace, 2500)
        db.query = faulting_query
        try:
            await storage._ensure_lexical_readiness()
            assert storage._lexical_backfill_task is not None
            await asyncio.wait_for(storage._lexical_backfill_task, timeout=120)
        finally:
            db.query = original_query

        # The injected fault fired on the second batch SELECT, and the first
        # batch (1000 rows) committed while the second left nothing behind.
        assert fault_calls["backfill_selects"] == 2
        assert await _covered(db, table, workspace) == 1000
        assert await _pending(db, table, workspace) == 1500
        assert any(
            "stopped after 1000 rows" in record.getMessage()
            and record.levelname == "WARNING"
            for record in lightrag_log_records
        )
    finally:
        await storage.finalize()

    restarted = _make_storage(NameSpace.VECTOR_STORE_CHUNKS, workspace)
    await restarted.initialize()
    try:
        assert restarted._lexical_backfill_task is not None
        await asyncio.wait_for(restarted._lexical_backfill_task, timeout=120)
        assert await _covered(restarted.db, table, workspace) == 2500
        assert await _pending(restarted.db, table, workspace) == 0
        assert await _row_count(restarted.db, table, workspace) == 2500
    finally:
        await restarted.finalize()


# ---------------------------------------------------------------------------
# 3. CONCURRENTLY does not lock writes: a committed INSERT overlaps the build
# ---------------------------------------------------------------------------


async def test_concurrent_index_build_does_not_lock_writes():
    workspace = f"lexnolock_{uuid4().hex[:8]}"
    storage = _make_storage(NameSpace.VECTOR_STORE_CHUNKS, workspace)
    await storage.initialize()
    table = storage.table_name
    db = storage.db
    stall_conn = await asyncpg.connect(**_raw_conn_kwargs())
    build_task = None
    try:
        await _seed_rows(db, NameSpace.VECTOR_STORE_CHUNKS, table, workspace, 2000)
        # Stage a pre-migration state: no lexical index. Dropping it here is
        # test setup; the production code only ever drops INVALID indexes.
        await db.execute(f"DROP INDEX IF EXISTS {_index_name(table)}")

        # A write transaction opened BEFORE the build forces CIC into its
        # wait phase, so the overlap window below is deterministic, not a
        # race against build speed.
        stall_tx = stall_conn.transaction()
        await stall_tx.start()
        await stall_conn.execute(
            f"INSERT INTO {table} (workspace, id, content, tokens, "
            "chunk_order_index, full_doc_id, file_path) "
            "VALUES ($1, 'stall-row', 'stalling writer', 1, 0, 'doc-stall', 's.txt')",
            workspace,
        )

        # Drive the index unit directly: ALTER TABLE takes an
        # AccessExclusiveLock even when ADD COLUMN IF NOT EXISTS is a no-op,
        # so the full readiness step would block behind the stall transaction
        # and never reach the build. The column is already ensured above.
        build_task = asyncio.create_task(
            PGVectorStorage._pg_ensure_lexical_index(db, table)
        )

        async def build_in_progress():
            return await db.query(
                "SELECT phase FROM pg_stat_progress_create_index "
                "WHERE relid = to_regclass($1)",
                [table.lower()],
            )

        progress = await _wait_until(build_in_progress, "CIC build to start")
        assert not build_task.done()

        # The writer commits WHILE the build is provably still running.
        insert_start = time.perf_counter()
        await db.execute(
            f"INSERT INTO {table} (workspace, id, content, tokens, "
            "chunk_order_index, full_doc_id, file_path) "
            "VALUES ($1, 'concurrent-row', '并发写入测试', 1, 0, 'doc-live', 'l.txt')",
            {"workspace": workspace},
        )
        insert_elapsed = time.perf_counter() - insert_start
        # 2000 seeded + the concurrent row; the stall writer's row is still
        # uncommitted and rightly invisible to other connections.
        assert await _row_count(db, table, workspace) == 2001
        assert await build_in_progress(), (
            "build finished before the write proved overlap"
        )
        assert not build_task.done()
        print(
            f"\n[no-lock proof] insert committed in {insert_elapsed:.3f}s while "
            f"CIC phase={progress['phase']!r} still in progress"
        )

        await stall_tx.commit()
        await asyncio.wait_for(build_task, timeout=120)
        build_task = None
        assert await _index_is_valid(db, table)
        assert await _row_count(db, table, workspace) == 2002
    finally:
        if build_task is not None:
            build_task.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await build_task
        await stall_conn.close()
        await storage.finalize()


# ---------------------------------------------------------------------------
# 4. Invalid index residue: real CIC failure -> detection -> drop -> retry
# ---------------------------------------------------------------------------


async def test_invalid_index_is_detected_dropped_and_rebuilt(lightrag_log_records):
    workspace = f"lexinvalid_{uuid4().hex[:8]}"
    storage = _make_storage(NameSpace.VECTOR_STORE_CHUNKS, workspace)
    await storage.initialize()
    table = storage.table_name
    db = storage.db
    index_name = _index_name(table)
    stall_conn = await asyncpg.connect(**_raw_conn_kwargs())
    cancel_conn = await asyncpg.connect(**_raw_conn_kwargs())
    build_task = None
    try:
        await _seed_rows(db, NameSpace.VECTOR_STORE_CHUNKS, table, workspace, 2000)
        await db.execute(f"DROP INDEX IF EXISTS {index_name}")

        stall_tx = stall_conn.transaction()
        await stall_tx.start()
        await stall_conn.execute(
            f"INSERT INTO {table} (workspace, id, content, tokens, "
            "chunk_order_index, full_doc_id, file_path) "
            "VALUES ($1, 'stall-row', 'stalling writer', 1, 0, 'doc-stall', 's.txt')",
            workspace,
        )

        # Index unit directly, for the same ALTER-lock reason as the no-lock
        # test above.
        build_task = asyncio.create_task(
            PGVectorStorage._pg_ensure_lexical_index(db, table)
        )

        async def cic_pid():
            return await cancel_conn.fetchval(
                "SELECT pid FROM pg_stat_activity "
                "WHERE query LIKE 'CREATE INDEX CONCURRENTLY%' "
                "AND query LIKE '%' || $1 || '%' "
                "AND pid <> pg_backend_pid() AND state = 'active'",
                index_name,
            )

        pid = await _wait_until(cic_pid, "the stalled CIC backend pid")
        # Real fault injection: cancel the building backend. CIC leaves the
        # index INVALID when canceled; the residue is what the code must heal.
        canceled = await cancel_conn.fetchval("SELECT pg_cancel_backend($1)", pid)
        assert canceled is True, "pg_cancel_backend did not signal the builder"
        await stall_tx.commit()

        await asyncio.wait_for(build_task, timeout=120)
        build_task = None

        messages = [record.getMessage() for record in lightrag_log_records]
        assert any(
            f"concurrent build of {index_name} failed (attempt 1/3)" in message
            for message in messages
        ), f"attempt-1 failure not logged: {messages}"
        assert any(
            f"dropping invalid lexical index {index_name}" in message
            for message in messages
        ), f"invalid-index detection not logged: {messages}"
        assert await _index_is_valid(db, table)
        invalid_residue = await db.query(
            "SELECT COUNT(*) AS n FROM pg_index "
            "WHERE indexrelid = to_regclass($1) AND NOT indisvalid",
            [index_name],
        )
        assert invalid_residue["n"] == 0
    finally:
        if build_task is not None:
            build_task.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await build_task
        await stall_conn.close()
        await cancel_conn.close()
        await storage.finalize()


# ---------------------------------------------------------------------------
# 5. >=4 concurrent workers inserting + backfilling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shared_workspace", [False, True], ids=["distinct", "shared"])
async def test_four_workers_race_migration_insert_and_backfill(
    shared_workspace, monkeypatch
):
    monkeypatch.setenv("POSTGRES_LEXICAL_BACKFILL_BATCH_SIZE", "100")
    run_id = uuid4().hex[:8]
    common_workspace = f"lexrace_{run_id}"
    barrier = asyncio.Barrier(4)
    table_holder: dict[str, str] = {}

    async def worker(i: int) -> int:
        workspace = common_workspace if shared_workspace else f"lexrace_{run_id}_{i}"
        storage = _make_storage(NameSpace.VECTOR_STORE_CHUNKS, workspace)
        try:
            await asyncio.wait_for(barrier.wait(), timeout=30)
            await storage.initialize()
            table = storage.table_name
            table_holder["table"] = table
            seed_prefix = f"w{i}"
            await storage.db.execute(
                f"INSERT INTO {table} (workspace, id, content, tokens, "
                "chunk_order_index, full_doc_id, file_path) "
                "SELECT $1, $2 || '-seed-' || g, '并发回填测试 ' || g, 1, 0, "
                "'doc-seed', 'seed.txt' "
                "FROM generate_series(1, $3) AS g ON CONFLICT DO NOTHING",
                {"workspace": workspace, "prefix": seed_prefix, "n": 300},
            )
            await storage._ensure_lexical_readiness()
            backfill_task = storage._lexical_backfill_task
            assert backfill_task is not None
            # Live writes race the backfill: upsert-maintained rows must not
            # be disturbed, and the backfill must still cover the seeded ones.
            await storage.upsert(
                {
                    f"{seed_prefix}-live-{j}": _upsert_item(
                        NameSpace.VECTOR_STORE_CHUNKS, f"实时写入 live write {j}"
                    )
                    for j in range(50)
                }
            )
            await storage.index_done_callback()
            await asyncio.wait_for(backfill_task, timeout=120)
            # In shared mode rival tasks cover other workers' rows too, and a
            # rival may not have seeded yet when this task exits: pin THIS
            # worker's own rows here, and the workspace total after the gather.
            own_covered = await storage.db.query(
                f"SELECT COUNT(*) AS n FROM {table} "
                f"WHERE workspace=$1 AND id LIKE $2 "
                f"AND {LEXICAL_TSV_COLUMN} IS NOT NULL",
                [workspace, f"{seed_prefix}-%"],
            )
            assert own_covered["n"] == 350, f"worker {i}: own rows {own_covered['n']}"
            return own_covered["n"]
        finally:
            await storage.finalize()

    results = await asyncio.gather(*(worker(i) for i in range(4)))
    assert results == [350] * 4

    conn = await asyncpg.connect(**_raw_conn_kwargs())
    try:
        table = table_holder["table"]
        if shared_workspace:
            total = await conn.fetchval(
                f"SELECT COUNT(*) FROM {table} WHERE workspace=$1 "
                f"AND {LEXICAL_TSV_COLUMN} IS NOT NULL",
                common_workspace,
            )
            pending = await conn.fetchval(
                f"SELECT COUNT(*) FROM {table} WHERE workspace=$1 "
                f"AND {LEXICAL_TSV_COLUMN} IS NULL",
                common_workspace,
            )
            assert (total, pending) == (1400, 0)
        valid = await conn.fetchval(
            "SELECT i.indisvalid FROM pg_index i WHERE i.indexrelid = to_regclass($1)",
            _index_name(table),
        )
        assert valid is True
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# 6. Upsert tsv consistency: stored == to_tsvector('simple', tokenized source)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("namespace", list(_BASE_TABLES))
@pytest.mark.parametrize(
    "text",
    [
        "乐高机器人搭建指南",
        "hello world hybrid retrieval",
        "LightRAG 混合检索 café 123",
        "",
    ],
    ids=["cjk", "latin", "mixed", "empty"],
)
async def test_upsert_maintains_tsv_consistency(namespace, text):
    workspace = f"lextsv_{uuid4().hex[:8]}"
    storage = _make_storage(namespace, workspace)
    await storage.initialize()
    table = storage.table_name
    try:
        item = _upsert_item(namespace, text)
        await storage.upsert({"row-1": item})
        await storage.index_done_callback()

        stored = await storage.db.query(
            f"SELECT {LEXICAL_TSV_COLUMN}::text AS tsv, "
            f"{LEXICAL_TSV_COLUMN} IS NOT NULL AS present "
            f"FROM {table} WHERE workspace=$1 AND id=$2",
            [workspace, "row-1"],
        )
        expected = await storage.db.query(
            "SELECT to_tsvector('simple', $1)::text AS tsv",
            [tokenize_for_index(_lexical_source(namespace, item))],
        )
        assert stored["present"] is True
        assert stored["tsv"] == expected["tsv"]
        if text == "乐高机器人搭建指南":
            assert "'乐高'" in stored["tsv"], "CJK bigram lexeme missing"
        if namespace == NameSpace.VECTOR_STORE_ENTITIES and text:
            # The entity lexical source is entity_name, NOT content.
            assert "zzz" not in (stored["tsv"] or "")

        # Overwrite: the same id re-upserted must refresh the tsv.
        updated = _upsert_item(namespace, "更新后的内容 updated content")
        await storage.upsert({"row-1": updated})
        await storage.index_done_callback()
        stored_after = await storage.db.query(
            f"SELECT {LEXICAL_TSV_COLUMN}::text AS tsv FROM {table} "
            "WHERE workspace=$1 AND id=$2",
            [workspace, "row-1"],
        )
        expected_after = await storage.db.query(
            "SELECT to_tsvector('simple', $1)::text AS tsv",
            [tokenize_for_index(_lexical_source(namespace, updated))],
        )
        assert stored_after["tsv"] == expected_after["tsv"]
    finally:
        await storage.finalize()
