"""Live-PostgreSQL hybrid query semantics.

Pins the hybrid query contract on a real pgvector server with controlled
embeddings: the vector
leg's cosine threshold is unchanged, RRF fusion order equals hand-computed
ranks, lexical-only records carry the exact distance sentinel, each leg
takes top_k and the union returns top_k, and a dropped/invalid GIN index or
a lexical query error degrades THAT query to the vector baseline with a
deduplicated warning. Reserved-character/injection strings run through
the live lexical leg, then empty queries, bounded >1000-token queries, fault
variants (real canceled-CIC invalid residue included), and four
concurrent workers querying while ingesting.

Markers and the env guard follow ``test_postgres_lexical_schema.py``: skipped
unless POSTGRES_HOST and POSTGRES_PASSWORD are set, skipped when
POSTGRES_WORKSPACE would remap the per-test workspaces. Every test uses its
own workspace on shared model-suffixed tables; an autouse fixture empties
those tables after each test.
"""

from __future__ import annotations

import asyncio
import os
from uuid import uuid4

import pytest

pytestmark = [pytest.mark.requires_db, pytest.mark.integration]

if not (os.environ.get("POSTGRES_HOST") and os.environ.get("POSTGRES_PASSWORD")):
    pytest.skip(
        "POSTGRES_HOST/POSTGRES_PASSWORD not set — skipping live hybrid query tests",
        allow_module_level=True,
    )

import asyncpg  # noqa: E402
import numpy as np  # noqa: E402

from lightrag.kg import shared_storage  # noqa: E402
from lightrag.kg.lexical_tokenizer import (  # noqa: E402
    MAX_TOKENS,
    tokenize,
    tokenize_for_query,
)
from lightrag.kg.postgres_impl import (  # noqa: E402
    LEXICAL_TSV_COLUMN,
    ClientManager,
    PGVectorStorage,
    _safe_index_name,
)
from lightrag.kg.shared_storage import initialize_share_data  # noqa: E402
from lightrag.namespace import NameSpace  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402

_MODEL_NAME = "hybridq"
_DIM = 8
_BASE_TABLES = {
    NameSpace.VECTOR_STORE_CHUNKS: "LIGHTRAG_VDB_CHUNKS",
    NameSpace.VECTOR_STORE_ENTITIES: "LIGHTRAG_VDB_ENTITY",
    NameSpace.VECTOR_STORE_RELATIONSHIPS: "LIGHTRAG_VDB_RELATION",
}

# Query embedding: cosine with each fixture vector is exact by construction.
_Q_VEC = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
_QUERY = "乐高机器人"
# content -> embedding. Cosines with _Q_VEC: 1.0, 0.6, 0.0, 0.0, 0.0.
_VECTORS = {
    "乐高机器人 搭建": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "完全无关的内容": [0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "乐高机器人 教程": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "乐高机器人 指南": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "无关文本": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
}
# id -> content. c-v* pass the 0.2 cosine threshold; c-l* match the query
# lexically but sit at cosine 0.0; c-x matches neither.
_CHUNK_FIXTURE = {
    "c-v1": "乐高机器人 搭建",
    "c-v2": "完全无关的内容",
    "c-l1": "乐高机器人 教程",
    "c-l2": "乐高机器人 指南",
    "c-x": "无关文本",
}
_VECTOR_ONLY_IDS = ["c-v1", "c-v2"]
# top_k=3 fusion, hand-computed: lexical leg (ts_rank_cd ties, id ASC)
# [c-l1, c-l2, c-v1]; vector leg [c-v1, c-v2].
# c-v1 = 1/61 + 1/63, c-l1 = 1/61, c-v2 = c-l2 = 1/62 with the vector-rank
# tie-break putting c-v2 first -> [c-v1, c-l1, c-v2].
_LEXICAL_MIX_IDS = ["c-v1", "c-l1", "c-v2"]


async def _embed(texts: list[str]) -> np.ndarray:
    return np.array([_VECTORS[t] for t in texts], dtype=np.float32)


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


def _chunk_item(content: str) -> dict:
    return {
        "tokens": 1,
        "chunk_order_index": 0,
        "full_doc_id": "doc-hyb",
        "content": content,
        "file_path": "hyb.txt",
    }


async def _seed_chunks(storage: PGVectorStorage) -> None:
    await storage.upsert(
        {cid: _chunk_item(content) for cid, content in _CHUNK_FIXTURE.items()}
    )
    await storage.index_done_callback()


async def _vector_baseline(storage: PGVectorStorage, top_k: int) -> list[dict]:
    return await storage.query(_QUERY, top_k=top_k, query_embedding=_Q_VEC)


async def _index_validity(db, table: str):
    return await db.query(
        "SELECT i.indisvalid FROM pg_index i WHERE i.indexrelid = to_regclass($1)",
        [_index_name(table)],
    )


@pytest.fixture(autouse=True)
async def _hybrid_test_env():
    """Shared-data init, workspace-remap guard, and post-test table cleanup."""
    if ClientManager.get_config()["workspace"]:
        pytest.skip(
            "POSTGRES_WORKSPACE / [postgres] workspace would remap the test "
            "workspaces onto an existing one"
        )
    initialize_share_data(workers=1)
    # asyncio.Lock binds to the loop of its first CONTENDED acquire; these
    # two primitives are process-wide while pytest loops are per-test (the
    # the shared fixture's reset pattern). Nothing holds them at setup.
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
# vector-leg threshold unchanged
# ---------------------------------------------------------------------------


async def test_vector_leg_threshold_unchanged_and_empty_lexical_union():
    workspace = f"hybthr_{uuid4().hex[:8]}"
    storage = _make_storage(NameSpace.VECTOR_STORE_CHUNKS, workspace)
    await storage.initialize()
    try:
        await _seed_chunks(storage)
        baseline = await _vector_baseline(storage, top_k=5)
        # Known cosines: 1.0 and 0.6 pass the 0.2 threshold; the three 0.0
        # rows are excluded even though two of them match the query text.
        assert [r["id"] for r in baseline] == _VECTOR_ONLY_IDS

        # A query with tokens but zero lexical matches: the fused result is
        # the vector baseline record-for-record (no sentinel, no reorder).
        fused = await storage.hybrid_query(
            "zzznomatch", top_k=5, query_embedding=_Q_VEC
        )
        assert fused == baseline
    finally:
        await storage.finalize()


# ---------------------------------------------------------------------------
# RRF fusion order == hand-computed ranks; top_k semantics; sentinel
# ---------------------------------------------------------------------------


async def test_rrf_fusion_order_matches_hand_computed_ranks():
    workspace = f"hybrrf_{uuid4().hex[:8]}"
    storage = _make_storage(NameSpace.VECTOR_STORE_CHUNKS, workspace)
    await storage.initialize()
    table = storage.table_name
    try:
        await _seed_chunks(storage)
        # Fixture premise: the three lexical matches carry the four query
        # bigrams at identical positions, so ts_rank_cd ties and the
        # documented id-ASC secondary order decides the lexical leg.
        ranks = await storage.db.query(
            f"SELECT id, ts_rank_cd({LEXICAL_TSV_COLUMN}, "
            "to_tsquery('simple', $1)) AS r "
            f"FROM {table} WHERE workspace=$2 "
            f"AND {LEXICAL_TSV_COLUMN} @@ to_tsquery('simple', $1) "
            "ORDER BY r DESC, id",
            [tokenize_for_query(_QUERY), workspace],
            multirows=True,
        )
        assert [row["id"] for row in ranks] == ["c-l1", "c-l2", "c-v1"]
        assert len({row["r"] for row in ranks}) == 1, "ts_rank_cd tie premise broke"

        # top_k=4: vector leg [c-v1, c-v2]; lexical leg [c-l1, c-l2, c-v1].
        # c-v1 = 1/61 + 1/63 > c-l1 = 1/61 > c-v2 = 1/62 (tie with c-l2,
        # vector rank wins) > c-l2. Union of both legs, returned in fusion
        # order, at most top_k.
        fused4 = await storage.hybrid_query(_QUERY, top_k=4, query_embedding=_Q_VEC)
        assert [r["id"] for r in fused4] == ["c-v1", "c-l1", "c-v2", "c-l2"]

        # top_k=2: each leg takes 2 (lexical leg drops c-v1), the union is
        # fused and truncated to 2: c-v1 = c-l1 = 1/61 (vector rank wins the
        # tie), then c-v2 = c-l2 = 1/62 (same rule).
        fused2 = await storage.hybrid_query(_QUERY, top_k=2, query_embedding=_Q_VEC)
        assert [r["id"] for r in fused2] == ["c-v1", "c-l1"]
    finally:
        await storage.finalize()


async def test_lexical_only_distance_sentinel_exact():
    workspace = f"hybsen_{uuid4().hex[:8]}"
    storage = _make_storage(NameSpace.VECTOR_STORE_CHUNKS, workspace)
    await storage.initialize()
    try:
        await _seed_chunks(storage)
        baseline = await _vector_baseline(storage, top_k=4)
        fused = await storage.hybrid_query(_QUERY, top_k=4, query_embedding=_Q_VEC)
        by_id = {r["id"]: r for r in fused}

        # Lexical-only records: distance exactly the cosine threshold, so a
        # downstream >= threshold similarity filter admits them. c-l1/c-l2
        # are at cosine 0.0 (absent from the vector baseline) yet retrieved.
        assert by_id["c-l1"]["distance"] == storage.cosine_better_than_threshold
        assert by_id["c-l2"]["distance"] == storage.cosine_better_than_threshold
        assert "c-l1" not in {r["id"] for r in baseline}
        # Vector-leg records keep their original fields untouched.
        assert by_id["c-v1"] == baseline[0]
        assert by_id["c-v2"] == baseline[1]
        assert "distance" not in by_id["c-v1"]
    finally:
        await storage.finalize()


async def test_entities_namespace_hybrid_fuses_on_entity_name():
    workspace = f"hybent_{uuid4().hex[:8]}"
    storage = _make_storage(NameSpace.VECTOR_STORE_ENTITIES, workspace)
    await storage.initialize()
    try:
        # entity_name is the lexical source; content is what gets embedded.
        await storage.upsert(
            {
                "e-v1": {
                    "entity_name": "乐高机器人",
                    "content": "乐高机器人 搭建",
                    "source_id": "chunk-hyb",
                    "file_path": "hyb.txt",
                },
                "e-v2": {
                    "entity_name": "无关实体",
                    "content": "完全无关的内容",
                    "source_id": "chunk-hyb",
                    "file_path": "hyb.txt",
                },
                "e-l1": {
                    "entity_name": "乐高机器人手册",
                    "content": "乐高机器人 教程",
                    "source_id": "chunk-hyb",
                    "file_path": "hyb.txt",
                },
            }
        )
        await storage.index_done_callback()
        baseline = await storage.query(_QUERY, top_k=3, query_embedding=_Q_VEC)
        assert [r["entity_name"] for r in baseline] == ["乐高机器人", "无关实体"]

        # Fixture premise: both names carry the four query bigrams at the
        # same positions, so ts_rank_cd ties (weight D, identical cover
        # geometry) and the entity_name-ASC secondary order puts the exact
        # name first: lexical leg = [乐高机器人, 乐高机器人手册].
        ranks = await storage.db.query(
            f"SELECT entity_name, ts_rank_cd({LEXICAL_TSV_COLUMN}, "
            "to_tsquery('simple', $1)) AS r "
            f"FROM {storage.table_name} WHERE workspace=$2 "
            f"AND {LEXICAL_TSV_COLUMN} @@ to_tsquery('simple', $1) "
            "ORDER BY r DESC, entity_name",
            [tokenize_for_query(_QUERY), workspace],
            multirows=True,
        )
        assert [row["entity_name"] for row in ranks] == ["乐高机器人", "乐高机器人手册"]
        assert ranks[0]["r"] == ranks[1]["r"], "ts_rank_cd tie premise broke"

        fused = await storage.hybrid_query(_QUERY, top_k=3, query_embedding=_Q_VEC)
        # Hand-computed: 乐高机器人 = 1/61 + 1/61; 无关实体 = 1/62 (vector
        # rank 2); 乐高机器人手册 = 1/62 (lexical rank 2). The 1/62 tie goes
        # to the record present in the vector leg (documented tie-break).
        assert [r["entity_name"] for r in fused] == [
            "乐高机器人",
            "无关实体",
            "乐高机器人手册",
        ]
        by_name = {r["entity_name"]: r for r in fused}
        assert by_name["乐高机器人手册"]["distance"] == 0.2
        assert by_name["乐高机器人"] == baseline[0]
    finally:
        await storage.finalize()


# ---------------------------------------------------------------------------
# Fault injection: dropped index -> degrade == vector baseline
# ---------------------------------------------------------------------------


async def test_drop_gin_degrades_to_vector_baseline_with_dedup_warning(
    lightrag_log_records,
):
    workspace = f"hybdrop_{uuid4().hex[:8]}"
    storage = _make_storage(NameSpace.VECTOR_STORE_CHUNKS, workspace)
    await storage.initialize()
    table = storage.table_name
    db = storage.db
    try:
        await _seed_chunks(storage)
        baseline = await _vector_baseline(storage, top_k=4)

        await db.execute(f"DROP INDEX IF EXISTS {_index_name(table)}")
        # The fault actually fired: the index is gone from the catalog.
        assert await _index_validity(db, table) is None

        for _ in range(3):
            degraded = await storage.hybrid_query(
                _QUERY, top_k=4, query_embedding=_Q_VEC
            )
            assert degraded == baseline
        warnings = [
            r
            for r in lightrag_log_records
            if r.levelname == "WARNING" and "degraded to vector-only" in r.getMessage()
        ]
        assert len(warnings) == 1, "degrade warning not deduplicated"
        assert "missing or invalid" in warnings[0].getMessage()

        # Rebuild through the production helper: hybrid retrieval returns.
        await PGVectorStorage._pg_ensure_lexical_index(db, table)
        healed = await storage.hybrid_query(_QUERY, top_k=4, query_embedding=_Q_VEC)
        assert [r["id"] for r in healed] == ["c-v1", "c-l1", "c-v2", "c-l2"]
        assert len(warnings) == 1

        # A new failure streak warns again (the success re-armed it).
        await db.execute(f"DROP INDEX IF EXISTS {_index_name(table)}")
        assert (
            await storage.hybrid_query(_QUERY, top_k=4, query_embedding=_Q_VEC)
            == baseline
        )
        warnings = [
            r
            for r in lightrag_log_records
            if r.levelname == "WARNING" and "degraded to vector-only" in r.getMessage()
        ]
        assert len(warnings) == 2
    finally:
        await storage.finalize()


async def test_invalid_index_residue_degrades_then_heals(lightrag_log_records):
    workspace = f"hybinv_{uuid4().hex[:8]}"
    storage = _make_storage(NameSpace.VECTOR_STORE_CHUNKS, workspace)
    await storage.initialize()
    table = storage.table_name
    db = storage.db
    index_name = _index_name(table)
    stall_conn = await asyncpg.connect(**_raw_conn_kwargs())
    cancel_conn = await asyncpg.connect(**_raw_conn_kwargs())
    try:
        await _seed_chunks(storage)
        baseline = await _vector_baseline(storage, top_k=4)
        await db.execute(f"DROP INDEX IF EXISTS {index_name}")

        # Real canceled-CIC residue: a pre-opened write transaction forces
        # the build into its wait phase, then pg_cancel_backend aborts it,
        # leaving the index INVALID (the shared fixture's stall trick).
        stall_tx = stall_conn.transaction()
        await stall_tx.start()
        await stall_conn.execute(
            f"INSERT INTO {table} (workspace, id, content, tokens, "
            "chunk_order_index, full_doc_id, file_path) "
            "VALUES ($1, 'stall-row', 'stalling writer', 1, 0, 'doc-stall', 's.txt')",
            workspace,
        )
        build_task = asyncio.create_task(
            db.execute(
                f"CREATE INDEX CONCURRENTLY {index_name} "
                f"ON {table} USING gin ({LEXICAL_TSV_COLUMN})"
            )
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
        canceled = await cancel_conn.fetchval("SELECT pg_cancel_backend($1)", pid)
        assert canceled is True, "pg_cancel_backend did not signal the builder"
        await stall_tx.commit()
        with pytest.raises(asyncpg.exceptions.QueryCanceledError):
            await asyncio.wait_for(build_task, timeout=60)

        # The residue is real: catalog says INVALID.
        validity = await _index_validity(db, table)
        assert validity is not None and validity["indisvalid"] is False

        degraded = await storage.hybrid_query(_QUERY, top_k=4, query_embedding=_Q_VEC)
        assert degraded == baseline
        assert any(
            r.levelname == "WARNING"
            and "degraded to vector-only" in r.getMessage()
            and "missing or invalid" in r.getMessage()
            for r in lightrag_log_records
        )

        # Production heal path drops the invalid index and rebuilds.
        await PGVectorStorage._pg_ensure_lexical_index(db, table)
        healed = await storage.hybrid_query(_QUERY, top_k=4, query_embedding=_Q_VEC)
        assert [r["id"] for r in healed] == ["c-v1", "c-l1", "c-v2", "c-l2"]
    finally:
        await stall_conn.close()
        await cancel_conn.close()
        await storage.finalize()


async def test_lexical_query_error_degrades_to_vector_baseline(lightrag_log_records):
    workspace = f"hyberr_{uuid4().hex[:8]}"
    storage = _make_storage(NameSpace.VECTOR_STORE_CHUNKS, workspace)
    await storage.initialize()
    db = storage.db
    original_query = db.query
    fault_calls = {"lexical": 0}

    async def faulting_query(sql, params=None, *args, **kwargs):
        if "ts_rank_cd" in sql:
            fault_calls["lexical"] += 1
            raise RuntimeError("injected live lexical failure")
        return await original_query(sql, params, *args, **kwargs)

    try:
        await _seed_chunks(storage)
        baseline = await _vector_baseline(storage, top_k=4)
        db.query = faulting_query
        try:
            degraded = await storage.hybrid_query(
                _QUERY, top_k=4, query_embedding=_Q_VEC
            )
        finally:
            db.query = original_query
        assert degraded == baseline
        assert fault_calls["lexical"] == 1, "injected fault never fired"
        assert any(
            r.levelname == "WARNING"
            and "injected live lexical failure" in r.getMessage()
            for r in lightrag_log_records
        )
    finally:
        await storage.finalize()


# ---------------------------------------------------------------------------
# reserved characters / injection strings through the LIVE lexical leg
# empty / whitespace queries (also offline in test_hybrid_fusion.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("query", "expected_ids"),
    [
        ("a & b", _VECTOR_ONLY_IDS),
        ("a | b", _VECTOR_ONLY_IDS),
        ("!a", _VECTOR_ONLY_IDS),
        ("(a)", _VECTOR_ONLY_IDS),
        ("a <-> b", _VECTOR_ONLY_IDS),
        ("it's", _VECTOR_ONLY_IDS),
        ("a\\b", _VECTOR_ONLY_IDS),
        ("a:*", _VECTOR_ONLY_IDS),
        ("'; DROP TABLE lightrag_vdb_chunks_hybridq_8d; --", _VECTOR_ONLY_IDS),
        ("a & !b | (c <-> d)", _VECTOR_ONLY_IDS),
        ("乐高 & 机器人", _LEXICAL_MIX_IDS),
        ("乐高|机器人", _LEXICAL_MIX_IDS),
        ("乐高机器人!", _LEXICAL_MIX_IDS),
        ("100% OR 1=1", _VECTOR_ONLY_IDS),
        ("<script>alert(1)</script>", _VECTOR_ONLY_IDS),
        ("乐高' AND '1'='1", _VECTOR_ONLY_IDS),
        ("", _VECTOR_ONLY_IDS),
        ("   ", _VECTOR_ONLY_IDS),
    ],
    ids=[
        "and",
        "or",
        "not",
        "parens",
        "follows",
        "quote",
        "backslash",
        "prefix",
        "drop-table",
        "mixed-operators",
        "cjk-and",
        "cjk-or",
        "cjk-not",
        "sql-or",
        "script",
        "sql-quote",
        "empty",
        "whitespace",
    ],
)
async def test_reserved_char_queries_parse_and_stay_sane(query, expected_ids):
    workspace = f"hybc1_{uuid4().hex[:8]}"
    storage = _make_storage(NameSpace.VECTOR_STORE_CHUNKS, workspace)
    await storage.initialize()
    table = storage.table_name
    try:
        await _seed_chunks(storage)
        fused = await storage.hybrid_query(query, top_k=3, query_embedding=_Q_VEC)
        assert [r["id"] for r in fused] == expected_ids
        assert all("content" in r for r in fused)
        # No injection side effect: the fixture rows are all still there.
        count = await storage.db.query(
            f"SELECT COUNT(*) AS n FROM {table} WHERE workspace=$1", [workspace]
        )
        assert count["n"] == len(_CHUNK_FIXTURE)
    finally:
        await storage.finalize()


# ---------------------------------------------------------------------------
# >1000-token queries stay bounded and complete
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_words", [1200, 6000], ids=["1200-tokens", "6000-tokens-capped"]
)
async def test_long_query_is_bounded_and_completes(n_words):
    workspace = f"hybc3_{uuid4().hex[:8]}"
    storage = _make_storage(NameSpace.VECTOR_STORE_CHUNKS, workspace)
    await storage.initialize()
    try:
        await _seed_chunks(storage)
        baseline = await _vector_baseline(storage, top_k=4)
        query = " ".join(f"term{i}" for i in range(n_words))
        assert len(tokenize(query)) == min(n_words, MAX_TOKENS)

        # No row contains every ANDed term, so the lexical leg is empty and
        # the result is the vector baseline; the wait is bounded.
        fused = await asyncio.wait_for(
            storage.hybrid_query(query, top_k=4, query_embedding=_Q_VEC),
            timeout=30,
        )
        assert fused == baseline
    finally:
        await storage.finalize()


# ---------------------------------------------------------------------------
# >=4 workers running hybrid queries while ingesting
# ---------------------------------------------------------------------------


async def test_four_workers_query_hybrid_while_ingesting():
    workspace = f"hybc5_{uuid4().hex[:8]}"
    barrier = asyncio.Barrier(4)

    async def worker(i: int) -> int:
        storage = _make_storage(NameSpace.VECTOR_STORE_CHUNKS, workspace)
        try:
            await asyncio.wait_for(barrier.wait(), timeout=30)
            await storage.initialize()
            queries = 0
            for round_no in range(5):
                await storage.upsert(
                    {
                        f"w{i}-r{round_no}-{j}": _chunk_item("乐高机器人 搭建")
                        for j in range(5)
                    }
                )
                await storage.index_done_callback()
                fused = await storage.hybrid_query(
                    _QUERY, top_k=4, query_embedding=_Q_VEC
                )
                assert len(fused) == 4
                assert all("content" in r and "id" in r for r in fused)
                queries += 1
            return queries
        finally:
            await storage.finalize()

    results = await asyncio.gather(*(worker(i) for i in range(4)))
    assert results == [5] * 4

    conn = await asyncpg.connect(**_raw_conn_kwargs())
    try:
        table = f"{_BASE_TABLES[NameSpace.VECTOR_STORE_CHUNKS]}_{_MODEL_NAME}_{_DIM}d".lower()
        total = await conn.fetchval(
            f"SELECT COUNT(*) FROM {table} WHERE workspace=$1", workspace
        )
        covered = await conn.fetchval(
            f"SELECT COUNT(*) FROM {table} WHERE workspace=$1 "
            f"AND {LEXICAL_TSV_COLUMN} IS NOT NULL",
            workspace,
        )
        assert (total, covered) == (100, 100)
        valid = await conn.fetchval(
            "SELECT i.indisvalid FROM pg_index i WHERE i.indexrelid = to_regclass($1)",
            _index_name(table),
        )
        assert valid is True
    finally:
        await conn.close()
