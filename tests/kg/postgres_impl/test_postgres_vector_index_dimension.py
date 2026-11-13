"""Vector index dimension limits and fresh-database table probing.

Regression for two first-run problems with the default configuration
(``text-embedding-3-large``, 3072 dimensions, plus ``POSTGRES_VECTOR_INDEX_TYPE=HNSW``):

- pgvector cannot build an HNSW / IVFFlat index above 2000 dimensions. The
  CREATE INDEX failure used to be logged and swallowed, so storage
  initialization succeeded and every vector table was served by a
  sequential scan. Index creation also drops the other index types first, so
  switching an existing HNSW_HALFVEC table to HNSW left it with no index.
  The dimension is now checked before any DDL and refused loudly.
- ``check_tables`` probed each table with ``SELECT 1 FROM <table>``; on a
  fresh database every probe failed through ``query()``, which logs an ERROR,
  right before the table was created.
"""

from unittest.mock import AsyncMock

import pytest

import lightrag.kg.postgres_impl as pg
from lightrag.kg.postgres_impl import (
    PGVectorStorage,
    PostgreSQLDB,
    _check_vector_index_dimension,
)

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    "index_type, dim",
    [
        ("HNSW", 2000),
        ("IVFFLAT", 2000),
        ("HNSW_HALFVEC", 3072),
        ("HNSW_HALFVEC", 4000),
        ("VCHORDRQ", 8192),
        (None, 8192),
        ("", 8192),
    ],
)
def test_supported_dimension_is_accepted(index_type, dim):
    _check_vector_index_dimension(index_type, dim)


@pytest.mark.parametrize(
    "index_type, dim, suggestion",
    [
        ("HNSW", 3072, "HNSW_HALFVEC"),
        ("IVFFLAT", 2001, "HNSW_HALFVEC"),
        ("ivfflat", 3072, "HNSW_HALFVEC"),
        ("HNSW_HALFVEC", 4096, "VCHORDRQ"),
    ],
)
def test_oversized_dimension_is_refused_with_a_remedy(index_type, dim, suggestion):
    with pytest.raises(ValueError, match=str(dim)) as exc:
        _check_vector_index_dimension(index_type, dim)
    assert suggestion in str(exc.value)


@pytest.mark.parametrize("index_type", ["HNSW", "IVFFLAT"])
def test_remedy_follows_the_dimension_not_the_refused_type(index_type):
    """Above 4000 dimensions HNSW_HALFVEC refuses too, so it must not be the
    advice for an HNSW / IVFFlat deployment that is that large."""
    with pytest.raises(ValueError) as exc:
        _check_vector_index_dimension(index_type, 4096)
    message = str(exc.value)
    assert "Set POSTGRES_VECTOR_INDEX_TYPE=HNSW_HALFVEC" not in message
    assert "VCHORDRQ" in message


def test_halfvec_remedy_discloses_the_lossy_conversion():
    """Switching an existing table to HNSW_HALFVEC rewrites stored vectors as
    float16 (ALTER COLUMN ... TYPE HALFVEC); the advice must not claim that
    only the index changes."""
    with pytest.raises(ValueError) as exc:
        _check_vector_index_dimension("HNSW", 3072)
    message = str(exc.value)
    assert "16-bit" in message
    assert "without touching stored vectors" not in message


def _bare_db(index_type: str) -> PostgreSQLDB:
    db = PostgreSQLDB.__new__(PostgreSQLDB)
    db.vector_index_type = index_type
    db.hnsw_m = 16
    db.hnsw_ef = 200
    db.ivfflat_lists = 100
    db.vchordrq_build_options = ""
    db.query = AsyncMock(return_value=None)
    db.execute = AsyncMock()
    db.check_table_exists = AsyncMock(return_value=True)
    return db


@pytest.mark.asyncio
async def test_create_vector_index_refuses_before_any_ddl():
    db = _bare_db("HNSW")

    with pytest.raises(ValueError, match="HNSW_HALFVEC"):
        await db._create_vector_index("LIGHTRAG_VDB_CHUNKS_model_3072d", 3072)

    # No index lookup, no DROP of the other index types, no ALTER COLUMN.
    db.query.assert_not_awaited()
    db.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_vector_index_still_builds_a_supported_index():
    db = _bare_db("HNSW_HALFVEC")

    await db._create_vector_index("LIGHTRAG_VDB_CHUNKS_model_3072d", 3072)

    executed = " ".join(call.args[0] for call in db.execute.await_args_list)
    assert "HALFVEC(3072)" in executed
    assert "halfvec_cosine_ops" in executed


@pytest.mark.asyncio
async def test_setup_table_refuses_before_creating_the_table():
    db = _bare_db("HNSW")

    with pytest.raises(ValueError, match="at most 2000"):
        await PGVectorStorage.setup_table(
            db,
            "LIGHTRAG_VDB_CHUNKS_model_3072d",
            "default",
            embedding_dim=3072,
            legacy_table_name="LIGHTRAG_VDB_CHUNKS",
            base_table="LIGHTRAG_VDB_CHUNKS",
        )

    db.check_table_exists.assert_not_awaited()
    db.execute.assert_not_awaited()


@pytest.mark.asyncio
async def test_check_tables_probes_existence_without_failing_queries(monkeypatch):
    monkeypatch.setattr(
        pg,
        "TABLES",
        {
            "LIGHTRAG_PRESENT": {"ddl": "CREATE TABLE LIGHTRAG_PRESENT (id int)"},
            "LIGHTRAG_MISSING": {"ddl": "CREATE TABLE LIGHTRAG_MISSING (id int)"},
        },
    )
    db = _bare_db("HNSW_HALFVEC")
    db.check_table_exists = AsyncMock(side_effect=lambda t: t == "LIGHTRAG_PRESENT")
    db.query = AsyncMock(return_value=[])
    # The migrations after the table loop are out of scope here.
    for name in dir(PostgreSQLDB):
        if name.startswith(("_migrate", "_check_llm_cache_needs_migration")):
            monkeypatch.setattr(db, name, AsyncMock(return_value=False), raising=False)

    await db.check_tables()

    probed = [call.args[0] for call in db.check_table_exists.await_args_list]
    assert probed[:2] == ["LIGHTRAG_PRESENT", "LIGHTRAG_MISSING"]
    created = [call.args[0] for call in db.execute.await_args_list]
    assert "CREATE TABLE LIGHTRAG_MISSING (id int)" in created
    assert "CREATE TABLE LIGHTRAG_PRESENT (id int)" not in created
    assert not any(
        str(call.args[0]).lstrip().upper().startswith("SELECT 1 FROM")
        for call in db.query.await_args_list
    )
