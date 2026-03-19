"""
Unit tests for PGKVStorage.upsert batch optimization (PR #2742 fixes).

Verifies:
1. Each namespace builds correct tuple ordering matching SQL positional params.
2. _run_with_retry is used (not the removed PostgreSQLDB.executemany wrapper).
3. Sub-batching splits data when len(data) > _max_batch_size.
4. Unknown namespace raises ValueError.
5. Empty data returns without any DB call.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from lightrag.kg.postgres_impl import PGKVStorage
from lightrag.namespace import NameSpace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GLOBAL_CONFIG = {"embedding_batch_num": 10}


def make_storage(namespace: str) -> PGKVStorage:
    """Construct a PGKVStorage instance with a mocked db."""
    db = MagicMock()
    captured: list[tuple] = []

    async def fake_run_with_retry(operation, **kwargs):
        """Call the closure with a mock connection to capture executemany args."""
        mock_conn = AsyncMock()
        await operation(mock_conn)
        # Store (sql, data) from each executemany call
        for call in mock_conn.executemany.call_args_list:
            captured.append((call.args[0], call.args[1]))

    db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)
    db.workspace = "test_ws"

    storage = PGKVStorage.__new__(PGKVStorage)
    storage.namespace = namespace
    storage.workspace = "test_ws"
    storage.global_config = GLOBAL_CONFIG
    storage.db = db
    storage.__post_init__()

    storage._captured = captured
    return storage


# ---------------------------------------------------------------------------
# 1. _max_batch_size is always 200 (not embedding_batch_num)
# ---------------------------------------------------------------------------


def test_max_batch_size_is_constant():
    storage = make_storage(NameSpace.KV_STORE_TEXT_CHUNKS)
    assert storage._max_batch_size == 200


# ---------------------------------------------------------------------------
# 2. Namespace: TEXT_CHUNKS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_text_chunks_tuple_order():
    storage = make_storage(NameSpace.KV_STORE_TEXT_CHUNKS)
    data = {
        "chunk-1": {
            "tokens": 42,
            "chunk_order_index": 0,
            "full_doc_id": "doc-1",
            "content": "hello world",
            "file_path": "/a/b.txt",
            "llm_cache_list": ["cache-key"],
        }
    }
    await storage.upsert(data)

    assert len(storage._captured) == 1
    sql, rows = storage._captured[0]
    assert "LIGHTRAG_DOC_CHUNKS" in sql
    assert len(rows) == 1
    row = rows[0]
    # SQL: (workspace, id, tokens, chunk_order_index, full_doc_id,
    #        content, file_path, llm_cache_list, create_time, update_time)
    assert row[0] == "test_ws"  # workspace
    assert row[1] == "chunk-1"  # id
    assert row[2] == 42  # tokens
    assert row[3] == 0  # chunk_order_index
    assert row[4] == "doc-1"  # full_doc_id
    assert row[5] == "hello world"  # content
    assert row[6] == "/a/b.txt"  # file_path
    assert json.loads(row[7]) == ["cache-key"]  # llm_cache_list


# ---------------------------------------------------------------------------
# 3. Namespace: FULL_DOCS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_full_docs_tuple_order():
    storage = make_storage(NameSpace.KV_STORE_FULL_DOCS)
    data = {"doc-1": {"content": "full text", "file_path": "/path/doc.pdf"}}
    await storage.upsert(data)

    assert len(storage._captured) == 1
    _, rows = storage._captured[0]
    row = rows[0]
    # SQL: (id, content, doc_name, workspace)
    assert row[0] == "doc-1"
    assert row[1] == "full text"
    assert row[2] == "/path/doc.pdf"
    assert row[3] == "test_ws"


# ---------------------------------------------------------------------------
# 4. Namespace: LLM_RESPONSE_CACHE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_llm_cache_tuple_order():
    storage = make_storage(NameSpace.KV_STORE_LLM_RESPONSE_CACHE)
    data = {
        "key-1": {
            "original_prompt": "what is X?",
            "return": "X is Y",
            "chunk_id": "chunk-1",
            "cache_type": "query",
            "queryparam": {"mode": "hybrid"},
        }
    }
    await storage.upsert(data)

    assert len(storage._captured) == 1
    _, rows = storage._captured[0]
    row = rows[0]
    # SQL: (workspace, id, original_prompt, return_value, chunk_id, cache_type, queryparam)
    assert row[0] == "test_ws"
    assert row[1] == "key-1"
    assert row[2] == "what is X?"
    assert row[3] == "X is Y"
    assert row[4] == "chunk-1"
    assert row[5] == "query"
    assert json.loads(row[6]) == {"mode": "hybrid"}


@pytest.mark.asyncio
async def test_upsert_llm_cache_null_queryparam():
    storage = make_storage(NameSpace.KV_STORE_LLM_RESPONSE_CACHE)
    data = {
        "key-2": {
            "original_prompt": "prompt",
            "return": "answer",
            "cache_type": "extract",
        }
    }
    await storage.upsert(data)
    _, rows = storage._captured[0]
    assert rows[0][6] is None  # queryparam should be None


# ---------------------------------------------------------------------------
# 5. Namespace: FULL_ENTITIES
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_full_entities_tuple_order():
    storage = make_storage(NameSpace.KV_STORE_FULL_ENTITIES)
    data = {"ent-1": {"entity_names": ["EntityA", "EntityB"], "count": 2}}
    await storage.upsert(data)

    _, rows = storage._captured[0]
    row = rows[0]
    # SQL: (workspace, id, entity_names, count, create_time, update_time)
    assert row[0] == "test_ws"
    assert row[1] == "ent-1"
    assert json.loads(row[2]) == ["EntityA", "EntityB"]
    assert row[3] == 2


# ---------------------------------------------------------------------------
# 6. Namespace: FULL_RELATIONS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_full_relations_tuple_order():
    storage = make_storage(NameSpace.KV_STORE_FULL_RELATIONS)
    data = {"rel-1": {"relation_pairs": [["A", "B"]], "count": 1}}
    await storage.upsert(data)

    _, rows = storage._captured[0]
    row = rows[0]
    # SQL: (workspace, id, relation_pairs, count, create_time, update_time)
    assert row[0] == "test_ws"
    assert row[1] == "rel-1"
    assert json.loads(row[2]) == [["A", "B"]]
    assert row[3] == 1


# ---------------------------------------------------------------------------
# 7. Namespace: ENTITY_CHUNKS / RELATION_CHUNKS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_entity_chunks_tuple_order():
    storage = make_storage(NameSpace.KV_STORE_ENTITY_CHUNKS)
    data = {"ec-1": {"chunk_ids": ["c1", "c2"], "count": 2}}
    await storage.upsert(data)

    _, rows = storage._captured[0]
    row = rows[0]
    # SQL: (workspace, id, chunk_ids, count, create_time, update_time)
    assert row[0] == "test_ws"
    assert row[1] == "ec-1"
    assert json.loads(row[2]) == ["c1", "c2"]
    assert row[3] == 2


@pytest.mark.asyncio
async def test_upsert_relation_chunks_tuple_order():
    storage = make_storage(NameSpace.KV_STORE_RELATION_CHUNKS)
    data = {"rc-1": {"chunk_ids": ["c3"], "count": 1}}
    await storage.upsert(data)

    _, rows = storage._captured[0]
    row = rows[0]
    assert row[0] == "test_ws"
    assert row[1] == "rc-1"
    assert json.loads(row[2]) == ["c3"]
    assert row[3] == 1


# ---------------------------------------------------------------------------
# 8. Sub-batching: data > _max_batch_size splits into multiple _run_with_retry calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sub_batching_splits_correctly():
    storage = make_storage(NameSpace.KV_STORE_FULL_DOCS)
    storage._max_batch_size = 3  # Override to small value for testing

    data = {f"doc-{i}": {"content": f"text {i}", "file_path": ""} for i in range(7)}
    await storage.upsert(data)

    # 7 records / batch_size 3 => 3 batches (3 + 3 + 1)
    assert len(storage._captured) == 3
    assert len(storage._captured[0][1]) == 3
    assert len(storage._captured[1][1]) == 3
    assert len(storage._captured[2][1]) == 1


@pytest.mark.asyncio
async def test_sub_batching_exact_multiple():
    storage = make_storage(NameSpace.KV_STORE_FULL_DOCS)
    storage._max_batch_size = 3

    data = {f"doc-{i}": {"content": f"text {i}", "file_path": ""} for i in range(6)}
    await storage.upsert(data)

    # 6 / 3 => exactly 2 batches
    assert len(storage._captured) == 2
    assert len(storage._captured[0][1]) == 3
    assert len(storage._captured[1][1]) == 3


# ---------------------------------------------------------------------------
# 9. Empty data: no DB call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_empty_data_no_db_call():
    storage = make_storage(NameSpace.KV_STORE_FULL_DOCS)
    await storage.upsert({})
    assert len(storage._captured) == 0
    storage.db._run_with_retry.assert_not_called()


# ---------------------------------------------------------------------------
# 10. Unknown namespace raises ValueError
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_unknown_namespace_raises():
    storage = make_storage("unknown_namespace")
    with pytest.raises(ValueError, match="Unknown namespace"):
        await storage.upsert({"k": {"v": 1}})


# ---------------------------------------------------------------------------
# 11. Multiple records go into one batch when within limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_multiple_records_single_batch():
    storage = make_storage(NameSpace.KV_STORE_FULL_DOCS)
    data = {
        "doc-1": {"content": "text 1", "file_path": "/a"},
        "doc-2": {"content": "text 2", "file_path": "/b"},
        "doc-3": {"content": "text 3", "file_path": "/c"},
    }
    await storage.upsert(data)

    # All 3 fit within default batch size of 200
    assert len(storage._captured) == 1
    _, rows = storage._captured[0]
    assert len(rows) == 3
    ids = {row[0] for row in rows}  # id is $1 for FULL_DOCS
    assert ids == {"doc-1", "doc-2", "doc-3"}
