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
import numpy as np
from unittest.mock import AsyncMock, MagicMock
from lightrag.kg.postgres_impl import PGDocStatusStorage, PGKVStorage, PGVectorStorage
from lightrag.namespace import NameSpace
from lightrag.utils import EmbeddingFunc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GLOBAL_CONFIG = {"embedding_batch_num": 10}


def make_storage(namespace: str) -> PGKVStorage:
    """Construct a PGKVStorage instance with a mocked db."""
    db = MagicMock()
    captured: list[tuple] = []
    retry_kwargs: list[dict] = []

    async def fake_run_with_retry(operation, **kwargs):
        """Call the closure with a mock connection to capture executemany args."""
        retry_kwargs.append(kwargs)
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
    storage._retry_kwargs = retry_kwargs
    return storage


def make_doc_status_storage() -> PGDocStatusStorage:
    """Construct a PGDocStatusStorage instance with a mocked db."""
    db = MagicMock()
    captured: list[tuple] = []
    retry_kwargs: list[dict] = []

    async def fake_run_with_retry(operation, **kwargs):
        retry_kwargs.append(kwargs)
        mock_conn = AsyncMock()
        tx = AsyncMock()
        tx.__aenter__.return_value = tx
        tx.__aexit__.return_value = False
        mock_conn.transaction = MagicMock(return_value=tx)
        await operation(mock_conn)
        for call in mock_conn.executemany.call_args_list:
            captured.append((call.args[0], call.args[1]))

    db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)
    db.workspace = "test_ws"

    storage = PGDocStatusStorage.__new__(PGDocStatusStorage)
    storage.namespace = NameSpace.DOC_STATUS
    storage.workspace = "test_ws"
    storage.global_config = GLOBAL_CONFIG
    storage.db = db
    storage._captured = captured
    storage._retry_kwargs = retry_kwargs
    return storage


def make_vector_storage(namespace: str) -> PGVectorStorage:
    """Construct a PGVectorStorage instance with a mocked db and embedding func."""
    db = MagicMock()
    captured: list[tuple] = []
    retry_kwargs: list[dict] = []

    async def fake_run_with_retry(operation, **kwargs):
        retry_kwargs.append(kwargs)
        mock_conn = AsyncMock()
        await operation(mock_conn)
        for call in mock_conn.executemany.call_args_list:
            captured.append((call.args[0], call.args[1]))

    db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)
    db.workspace = "test_ws"

    async def embed_func(texts, **kwargs):
        return np.array([[0.1, 0.2, 0.3] for _ in texts], dtype=np.float32)

    embedding = EmbeddingFunc(
        embedding_dim=3,
        func=embed_func,
        model_name="test_model",
    )
    storage = PGVectorStorage(
        namespace=namespace,
        workspace="test_ws",
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.5,
            },
        },
        embedding_func=embedding,
    )
    storage.db = db
    storage._captured = captured
    storage._retry_kwargs = retry_kwargs
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
            "heading": {"level": 2, "text": "Section A"},
            "sidecar": {"type": "drawing", "id": "img-1", "refs": []},
        }
    }
    await storage.upsert(data)

    assert len(storage._captured) == 1
    sql, rows = storage._captured[0]
    assert "LIGHTRAG_DOC_CHUNKS" in sql
    assert len(rows) == 1
    row = rows[0]
    # SQL: (workspace, id, tokens, chunk_order_index, full_doc_id,
    #        content, file_path, llm_cache_list, heading, sidecar,
    #        create_time, update_time)
    assert row[0] == "test_ws"  # workspace
    assert row[1] == "chunk-1"  # id
    assert row[2] == 42  # tokens
    assert row[3] == 0  # chunk_order_index
    assert row[4] == "doc-1"  # full_doc_id
    assert row[5] == "hello world"  # content
    assert row[6] == "/a/b.txt"  # file_path
    assert json.loads(row[7]) == ["cache-key"]  # llm_cache_list
    assert json.loads(row[8]) == {"level": 2, "text": "Section A"}  # heading
    assert json.loads(row[9]) == {
        "type": "drawing",
        "id": "img-1",
        "refs": [],
    }  # sidecar


@pytest.mark.asyncio
async def test_upsert_text_chunks_missing_heading_sidecar_defaults_to_empty_dict():
    """Plain-text chunks without heading/sidecar should serialize to '{}'."""
    storage = make_storage(NameSpace.KV_STORE_TEXT_CHUNKS)
    data = {
        "chunk-1": {
            "tokens": 10,
            "chunk_order_index": 0,
            "full_doc_id": "doc-1",
            "content": "plain text",
            "file_path": "/a/b.txt",
        }
    }
    await storage.upsert(data)

    _, rows = storage._captured[0]
    row = rows[0]
    assert json.loads(row[8]) == {}  # heading
    assert json.loads(row[9]) == {}  # sidecar


@pytest.mark.asyncio
async def test_upsert_text_chunks_none_heading_sidecar_defaults_to_empty_dict():
    """Explicit None values should be coerced to '{}' to avoid type errors."""
    storage = make_storage(NameSpace.KV_STORE_TEXT_CHUNKS)
    data = {
        "chunk-1": {
            "tokens": 10,
            "chunk_order_index": 0,
            "full_doc_id": "doc-1",
            "content": "plain text",
            "file_path": "/a/b.txt",
            "heading": None,
            "sidecar": None,
        }
    }
    await storage.upsert(data)

    _, rows = storage._captured[0]
    row = rows[0]
    assert json.loads(row[8]) == {}
    assert json.loads(row[9]) == {}


# ---------------------------------------------------------------------------
# 3. Namespace: FULL_DOCS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_full_docs_tuple_order():
    storage = make_storage(NameSpace.KV_STORE_FULL_DOCS)
    data = {
        "doc-1": {
            "content": "full text",
            "file_path": "/path/doc.pdf",
            "sidecar_location": "lightrag://sidecar/doc-1",
            "parse_format": "lightrag",
            "content_hash": "deadbeef",
            "process_options": "Fi",
            "chunk_options": {"chunk_token_size": 1200, "chunk_overlap": 100},
            "parse_engine": "mineru",
        }
    }
    await storage.upsert(data)

    assert len(storage._captured) == 1
    _, rows = storage._captured[0]
    row = rows[0]
    # SQL: (id, content, doc_name, workspace, sidecar_location, parse_format,
    #       content_hash, process_options, chunk_options, parse_engine)
    assert row[0] == "doc-1"
    assert row[1] == "full text"
    assert row[2] == "/path/doc.pdf"
    assert row[3] == "test_ws"
    assert row[4] == "lightrag://sidecar/doc-1"
    assert row[5] == "lightrag"
    assert row[6] == "deadbeef"
    assert row[7] == "Fi"
    assert json.loads(row[8]) == {"chunk_token_size": 1200, "chunk_overlap": 100}
    assert row[9] == "mineru"


@pytest.mark.asyncio
async def test_upsert_full_docs_missing_pipeline_fields_pass_through_as_none():
    """Missing pipeline-derived fields must serialize as None at the Python
    layer so the SQL-level COALESCE guard can distinguish "caller did not
    supply" from "caller supplied a real value".

    The 'raw' default for parse_format is provided by the column DDL on
    initial insert; the Python layer must NOT inject it, otherwise the
    COALESCE guard never triggers on subsequent partial writes (a follow-up
    upsert with no parse_format would re-stamp the column with 'raw' and
    blow away a previously-set 'lightrag').
    """
    storage = make_storage(NameSpace.KV_STORE_FULL_DOCS)
    data = {"doc-1": {"content": "full text", "file_path": "/path/doc.pdf"}}
    await storage.upsert(data)

    _, rows = storage._captured[0]
    row = rows[0]
    assert row[4] is None  # sidecar_location
    assert row[5] is None  # parse_format — DDL supplies 'raw' default on insert
    assert row[6] is None  # content_hash
    assert row[7] is None  # process_options
    assert json.loads(row[8]) == {}  # chunk_options default
    assert row[9] is None  # parse_engine


@pytest.mark.asyncio
async def test_upsert_full_docs_none_chunk_options_defaults_to_empty_dict():
    storage = make_storage(NameSpace.KV_STORE_FULL_DOCS)
    data = {
        "doc-1": {
            "content": "full text",
            "file_path": "/path/doc.pdf",
            "chunk_options": None,
        }
    }
    await storage.upsert(data)

    _, rows = storage._captured[0]
    assert json.loads(rows[0][8]) == {}


@pytest.mark.asyncio
async def test_upsert_full_docs_sql_protects_partial_writes():
    """The ON CONFLICT clause must COALESCE+NULLIF every pipeline-derived
    column so a follow-up upsert that only carries ``content`` + ``doc_name``
    does not silently overwrite previously-recorded metadata back to defaults.

    We assert this at the SQL-template level since the actual COALESCE
    behavior is executed by Postgres. The presence of the protective
    expression in the SQL is the single source of truth for the guarantee.
    """
    storage = make_storage(NameSpace.KV_STORE_FULL_DOCS)
    await storage.upsert(
        {"doc-1": {"content": "full text", "file_path": "/path/doc.pdf"}}
    )
    sql, _ = storage._captured[0]
    normalized = " ".join(sql.split()).lower()

    # Each pipeline-derived string column must be COALESCE/NULLIF-guarded
    for col in (
        "sidecar_location",
        "parse_format",
        "content_hash",
        "process_options",
        "parse_engine",
    ):
        assert (
            f"coalesce( nullif(excluded.{col}, '')" in normalized
        ), f"upsert_doc_full must guard {col} via COALESCE+NULLIF"
        assert (
            f"lightrag_doc_full.{col}" in normalized
        ), f"upsert_doc_full must preserve existing {col} on partial write"

    # chunk_options (JSONB) is guarded via CASE on NULL/empty-object literal
    assert "excluded.chunk_options is null" in normalized
    assert "excluded.chunk_options = '{}'::jsonb" in normalized
    assert "lightrag_doc_full.chunk_options" in normalized

    # content / doc_name remain straight overwrites — they ARE the payload
    assert "content = excluded.content" in normalized
    assert "doc_name = excluded.doc_name" in normalized


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


@pytest.mark.asyncio
async def test_kv_upsert_passes_timing_label():
    storage = make_storage(NameSpace.KV_STORE_FULL_DOCS)
    await storage.upsert({"doc-1": {"content": "text 1", "file_path": "/a"}})

    assert storage._retry_kwargs[0]["timing_label"] == (
        f"test_ws PGKVStorage.upsert[{NameSpace.KV_STORE_FULL_DOCS}]"
    )


@pytest.mark.asyncio
async def test_doc_status_upsert_passes_timing_label():
    storage = make_doc_status_storage()
    await storage.upsert(
        {
            "doc-1": {
                "content_summary": "summary",
                "content_length": 12,
                "chunks_count": 1,
                "status": "processed",
                "file_path": "/a.txt",
                "chunks_list": ["chunk-1"],
                "metadata": {"source": "test"},
                "created_at": "2024-01-01T00:00:00+00:00",
                "updated_at": "2024-01-01T00:00:00+00:00",
            }
        }
    )

    assert storage._retry_kwargs[0]["timing_label"] == (
        "test_ws PGDocStatusStorage.upsert"
    )


# ---------------------------------------------------------------------------
# doc_status: content_hash tuple + COALESCE SQL guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_doc_status_upsert_includes_content_hash():
    storage = make_doc_status_storage()
    await storage.upsert(
        {
            "doc-1": {
                "content_summary": "summary",
                "content_length": 12,
                "chunks_count": 1,
                "status": "processed",
                "file_path": "/a.txt",
                "chunks_list": ["chunk-1"],
                "metadata": {"source": "test"},
                "content_hash": "abc123",
                "created_at": "2024-01-01T00:00:00+00:00",
                "updated_at": "2024-01-01T00:00:00+00:00",
            }
        }
    )

    sql, rows = storage._captured[0]
    # content_hash should be present in the INSERT column list and tuple
    assert "content_hash" in sql
    row = rows[0]
    # Tuple layout: workspace, id, content_summary, content_length, chunks_count,
    # status, file_path, chunks_list, track_id, metadata, error_msg,
    # content_hash, created_at, updated_at
    assert row[11] == "abc123"


@pytest.mark.asyncio
async def test_doc_status_upsert_missing_content_hash_is_none():
    """Existing callers that do not pass content_hash still produce valid tuples."""
    storage = make_doc_status_storage()
    await storage.upsert(
        {
            "doc-1": {
                "content_summary": "summary",
                "content_length": 12,
                "chunks_count": 1,
                "status": "processed",
                "file_path": "/a.txt",
                "chunks_list": ["chunk-1"],
                "metadata": {"source": "test"},
                "created_at": "2024-01-01T00:00:00+00:00",
                "updated_at": "2024-01-01T00:00:00+00:00",
            }
        }
    )

    _, rows = storage._captured[0]
    assert rows[0][11] is None


@pytest.mark.asyncio
async def test_doc_status_upsert_sql_protects_existing_content_hash():
    """The ON CONFLICT clause must COALESCE+NULLIF to preserve a previously
    set content_hash when a subsequent state-transition upsert carries no
    hash (None) or an empty string.

    We assert this at the SQL-template level since the actual COALESCE
    behavior is executed by Postgres. The presence of the protective
    expression in the SQL is the single source of truth for the guarantee.
    """
    storage = make_doc_status_storage()
    await storage.upsert(
        {
            "doc-1": {
                "content_summary": "summary",
                "content_length": 12,
                "chunks_count": 1,
                "status": "processed",
                "file_path": "/a.txt",
                "chunks_list": [],
                "metadata": {},
                "created_at": "2024-01-01T00:00:00+00:00",
                "updated_at": "2024-01-01T00:00:00+00:00",
            }
        }
    )

    sql, _ = storage._captured[0]
    normalized = " ".join(sql.split()).lower()
    assert "coalesce(" in normalized
    assert "nullif(excluded.content_hash, '')" in normalized
    assert "lightrag_doc_status.content_hash" in normalized


@pytest.mark.asyncio
async def test_vector_upsert_passes_timing_label():
    storage = make_vector_storage(NameSpace.VECTOR_STORE_CHUNKS)
    await storage.upsert(
        {
            "chunk-1": {
                "tokens": 42,
                "chunk_order_index": 0,
                "full_doc_id": "doc-1",
                "content": "hello world",
                "file_path": "/a/b.txt",
            }
        }
    )

    assert storage._retry_kwargs[0]["timing_label"] == (
        f"test_ws PGVectorStorage.upsert[{NameSpace.VECTOR_STORE_CHUNKS}]"
    )
