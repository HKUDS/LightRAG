"""Unit tests for PGDocStatusStorage database-native overrides.

Covers the PG-specific implementations of:
  * get_doc_by_file_basename
  * get_doc_by_content_hash

Both override the base-class full-table scan with indexed SQL queries.
"""

from datetime import datetime
import pytest
from unittest.mock import AsyncMock, MagicMock

from lightrag.kg.postgres_impl import PGDocStatusStorage
from lightrag.namespace import NameSpace


def _make_storage():
    storage = PGDocStatusStorage.__new__(PGDocStatusStorage)
    storage.namespace = NameSpace.DOC_STATUS
    storage.workspace = "test_ws"
    storage.global_config = {"embedding_batch_num": 10}
    storage.db = MagicMock()
    storage.db.query = AsyncMock()
    return storage


def _row(**overrides):
    base = {
        "id": "doc-1",
        "content_summary": "summary",
        "content_length": 12,
        "chunks_count": 1,
        "status": "processed",
        "file_path": "report.pdf",
        "chunks_list": "[]",
        "metadata": "{}",
        "error_msg": None,
        "track_id": None,
        "content_hash": "abc123",
        "created_at": datetime(2024, 1, 1, 0, 0, 0),
        "updated_at": datetime(2024, 1, 1, 0, 0, 0),
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# get_doc_by_file_basename
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_doc_by_file_basename_empty_returns_none():
    storage = _make_storage()
    assert await storage.get_doc_by_file_basename("") is None
    storage.db.query.assert_not_called()


@pytest.mark.asyncio
async def test_get_doc_by_file_basename_unknown_source_returns_none():
    storage = _make_storage()
    # normalize_document_file_path returns "unknown_source" for None-ish inputs
    assert await storage.get_doc_by_file_basename("unknown_source") is None
    storage.db.query.assert_not_called()


@pytest.mark.asyncio
async def test_get_doc_by_file_basename_exact_match():
    storage = _make_storage()
    storage.db.query.return_value = [_row(file_path="report.pdf")]

    result = await storage.get_doc_by_file_basename("report.pdf")

    assert result is not None
    doc_id, doc = result
    assert doc_id == "doc-1"
    assert doc["file_path"] == "report.pdf"
    assert doc["content_hash"] == "abc123"

    call = storage.db.query.call_args
    sql = call.args[0]
    params = call.args[1]
    assert "LIGHTRAG_DOC_STATUS" in sql
    assert "workspace=$1" in sql
    assert params[0] == "test_ws"
    assert params[1] == "report.pdf"


@pytest.mark.asyncio
async def test_get_doc_by_file_basename_legacy_hint_fallback_via_like():
    """Legacy rows still carrying a parser hint segment (a.[native].docx)
    must be matched when the canonical form (a.docx) is provided."""
    storage = _make_storage()
    storage.db.query.return_value = [
        _row(id="doc-legacy", file_path="report.[native].pdf")
    ]

    result = await storage.get_doc_by_file_basename("report.pdf")

    assert result is not None
    assert result[0] == "doc-legacy"

    call = storage.db.query.call_args
    sql = call.args[0]
    params = call.args[1]
    # The SQL must push the fallback to the DB rather than scan in Python
    assert "LIKE" in sql
    assert params[2] == "report.[%].pdf"


@pytest.mark.asyncio
async def test_get_doc_by_file_basename_no_match_returns_none():
    storage = _make_storage()
    storage.db.query.return_value = []
    assert await storage.get_doc_by_file_basename("missing.pdf") is None


# ---------------------------------------------------------------------------
# get_doc_by_content_hash
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_doc_by_content_hash_empty_returns_none():
    storage = _make_storage()
    assert await storage.get_doc_by_content_hash("") is None
    storage.db.query.assert_not_called()


@pytest.mark.asyncio
async def test_get_doc_by_content_hash_match():
    storage = _make_storage()
    storage.db.query.return_value = [_row(content_hash="hash-abc")]

    result = await storage.get_doc_by_content_hash("hash-abc")

    assert result is not None
    doc_id, doc = result
    assert doc_id == "doc-1"
    assert doc["content_hash"] == "hash-abc"

    call = storage.db.query.call_args
    sql = call.args[0]
    params = call.args[1]
    assert "content_hash=$2" in sql
    # Stable ordering for repeatability across re-runs / replicas
    assert "ORDER BY created_at ASC, id ASC" in sql
    assert "LIMIT 1" in sql
    assert params == ["test_ws", "hash-abc"]


@pytest.mark.asyncio
async def test_get_doc_by_content_hash_no_match_returns_none():
    storage = _make_storage()
    storage.db.query.return_value = []
    assert await storage.get_doc_by_content_hash("nope") is None
