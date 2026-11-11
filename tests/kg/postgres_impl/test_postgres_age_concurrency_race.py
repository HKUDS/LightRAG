"""Tests for Apache AGE concurrency race transient retry handling in PGGraphStorage.

Issue #4211: Under concurrent document ingestion, Apache AGE catalog lookups
can race during graph writes (DDL/DML) and produce corrupted relation-not-found
messages:
- Shape 1 (undecodable / invalid UTF-8 bytes): Python raises UnicodeDecodeError
  while decoding the error packet from PostgreSQL, with raw bytes containing
  b'relation "' and b'does not exist'.
- Shape 2 (valid UTF-8): asyncpg raises PostgresError (such as UndefinedTableError)
  quoting the corrupted relation name (`relation "<graph>.<garbage>" does not exist`).

These transient errors should be recognized by _is_transient_graph_write_error
and retried by the @retry loop on graph upserts rather than failing document ingestion.
"""

from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest
from tenacity import wait_none

from lightrag.kg.postgres_impl import (
    PGGraphQueryException,
    PGGraphStorage,
    _is_age_concurrency_race,
    _is_transient_graph_write_error,
)


class _FakeTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeConnection:
    def __init__(self, edge_rows: list | None = None):
        self.calls: list[dict] = []
        self._edge_rows = [{"r": "edge"}] if edge_rows is None else edge_rows

    def transaction(self):
        return _FakeTransaction()

    async def execute(self, sql, *args):
        self.calls.append({"sql": sql, "args": args})
        return ""

    async def fetch(self, sql, *args):
        self.calls.append({"sql": sql, "args": args})
        return self._edge_rows


def _make_graph_storage(workspace="test_ws", graph_name="test_graph"):
    """Instantiate a minimal PGGraphStorage double without connecting to PostgreSQL."""
    storage = PGGraphStorage.__new__(PGGraphStorage)
    storage.workspace = workspace
    storage.namespace = "test_ns"
    storage.graph_name = graph_name
    storage._entity_type_cache = set()
    storage._relation_type_cache = set()
    storage.__post_init__()
    storage.db = MagicMock()
    return storage


class TestIsAgeConcurrencyRacePredicate:
    """Direct unit tests for the _is_age_concurrency_race helper and

    _is_transient_graph_write_error predicate.
    """

    def test_unicode_decode_error_with_corrupted_relation_not_found(self):
        # Shape 1: raw bytes contain relation " and does not exist with invalid UTF-8 bytes
        raw_bytes = b'relation "testgraph.\x9c\xff\xfe" does not exist'
        exc = UnicodeDecodeError("utf-8", raw_bytes, 20, 21, "invalid start byte")

        assert _is_age_concurrency_race(exc) is True

        wrapped = PGGraphQueryException({"message": "test"})
        wrapped.__cause__ = exc
        assert _is_transient_graph_write_error(wrapped) is True

    def test_unicode_decode_error_chained_under_another_exception(self):
        # When asyncpg or Python wraps the UnicodeDecodeError in another exception
        raw_bytes = b'relation "mygraph.\x80\x81" does not exist'
        inner = UnicodeDecodeError("utf-8", raw_bytes, 18, 19, "invalid start byte")
        outer = asyncpg.InterfaceError("cannot decode message")
        outer.__cause__ = inner

        assert _is_age_concurrency_race(outer) is True

        wrapped = PGGraphQueryException({"message": "test"})
        wrapped.__cause__ = outer
        assert _is_transient_graph_write_error(wrapped) is True

    def test_undefined_table_error_with_corrupted_relation_name(self):
        # Shape 2: valid UTF-8, asyncpg raises UndefinedTableError
        exc = asyncpg.exceptions.UndefinedTableError(
            'relation "testgraph._ag_label_vertex_corrupted" does not exist'
        )

        assert _is_age_concurrency_race(exc) is True

        wrapped = PGGraphQueryException({"message": "test"})
        wrapped.__cause__ = exc
        assert _is_transient_graph_write_error(wrapped) is True

    def test_generic_postgres_error_with_corrupted_relation_name(self):
        # Shape 2 variant: general PostgresError quoting corrupted relation name
        exc = asyncpg.exceptions.PostgresError('relation "testgraph.\ufffd\ufffd" does not exist')

        assert _is_age_concurrency_race(exc) is True

        wrapped = PGGraphQueryException({"message": "test"})
        wrapped.__cause__ = exc
        assert _is_transient_graph_write_error(wrapped) is True

    def test_unrelated_unicode_decode_error_is_rejected(self):
        # UnicodeDecodeError on non-relation payload must NOT be classified as AGE race
        raw_bytes = b'{"key": "some invalid \x9c byte"}'
        exc = UnicodeDecodeError("utf-8", raw_bytes, 22, 23, "invalid start byte")

        assert _is_age_concurrency_race(exc) is False

        wrapped = PGGraphQueryException({"message": "test"})
        wrapped.__cause__ = exc
        assert _is_transient_graph_write_error(wrapped) is False

    def test_unrelated_postgres_error_is_rejected(self):
        # Non-relation Postgres errors (e.g. syntax errors) must NOT be retried
        exc = asyncpg.exceptions.PostgresSyntaxError('syntax error at or near "SELECT"')

        assert _is_age_concurrency_race(exc) is False

        wrapped = PGGraphQueryException({"message": "test"})
        wrapped.__cause__ = exc
        assert _is_transient_graph_write_error(wrapped) is False

    def test_non_pggraphquery_exception_is_rejected(self):
        # Predicate requires top-level PGGraphQueryException
        exc = UnicodeDecodeError(
            "utf-8", b'relation "foo" does not exist', 0, 1, "test"
        )
        assert _is_transient_graph_write_error(exc) is False

    def test_none_cause_is_rejected(self):
        wrapped = PGGraphQueryException({"message": "test"})
        assert _is_transient_graph_write_error(wrapped) is False


class TestPGGraphStorageAgeRaceRetries:
    """Test that PGGraphStorage write operations retry transient AGE concurrency races."""

    @pytest.mark.asyncio
    async def test_upsert_edge_retries_on_undecodable_relation_race(self, monkeypatch):
        monkeypatch.setattr(PGGraphStorage.upsert_edge.retry, "wait", wait_none())

        storage = _make_graph_storage()
        raw_bytes = b'relation "testgraph.\x9c\xff" does not exist'
        corrupt_exc = UnicodeDecodeError("utf-8", raw_bytes, 20, 21, "invalid start byte")

        call_count = 0

        async def fake_run_with_retry(_operation, **_kwargs):
            nonlocal call_count
            call_count += 1
            raise corrupt_exc

        storage.db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)

        with pytest.raises(PGGraphQueryException) as excinfo:
            await storage.upsert_edge("Alice", "Bob", {"weight": "1.0"})

        assert excinfo.value.__cause__ is corrupt_exc
        assert _is_transient_graph_write_error(excinfo.value) is True
        # Proves @retry fired all 3 attempts
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_upsert_edge_retries_on_undefined_table_race(self, monkeypatch):
        monkeypatch.setattr(PGGraphStorage.upsert_edge.retry, "wait", wait_none())

        storage = _make_graph_storage()
        corrupt_exc = asyncpg.exceptions.UndefinedTableError(
            'relation "testgraph.corrupted_edge_label" does not exist'
        )

        call_count = 0

        async def fake_run_with_retry(_operation, **_kwargs):
            nonlocal call_count
            call_count += 1
            raise corrupt_exc

        storage.db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)

        with pytest.raises(PGGraphQueryException) as excinfo:
            await storage.upsert_edge("Alice", "Bob", {"weight": "1.0"})

        assert excinfo.value.__cause__ is corrupt_exc
        assert _is_transient_graph_write_error(excinfo.value) is True
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_upsert_edge_recovers_after_transient_age_race(self, monkeypatch):
        """Simulate transient failure on attempt 1, followed by success on attempt 2."""
        monkeypatch.setattr(PGGraphStorage.upsert_edge.retry, "wait", wait_none())

        storage = _make_graph_storage()
        raw_bytes = b'relation "testgraph.\x9c" does not exist'
        corrupt_exc = UnicodeDecodeError("utf-8", raw_bytes, 20, 21, "invalid start byte")

        call_count = 0
        conn = _FakeConnection()

        async def fake_run_with_retry(operation, **_kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise corrupt_exc
            return await operation(conn)

        storage.db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)

        # Must succeed without raising
        await storage.upsert_edge("Alice", "Bob", {"weight": "1.0"})
        assert call_count == 2
        assert len(conn.calls) > 0

    @pytest.mark.asyncio
    async def test_upsert_node_retries_on_age_concurrency_race(self, monkeypatch):
        monkeypatch.setattr(PGGraphStorage.upsert_node.retry, "wait", wait_none())

        storage = _make_graph_storage()
        raw_bytes = b'relation "testgraph.\x9c" does not exist'
        corrupt_exc = UnicodeDecodeError("utf-8", raw_bytes, 20, 21, "invalid start byte")

        call_count = 0

        async def fake_query(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            wrapped = PGGraphQueryException({"message": "test"})
            wrapped.__cause__ = corrupt_exc
            raise wrapped

        storage._query = AsyncMock(side_effect=fake_query)

        with pytest.raises(PGGraphQueryException) as excinfo:
            await storage.upsert_node("Alice", {"entity_id": "Alice", "entity_type": "Person"})

        assert excinfo.value.__cause__ is corrupt_exc
        assert _is_transient_graph_write_error(excinfo.value) is True
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_upsert_node_chunk_retries_on_age_race(self, monkeypatch):
        monkeypatch.setattr(PGGraphStorage._upsert_node_chunk.retry, "wait", wait_none())

        storage = _make_graph_storage()
        corrupt_exc = asyncpg.exceptions.UndefinedTableError(
            'relation "testgraph._ag_label_vertex" does not exist'
        )

        call_count = 0

        async def fake_run_with_retry(_operation, **_kwargs):
            nonlocal call_count
            call_count += 1
            raise corrupt_exc

        storage.db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)

        with pytest.raises(PGGraphQueryException) as excinfo:
            await storage._upsert_node_chunk([("Alice", {"entity_id": "Alice", "entity_type": "Person"})])

        assert excinfo.value.__cause__ is corrupt_exc
        assert _is_transient_graph_write_error(excinfo.value) is True
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_upsert_edge_chunk_retries_on_age_race(self, monkeypatch):
        monkeypatch.setattr(PGGraphStorage._upsert_edge_chunk.retry, "wait", wait_none())

        storage = _make_graph_storage()
        raw_bytes = b'relation "testgraph.\x80\x81" does not exist'
        corrupt_exc = UnicodeDecodeError("utf-8", raw_bytes, 20, 21, "invalid start byte")

        call_count = 0

        async def fake_run_with_retry(_operation, **_kwargs):
            nonlocal call_count
            call_count += 1
            raise corrupt_exc

        storage.db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)

        with pytest.raises(PGGraphQueryException) as excinfo:
            await storage._upsert_edge_chunk([("Alice", "Bob", {"weight": "1.0"})])

        assert excinfo.value.__cause__ is corrupt_exc
        assert _is_transient_graph_write_error(excinfo.value) is True
        assert call_count == 3
