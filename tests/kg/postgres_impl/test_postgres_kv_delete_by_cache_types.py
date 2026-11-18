"""``PGKVStorage.delete_by_cache_types``: one parameterized DELETE.

The offline CLI this replaces built ``id LIKE '{mode}:query:%'`` clauses by
f-string interpolation. The types now travel as a bind parameter, the workspace
scopes the statement, and the count comes from the command tag.
"""

import pytest

pytest.importorskip("asyncpg")

from lightrag.kg.postgres_impl import PGKVStorage  # noqa: E402
from lightrag.namespace import NameSpace  # noqa: E402

pytestmark = pytest.mark.offline


class _RecordingConnection:
    def __init__(self, tag: str):
        self.tag = tag
        self.calls: list[tuple] = []

    async def execute(self, sql, *params):
        self.calls.append((sql, params))
        return self.tag


class _FakeDB:
    def __init__(self, connection):
        self._connection = connection

    async def _run_with_retry(self, operation):
        return await operation(self._connection)


def _storage(tag: str):
    storage = object.__new__(PGKVStorage)
    storage.namespace = NameSpace.KV_STORE_LLM_RESPONSE_CACHE
    storage.workspace = "test_ws"
    connection = _RecordingConnection(tag)
    storage.db = _FakeDB(connection)
    return storage, connection


async def test_types_are_bound_not_interpolated_and_the_count_is_returned():
    storage, connection = _storage("DELETE 12")

    deleted = await storage.delete_by_cache_types({"query", "keywords"})

    assert deleted == 12
    [(sql, params)] = connection.calls
    assert params == ("test_ws", ["keywords", "query"])
    assert "keywords" not in sql and "query" not in sql.replace("LIGHTRAG", "")
    assert "workspace=$1" in sql and "ANY($2::text[])" in sql
    assert sql.startswith("DELETE FROM LIGHTRAG_LLM_CACHE ")


async def test_nothing_matched_is_zero():
    storage, _ = _storage("DELETE 0")

    assert await storage.delete_by_cache_types({"query"}) == 0


async def test_no_types_issues_no_statement():
    storage, connection = _storage("DELETE 5")

    assert await storage.delete_by_cache_types(set()) == 0
    assert connection.calls == []
