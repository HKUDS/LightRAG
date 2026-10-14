"""``_SIDECAR_DEGRADE_ERRORS`` covers optional KV drivers and degrades without abort."""

from __future__ import annotations

import importlib

import pytest

from lightrag.sidecar.query_attachments import (
    _SIDECAR_DEGRADE_ERRORS,
    _fetch_chunk_records,
    enrich_raw_data_attachments,
)


class SimulatedDriverConnectionError(Exception):
    """Like ``redis.exceptions.ConnectionError``: not a builtin ``ConnectionError``."""


@pytest.mark.offline
@pytest.mark.parametrize(
    ("package", "module_path", "exception_name"),
    [
        ("redis", "redis.exceptions", "ConnectionError"),
        ("pymongo", "pymongo.errors", "PyMongoError"),
        ("asyncpg", "asyncpg.exceptions", "PostgresConnectionError"),
        ("opensearchpy", "opensearchpy.exceptions", "OpenSearchException"),
    ],
)
def test_sidecar_degrade_tuple_includes_kv_driver_family(
    package: str, module_path: str, exception_name: str
) -> None:
    pytest.importorskip(package)
    module = importlib.import_module(module_path)
    exc_type = getattr(module, exception_name)
    assert issubclass(exc_type, _SIDECAR_DEGRADE_ERRORS)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_fetch_chunk_records_degrades_on_non_builtin_driver_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: driver transport errors must not be limited to builtin ConnectionError."""

    monkeypatch.setattr(
        "lightrag.sidecar.query_attachments._SIDECAR_DEGRADE_ERRORS",
        _SIDECAR_DEGRADE_ERRORS + (SimulatedDriverConnectionError,),
    )

    class BrokenTextChunks:
        async def get_by_ids(self, ids: list[str]):
            raise SimulatedDriverConnectionError("kv transport down")

    result = await _fetch_chunk_records(BrokenTextChunks(), ["chunk-a"])
    assert result == {}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_enrich_degrades_on_non_builtin_driver_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lightrag.sidecar.query_attachments._SIDECAR_DEGRADE_ERRORS",
        _SIDECAR_DEGRADE_ERRORS + (SimulatedDriverConnectionError,),
    )

    class BrokenFullDocs:
        async def get_by_id(self, doc_id: str):
            raise SimulatedDriverConnectionError("full_docs transport down")

    raw = {
        "data": {"chunks": []},
        "metadata": {"drawing_candidate_whitelist": ["im-deadbeef-0001"]},
    }
    result = await enrich_raw_data_attachments(raw, BrokenFullDocs())
    assert result["data"]["attachments"] == []


@pytest.mark.offline
@pytest.mark.asyncio
async def test_enrich_degrades_on_redis_driver_connection_error() -> None:
    pytest.importorskip("redis")
    from redis.exceptions import ConnectionError as RedisConnectionError

    assert not issubclass(RedisConnectionError, ConnectionError)

    class BrokenFullDocs:
        async def get_by_id(self, doc_id: str):
            raise RedisConnectionError("redis down")

    raw = {
        "data": {"chunks": []},
        "metadata": {"drawing_candidate_whitelist": ["im-deadbeef-0001"]},
    }
    result = await enrich_raw_data_attachments(raw, BrokenFullDocs())
    assert result["data"]["attachments"] == []
