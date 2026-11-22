"""Regression tests for Neo4j write retry exhaustion behavior."""

import pytest
from neo4j import exceptions as neo4jExceptions
from tenacity import wait_none

from lightrag.kg.neo4j_impl import Neo4JStorage


pytestmark = pytest.mark.offline


class _FakeResult:
    async def consume(self):
        return None


class _FakeTransaction:
    def __init__(self, driver):
        self._driver = driver

    async def run(self, query, **params):
        self._driver.writes += 1
        return _FakeResult()


class _FakeSession:
    def __init__(self, driver):
        self._driver = driver

    async def __aenter__(self):
        self._driver.attempts += 1
        if self._driver.attempts <= self._driver.failures:
            raise neo4jExceptions.ServiceUnavailable("Neo4j unavailable")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def execute_write(self, callback):
        return await callback(_FakeTransaction(self._driver))


class _FakeDriver:
    def __init__(self, failures):
        self.failures = failures
        self.attempts = 0
        self.writes = 0

    def session(self, **kwargs):
        return _FakeSession(self)


def _make_storage(failures):
    storage = Neo4JStorage(
        namespace="chunk_entity_relation",
        global_config={"max_graph_nodes": 1000},
        embedding_func=None,
        workspace="test",
    )
    driver = _FakeDriver(failures)
    storage._driver = driver
    storage._DATABASE = "neo4j"
    return storage, driver


WRITE_CASES = [
    pytest.param("upsert_node", ("node", {"entity_id": "node"}), id="upsert_node"),
    pytest.param(
        "upsert_nodes_batch",
        ([("node", {"entity_id": "node"})],),
        id="upsert_nodes_batch",
    ),
    pytest.param(
        "upsert_edges_batch",
        ([("source", "target", {})],),
        id="upsert_edges_batch",
    ),
    pytest.param("upsert_edge", ("source", "target", {}), id="upsert_edge"),
    pytest.param("delete_node", ("node",), id="delete_node"),
    pytest.param("remove_edges", ([("source", "target")],), id="remove_edges"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name,args", WRITE_CASES)
async def test_write_retry_exhaustion_reraises_original_exception(
    monkeypatch, method_name, args
):
    storage, driver = _make_storage(failures=3)
    method = getattr(Neo4JStorage, method_name)
    monkeypatch.setattr(method.retry, "wait", wait_none())

    with pytest.raises(neo4jExceptions.ServiceUnavailable) as exc_info:
        await getattr(storage, method_name)(*args)

    assert type(exc_info.value) is neo4jExceptions.ServiceUnavailable
    assert driver.attempts == 3


@pytest.mark.asyncio
async def test_remove_nodes_does_not_multiply_delete_node_retries(monkeypatch):
    storage, driver = _make_storage(failures=3)
    monkeypatch.setattr(Neo4JStorage.delete_node.retry, "wait", wait_none())

    with pytest.raises(neo4jExceptions.ServiceUnavailable) as exc_info:
        await storage.remove_nodes(["node"])

    assert type(exc_info.value) is neo4jExceptions.ServiceUnavailable
    assert driver.attempts == 3


@pytest.mark.asyncio
async def test_write_retry_transient_failure_then_success(monkeypatch):
    storage, driver = _make_storage(failures=1)
    monkeypatch.setattr(Neo4JStorage.upsert_node.retry, "wait", wait_none())

    await storage.upsert_node("node", {"entity_id": "node"})

    assert driver.attempts == 2
    assert driver.writes == 1
