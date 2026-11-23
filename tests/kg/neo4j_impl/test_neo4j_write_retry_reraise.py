"""Regression tests for Neo4j write retry exhaustion behavior."""

import pytest
from neo4j import exceptions as neo4jExceptions
from tenacity import wait_none

from lightrag.kg.neo4j_impl import Neo4JStorage


pytestmark = pytest.mark.offline


class _UnavailableDriver:
    """Driver whose every session fails to open, counting the attempts."""

    def __init__(self):
        self.attempts = 0

    def session(self, **kwargs):
        return self

    async def __aenter__(self):
        self.attempts += 1
        raise neo4jExceptions.ServiceUnavailable("Neo4j unavailable")

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _make_storage():
    storage = Neo4JStorage(
        namespace="chunk_entity_relation",
        global_config={"max_graph_nodes": 1000},
        embedding_func=None,
        workspace="test",
    )
    driver = _UnavailableDriver()
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
    storage, driver = _make_storage()
    method = getattr(Neo4JStorage, method_name)
    monkeypatch.setattr(method.retry, "wait", wait_none())

    with pytest.raises(neo4jExceptions.ServiceUnavailable):
        await getattr(storage, method_name)(*args)

    assert driver.attempts == 3


@pytest.mark.asyncio
async def test_remove_nodes_does_not_multiply_delete_node_retries(monkeypatch):
    storage, driver = _make_storage()
    monkeypatch.setattr(Neo4JStorage.delete_node.retry, "wait", wait_none())

    with pytest.raises(neo4jExceptions.ServiceUnavailable):
        await storage.remove_nodes(["node"])

    assert driver.attempts == 3
