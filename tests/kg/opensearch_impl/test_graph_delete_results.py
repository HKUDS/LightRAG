"""Graph deletes honor per-item failures without rejecting success or absence."""

import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from opensearchpy.exceptions import OpenSearchException

import lightrag.kg.opensearch_impl as opensearch_module
from lightrag.kg.opensearch_impl import (
    OpenSearchDocStatusStorage,
    OpenSearchGraphStorage,
)

pytestmark = pytest.mark.offline


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["remove_edges", "remove_nodes"])
@pytest.mark.parametrize("status", [200, 404, 403])
async def test_graph_delete_item_results(monkeypatch, method, status):
    item = {"delete": {"_id": "row", "status": status}}
    storage = OpenSearchGraphStorage("graph", {}, None, "test")
    storage._indices_ready = True
    storage.client = SimpleNamespace(
        bulk=AsyncMock(return_value={"errors": status >= 400, "items": [item]}),
        delete_by_query=AsyncMock(return_value={"failures": []}),
    )
    monkeypatch.setattr(
        opensearch_module,
        "_run_chunked_async_bulk",
        AsyncMock(return_value=(int(status == 200), [] if status == 200 else [item])),
    )
    values = [("A", "B")] if method == "remove_edges" else ["A", "B"]
    if status in (200, 404):
        await getattr(storage, method)(values)
    else:
        with pytest.raises(OpenSearchException, match="bulk deletion failed"):
            await getattr(storage, method)(values)
    # A rejected item can follow deletes that already landed, so the search
    # views are marked stale whatever the outcome.
    assert storage._edges_dirty
    assert storage._nodes_dirty == (method == "remove_nodes")


@pytest.mark.asyncio
async def test_delete_node_marks_views_dirty_when_node_delete_fails():
    storage = OpenSearchGraphStorage("graph", {}, None, "test")
    storage._indices_ready = True
    storage.client = SimpleNamespace(
        delete_by_query=AsyncMock(return_value={"deleted": 1, "failures": []}),
        delete=AsyncMock(side_effect=OpenSearchException("unavailable")),
    )
    with pytest.raises(OpenSearchException, match="unavailable"):
        await storage.delete_node("A")
    assert storage._edges_dirty
    assert storage._nodes_dirty


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 404, 403])
async def test_status_delete_item_results(monkeypatch, status):
    item = {"delete": {"_id": "row", "status": status}}
    storage = OpenSearchDocStatusStorage("doc_status", {}, None, "test")
    storage._index_ready = True
    storage.client = SimpleNamespace()
    monkeypatch.setattr(
        opensearch_module,
        "_run_chunked_async_bulk",
        AsyncMock(return_value=(int(status == 200), [] if status == 200 else [item])),
    )
    if status in (200, 404):
        await storage.delete(["row"])
    else:
        with pytest.raises(OpenSearchException, match="bulk deletion failed"):
            await storage.delete(["row"])


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["remove_edges", "remove_nodes", "delete_node"])
async def test_refresh_during_in_flight_delete_leaves_views_dirty(monkeypatch, method):
    """A concurrent read's refresh while the delete is in flight must not
    leave the views marked clean once the delete has landed."""
    monkeypatch.setattr(
        opensearch_module, "get_data_init_lock", lambda: contextlib.AsyncExitStack()
    )
    storage = OpenSearchGraphStorage("graph", {}, None, "test")
    storage._indices_ready = True
    refreshes: list[str] = []

    async def concurrent_read_refresh(*_args, **_kwargs):
        # The refresh runs before this in-flight write lands, so the views
        # it leaves clean are about to go stale again.
        await storage._refresh_graph_indices_if_dirty(
            refresh_nodes=True, refresh_edges=True
        )
        return {"errors": False, "items": [{"delete": {"_id": "e", "status": 200}}]}

    storage.client = SimpleNamespace(
        bulk=AsyncMock(side_effect=concurrent_read_refresh),
        delete_by_query=AsyncMock(side_effect=concurrent_read_refresh),
        delete=AsyncMock(return_value={"result": "deleted"}),
        indices=SimpleNamespace(
            refresh=AsyncMock(side_effect=lambda index: refreshes.append(index))
        ),
    )

    async def node_bulk_with_concurrent_refresh(*args, **kwargs):
        await concurrent_read_refresh()
        return 1, []

    monkeypatch.setattr(
        opensearch_module,
        "_run_chunked_async_bulk",
        node_bulk_with_concurrent_refresh,
    )
    storage._nodes_dirty = True
    storage._edges_dirty = True

    if method == "remove_edges":
        await storage.remove_edges([("A", "B")])
    elif method == "remove_nodes":
        await storage.remove_nodes(["A"])
    else:
        await storage.delete_node("A")

    assert refreshes, "the concurrent refresh did not run"
    assert storage._edges_dirty
    assert storage._nodes_dirty == (method != "remove_edges")
