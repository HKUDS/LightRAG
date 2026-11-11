"""Graph deletes honor per-item failures without rejecting success or absence."""

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
@pytest.mark.parametrize("status", [200, 404, 403, 503])
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


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 404, 403, 503])
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
