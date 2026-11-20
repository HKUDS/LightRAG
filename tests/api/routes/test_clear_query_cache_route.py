"""``DELETE /documents/query_cache``: clear the query-answer cache only.

The route is the REST face of ``LightRAG.aclear_query_cache``, whose storage
contract (types, ordered commit, failure on a commit that did not land) is
pinned in ``tests/test_aclear_query_cache.py``. These tests pin only what is
the route's own, against a stub: where it is mounted, the selector and its
default, that it is accepted while the pipeline is busy -- it takes no
reservation, unlike ``DELETE /documents`` -- and that a failure reaches the
client as an HTTP error rather than as a success.

See HKUDS/LightRAG#4082.
"""

import importlib
import sys
from uuid import uuid4

import pytest
from fastapi import HTTPException

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

DocumentManager = _document_routes.DocumentManager
create_document_routes = _document_routes.create_document_routes

pytestmark = pytest.mark.offline


class _StubRag:
    def __init__(self, workspace: str, error: Exception | None = None):
        self.workspace = workspace
        self.error = error
        self.calls: list[list[str]] = []

    async def aclear_query_cache(self, cache_types):
        self.calls.append(list(cache_types))
        if self.error is not None:
            raise self.error
        return 7


def _route(rag, input_dir):
    router = create_document_routes(rag, DocumentManager(str(input_dir)))
    [route] = [
        r for r in router.routes if getattr(r, "name", "") == "clear_query_cache"
    ]
    return route


async def _busy_workspace() -> tuple[str, dict]:
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    workspace = f"querycacheroute-{uuid4().hex[:8]}"
    shared_storage.initialize_share_data()
    await shared_storage.initialize_pipeline_status(workspace=workspace)
    status = await shared_storage.get_namespace_data(
        "pipeline_status", workspace=workspace
    )
    return workspace, status


@pytest.mark.parametrize(
    ("cache_type", "expected_types"),
    [(None, ["keywords", "query"]), (["query", "query"], ["query"])],
    ids=["default-both", "explicit"],
)
async def test_the_route_clears_the_selected_types_while_the_pipeline_is_busy(
    tmp_path, cache_type, expected_types
):
    workspace, status = await _busy_workspace()
    status.update(busy=True, destructive_busy=True)
    try:
        rag = _StubRag(workspace)
        route = _route(rag, tmp_path)

        response = await route.endpoint(cache_type=cache_type)

        assert (route.path, route.methods) == ("/documents/query_cache", {"DELETE"})
        assert rag.calls == [expected_types]
        assert (response.status, response.deleted, response.cache_types) == (
            "success",
            7,
            expected_types,
        )
    finally:
        status.update(busy=False, destructive_busy=False)


async def test_a_failure_is_an_http_error_not_a_success(tmp_path):
    rag = _StubRag("unused", error=RuntimeError("the commit did not land"))

    with pytest.raises(HTTPException) as exc_info:
        await _route(rag, tmp_path).endpoint(cache_type=None)

    assert exc_info.value.status_code == 500
    assert rag.calls == [["keywords", "query"]]
