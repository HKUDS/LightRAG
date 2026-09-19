"""``/documents`` DELETE (clear all) and the workspace's configuration records.

The records go LAST, and only when every data drop succeeded: *data gone,
configuration remains* is loud and recoverable, *configuration gone, data
remains* lets the next startup adopt a wrong baseline over surviving vectors.
So a partial drop keeps all three records rather than deleting "the ones for
the parts that did drop". Scenarios 13 and 14 in
docs/design/ConfigurationStorage.md.
"""

from __future__ import annotations

import importlib
import sys
from uuid import uuid4

import pytest

from lightrag import config_store as cs

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_document_routes = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

DocumentManager = _document_routes.DocumentManager
create_document_routes = _document_routes.create_document_routes

pytestmark = pytest.mark.offline


class _NoopStorage:
    namespace = "noop"

    async def drop(self):
        return {"status": "success", "message": "data dropped"}


class _FailingStorage:
    namespace = "failing"

    async def drop(self):
        return {"status": "error", "message": "backend refused the drop"}


class _FakeConfigKV:
    supports_strict_point_reads = True

    def __init__(self, workspace):
        self.rows = {
            cs.embedding_baseline_key(workspace, t): cs.make_config_row(
                scope_workspace=workspace,
                suffix=cs.embedding_baseline_suffix(t),
                value={"model": "m", "dim": 8, "origin": "probe"},
                updated_by="test",
            )
            for t in cs.EMBEDDING_TARGETS
        }
        self.deleted: list[list[str]] = []
        self.flushes = 0

    async def get_by_id_strict(self, key):
        row = self.rows.get(key)
        return None if row is None else dict(row)

    async def delete(self, ids):
        self.deleted.append(list(ids))
        for key in ids:
            self.rows.pop(key, None)

    async def index_done_callback(self):
        self.flushes += 1


class _ClearRag:
    def __init__(
        self,
        workspace: str,
        *,
        failing_chunks: bool = False,
        failing_cache: bool = False,
    ):
        self.workspace = workspace
        self.failing_cache = failing_cache
        self.cache_cleared = 0
        storage = _NoopStorage()
        storage.workspace = workspace
        if failing_chunks:
            chunks = _FailingStorage()
            chunks.workspace = workspace
            self.text_chunks = chunks
        else:
            self.text_chunks = storage
        self.full_docs = storage
        self.full_entities = storage
        self.full_relations = storage
        self.entity_chunks = storage
        self.relation_chunks = storage
        self.entities_vdb = storage
        self.relationships_vdb = storage
        self.chunks_vdb = storage
        self.chunk_entity_relation_graph = storage
        self.doc_status = storage
        self.configuration_storage = _FakeConfigKV(workspace)

    async def aclear_cache(self):
        if self.failing_cache:
            raise RuntimeError("cache backend refused the drop")
        self.cache_cleared += 1


def _clear_endpoint(rag, input_dir):
    router = create_document_routes(rag, DocumentManager(str(input_dir)))
    return [
        route.endpoint
        for route in router.routes
        if getattr(route, "name", "") == "clear_documents"
    ][-1]


async def _init_workspace(workspace):
    shared_storage = importlib.import_module("lightrag.kg.shared_storage")
    shared_storage.initialize_share_data()
    await shared_storage.initialize_pipeline_status(workspace=workspace)


async def test_every_drop_succeeding_deletes_the_three_records_last(tmp_path):
    """Scenario 14, the endpoint's half."""
    workspace = f"clear-config-ok-{uuid4().hex[:8]}"
    await _init_workspace(workspace)
    rag = _ClearRag(workspace)

    response = await _clear_endpoint(rag, tmp_path)()

    assert response.status == "success"
    config = rag.configuration_storage
    assert config.rows == {}
    assert len(config.deleted) == 1
    assert set(config.deleted[0]) == {
        cs.embedding_baseline_key(workspace, t) for t in cs.EMBEDDING_TARGETS
    }
    assert config.flushes == 1


async def test_a_partial_drop_keeps_all_three_records(tmp_path):
    """Scenario 13. text_chunks refused to drop; the records for the ten
    storages that DID drop are kept too -- deleting them would let a later
    start bootstrap over the surviving chunk rows."""
    workspace = f"clear-config-partial-{uuid4().hex[:8]}"
    await _init_workspace(workspace)
    rag = _ClearRag(workspace, failing_chunks=True)

    response = await _clear_endpoint(rag, tmp_path)()

    assert response.status == "partial_success"
    config = rag.configuration_storage
    assert config.deleted == []
    assert len(config.rows) == 3


async def test_a_failed_record_delete_is_reported_not_hidden(tmp_path):
    class _Stubborn(_FakeConfigKV):
        async def delete(self, ids):
            self.deleted.append(list(ids))  # ...and nothing is removed

    workspace = f"clear-config-stuck-{uuid4().hex[:8]}"
    await _init_workspace(workspace)
    rag = _ClearRag(workspace)
    rag.configuration_storage = _Stubborn(workspace)

    response = await _clear_endpoint(rag, tmp_path)()

    assert response.status == "partial_success"
    assert "configuration records could not be deleted" in response.message


async def test_a_failed_cache_drop_keeps_all_three_records(tmp_path):
    """The opt-in cache drop is a data drop too. Every storage dropped, but
    ``aclear_cache`` raised: the cache rows survive in this workspace, so the
    records stay with them rather than being deleted first and leaving the
    never-acceptable residue (configuration gone, data remains)."""
    workspace = f"clear-config-cache-{uuid4().hex[:8]}"
    await _init_workspace(workspace)
    rag = _ClearRag(workspace, failing_cache=True)

    response = await _clear_endpoint(rag, tmp_path)(clear_llm_cache=True)

    assert response.status == "partial_success"
    assert "LLM response cache could not be cleared" in response.message
    config = rag.configuration_storage
    assert config.deleted == []
    assert len(config.rows) == 3


async def test_a_successful_cache_drop_still_lets_the_records_go(tmp_path):
    """Ordering only: the records are deleted AFTER the cache drop, not
    skipped because one was requested."""
    workspace = f"clear-config-cache-ok-{uuid4().hex[:8]}"
    await _init_workspace(workspace)
    rag = _ClearRag(workspace)

    response = await _clear_endpoint(rag, tmp_path)(clear_llm_cache=True)

    assert response.status == "success"
    assert rag.cache_cleared == 1
    assert rag.configuration_storage.rows == {}
    assert len(rag.configuration_storage.deleted) == 1
