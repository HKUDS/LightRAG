"""Strict-read contracts for the OpenSearch backend (Phase 0 groundwork).

Two contracts introduced for the pipeline scheduling control-plane:

* ``get_docs_by_statuses(..., strict=True)`` — complete-or-raise: a PIT
  creation failure, any page failure and any hit that cannot be parsed into
  ``DocProcessingStatus`` must raise instead of degrading to a partial result
  (the historical behavior, kept for ``strict=False`` UI paths, logged and
  returned whatever had been collected).
* KV ``get_by_id`` / ``get_by_ids`` scheduling-safe reads — ``None`` means
  CONFIRMED absent; a non-missing-index transport error must raise, because
  the pipeline's consistency validator interprets ``None`` as "content
  missing" and would delete a live doc_status row on a transient failure.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from opensearchpy import AsyncOpenSearch
from opensearchpy.exceptions import NotFoundError, TransportError

from lightrag.base import DocStatus
from lightrag.kg.opensearch_impl import (
    ClientManager,
    OpenSearchDocStatusStorage,
    OpenSearchKVStorage,
)

pytestmark = pytest.mark.offline


class _mock_lock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


def _missing_index_error() -> NotFoundError:
    return NotFoundError(404, "index_not_found_exception", "no such index")


def _transient_error() -> TransportError:
    return TransportError(503, "search_phase_execution_exception", "shard failure")


@pytest.fixture(autouse=True)
def patch_data_init_lock():
    with patch(
        "lightrag.kg.opensearch_impl.get_data_init_lock", side_effect=_mock_lock
    ):
        yield


@pytest.fixture(autouse=True)
def patch_namespace_lock():
    cache: dict[tuple[str, str | None], asyncio.Lock] = {}

    def factory(namespace, workspace=None, enable_logging=False):
        key = (namespace, workspace or "")
        lock = cache.get(key)
        if lock is None:
            lock = asyncio.Lock()
            cache[key] = lock
        return lock

    with patch("lightrag.kg.opensearch_impl.get_namespace_lock", side_effect=factory):
        yield


@pytest.fixture(autouse=True)
def patch_shard_doc_supported():
    with patch("lightrag.kg.opensearch_impl._shard_doc_supported", True):
        yield


@pytest.fixture
def global_config():
    return {
        "embedding_batch_num": 10,
        "max_graph_nodes": 1000,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
    }


class _EmbedFunc:
    embedding_dim = 8
    max_token_size = 512
    model_name = "mock-embed"

    async def __call__(self, texts, **kwargs):
        raise AssertionError("doc-status/kv reads must not embed")


def _make_client() -> AsyncMock:
    client = AsyncMock(spec=AsyncOpenSearch)
    client.indices = AsyncMock()
    client.indices.exists = AsyncMock(return_value=True)
    client.indices.create = AsyncMock()
    client.indices.get_mapping = AsyncMock(return_value={})
    client.create_pit = AsyncMock(return_value={"pit_id": "pit-1"})
    client.delete_pit = AsyncMock()
    return client


def _status_source(doc_id: str, status: str = "failed") -> dict:
    return {
        "__mirrored_id": doc_id,
        "content_summary": f"summary-{doc_id}",
        "content_length": 10,
        "file_path": f"{doc_id}.txt",
        "status": status,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }


def _hit(doc_id: str, source: dict) -> dict:
    return {"_id": doc_id, "_source": source, "sort": [doc_id]}


def _page(hits: list[dict]) -> dict:
    return {"hits": {"hits": hits}}


async def _make_doc_status(global_config, client) -> OpenSearchDocStatusStorage:
    with patch.object(ClientManager, "get_client", return_value=client):
        storage = OpenSearchDocStatusStorage(
            namespace="doc_status",
            global_config=global_config,
            embedding_func=_EmbedFunc(),
            workspace="strictws",
        )
        await storage.initialize()
    return storage


async def _make_kv(global_config, client) -> OpenSearchKVStorage:
    with patch.object(ClientManager, "get_client", return_value=client):
        storage = OpenSearchKVStorage(
            namespace="full_docs",
            global_config=global_config,
            embedding_func=_EmbedFunc(),
            workspace="strictws",
        )
        await storage.initialize()
    return storage


# ---------------------------------------------------------------------------
# get_docs_by_statuses strict contract
# ---------------------------------------------------------------------------


async def test_second_page_failure_raises_in_strict_partial_in_relaxed(
    global_config,
):
    """A mid-pagination transport error must not silently drop the tail."""
    client = _make_client()
    storage = await _make_doc_status(global_config, client)

    client.search = AsyncMock(side_effect=_transient_error())
    with pytest.raises(TransportError):
        await storage.get_docs_by_statuses([DocStatus.FAILED], strict=True)

    # Relaxed mode keeps the historical partial behavior (here: nothing).
    client.search = AsyncMock(side_effect=_transient_error())
    assert await storage.get_docs_by_statuses([DocStatus.FAILED]) == {}

    # Multi-page: page 1 succeeds (full batch), page 2 fails.
    big_page = [
        _hit(f"d{i}", _status_source(f"d{i}"))
        for i in range(10000)  # batch_size, forces a second page
    ]
    client.search = AsyncMock(side_effect=[_page(big_page), _transient_error()])
    with pytest.raises(TransportError):
        await storage.get_docs_by_statuses([DocStatus.FAILED], strict=True)

    client.search = AsyncMock(side_effect=[_page(big_page), _transient_error()])
    partial = await storage.get_docs_by_statuses([DocStatus.FAILED])
    assert len(partial) == 10000  # relaxed mode returned the partial result


async def test_pit_creation_failure_raises_in_strict(global_config):
    client = _make_client()
    client.create_pit = AsyncMock(side_effect=_transient_error())
    storage = await _make_doc_status(global_config, client)
    with pytest.raises(TransportError):
        await storage.get_docs_by_statuses([DocStatus.PENDING], strict=True)
    assert await storage.get_docs_by_statuses([DocStatus.PENDING]) == {}


async def test_undeserializable_record_raises_in_strict_skips_in_relaxed(
    global_config,
):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    bad = _hit("bad", {"__mirrored_id": "bad", "status": "failed"})  # missing fields
    good = _hit("good", _status_source("good"))

    client.search = AsyncMock(return_value=_page([good, bad]))
    with pytest.raises(TypeError):
        await storage.get_docs_by_statuses([DocStatus.FAILED], strict=True)

    client.search = AsyncMock(return_value=_page([good, bad]))
    relaxed = await storage.get_docs_by_statuses([DocStatus.FAILED])
    assert set(relaxed) == {"good"}  # relaxed mode skips the bad record


async def test_missing_index_is_complete_empty_in_both_modes(global_config):
    client = _make_client()
    client.search = AsyncMock(side_effect=_missing_index_error())
    storage = await _make_doc_status(global_config, client)
    assert await storage.get_docs_by_statuses([DocStatus.PENDING], strict=True) == {}
    storage._index_ready = True  # re-arm after _mark_index_missing
    assert await storage.get_docs_by_statuses([DocStatus.PENDING]) == {}


async def test_pit_delete_failure_stays_best_effort(global_config):
    client = _make_client()
    client.search = AsyncMock(return_value=_page([_hit("d1", _status_source("d1"))]))
    client.delete_pit = AsyncMock(side_effect=_transient_error())
    storage = await _make_doc_status(global_config, client)
    result = await storage.get_docs_by_statuses([DocStatus.FAILED], strict=True)
    assert set(result) == {"d1"}  # collected result is complete; cleanup best-effort


# ---------------------------------------------------------------------------
# KV scheduling-safe reads
# ---------------------------------------------------------------------------


async def test_kv_get_by_id_raises_on_transient_error(global_config):
    client = _make_client()
    client.mget = AsyncMock(side_effect=_transient_error())
    storage = await _make_kv(global_config, client)
    with pytest.raises(TransportError):
        await storage.get_by_id("doc-1")


async def test_kv_get_by_id_none_only_for_confirmed_absent(global_config):
    client = _make_client()
    storage = await _make_kv(global_config, client)
    # Confirmed absent: mget answers found=False.
    client.mget = AsyncMock(return_value={"docs": [{"_id": "doc-1", "found": False}]})
    assert await storage.get_by_id("doc-1") is None
    # Missing index: index gone == nothing stored, confirmed absent.
    client.mget = AsyncMock(side_effect=_missing_index_error())
    assert await storage.get_by_id("doc-1") is None


async def test_kv_get_by_ids_raises_on_transient_error(global_config):
    client = _make_client()
    client.mget = AsyncMock(side_effect=_transient_error())
    storage = await _make_kv(global_config, client)
    with pytest.raises(TransportError):
        await storage.get_by_ids(["doc-1", "doc-2"])
