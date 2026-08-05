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
    OpenSearchGraphStorage,
    OpenSearchKVStorage,
    OpenSearchVectorDBStorage,
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


def _migrated_mapping(index: str, **_kwargs) -> dict:
    """Mapping of an index that already carries the __mirrored_id tiebreaker."""
    return {index: {"mappings": {"properties": {"__mirrored_id": {"type": "keyword"}}}}}


def _make_client() -> AsyncMock:
    client = AsyncMock(spec=AsyncOpenSearch)
    client.indices = AsyncMock()
    client.indices.exists = AsyncMock(return_value=True)
    client.indices.create = AsyncMock()
    client.indices.get_mapping = AsyncMock(side_effect=_migrated_mapping)
    client.indices.put_mapping = AsyncMock()
    # A healthy index: startup's unconditional __mirrored_id coverage audit
    # finds zero gaps. ``query_params`` wraps client methods in a SYNC wrapper,
    # so spec'd children default to MagicMock — async ones must be set here or
    # ``await client.count(...)`` fails with "MagicMock can't be used in await".
    client.count = AsyncMock(return_value={"count": 0})
    client.update_by_query = AsyncMock(return_value={"updated": 0, "failures": []})
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


def _make_graph_client() -> AsyncMock:
    """Graph indices don't exist yet -- skips canonical-edge migration and
    PPL auto-detect cleanly falls back to client-side BFS."""
    client = _make_client()
    client.indices.exists = AsyncMock(return_value=False)
    client.transport = AsyncMock()
    client.transport.perform_request = AsyncMock(side_effect=Exception("no PPL"))
    return client


async def _make_graph(global_config, client) -> OpenSearchGraphStorage:
    with patch.object(ClientManager, "get_client", return_value=client):
        storage = OpenSearchGraphStorage(
            namespace="chunk_entity_relation",
            global_config=global_config,
            embedding_func=_EmbedFunc(),
            workspace="strictws",
        )
        await storage.initialize()
    return storage


async def _make_vector(global_config, client) -> OpenSearchVectorDBStorage:
    with patch.object(ClientManager, "get_client", return_value=client):
        storage = OpenSearchVectorDBStorage(
            namespace="chunks",
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


# ---------------------------------------------------------------------------
# KV item-level mget errors (HTTP 200, but a single item failed)
#
# OpenSearch answers an ``mget`` with HTTP 200 even when individual items
# failed: such an item carries an ``error`` object and NO ``found`` flag.
# Reading missing/false ``found`` as "absent" turns a transient per-item
# failure into a CONFIRMED miss, which the consistency validator then deletes a
# live doc_status row over.  ``None`` must mean confirmed absent and nothing
# else.
# ---------------------------------------------------------------------------


async def test_kv_get_by_id_raises_on_item_level_error(global_config):
    client = _make_client()
    storage = await _make_kv(global_config, client)
    client.mget = AsyncMock(
        return_value={
            "docs": [
                {
                    "_id": "doc-1",
                    "error": {"type": "shard", "reason": "x"},
                    "status": 503,
                }
            ]
        }
    )
    with pytest.raises(RuntimeError, match="item error"):
        await storage.get_by_id("doc-1")


async def test_kv_get_by_id_raises_on_malformed_or_missing_item(global_config):
    client = _make_client()
    storage = await _make_kv(global_config, client)
    # Empty docs array — no answer at all for the requested id.
    client.mget = AsyncMock(return_value={"docs": []})
    with pytest.raises(RuntimeError):
        await storage.get_by_id("doc-1")
    # 'found' flag absent entirely — cannot tell present from absent.
    client.mget = AsyncMock(return_value={"docs": [{"_id": "doc-1"}]})
    with pytest.raises(RuntimeError):
        await storage.get_by_id("doc-1")
    # found=True but no _source — malformed.
    client.mget = AsyncMock(return_value={"docs": [{"_id": "doc-1", "found": True}]})
    with pytest.raises(RuntimeError):
        await storage.get_by_id("doc-1")
    # Response id does not match the requested id.
    client.mget = AsyncMock(
        return_value={"docs": [{"_id": "other", "found": True, "_source": {"k": 1}}]}
    )
    with pytest.raises(RuntimeError):
        await storage.get_by_id("doc-1")


async def test_kv_get_by_id_none_only_for_explicit_found_false(global_config):
    client = _make_client()
    storage = await _make_kv(global_config, client)
    client.mget = AsyncMock(return_value={"docs": [{"_id": "doc-1", "found": False}]})
    assert await storage.get_by_id("doc-1") is None


async def test_kv_get_by_ids_raises_on_item_level_error(global_config):
    client = _make_client()
    storage = await _make_kv(global_config, client)
    client.mget = AsyncMock(
        return_value={
            "docs": [
                {"_id": "doc-1", "found": True, "_source": {"k": "v"}},
                {"_id": "doc-2", "error": {"type": "shard"}, "status": 503},
            ]
        }
    )
    with pytest.raises(RuntimeError, match="item error"):
        await storage.get_by_ids(["doc-1", "doc-2"])


async def test_kv_get_by_ids_raises_on_omitted_id(global_config):
    """A short docs list (an id silently dropped) must raise, not report the
    missing id as absent."""
    client = _make_client()
    storage = await _make_kv(global_config, client)
    client.mget = AsyncMock(
        return_value={"docs": [{"_id": "doc-1", "found": True, "_source": {"k": "v"}}]}
    )
    with pytest.raises(RuntimeError):
        await storage.get_by_ids(["doc-1", "doc-2"])


async def test_kv_get_by_ids_maps_found_and_confirmed_absent(global_config):
    """Happy path preserved: found → materialized in position, explicit
    found=False → None in position."""
    client = _make_client()
    storage = await _make_kv(global_config, client)
    client.mget = AsyncMock(
        return_value={
            "docs": [
                {"_id": "doc-1", "found": True, "_source": {"k": "v1"}},
                {"_id": "doc-2", "found": False},
            ]
        }
    )
    result = await storage.get_by_ids(["doc-1", "doc-2"])
    assert result[0]["k"] == "v1"
    assert result[0]["_id"] == "doc-1"
    assert result[1] is None


# ---------------------------------------------------------------------------
# filter_keys / doc-status batch reads: the SAME item-level contract
#
# ``filter_keys`` issues ``_source=False`` mget, so a found item legitimately
# carries no ``_source``.  A per-item shard error still must NOT be read as
# "missing" — otherwise an existing doc_id lands in the missing set and the
# enqueue dedup (pipeline / custom-chunk create) re-ingests it.  Same contract
# for OpenSearchDocStatusStorage.get_by_ids.
# ---------------------------------------------------------------------------


def _mget_error_for(bad_id: str, *, source: dict | None = None):
    """Build an mget side_effect echoing requested-id order: ``bad_id`` gets an
    item-level error, every other id is found (with ``source`` if given)."""

    async def _side_effect(index=None, body=None, **kwargs):
        docs = []
        for doc_id in (body or {}).get("ids", []):
            if doc_id == bad_id:
                docs.append({"_id": doc_id, "error": {"type": "shard"}, "status": 503})
            elif source is None:
                docs.append({"_id": doc_id, "found": True})  # _source=False query
            else:
                docs.append({"_id": doc_id, "found": True, "_source": dict(source)})
        return {"docs": docs}

    return _side_effect


async def test_kv_filter_keys_raises_on_item_level_error(global_config):
    client = _make_client()
    storage = await _make_kv(global_config, client)
    storage._index_ready = True
    client.mget = AsyncMock(side_effect=_mget_error_for("bad"))
    with pytest.raises(RuntimeError, match="item error"):
        await storage.filter_keys({"good", "bad"})


async def test_kv_filter_keys_returns_only_confirmed_absent(global_config):
    client = _make_client()
    storage = await _make_kv(global_config, client)
    storage._index_ready = True

    async def _side_effect(index=None, body=None, **kwargs):
        # "here" exists (found, no _source under _source=False); "gone" absent.
        return {
            "docs": [
                {"_id": doc_id, "found": doc_id == "here"} for doc_id in body["ids"]
            ]
        }

    client.mget = AsyncMock(side_effect=_side_effect)
    assert await storage.filter_keys({"here", "gone"}) == {"gone"}


async def test_doc_status_get_by_ids_raises_on_item_level_error(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    storage._index_ready = True
    client.mget = AsyncMock(
        side_effect=_mget_error_for("bad", source={"status": "failed"})
    )
    with pytest.raises(RuntimeError, match="item error"):
        await storage.get_by_ids(["good", "bad"])


async def test_doc_status_filter_keys_raises_on_item_level_error(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    storage._index_ready = True
    client.mget = AsyncMock(side_effect=_mget_error_for("bad"))
    with pytest.raises(RuntimeError, match="item error"):
        await storage.filter_keys({"good", "bad"})


async def test_kv_filter_keys_raises_on_whole_call_transient(global_config):
    """A whole-call transport error must NOT be read as 'all keys new': the
    dedup gate raises so the caller aborts before any upsert (fail-closed)."""
    client = _make_client()
    storage = await _make_kv(global_config, client)
    storage._index_ready = True
    client.mget = AsyncMock(side_effect=_transient_error())
    with pytest.raises(TransportError):
        await storage.filter_keys({"a", "b"})


async def test_doc_status_filter_keys_raises_on_whole_call_transient(global_config):
    """First-layer enqueue dedup (pipeline.py): a transient whole-call failure
    must raise, not report existing docs as new and re-schedule them."""
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    storage._index_ready = True
    client.mget = AsyncMock(side_effect=_transient_error())
    with pytest.raises(TransportError):
        await storage.filter_keys({"a", "b"})


async def test_filter_keys_missing_index_returns_all_keys(global_config):
    """A genuinely missing index is confirmed-empty — every key is new. This
    stays fail-open (unlike a transport error) for both KV and doc_status."""
    kv_client = _make_client()
    kv = await _make_kv(global_config, kv_client)
    kv._index_ready = True
    kv_client.mget = AsyncMock(side_effect=_missing_index_error())
    assert await kv.filter_keys({"a", "b"}) == {"a", "b"}

    ds_client = _make_client()
    ds = await _make_doc_status(global_config, ds_client)
    ds._index_ready = True
    ds_client.mget = AsyncMock(side_effect=_missing_index_error())
    assert await ds.filter_keys({"a", "b"}) == {"a", "b"}


async def test_doc_status_get_by_ids_raises_on_whole_call_transient(global_config):
    """A whole-call transport error (not an item-level error) is NOT the same
    as 'every id is absent' -- unlike the item-level contract above, this used
    to be swallowed into an all-None result before the fix."""
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    storage._index_ready = True
    client.mget = AsyncMock(side_effect=_transient_error())
    with pytest.raises(TransportError):
        await storage.get_by_ids(["doc-1", "doc-2"])


# ---------------------------------------------------------------------------
# Graph reads: same scheduling-safe contract as KV/doc_status.
#
# A whole-call transport error must not be reported as "node/edge doesn't
# exist" (has_node/has_edge/get_node/...), "no neighbors"
# (get_nodes_batch/get_node_edges/...), or "empty graph" (get_all_*). Entity
# merge/dedup and knowledge-graph queries treat those as confirmed facts.
# ---------------------------------------------------------------------------


async def test_graph_has_node_raises_on_transient_error(global_config):
    client = _make_graph_client()
    storage = await _make_graph(global_config, client)
    client.exists = AsyncMock(side_effect=_transient_error())
    with pytest.raises(TransportError):
        await storage.has_node("Alice")


async def test_graph_get_node_raises_on_transient_error(global_config):
    client = _make_graph_client()
    storage = await _make_graph(global_config, client)
    client.mget = AsyncMock(side_effect=_transient_error())
    with pytest.raises(TransportError):
        await storage.get_node("Alice")


async def test_graph_get_nodes_batch_raises_on_transient_error(global_config):
    client = _make_graph_client()
    storage = await _make_graph(global_config, client)
    client.mget = AsyncMock(side_effect=_transient_error())
    with pytest.raises(TransportError):
        await storage.get_nodes_batch(["Alice", "Bob"])


async def test_graph_has_nodes_batch_raises_on_transient_error(global_config):
    client = _make_graph_client()
    storage = await _make_graph(global_config, client)
    client.mget = AsyncMock(side_effect=_transient_error())
    with pytest.raises(TransportError):
        await storage.has_nodes_batch(["Alice", "Bob"])


async def test_graph_get_all_nodes_raises_on_transient_error(global_config):
    client = _make_graph_client()
    storage = await _make_graph(global_config, client)
    client.search = AsyncMock(side_effect=_transient_error())
    with pytest.raises(TransportError):
        await storage.get_all_nodes()


async def test_graph_missing_index_still_returns_empty(global_config):
    """The confirmed-empty case (missing index) is unaffected by the fix."""
    client = _make_graph_client()
    storage = await _make_graph(global_config, client)
    client.exists = AsyncMock(side_effect=_missing_index_error())
    assert await storage.has_node("Alice") is False


async def test_bfs_subgraph_transient_error_raises_not_reports_false_complete(
    global_config,
):
    """Regression for the truncation-flag bug: a mid-BFS transport error used
    to ``break`` the level loop silently, then compute
    ``is_truncated = len(seen_nodes) >= max_nodes`` -- False, since max_nodes
    was never reached -- reporting a partial subgraph as a complete one. It
    must now raise instead of returning a falsely-complete KnowledgeGraph."""
    client = _make_graph_client()
    storage = await _make_graph(global_config, client)
    storage._ppl_graphlookup_available = False
    client.mget = AsyncMock(
        return_value={
            "docs": [
                {
                    "_id": "start",
                    "found": True,
                    "_source": {"entity_type": "person"},
                }
            ]
        }
    )
    client.search = AsyncMock(side_effect=_transient_error())
    with pytest.raises(TransportError):
        await storage.get_knowledge_graph("start", max_depth=2, max_nodes=100)


def _bfs_mget_side_effect(real_nodes: dict):
    """Mock ``client.mget`` for both the single-id start-node lookup and the
    batched per-level neighbor resolution. Ids absent from `real_nodes` come
    back ``found: False``, mirroring a dangling edge endpoint. Writes now
    materialize both endpoints, so new data cannot produce one, but documents
    written before that change still can and the traversal must keep tolerating
    them."""

    async def _mget(index=None, body=None, **kwargs):
        docs = []
        for node_id in body["ids"]:
            if node_id in real_nodes:
                docs.append(
                    {"_id": node_id, "found": True, "_source": real_nodes[node_id]}
                )
            else:
                docs.append({"_id": node_id, "found": False})
        return {"docs": docs}

    return _mget


def _bfs_search_side_effect(edges: list):
    """Mock ``client.search`` for both the per-level edge scan (``should``,
    at least one endpoint in the frontier) and the final PIT-scrolled
    edge fetch (``must``, both endpoints in the seen-node set)."""

    async def _search(index=None, body=None, **kwargs):
        bool_query = body["query"]["bool"]
        if "should" in bool_query:
            ids = set(bool_query["should"][0]["terms"]["source_node_id"])
            hits = [
                {"_source": e}
                for e in edges
                if e["source_node_id"] in ids or e["target_node_id"] in ids
            ]
        else:
            ids = set(bool_query["must"][0]["terms"]["source_node_id"])
            hits = [
                {"_source": e}
                for e in edges
                if e["source_node_id"] in ids and e["target_node_id"] in ids
            ]
        return {"hits": {"hits": hits}}

    return _search


async def _make_bfs_storage(global_config, real_nodes: dict, edges: list):
    client = _make_graph_client()
    storage = await _make_graph(global_config, client)
    storage._ppl_graphlookup_available = False
    client.mget = AsyncMock(side_effect=_bfs_mget_side_effect(real_nodes))
    client.search = AsyncMock(side_effect=_bfs_search_side_effect(edges))
    return storage


async def test_bfs_subgraph_counter_example_not_falsely_truncated(global_config):
    """A only connects to B and C, and B/C's only neighbor is the already-
    visited A. Filling max_nodes=3 exactly on round 1 must not make round 2's
    top-of-loop capacity check falsely declare truncation before confirming
    there is nothing left to explore."""
    real_nodes = {n: {"entity_type": "person"} for n in ["A", "B", "C"]}
    edges = [
        {"source_node_id": "A", "target_node_id": "B"},
        {"source_node_id": "A", "target_node_id": "C"},
    ]
    storage = await _make_bfs_storage(global_config, real_nodes, edges)

    result = await storage.get_knowledge_graph("A", max_depth=2, max_nodes=3)

    assert {n.id for n in result.nodes} == {"A", "B", "C"}
    assert result.is_truncated is False


async def test_bfs_subgraph_diamond_not_truncated(global_config):
    real_nodes = {n: {"entity_type": "person"} for n in ["A", "B", "C", "D"]}
    edges = [
        {"source_node_id": "A", "target_node_id": "B"},
        {"source_node_id": "A", "target_node_id": "C"},
        {"source_node_id": "B", "target_node_id": "D"},
        {"source_node_id": "C", "target_node_id": "D"},
    ]
    storage = await _make_bfs_storage(global_config, real_nodes, edges)

    result = await storage.get_knowledge_graph("A", max_depth=2, max_nodes=4)

    assert {n.id for n in result.nodes} == {"A", "B", "C", "D"}
    assert result.is_truncated is False


async def test_bfs_subgraph_star_reports_truncated_and_respects_cap(global_config):
    leaves = ["B", "C", "D", "E", "F"]
    real_nodes = {n: {"entity_type": "person"} for n in ["A"] + leaves}
    edges = [{"source_node_id": "A", "target_node_id": leaf} for leaf in leaves]
    storage = await _make_bfs_storage(global_config, real_nodes, edges)

    result = await storage.get_knowledge_graph("A", max_depth=2, max_nodes=3)

    assert result.is_truncated is True
    assert len(result.nodes) <= 3


async def test_bfs_subgraph_dangling_only_neighbor_not_falsely_truncated(
    global_config,
):
    """The only neighbor is a dangling id (edge-referenced, no node
    document) -- nothing real was cut, so this must not be truncated."""
    real_nodes = {"A": {"entity_type": "person"}}
    edges = [{"source_node_id": "A", "target_node_id": "X"}]
    storage = await _make_bfs_storage(global_config, real_nodes, edges)

    result = await storage.get_knowledge_graph("A", max_depth=2, max_nodes=1)

    assert {n.id for n in result.nodes} == {"A"}
    assert result.is_truncated is False


async def test_bfs_subgraph_dangling_candidate_does_not_steal_real_node_slot(
    global_config,
):
    """A occupies one of max_nodes=2 slots; the real neighbor B must fill the
    second slot even though a dangling candidate X is also in the frontier."""
    real_nodes = {"A": {"entity_type": "person"}, "B": {"entity_type": "person"}}
    edges = [
        {"source_node_id": "A", "target_node_id": "X"},
        {"source_node_id": "A", "target_node_id": "B"},
    ]
    storage = await _make_bfs_storage(global_config, real_nodes, edges)

    result = await storage.get_knowledge_graph("A", max_depth=2, max_nodes=2)

    assert {n.id for n in result.nodes} == {"A", "B"}
    assert result.is_truncated is False


async def test_vector_query_raises_on_whole_call_transient(global_config):
    """Matches the raise-on-unexpected-error convention every other vector
    backend's query() follows (Postgres/Milvus/Qdrant/Mongo)."""
    client = _make_client()
    storage = await _make_vector(global_config, client)
    client.search = AsyncMock(side_effect=_transient_error())
    with pytest.raises(TransportError):
        await storage.query("q", top_k=5, query_embedding=[0.1] * 8)


async def test_vector_query_missing_index_still_returns_empty(global_config):
    client = _make_client()
    storage = await _make_vector(global_config, client)
    client.search = AsyncMock(side_effect=_missing_index_error())
    assert await storage.query("q", top_k=5, query_embedding=[0.1] * 8) == []


# ---------------------------------------------------------------------------
# Graph writes: has_node()/has_nodes_batch() now raise on a transient error
# (see above). upsert_edge()/upsert_edges_batch() call them internally as an
# existence check before writing -- if the outer write's own except block
# swallowed that propagated exception, the edge would silently never be
# written while the caller believed the upsert succeeded. Every graph write
# method must raise on failure so the pipeline marks the document failed and
# retries, matching upsert_node/upsert_edge/delete_node in postgres_impl.py
# and neo4j_impl.py.
# ---------------------------------------------------------------------------


async def test_upsert_edge_raises_when_endpoint_existence_check_fails(global_config):
    # upsert_edge materializes BOTH endpoints and probes them with one mget
    # (has_nodes_batch), like upsert_edges_batch does.
    client = _make_graph_client()
    storage = await _make_graph(global_config, client)
    client.mget = AsyncMock(side_effect=_transient_error())
    with pytest.raises(TransportError):
        await storage.upsert_edge("A", "B", {})


async def test_upsert_edges_batch_raises_when_has_nodes_batch_fails(global_config):
    client = _make_graph_client()
    storage = await _make_graph(global_config, client)
    client.mget = AsyncMock(side_effect=_transient_error())
    with pytest.raises(TransportError):
        await storage.upsert_edges_batch([("A", "B", {})])


async def test_upsert_node_raises_on_transient_error(global_config):
    client = _make_graph_client()
    storage = await _make_graph(global_config, client)
    client.index = AsyncMock(side_effect=_transient_error())
    with pytest.raises(TransportError):
        await storage.upsert_node("A", {})


async def test_delete_node_raises_on_transient_error(global_config):
    client = _make_graph_client()
    storage = await _make_graph(global_config, client)
    client.delete_by_query = AsyncMock(side_effect=_transient_error())
    with pytest.raises(TransportError):
        await storage.delete_node("A")


async def test_get_nodes_edges_batch_success_path_returns_dict_not_none(
    global_config,
):
    """Regression: the swallow -> raise cleanup at the whole-call except block
    dropped the trailing ``return result`` that used to run unconditionally
    after the try/except (the try body itself never returned). Without it, a
    successful call fell off the end of the function and implicitly returned
    None -- adelete_by_doc_id then crashed on ``nodes_edges_dict.items()``."""
    client = _make_graph_client()
    storage = await _make_graph(global_config, client)
    client.search = AsyncMock(
        return_value={
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "source_node_id": "A",
                            "target_node_id": "B",
                        },
                        "sort": ["A", "B"],
                    }
                ]
            }
        }
    )
    result = await storage.get_nodes_edges_batch(["A", "B"])
    assert result is not None
    assert result["A"] == [("A", "B")]
    assert result["B"] == [("A", "B")]
