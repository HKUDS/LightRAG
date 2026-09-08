"""OpenSearch Phase-1 scheduling contracts (memory-bounding plan).

Offline coverage for the OpenSearch slice of the bounded-scheduling API:

* ``get_docs_by_statuses_page`` — native ``search_after`` keyset pages
  sorted ``(created_at ASC, __mirrored_id ASC)`` with ``created_at``
  ``missing=_first`` (legacy rows sort FIRST and stay reachable), live-view
  (no PIT), a plain status terms filter, cursor = the last HIT's sort values
  verbatim, ``hits < limit`` == CURSOR_END.
* ``count_docs_by_statuses`` — fail-closed ``_count``.
* ``update_doc_status_fields`` — immutable ``created_at`` + 404 mapping.
* ``get_docs_by_ids`` — strict batch mget, lightweight projection, confirmed
  absence vs never-absent item errors.
* ``resolve_doc_source_strict`` — conflict-aware Absent/Unique/Conflict with
  a cheap exact ``_count``; fail-closed where the legacy lookup swallows.
* ``list_source_conflicts_page`` / ``repair_source_conflict`` — operator
  conflict flow with a recompute-and-compare CAS.
* strict point reads — three-state ``get_by_id_strict`` on both the KV and
  doc-status classes.
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from opensearchpy import AsyncOpenSearch
from opensearchpy.exceptions import NotFoundError, TransportError

from lightrag.base import (
    CURSOR_END,
    CursorAfter,
    DocStatus,
    SourceAbsent,
    SourceConflict,
    SourceConflictSummary,
    SourceUnique,
)
from lightrag.exceptions import (
    StorageControlPlaneError,
    StorageRecordNotFoundError,
)
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


def _doc_missing_error() -> NotFoundError:
    # A document-level 404 (GET/update on an existing index): the error body
    # never mentions index_not_found_exception.
    return NotFoundError(404, "document_missing_exception", "no such document")


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
        raise AssertionError("doc-status/kv scheduling paths must not embed")


def _migrated_mapping(index: str, **_kwargs) -> dict:
    """Mapping of an index that already carries the __mirrored_id tiebreaker, so
    startup runs no legacy value backfill (see _ensure_scheduling_fields_mapping)."""
    return {index: {"mappings": {"properties": {"__mirrored_id": {"type": "keyword"}}}}}


def _make_legacy_client() -> AsyncMock:
    """Client for an index predating __mirrored_id: no mapping, so no values."""
    client = _make_client()
    client.indices.get_mapping = AsyncMock(return_value={})
    return client


def _make_client() -> AsyncMock:
    client = AsyncMock(spec=AsyncOpenSearch)
    client.indices = AsyncMock()
    client.indices.exists = AsyncMock(return_value=True)
    client.indices.create = AsyncMock()
    client.indices.get_mapping = AsyncMock(side_effect=_migrated_mapping)
    client.indices.put_mapping = AsyncMock()
    client.get = AsyncMock(return_value={})
    client.index = AsyncMock(return_value={})
    client.update = AsyncMock(return_value={})
    client.count = AsyncMock(return_value={"count": 0})
    client.search = AsyncMock(return_value={"hits": {"hits": []}})
    # query_params wraps the client methods in a SYNC wrapper, so spec'd
    # children default to MagicMock — async ones must be set explicitly.
    client.update_by_query = AsyncMock(return_value={"updated": 0, "failures": []})
    return client


def _status_source(doc_id: str, status: str = "pending", **extra) -> dict:
    source = {
        "__mirrored_id": doc_id,
        "content_summary": f"summary-{doc_id}",
        "content_length": 10,
        "file_path": f"{doc_id}.txt",
        "status": status,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    source.update(extra)
    return source


def _hit(doc_id: str, source: dict, sort: list | None = None) -> dict:
    return {"_id": doc_id, "_source": source, "sort": sort or [1767225600000, doc_id]}


def _page(hits: list[dict]) -> dict:
    return {"hits": {"hits": hits}}


async def _make_doc_status(global_config, client) -> OpenSearchDocStatusStorage:
    with patch.object(ClientManager, "get_client", return_value=client):
        storage = OpenSearchDocStatusStorage(
            namespace="doc_status",
            global_config=global_config,
            embedding_func=_EmbedFunc(),
            workspace="schedws",
        )
        await storage.initialize()
    return storage


async def _make_kv(global_config, client) -> OpenSearchKVStorage:
    with patch.object(ClientManager, "get_client", return_value=client):
        storage = OpenSearchKVStorage(
            namespace="full_docs",
            global_config=global_config,
            embedding_func=_EmbedFunc(),
            workspace="schedws",
        )
        await storage.initialize()
    return storage


# ---------------------------------------------------------------------------
# Capability flags
# ---------------------------------------------------------------------------


def test_scheduling_api_is_mandatory_no_flags():
    # The doc_status scheduling flags were removed: the paging/batch/resolver
    # API is @abstractmethod, so an instantiable backend implements it by
    # definition. get_by_id_strict stays an optional KV capability flag.
    for name in (
        "supports_bounded_scheduling_pages",
        "supports_strict_doc_status_batch_reads",
        "supports_strict_doc_source_resolution",
        "supports_failure_generation",
    ):
        assert not hasattr(OpenSearchDocStatusStorage, name)
        assert not hasattr(OpenSearchKVStorage, name)
    # strict point reads remain a declared capability.
    assert OpenSearchDocStatusStorage.supports_strict_point_reads is True
    assert OpenSearchKVStorage.supports_strict_point_reads is True
    # Fully concrete → no abstract methods left → instantiable.
    assert not OpenSearchDocStatusStorage.__abstractmethods__
    assert not OpenSearchKVStorage.__abstractmethods__


# ---------------------------------------------------------------------------
# get_docs_by_statuses_page: body construction
# ---------------------------------------------------------------------------


async def test_page_body_sort_size_and_status_terms(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.search = AsyncMock(return_value=_page([]))

    await storage.get_docs_by_statuses_page(
        [DocStatus.PENDING, DocStatus.FAILED], limit=25, strict=True
    )

    kwargs = client.search.call_args.kwargs
    assert kwargs["index"] == storage._index_name
    body = kwargs["body"]
    # created_at sorts missing rows FIRST (NULLS-FIRST parity with the other
    # backends), tie-broken on the mirrored keyword id.
    assert body["sort"] == [
        {"created_at": {"order": "asc", "missing": "_first"}},
        {"__mirrored_id": {"order": "asc"}},
    ]
    assert body["size"] == 25
    assert "search_after" not in body
    assert body["query"] == {"terms": {"status": ["failed", "pending"]}}
    # No failure-generation predicate exists anywhere in the query anymore.
    assert "failure_generation" not in json.dumps(body)


async def test_page_search_after_passthrough(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.search = AsyncMock(return_value=_page([]))

    opaque = json.dumps([1767225600000, "doc-9"])
    await storage.get_docs_by_statuses_page(
        [DocStatus.PENDING], limit=5, position=CursorAfter(opaque), strict=True
    )

    body = client.search.call_args.kwargs["body"]
    assert body["search_after"] == [1767225600000, "doc-9"]


# ---------------------------------------------------------------------------
# get_docs_by_statuses_page: cursor advance / termination
# ---------------------------------------------------------------------------


async def test_page_full_page_returns_cursor_after_last_hit_sort(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    hits = [
        _hit("d1", _status_source("d1"), sort=[100, "d1"]),
        _hit("d2", _status_source("d2"), sort=[200, "d2"]),
    ]
    client.search = AsyncMock(return_value=_page(hits))

    page = await storage.get_docs_by_statuses_page(
        [DocStatus.PENDING], limit=2, strict=True
    )

    assert set(page.docs) == {"d1", "d2"}
    assert isinstance(page.next_position, CursorAfter)
    # The cursor is the LAST HIT's sort values verbatim — the natural
    # consumed frontier for a server-side-filtered search_after sweep.
    assert page.next_position.opaque == json.dumps([200, "d2"])
    record = page.docs["d1"]
    assert record.status is DocStatus.PENDING
    assert record.created_at == "2026-01-01T00:00:00+00:00"
    assert record.file_path == "d1.txt"
    assert record.has_custom_chunk_journal is False


async def test_page_short_page_terminates(global_config):
    """Server-side filtering means returned < size proves exhaustion."""
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.search = AsyncMock(return_value=_page([_hit("d1", _status_source("d1"))]))

    page = await storage.get_docs_by_statuses_page(
        [DocStatus.PENDING], limit=2, strict=True
    )
    assert set(page.docs) == {"d1"}
    assert page.next_position is CURSOR_END


async def test_page_empty_page_terminates_and_end_position_short_circuits(
    global_config,
):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.search = AsyncMock(return_value=_page([]))

    page = await storage.get_docs_by_statuses_page(
        [DocStatus.PENDING], limit=2, strict=True
    )
    assert page.docs == {}
    assert page.next_position is CURSOR_END

    client.search.reset_mock()
    page = await storage.get_docs_by_statuses_page(
        [DocStatus.PENDING], limit=2, position=CURSOR_END, strict=True
    )
    assert page.docs == {}
    assert page.next_position is CURSOR_END
    client.search.assert_not_awaited()


async def test_page_invalid_limit_raises(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    with pytest.raises(ValueError):
        await storage.get_docs_by_statuses_page([DocStatus.PENDING], limit=0)


# ---------------------------------------------------------------------------
# get_docs_by_statuses_page: malformed cursor / strict failure semantics
# ---------------------------------------------------------------------------


async def test_page_malformed_cursor_raises_before_any_rpc(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.search = AsyncMock(return_value=_page([]))

    for bad_opaque in ("not-json", '{"a": 1}', "[1]", '["a", "b", "c"]'):
        with pytest.raises(StorageControlPlaneError):
            await storage.get_docs_by_statuses_page(
                [DocStatus.PENDING],
                limit=5,
                position=CursorAfter(bad_opaque),
                strict=True,
            )
    client.search.assert_not_awaited()


# ---------------------------------------------------------------------------
# __mirrored_id tiebreaker readiness (startup)
# ---------------------------------------------------------------------------


def _missing_mirrored_id_query() -> dict:
    return {"bool": {"must_not": {"exists": {"field": "__mirrored_id"}}}}


async def test_startup_audits_coverage_even_when_the_mapping_is_present(global_config):
    """Fix-proof: the audit used to be gated on "we just added the mapping", but
    the mapping present does not imply the VALUES are present — a snapshot
    restore, an external writer, or this very migration crashing between its
    put_mapping and its backfill all produce that state, and the gate then
    skipped the audit for good. A healthy index still costs only one _count and
    no backfill."""
    client = _make_client()
    await _make_doc_status(global_config, client)
    client.count.assert_awaited_once()
    assert client.count.call_args.kwargs["body"]["query"] == (
        _missing_mirrored_id_query()
    )
    client.update_by_query.assert_not_awaited()


async def test_startup_backfills_a_mapped_index_whose_values_are_missing(global_config):
    """The state the old gate declared impossible: mapping present, values
    absent. It must be repaired, not trusted."""
    client = _make_client()
    client.count = AsyncMock(side_effect=[{"count": 4}, {"count": 0}])
    await _make_doc_status(global_config, client)
    client.update_by_query.assert_awaited_once()


async def test_startup_skips_backfill_when_coverage_is_complete(global_config):
    client = _make_legacy_client()
    client.count = AsyncMock(return_value={"count": 0})
    await _make_doc_status(global_config, client)
    client.update_by_query.assert_not_awaited()


async def test_startup_backfills_legacy_docs_missing_mirrored_id(global_config):
    """Fix-proof: a legacy index whose docs carry NO __mirrored_id value used to
    be accepted silently on OpenSearch >= 3.3.0 (the mapping-only check is
    skipped there), leaving the no-PIT scheduling page with no tiebreaker — so
    docs sharing a created_at were dropped from the sweep. Startup must now
    repair the values before serving."""
    client = _make_legacy_client()
    # 3 legacy docs before the backfill, full coverage after it.
    client.count = AsyncMock(side_effect=[{"count": 3}, {"count": 0}])
    await _make_doc_status(global_config, client)

    client.update_by_query.assert_awaited_once()
    kwargs = client.update_by_query.call_args.kwargs
    body = kwargs["body"]
    assert body["query"] == _missing_mirrored_id_query()
    assert body["script"]["source"] == "ctx._source.__mirrored_id = ctx._id"
    assert body["conflicts"] == "proceed"
    assert kwargs["refresh"] is True


async def test_startup_raises_when_mirrored_id_backfill_leaves_gaps(global_config):
    """The outcome is VERIFIED, not assumed: a cluster that refuses the script
    (or races new legacy writes in) must fail startup loudly rather than serve a
    sweep that silently skips documents."""
    client = _make_legacy_client()
    client.count = AsyncMock(side_effect=[{"count": 3}, {"count": 2}])
    with pytest.raises(RuntimeError, match="__mirrored_id"):
        await _make_doc_status(global_config, client)


async def test_startup_raises_on_malformed_mirrored_id_audit(global_config):
    client = _make_legacy_client()
    client.count = AsyncMock(return_value={"count": None})
    with pytest.raises(RuntimeError, match="malformed"):
        await _make_doc_status(global_config, client)


async def test_page_strict_raises_on_transport_error(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.search = AsyncMock(side_effect=_transient_error())
    with pytest.raises(TransportError):
        await storage.get_docs_by_statuses_page(
            [DocStatus.PENDING], limit=5, strict=True
        )


async def test_page_strict_raises_when_index_not_ready(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    storage._index_ready = False
    client.search = AsyncMock(return_value=_page([]))

    with pytest.raises(StorageControlPlaneError):
        await storage.get_docs_by_statuses_page(
            [DocStatus.PENDING], limit=5, strict=True
        )
    client.search.assert_not_awaited()

    # Relaxed mode keeps the best-effort empty-terminal behavior.
    page = await storage.get_docs_by_statuses_page([DocStatus.PENDING], limit=5)
    assert page.docs == {}
    assert page.next_position is CURSOR_END


async def test_page_strict_raises_when_index_vanishes_mid_sweep(global_config):
    """A live index_not_found is a lost index, NOT a completed sweep.

    Fail-proof for the fail-open bug: strict paging used to return an empty
    terminal page here, which the sweep reads as "no work left" — stranding
    every document. It must raise, exactly like the already-known
    ``_index_ready=False`` case, and must not differ by which call noticed.
    """
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.search = AsyncMock(side_effect=_missing_index_error())

    with pytest.raises(StorageControlPlaneError):
        await storage.get_docs_by_statuses_page(
            [DocStatus.PENDING], limit=5, strict=True
        )
    assert storage._index_ready is False


async def test_page_relaxed_missing_index_stays_best_effort(global_config):
    """Relaxed mode keeps the best-effort empty terminal page."""
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.search = AsyncMock(side_effect=_missing_index_error())

    page = await storage.get_docs_by_statuses_page([DocStatus.PENDING], limit=5)
    assert page.docs == {}
    assert page.next_position is CURSOR_END
    assert storage._index_ready is False


async def test_page_relaxed_skips_bad_row_strict_raises(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    good = _hit("good", _status_source("good"), sort=[100, "good"])
    bad_source = _status_source("bad")
    del bad_source["created_at"]  # unusable scheduling row
    bad = _hit("bad", bad_source, sort=[200, "bad"])

    client.search = AsyncMock(return_value=_page([good, bad]))
    with pytest.raises((KeyError, TypeError)):
        await storage.get_docs_by_statuses_page(
            [DocStatus.PENDING], limit=2, strict=True
        )

    client.search = AsyncMock(return_value=_page([good, bad]))
    page = await storage.get_docs_by_statuses_page([DocStatus.PENDING], limit=2)
    assert set(page.docs) == {"good"}
    # The skipped row is still CONSUMED: the cursor is the last hit's sort,
    # so the sweep never re-reads (or worse, loops on) the bad record.
    assert isinstance(page.next_position, CursorAfter)
    assert page.next_position.opaque == json.dumps([200, "bad"])


# ---------------------------------------------------------------------------
# count_docs_by_statuses
# ---------------------------------------------------------------------------


async def test_count_uses_count_api_with_status_terms(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.count = AsyncMock(return_value={"count": 42})

    assert (
        await storage.count_docs_by_statuses([DocStatus.PENDING, DocStatus.FAILED])
        == 42
    )
    kwargs = client.count.call_args.kwargs
    assert kwargs["index"] == storage._index_name
    assert kwargs["body"] == {"query": {"terms": {"status": ["failed", "pending"]}}}

    assert await storage.count_docs_by_statuses([]) == 0


async def test_count_fails_closed(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)

    client.count = AsyncMock(side_effect=_transient_error())
    with pytest.raises(TransportError):
        await storage.count_docs_by_statuses([DocStatus.PENDING])

    client.count = AsyncMock(return_value={"count": "not-a-number"})
    with pytest.raises(StorageControlPlaneError):
        await storage.count_docs_by_statuses([DocStatus.PENDING])

    storage._index_ready = False
    with pytest.raises(StorageControlPlaneError):
        await storage.count_docs_by_statuses([DocStatus.PENDING])


async def test_count_missing_index_raises_not_zero(global_config):
    """Fail-proof for the fail-open bug: a live index_not_found used to return
    0, so the FIRST admission check after an external drop reported full
    capacity (only later ones, seeing _index_ready=False, failed closed)."""
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.count = AsyncMock(side_effect=_missing_index_error())
    with pytest.raises(StorageControlPlaneError):
        await storage.count_docs_by_statuses([DocStatus.PENDING])
    assert storage._index_ready is False


# ---------------------------------------------------------------------------
# update_doc_status_fields
# ---------------------------------------------------------------------------


async def test_update_fields_rejects_created_at(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    with pytest.raises(ValueError, match="created_at"):
        await storage.update_doc_status_fields(
            "doc-1", {"created_at": "2026-02-02T00:00:00+00:00"}
        )
    client.update.assert_not_awaited()


async def test_update_fields_partial_doc_merge(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    await storage.update_doc_status_fields("doc-1", {"status": "processing"})
    kwargs = client.update.call_args.kwargs
    assert kwargs["index"] == storage._index_name
    assert kwargs["id"] == "doc-1"
    assert kwargs["body"] == {"doc": {"status": "processing"}}
    assert kwargs["refresh"] == "wait_for"


async def test_update_fields_missing_record(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.update = AsyncMock(side_effect=_doc_missing_error())
    with pytest.raises(StorageRecordNotFoundError):
        await storage.update_doc_status_fields("ghost", {"status": "failed"})
    # missing_ok downgrades the 404 to a no-op.
    await storage.update_doc_status_fields(
        "ghost", {"status": "failed"}, missing_ok=True
    )


# ---------------------------------------------------------------------------
# get_docs_by_ids: strict batch read
# ---------------------------------------------------------------------------


async def test_get_docs_by_ids_present_and_confirmed_absent(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.mget = AsyncMock(
        return_value={
            "docs": [
                {"_id": "d1", "found": True, "_source": _status_source("d1")},
                {"_id": "d2", "found": False},
            ]
        }
    )
    result = await storage.get_docs_by_ids(["d1", "d2"])
    assert set(result) == {"d1"}  # found=False is CONFIRMED absent → omitted
    assert result["d1"].status is DocStatus.PENDING
    assert result["d1"].file_path == "d1.txt"
    # Lightweight projection: never fetch chunks_list.
    kwargs = client.mget.call_args.kwargs
    assert kwargs["body"] == {"ids": ["d1", "d2"]}
    assert (
        kwargs["_source_includes"]
        == OpenSearchDocStatusStorage._SCHEDULING_SOURCE_FIELDS
    )


async def test_get_docs_by_ids_empty_short_circuits(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.mget = AsyncMock()
    assert await storage.get_docs_by_ids([]) == {}
    client.mget.assert_not_awaited()


async def test_get_docs_by_ids_item_level_error_never_absent(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.mget = AsyncMock(
        return_value={
            "docs": [{"_id": "d1", "error": {"type": "shard"}, "status": 503}]
        }
    )
    with pytest.raises(RuntimeError, match="item error"):
        await storage.get_docs_by_ids(["d1"])


async def test_get_docs_by_ids_strict_raises_relaxed_skips_bad_row(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    bad_source = _status_source("d1")
    del bad_source["created_at"]

    client.mget = AsyncMock(
        return_value={"docs": [{"_id": "d1", "found": True, "_source": bad_source}]}
    )
    with pytest.raises((KeyError, TypeError)):
        await storage.get_docs_by_ids(["d1"], strict=True)

    client.mget = AsyncMock(
        return_value={"docs": [{"_id": "d1", "found": True, "_source": bad_source}]}
    )
    assert await storage.get_docs_by_ids(["d1"]) == {}


async def test_get_docs_by_ids_index_not_ready_raises(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    storage._index_ready = False
    client.mget = AsyncMock()
    with pytest.raises(StorageControlPlaneError):
        await storage.get_docs_by_ids(["d1"])
    client.mget.assert_not_awaited()


async def test_get_docs_by_ids_missing_index_raises(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.mget = AsyncMock(side_effect=_missing_index_error())
    with pytest.raises(StorageControlPlaneError):
        await storage.get_docs_by_ids(["d1"])
    assert storage._index_ready is False


# ---------------------------------------------------------------------------
# get_full_docs_by_ids: FULL-record batch hydration
# ---------------------------------------------------------------------------


async def test_get_full_docs_by_ids_present_and_missing_full_projection(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    d1_source = _status_source(
        "d1", status="failed", chunks_list=["c1", "c2"], error="boom"
    )
    d2_source = _status_source("d2", status="processed", chunks_list=["c3"])
    client.mget = AsyncMock(
        return_value={
            "docs": [
                {"_id": "d1", "found": True, "_source": d1_source},
                {"_id": "d2", "found": True, "_source": d2_source},
                {"_id": "ghost", "found": False},
            ]
        }
    )

    result = await storage.get_full_docs_by_ids(["d1", "d2", "ghost"], strict=True)

    # found=False is CONFIRMED absent -> omitted.
    assert set(result) == {"d1", "d2"}
    # FULL DocProcessingStatus leaves status as the raw str value (== not is).
    assert result["d1"].status == DocStatus.FAILED
    # FULL projection: fields the lightweight scheduling record never carries.
    assert result["d1"].content_summary == "summary-d1"
    assert result["d1"].content_length == 10
    assert result["d1"].chunks_list == ["c1", "c2"]
    assert result["d1"].error_msg == "boom"  # legacy `error` -> error_msg
    assert result["d2"].chunks_list == ["c3"]
    # Full fetch: no _source_includes projection is applied (unlike get_docs_by_ids).
    kwargs = client.mget.call_args.kwargs
    assert kwargs["body"] == {"ids": ["d1", "d2", "ghost"]}
    assert "_source_includes" not in kwargs


async def test_get_full_docs_by_ids_empty_short_circuits(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.mget = AsyncMock()
    assert await storage.get_full_docs_by_ids([]) == {}
    client.mget.assert_not_awaited()


async def test_get_full_docs_by_ids_strict_raises_on_transport_error(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.mget = AsyncMock(side_effect=_transient_error())
    with pytest.raises(TransportError):
        await storage.get_full_docs_by_ids(["d1"], strict=True)


async def test_get_full_docs_by_ids_strict_raises_relaxed_skips_bad_row(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    good_source = _status_source("d1", status="processed")
    bad_source = _status_source("d2")
    del bad_source["created_at"]  # DocProcessingStatus(**data) -> TypeError

    docs = {
        "docs": [
            {"_id": "d1", "found": True, "_source": good_source},
            {"_id": "d2", "found": True, "_source": bad_source},
        ]
    }
    client.mget = AsyncMock(return_value=docs)
    # strict: the WHOLE call fails rather than return a partial map.
    with pytest.raises((KeyError, TypeError)):
        await storage.get_full_docs_by_ids(["d1", "d2"], strict=True)

    client.mget = AsyncMock(return_value=docs)
    # relaxed: skip the unusable row, keep the good one.
    result = await storage.get_full_docs_by_ids(["d1", "d2"])
    assert set(result) == {"d1"}
    assert result["d1"].content_summary == "summary-d1"


async def test_get_full_docs_by_ids_index_not_ready_raises(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    storage._index_ready = False
    client.mget = AsyncMock()
    with pytest.raises(StorageControlPlaneError):
        await storage.get_full_docs_by_ids(["d1"])
    client.mget.assert_not_awaited()


# ---------------------------------------------------------------------------
# resolve_doc_source_strict: conflict-aware source resolution
# ---------------------------------------------------------------------------


async def test_resolve_source_absent_on_zero_hits(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.search = AsyncMock(return_value=_page([]))

    assert isinstance(
        await storage.resolve_doc_source_strict("report.pdf"), SourceAbsent
    )
    body = client.search.call_args.kwargs["body"]
    assert body["size"] == 2
    assert body["sort"] == [{"__mirrored_id": {"order": "asc"}}]
    assert body["query"]["bool"]["filter"] == [{"term": {"file_path": "report.pdf"}}]
    assert body["query"]["bool"]["must_not"] == [
        {"term": {"metadata.is_duplicate": True}}
    ]


async def test_resolve_source_unique(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    # Startup's tiebreaker coverage audit spends one _count; this test is about
    # the resolver's own follow-up count, so measure from zero.
    client.count.reset_mock()
    source = _status_source("doc-1", status="processed", file_path="report.pdf")
    client.search = AsyncMock(return_value=_page([_hit("doc-1", source)]))

    result = await storage.resolve_doc_source_strict("report.pdf")
    assert isinstance(result, SourceUnique)
    assert result.doc_id == "doc-1"
    assert result.doc.status is DocStatus.PROCESSED
    assert result.doc.file_path == "report.pdf"
    # A single hit needs no _count follow-up.
    client.count.assert_not_awaited()


async def test_resolve_source_conflict_uses_exact_count(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    hits = [
        _hit("doc-b", _status_source("doc-b", file_path="report.pdf")),
        _hit("doc-a", _status_source("doc-a", file_path="report.pdf")),
    ]
    client.search = AsyncMock(return_value=_page(hits))
    client.count = AsyncMock(return_value={"count": 5})

    result = await storage.resolve_doc_source_strict("report.pdf")
    assert isinstance(result, SourceConflict)
    assert result.candidate_count == 5
    assert result.sample_doc_ids == ("doc-a", "doc-b")  # sorted sample of two
    # The exact count reuses the same primary-only query.
    count_body = client.count.call_args.kwargs["body"]
    assert count_body["query"] == storage._primary_basename_query("report.pdf")["query"]


async def test_resolve_source_sentinels_are_absent_without_lookup(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.search = AsyncMock(return_value=_page([]))
    assert isinstance(await storage.resolve_doc_source_strict(""), SourceAbsent)
    assert isinstance(
        await storage.resolve_doc_source_strict("unknown_source"), SourceAbsent
    )
    client.search.assert_not_awaited()


async def test_resolve_source_fail_closed(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)

    client.search = AsyncMock(side_effect=_transient_error())
    with pytest.raises(TransportError):
        await storage.resolve_doc_source_strict("report.pdf")

    client.search = AsyncMock(side_effect=_missing_index_error())
    with pytest.raises(StorageControlPlaneError):
        await storage.resolve_doc_source_strict("report.pdf")
    assert storage._index_ready is False

    storage._index_ready = False
    with pytest.raises(StorageControlPlaneError):
        await storage.resolve_doc_source_strict("report.pdf")


# ---------------------------------------------------------------------------
# list_source_conflicts_page
# ---------------------------------------------------------------------------


def _composite_response(buckets: list[dict], after_key: dict | None) -> dict:
    agg: dict = {"buckets": buckets}
    if after_key is not None:
        agg["after_key"] = after_key
    return {"aggregations": {"conflicts": agg}}


def _bucket(file_path: str, doc_count: int, sample_ids: list[str]) -> dict:
    return {
        "key": {"file_path": file_path},
        "doc_count": doc_count,
        "sample_ids": {"hits": {"hits": [{"_id": i} for i in sample_ids]}},
    }


async def test_list_conflicts_filters_single_primary_and_continues(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    buckets = [
        _bucket("a.pdf", 3, ["a3", "a1", "a2"]),
        _bucket("b.pdf", 1, ["b1"]),  # single primary → not a conflict
    ]
    client.search = AsyncMock(
        return_value=_composite_response(buckets, after_key={"file_path": "b.pdf"})
    )

    page = await storage.list_source_conflicts_page(limit=2)
    assert page.conflicts == (
        SourceConflictSummary(
            canonical_source_key="a.pdf",
            candidate_count=3,
            sample_doc_ids=("a1", "a2", "a3"),
        ),
    )
    # Full composite page (len == limit) with an after_key → keep sweeping.
    assert isinstance(page.next_position, CursorAfter)
    assert page.next_position.opaque == json.dumps({"file_path": "b.pdf"})

    # The scan restricts to primaries and real basenames.
    body = client.search.call_args.kwargs["body"]
    assert body["query"] == storage._conflict_scan_query()
    assert body["aggs"]["conflicts"]["composite"]["size"] == 2


async def test_list_conflicts_short_page_terminates(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.search = AsyncMock(
        return_value=_composite_response(
            [_bucket("a.pdf", 2, ["a1", "a2"])], after_key={"file_path": "a.pdf"}
        )
    )
    page = await storage.list_source_conflicts_page(limit=5)
    assert len(page.conflicts) == 1
    # Fewer buckets than the limit proves the composite is exhausted.
    assert page.next_position is CURSOR_END


async def test_list_conflicts_end_position_short_circuits(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.search = AsyncMock()
    page = await storage.list_source_conflicts_page(limit=5, position=CURSOR_END)
    assert page.conflicts == ()
    assert page.next_position is CURSOR_END
    client.search.assert_not_awaited()


async def test_list_conflicts_cursor_passthrough_and_bad_cursor(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.search = AsyncMock(return_value=_composite_response([], after_key=None))

    await storage.list_source_conflicts_page(
        limit=3, position=CursorAfter(json.dumps({"file_path": "a.pdf"}))
    )
    composite = client.search.call_args.kwargs["body"]["aggs"]["conflicts"]["composite"]
    assert composite["after"] == {"file_path": "a.pdf"}

    with pytest.raises(StorageControlPlaneError):
        await storage.list_source_conflicts_page(
            limit=3, position=CursorAfter("[1, 2]")
        )


# ---------------------------------------------------------------------------
# repair_source_conflict
# ---------------------------------------------------------------------------


def _candidate_scan_page(doc_ids: list[str]) -> dict:
    return {"hits": {"hits": [{"_id": d, "sort": [d]} for d in doc_ids]}}


async def test_repair_dry_run_reports_count_and_fingerprint(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.search = AsyncMock(return_value=_candidate_scan_page(["d3", "d1", "d2"]))

    result = await storage.repair_source_conflict(
        "report.pdf",
        primary_doc_id="d1",
        expected_candidate_count=0,
        expected_candidate_fingerprint="ignored-on-dry-run",
        dry_run=True,
    )
    assert result.committed is False
    assert result.candidate_count == 3
    assert result.fingerprint == storage._conflict_fingerprint(["d1", "d2", "d3"])
    assert result.demoted_sample_doc_ids == ("d2", "d3")
    client.update.assert_not_awaited()


async def test_repair_commit_demotes_losers_with_cas(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.search = AsyncMock(return_value=_candidate_scan_page(["d1", "d2", "d3"]))
    fingerprint = storage._conflict_fingerprint(["d1", "d2", "d3"])

    result = await storage.repair_source_conflict(
        "report.pdf",
        primary_doc_id="d1",
        expected_candidate_count=3,
        expected_candidate_fingerprint=fingerprint,
        dry_run=False,
    )
    assert result.committed is True
    assert result.candidate_count == 3
    demoted = {c.kwargs["id"] for c in client.update.await_args_list}
    assert demoted == {"d2", "d3"}
    for call in client.update.await_args_list:
        assert call.kwargs["body"] == {
            "doc": {"metadata": {"is_duplicate": True, "original_doc_id": "d1"}}
        }
        assert call.kwargs["refresh"] == "wait_for"


async def test_repair_commit_cas_mismatch_raises(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.search = AsyncMock(return_value=_candidate_scan_page(["d1", "d2", "d3"]))

    with pytest.raises(StorageControlPlaneError):
        await storage.repair_source_conflict(
            "report.pdf",
            primary_doc_id="d1",
            expected_candidate_count=2,  # stale expectation
            expected_candidate_fingerprint="stale",
            dry_run=False,
        )
    client.update.assert_not_awaited()


async def test_repair_primary_not_a_candidate_raises(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.search = AsyncMock(return_value=_candidate_scan_page(["d1", "d2"]))

    with pytest.raises(ValueError, match="not a current primary"):
        await storage.repair_source_conflict(
            "report.pdf",
            primary_doc_id="zzz",
            expected_candidate_count=2,
            expected_candidate_fingerprint="whatever",
            dry_run=True,
        )


async def test_repair_index_not_ready_raises(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    storage._index_ready = False
    with pytest.raises(StorageControlPlaneError):
        await storage.repair_source_conflict(
            "report.pdf",
            primary_doc_id="d1",
            expected_candidate_count=2,
            expected_candidate_fingerprint="x",
            dry_run=True,
        )


# ---------------------------------------------------------------------------
# KV get_by_id_strict: three-state contract
# ---------------------------------------------------------------------------


async def test_kv_strict_pending_delete_is_confirmed_absent(global_config):
    client = _make_client()
    storage = await _make_kv(global_config, client)
    client.mget = AsyncMock()
    await storage.upsert({"doc-1": {"content": "x"}})
    await storage.delete(["doc-1"])
    assert await storage.get_by_id_strict("doc-1") is None
    client.mget.assert_not_awaited()


async def test_kv_strict_pending_upsert_is_present(global_config):
    client = _make_client()
    storage = await _make_kv(global_config, client)
    client.mget = AsyncMock()
    await storage.upsert({"doc-1": {"content": "x"}})
    doc = await storage.get_by_id_strict("doc-1")
    assert doc["content"] == "x"
    assert doc["_id"] == "doc-1"
    client.mget.assert_not_awaited()


async def test_kv_strict_index_not_ready_raises(global_config):
    client = _make_client()
    storage = await _make_kv(global_config, client)
    storage._index_ready = False
    with pytest.raises(StorageControlPlaneError):
        await storage.get_by_id_strict("doc-1")
    # The relaxed read keeps its best-effort miss behavior.
    assert await storage.get_by_id("doc-1") is None


async def test_kv_strict_transport_error_raises(global_config):
    client = _make_client()
    storage = await _make_kv(global_config, client)
    client.mget = AsyncMock(side_effect=_transient_error())
    with pytest.raises(TransportError):
        await storage.get_by_id_strict("doc-1")


async def test_kv_strict_missing_index_raises(global_config):
    """get_by_id reads a missing index as a miss; the strict variant cannot
    positively confirm absence there and must raise."""
    client = _make_client()
    storage = await _make_kv(global_config, client)
    client.mget = AsyncMock(side_effect=_missing_index_error())
    with pytest.raises(StorageControlPlaneError):
        await storage.get_by_id_strict("doc-1")
    assert storage._index_ready is False


async def test_kv_strict_healthy_miss_is_none(global_config):
    client = _make_client()
    storage = await _make_kv(global_config, client)
    client.mget = AsyncMock(return_value={"docs": [{"_id": "doc-1", "found": False}]})
    assert await storage.get_by_id_strict("doc-1") is None


# ---------------------------------------------------------------------------
# Doc-status get_by_id_strict
# ---------------------------------------------------------------------------


async def test_doc_status_strict_point_read_three_state(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)

    client.mget = AsyncMock(return_value={"docs": [{"_id": "doc-1", "found": False}]})
    assert await storage.get_by_id_strict("doc-1") is None

    client.mget = AsyncMock(side_effect=_transient_error())
    with pytest.raises(TransportError):
        await storage.get_by_id_strict("doc-1")
    # The relaxed read swallows the same error as a miss.
    client.mget = AsyncMock(side_effect=_transient_error())
    assert await storage.get_by_id("doc-1") is None

    client.mget = AsyncMock(side_effect=_missing_index_error())
    with pytest.raises(StorageControlPlaneError):
        await storage.get_by_id_strict("doc-1")

    storage._index_ready = False
    with pytest.raises(StorageControlPlaneError):
        await storage.get_by_id_strict("doc-1")


# ---------------------------------------------------------------------------
# Legacy basename lookup: primary-only best-effort (unchanged)
# ---------------------------------------------------------------------------


async def test_legacy_basename_query_excludes_duplicate_rows(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.search = AsyncMock(return_value=_page([]))

    assert await storage.get_doc_by_file_basename("report.pdf") is None
    body = client.search.call_args.kwargs["body"]
    assert body["query"]["bool"]["filter"] == [{"term": {"file_path": "report.pdf"}}]
    assert body["query"]["bool"]["must_not"] == [
        {"term": {"metadata.is_duplicate": True}}
    ]


async def test_legacy_basename_swallows_errors(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.search = AsyncMock(side_effect=_transient_error())
    assert await storage.get_doc_by_file_basename("report.pdf") is None


@pytest.mark.asyncio
async def test_get_doc_by_content_hash_exclude_doc_id_must_not_ids():
    # LR2 Phase 2.5: exclude_doc_id becomes a bool must_not ids clause, still
    # served by the content_hash keyword term — never a scan.
    s = OpenSearchDocStatusStorage.__new__(OpenSearchDocStatusStorage)
    s.workspace = "ws"
    s._index_name = "idx"
    s._index_ready = True
    s.client = AsyncMock()
    s.client.search = AsyncMock(
        return_value={
            "hits": {"hits": [{"_id": "doc-2", "_source": {"content_hash": "h"}}]}
        }
    )

    result = await s.get_doc_by_content_hash("h", exclude_doc_id="doc-1")
    assert result is not None and result[0] == "doc-2"
    body = s.client.search.call_args.kwargs["body"]
    assert body["query"]["bool"]["must"] == [{"term": {"content_hash": "h"}}]
    assert body["query"]["bool"]["must_not"] == [{"ids": {"values": ["doc-1"]}}]

    # No exclusion → plain term query.
    s.client.search.reset_mock()
    s.client.search.return_value = {"hits": {"hits": []}}
    await s.get_doc_by_content_hash("h")
    body = s.client.search.call_args.kwargs["body"]
    assert body["query"] == {"term": {"content_hash": "h"}}


def _content_hash_storage() -> OpenSearchDocStatusStorage:
    s = OpenSearchDocStatusStorage.__new__(OpenSearchDocStatusStorage)
    s.workspace = "ws"
    s._index_name = "idx"
    s._index_ready = True
    s.client = AsyncMock()
    return s


@pytest.mark.asyncio
async def test_content_hash_lookup_sorts_for_the_earliest_holder():
    """The base contract promises the EARLIEST other holder: that id is
    persisted as a duplicate's original_doc_id, so an unsorted size=1 query
    would make the attribution vary between runs over identical data."""
    s = _content_hash_storage()
    s.client.search = AsyncMock(
        return_value={
            "hits": {"hits": [{"_id": "doc-2", "_source": {"content_hash": "h"}}]}
        }
    )

    await s.get_doc_by_content_hash("h", exclude_doc_id="doc-1")

    body = s.client.search.call_args.kwargs["body"]
    assert body["sort"] == [
        {"created_at": {"order": "asc", "missing": "_first"}},
        {"__mirrored_id": {"order": "asc"}},
    ]
    # Ordered window (see test_content_hash_skips_pointer_rows_over_a_bounded_window):
    # every hit past the first is only reached when it has to skip a pointer row,
    # and the FIRST surviving hit is still the earliest holder.
    assert body["size"] == s._CONTENT_HASH_POINTER_WINDOW


@pytest.mark.asyncio
async def test_content_hash_lookup_is_fail_closed():
    """Fix-proof: a not-ready index, a vanished index and a transport error all
    used to return None, which the dedup callers read as "no duplicate" — so a
    storage failure enqueued a duplicate row and re-ingested its content. All
    three must raise; only a completed query with no hits is None.
    """
    # Not ready: absence cannot be confirmed.
    s = _content_hash_storage()
    s._index_ready = False
    s.client.search = AsyncMock()
    with pytest.raises(StorageControlPlaneError):
        await s.get_doc_by_content_hash("h", exclude_doc_id="doc-1")
    s.client.search.assert_not_awaited()

    # Vanished mid-query.
    s = _content_hash_storage()
    s.client.search = AsyncMock(side_effect=_missing_index_error())
    with pytest.raises(StorageControlPlaneError):
        await s.get_doc_by_content_hash("h", exclude_doc_id="doc-1")
    assert s._index_ready is False

    # Any other transport error propagates unchanged.
    s = _content_hash_storage()
    s.client.search = AsyncMock(side_effect=_transient_error())
    with pytest.raises(TransportError):
        await s.get_doc_by_content_hash("h", exclude_doc_id="doc-1")

    # A completed query with no hits IS a confirmed absence.
    s = _content_hash_storage()
    s.client.search = AsyncMock(return_value={"hits": {"hits": []}})
    assert await s.get_doc_by_content_hash("h", exclude_doc_id="doc-1") is None


@pytest.mark.asyncio
async def test_content_hash_skips_pointer_rows_over_a_bounded_window():
    """Second half of ``exclude_doc_id`` (base contract): a row marked
    ``is_duplicate`` naming the excluded id as its original records that the
    content belongs to the asking document, so returning it would close an
    is_duplicate cycle and leave their shared source with no primary.

    OpenSearch applies it to ``_source`` over a bounded ordered window rather
    than as a term query: ``metadata`` lives under ``dynamic: true`` with no
    declared mapping for ``original_doc_id``, so a term query on the bare path
    depends on whatever dynamic mapping produced (a string becomes ``text`` plus
    a ``.keyword`` subfield) and would silently match nothing — failing OPEN
    exactly where the pointer has to be excluded. The window must not truncate
    the search: the third holder below is returned, not swallowed.
    """
    s = _content_hash_storage()
    s.client.search = AsyncMock(
        return_value={
            "hits": {
                "hits": [
                    {
                        "_id": "doc-pointer",
                        "_source": {
                            "content_hash": "h",
                            "metadata": {
                                "is_duplicate": True,
                                "original_doc_id": "doc-1",
                            },
                        },
                    },
                    {"_id": "doc-3", "_source": {"content_hash": "h"}},
                ]
            }
        }
    )

    result = await s.get_doc_by_content_hash("h", exclude_doc_id="doc-1")

    assert result is not None and result[0] == "doc-3"
    body = s.client.search.call_args.kwargs["body"]
    assert body["size"] == s._CONTENT_HASH_POINTER_WINDOW
    assert body["sort"][0] == {"created_at": {"order": "asc", "missing": "_first"}}
    # Without an id to point AT there is nothing to skip, so one hit still suffices.
    s.client.search.return_value = {"hits": {"hits": []}}
    await s.get_doc_by_content_hash("h")
    assert s.client.search.call_args.kwargs["body"]["size"] == 1


@pytest.mark.asyncio
async def test_a_window_of_nothing_but_pointers_raises_instead_of_reporting_absence():
    """``None`` means CONFIRMED no other holder and the dedup callers act on it
    destructively, so a window that cannot answer must raise, not report absence."""
    s = _content_hash_storage()
    pointer = {
        "content_hash": "h",
        "metadata": {"is_duplicate": True, "original_doc_id": "doc-1"},
    }
    s.client.search = AsyncMock(
        return_value={
            "hits": {
                "hits": [
                    {"_id": f"doc-p{i}", "_source": pointer}
                    for i in range(
                        OpenSearchDocStatusStorage._CONTENT_HASH_POINTER_WINDOW
                    )
                ]
            }
        }
    )

    with pytest.raises(StorageControlPlaneError, match="bounded window"):
        await s.get_doc_by_content_hash("h", exclude_doc_id="doc-1")
