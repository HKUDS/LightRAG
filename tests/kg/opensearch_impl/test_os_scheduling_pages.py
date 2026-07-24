"""OpenSearch Phase-1 scheduling contracts (memory-bounding plan).

Offline coverage for the OpenSearch slice of the bounded-scheduling API:

* ``get_docs_by_statuses_page`` — native ``search_after`` keyset pages
  sorted ``(created_at ASC, __mirrored_id ASC)``, live-view (no PIT),
  server-side generation-cohort predicate whose ``must_not exists`` arm
  keeps legacy docs (no ``failure_generation`` field) eligible, cursor =
  the last HIT's sort values verbatim, ``hits < limit`` == CURSOR_END.
* ``count_docs_by_statuses`` — fail-closed ``_count``.
* ``update_doc_status_fields`` — immutable ``created_at`` + 404 mapping.
* failure-generation control plane — independent ctrl index, realtime GET
  + ``if_seq_no/if_primary_term`` CAS with bounded 409 retries, mode
  marker that NEVER degrades to LEGACY on read failure.
* ``mark_doc_failed`` — per-attempt idempotency (no counter touch) and
  CAS retry on conflict (reserved generations become holes, never reused).
* strict point/identity reads — three-state ``get_by_id_strict`` on both
  the KV and doc-status classes, and ``get_doc_by_file_basename_strict``
  raising where the legacy lookup swallows.
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

from opensearchpy import AsyncOpenSearch
from opensearchpy.exceptions import ConflictError, NotFoundError, TransportError

from lightrag.base import (
    CURSOR_END,
    CursorAfter,
    DocStatus,
    FailureGenerationMode,
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


def _conflict_error() -> ConflictError:
    return ConflictError(409, "version_conflict_engine_exception", "seq_no mismatch")


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


def _ctrl_response(
    counter: int = 0,
    mode: str = "enforced",
    schema_version: int = 1,
    seq_no: int = 0,
    primary_term: int = 1,
) -> dict:
    return {
        "_id": "scheduling_ctrl",
        "found": True,
        "_seq_no": seq_no,
        "_primary_term": primary_term,
        "_source": {
            "schema_version": schema_version,
            "mode": mode,
            "failure_generation_counter": counter,
        },
    }


def _make_client() -> AsyncMock:
    client = AsyncMock(spec=AsyncOpenSearch)
    client.indices = AsyncMock()
    client.indices.exists = AsyncMock(return_value=True)
    client.indices.create = AsyncMock()
    client.indices.get_mapping = AsyncMock(return_value={})
    client.indices.put_mapping = AsyncMock()
    # Ctrl bootstrap during initialize(): valid marker, counter already
    # calibrated (max aggregation answers "no persisted generations").
    client.get = AsyncMock(return_value=_ctrl_response())
    client.index = AsyncMock(return_value={})
    client.update = AsyncMock(return_value={})
    client.count = AsyncMock(return_value={"count": 0})
    client.search = AsyncMock(
        return_value={
            "hits": {"hits": []},
            "aggregations": {"max_failure_generation": {"value": None}},
        }
    )
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


def test_capability_flags():
    assert OpenSearchDocStatusStorage.supports_bounded_scheduling_pages is True
    assert OpenSearchDocStatusStorage.supports_failure_generation is True
    assert OpenSearchDocStatusStorage.supports_strict_doc_identity_lookup is True
    assert OpenSearchDocStatusStorage.supports_strict_point_reads is True
    assert OpenSearchKVStorage.supports_strict_point_reads is True


# ---------------------------------------------------------------------------
# get_docs_by_statuses_page: body construction
# ---------------------------------------------------------------------------


async def test_page_body_sort_size_and_no_generation_predicate(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.search = AsyncMock(return_value=_page([]))

    await storage.get_docs_by_statuses_page(
        [DocStatus.PENDING, DocStatus.FAILED], limit=25, strict=True
    )

    kwargs = client.search.call_args.kwargs
    assert kwargs["index"] == storage._index_name
    body = kwargs["body"]
    assert body["sort"] == [
        {"created_at": {"order": "asc"}},
        {"__mirrored_id": {"order": "asc"}},
    ]
    assert body["size"] == 25
    assert "search_after" not in body
    assert body["query"] == {"terms": {"status": ["failed", "pending"]}}
    # No cutoff → no generation predicate anywhere in the body.
    assert "failure_generation" not in json.dumps(body)


async def test_page_generation_predicate_includes_must_not_exists(global_config):
    """Legacy docs lack the failure_generation field entirely — the cohort
    query MUST carry the must_not-exists arm (mapping alone does not make
    missing read as 0), or every legacy FAILED row silently drops out of the
    first manual cohort."""
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.search = AsyncMock(return_value=_page([]))

    await storage.get_docs_by_statuses_page(
        [DocStatus.FAILED, DocStatus.PENDING],
        limit=10,
        max_failure_generation=7,
        strict=True,
    )

    body = client.search.call_args.kwargs["body"]
    filters = body["query"]["bool"]["filter"]
    assert filters[0] == {"terms": {"status": ["failed", "pending"]}}
    predicate = filters[1]["bool"]
    assert predicate["minimum_should_match"] == 1
    should = predicate["should"]
    # Non-FAILED rows are unaffected by the cohort cutoff.
    assert {"bool": {"must_not": {"term": {"status": "failed"}}}} in should
    # Migrated FAILED rows: generation <= cutoff.
    assert {"range": {"failure_generation": {"lte": 7}}} in should
    # Legacy FAILED rows WITHOUT the field stay eligible (logical 0).
    assert {"bool": {"must_not": {"exists": {"field": "failure_generation"}}}} in should
    assert len(should) == 3


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


async def test_page_missing_index_is_legitimately_empty(global_config):
    """Mirrors the existing get_docs_by_statuses strict decision: a live
    index_not_found is a legitimately empty (complete) sweep."""
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.search = AsyncMock(side_effect=_missing_index_error())

    page = await storage.get_docs_by_statuses_page(
        [DocStatus.PENDING], limit=5, strict=True
    )
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


async def test_count_missing_index_is_zero(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.count = AsyncMock(side_effect=_missing_index_error())
    assert await storage.count_docs_by_statuses([DocStatus.PENDING]) == 0
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
# get_failure_generation_mode
# ---------------------------------------------------------------------------


async def test_mode_missing_marker_is_migrating(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.get = AsyncMock(side_effect=_doc_missing_error())
    assert (
        await storage.get_failure_generation_mode() is FailureGenerationMode.MIGRATING
    )


async def test_mode_bad_schema_or_mode_is_migrating(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)

    client.get = AsyncMock(return_value=_ctrl_response(schema_version=99))
    assert (
        await storage.get_failure_generation_mode() is FailureGenerationMode.MIGRATING
    )

    client.get = AsyncMock(return_value=_ctrl_response(mode="bogus"))
    assert (
        await storage.get_failure_generation_mode() is FailureGenerationMode.MIGRATING
    )


async def test_mode_valid_values_map_through(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)

    client.get = AsyncMock(return_value=_ctrl_response(mode="enforced"))
    assert await storage.get_failure_generation_mode() is FailureGenerationMode.ENFORCED
    kwargs = client.get.call_args.kwargs
    assert kwargs["index"] == storage._ctrl_index_name

    client.get = AsyncMock(return_value=_ctrl_response(mode="legacy"))
    assert await storage.get_failure_generation_mode() is FailureGenerationMode.LEGACY


async def test_mode_transport_error_raises_never_degrades(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.get = AsyncMock(side_effect=_transient_error())
    with pytest.raises(StorageControlPlaneError):
        await storage.get_failure_generation_mode()


# ---------------------------------------------------------------------------
# reserve_failure_generation
# ---------------------------------------------------------------------------


async def test_reserve_cas_increments_counter(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.get = AsyncMock(
        return_value=_ctrl_response(counter=5, seq_no=3, primary_term=2)
    )
    client.index = AsyncMock(return_value={})

    assert await storage.reserve_failure_generation() == 6

    kwargs = client.index.call_args.kwargs
    assert kwargs["index"] == storage._ctrl_index_name
    assert kwargs["body"]["failure_generation_counter"] == 6
    assert kwargs["if_seq_no"] == 3
    assert kwargs["if_primary_term"] == 2


async def test_reserve_retries_on_conflict(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.get = AsyncMock(
        side_effect=[
            _ctrl_response(counter=5, seq_no=3),
            _ctrl_response(counter=6, seq_no=4),
        ]
    )
    client.index = AsyncMock(side_effect=[_conflict_error(), {}])

    # Lost the first CAS race to a concurrent reservation; the retry re-reads
    # the fresh counter and lands 7.
    assert await storage.reserve_failure_generation() == 7
    assert client.index.await_count == 2


async def test_reserve_missing_or_invalid_marker_raises(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)

    client.get = AsyncMock(side_effect=_doc_missing_error())
    with pytest.raises(StorageControlPlaneError):
        await storage.reserve_failure_generation()

    client.get = AsyncMock(return_value=_ctrl_response(schema_version=99))
    with pytest.raises(StorageControlPlaneError):
        await storage.reserve_failure_generation()


# ---------------------------------------------------------------------------
# mark_doc_failed
# ---------------------------------------------------------------------------


def _existing_row(**extra) -> dict:
    row = {
        "__mirrored_id": "doc-1",
        "content_summary": "s",
        "content_length": 1,
        "file_path": "doc-1.txt",
        "status": "processing",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "processing_attempt_id": "att-1",
    }
    row.update(extra)
    return row


async def test_mark_doc_failed_same_attempt_is_idempotent(global_config):
    """Already-FAILED with the same attempt returns the existing generation
    and never touches the counter (no reserve, no publish)."""
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.get = AsyncMock(
        return_value={
            "_id": "doc-1",
            "found": True,
            "_seq_no": 7,
            "_primary_term": 1,
            "_source": _existing_row(
                status="failed", failure_attempt_id="att-1", failure_generation=4
            ),
        }
    )
    client.index = AsyncMock(return_value={})

    generation = await storage.mark_doc_failed(
        "doc-1", {"processing_attempt_id": "att-1", "error_msg": "boom"}
    )
    assert generation == 4
    client.index.assert_not_awaited()


async def test_mark_doc_failed_cas_retry_on_conflict(global_config):
    """A 409 on the FAILED publish re-GETs, re-checks idempotency and
    retries; the first reserved generation becomes a permanent hole."""
    client = _make_client()
    storage = await _make_doc_status(global_config, client)

    ctrl_state = {"counter": 0, "seq": 0}
    doc_writes: list[dict] = []
    doc_write_kwargs: list[dict] = []

    async def _route_get(index=None, id=None, **kw):
        if index == storage._ctrl_index_name:
            return _ctrl_response(
                counter=ctrl_state["counter"], seq_no=ctrl_state["seq"]
            )
        return {
            "_id": id,
            "found": True,
            "_seq_no": 7,
            "_primary_term": 1,
            "_source": _existing_row(),
        }

    async def _route_index(index=None, id=None, body=None, **kw):
        if index == storage._ctrl_index_name:
            ctrl_state["counter"] = body["failure_generation_counter"]
            ctrl_state["seq"] += 1
            return {}
        doc_writes.append(body)
        doc_write_kwargs.append(kw)
        if len(doc_writes) == 1:
            raise _conflict_error()
        return {}

    client.get = AsyncMock(side_effect=_route_get)
    client.index = AsyncMock(side_effect=_route_index)

    generation = await storage.mark_doc_failed(
        "doc-1",
        {
            "processing_attempt_id": "att-1",
            "error_msg": "boom",
            # Caller-supplied created_at must be IGNORED for existing rows.
            "created_at": "2030-12-31T00:00:00+00:00",
        },
    )

    # First reservation (1) was published into the conflicted write and is a
    # hole; the retry reserved and published 2.
    assert generation == 2
    assert ctrl_state["counter"] == 2
    assert len(doc_writes) == 2
    final = doc_writes[-1]
    assert final["status"] == "failed"
    assert final["failure_generation"] == 2
    assert final["failure_attempt_id"] == "att-1"
    assert final["created_at"] == "2026-01-01T00:00:00+00:00"
    assert doc_write_kwargs[-1]["if_seq_no"] == 7
    assert doc_write_kwargs[-1]["if_primary_term"] == 1
    assert doc_write_kwargs[-1]["refresh"] == "wait_for"


async def test_mark_doc_failed_missing_row_conditional_create(global_config):
    """Parse/enqueue errors can fail before the PENDING row landed: the
    FAILED record is created with op_type=create so a concurrent create can
    never be clobbered."""
    client = _make_client()
    storage = await _make_doc_status(global_config, client)

    async def _route_get(index=None, id=None, **kw):
        if index == storage._ctrl_index_name:
            return _ctrl_response(counter=0, seq_no=0)
        raise _doc_missing_error()

    create_kwargs: list[dict] = []

    async def _route_index(index=None, id=None, body=None, **kw):
        if index == storage._ctrl_index_name:
            return {}
        create_kwargs.append({"body": body, **kw})
        return {}

    client.get = AsyncMock(side_effect=_route_get)
    client.index = AsyncMock(side_effect=_route_index)

    generation = await storage.mark_doc_failed(
        "doc-new",
        {
            "processing_attempt_id": "att-9",
            "error_msg": "parse exploded",
            "created_at": "2026-03-01T00:00:00+00:00",
        },
    )
    assert generation == 1
    assert len(create_kwargs) == 1
    assert create_kwargs[0]["op_type"] == "create"
    body = create_kwargs[0]["body"]
    assert body["status"] == "failed"
    assert body["failure_generation"] == 1
    assert body["failure_attempt_id"] == "att-9"
    assert body["__mirrored_id"] == "doc-new"
    assert body["chunks_list"] == []


# ---------------------------------------------------------------------------
# ensure_processing_attempt_id
# ---------------------------------------------------------------------------


async def test_ensure_attempt_id_reuses_persisted(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.get = AsyncMock(
        return_value={
            "_id": "doc-1",
            "found": True,
            "_seq_no": 1,
            "_primary_term": 1,
            "_source": _existing_row(processing_attempt_id="att-keep"),
        }
    )
    assert await storage.ensure_processing_attempt_id("doc-1") == "att-keep"
    client.update.assert_not_awaited()


async def test_ensure_attempt_id_mints_with_cas(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    row = _existing_row()
    row.pop("processing_attempt_id")
    client.get = AsyncMock(
        return_value={
            "_id": "doc-1",
            "found": True,
            "_seq_no": 5,
            "_primary_term": 2,
            "_source": row,
        }
    )
    minted = await storage.ensure_processing_attempt_id("doc-1")
    assert isinstance(minted, str) and len(minted) == 32
    kwargs = client.update.call_args.kwargs
    assert kwargs["body"] == {"doc": {"processing_attempt_id": minted}}
    assert kwargs["if_seq_no"] == 5
    assert kwargs["if_primary_term"] == 2


async def test_ensure_attempt_id_conflict_reuses_winner(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    bare = _existing_row()
    bare.pop("processing_attempt_id")
    client.get = AsyncMock(
        side_effect=[
            {
                "_id": "doc-1",
                "found": True,
                "_seq_no": 5,
                "_primary_term": 2,
                "_source": bare,
            },
            {
                "_id": "doc-1",
                "found": True,
                "_seq_no": 6,
                "_primary_term": 2,
                "_source": _existing_row(processing_attempt_id="att-winner"),
            },
        ]
    )
    client.update = AsyncMock(side_effect=_conflict_error())
    assert await storage.ensure_processing_attempt_id("doc-1") == "att-winner"


async def test_ensure_attempt_id_missing_record(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.get = AsyncMock(side_effect=_doc_missing_error())
    with pytest.raises(StorageRecordNotFoundError):
        await storage.ensure_processing_attempt_id("ghost")


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
# Basename lookups: primary-only + strict variant
# ---------------------------------------------------------------------------


async def test_basename_queries_exclude_duplicate_rows(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    client.search = AsyncMock(return_value=_page([]))

    assert await storage.get_doc_by_file_basename("report.pdf") is None
    body = client.search.call_args.kwargs["body"]
    assert body["query"]["bool"]["filter"] == [{"term": {"file_path": "report.pdf"}}]
    assert body["query"]["bool"]["must_not"] == [
        {"term": {"metadata.is_duplicate": True}}
    ]

    client.search.reset_mock()
    assert await storage.get_doc_by_file_basename_strict("report.pdf") is None
    strict_body = client.search.call_args.kwargs["body"]
    assert strict_body == body


async def test_basename_strict_raises_where_legacy_swallows(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)

    client.search = AsyncMock(side_effect=_transient_error())
    assert await storage.get_doc_by_file_basename("report.pdf") is None

    client.search = AsyncMock(side_effect=_transient_error())
    with pytest.raises(TransportError):
        await storage.get_doc_by_file_basename_strict("report.pdf")

    client.search = AsyncMock(side_effect=_missing_index_error())
    with pytest.raises(StorageControlPlaneError):
        await storage.get_doc_by_file_basename_strict("report.pdf")

    storage._index_ready = False
    with pytest.raises(StorageControlPlaneError):
        await storage.get_doc_by_file_basename_strict("report.pdf")

    # Sentinel basenames are confirmed-absent without any lookup.
    assert await storage.get_doc_by_file_basename_strict("") is None
    assert await storage.get_doc_by_file_basename_strict("unknown_source") is None


async def test_basename_strict_returns_primary_hit(global_config):
    client = _make_client()
    storage = await _make_doc_status(global_config, client)
    source = _status_source("doc-1", status="processed")
    client.search = AsyncMock(return_value=_page([_hit("doc-1", source)]))
    result = await storage.get_doc_by_file_basename_strict("doc-1.txt")
    assert result == ("doc-1", source)


# ---------------------------------------------------------------------------
# Ctrl bootstrap at initialize()
# ---------------------------------------------------------------------------


async def test_bootstrap_seeds_marker_from_max_aggregation(global_config):
    client = _make_client()
    client.get = AsyncMock(side_effect=_doc_missing_error())
    client.search = AsyncMock(
        return_value={"aggregations": {"max_failure_generation": {"value": 9.0}}}
    )
    storage = await _make_doc_status(global_config, client)

    ctrl_creates = [
        call.kwargs
        for call in client.index.await_args_list
        if call.kwargs.get("index") == storage._ctrl_index_name
    ]
    assert len(ctrl_creates) == 1
    assert ctrl_creates[0]["op_type"] == "create"
    assert ctrl_creates[0]["body"] == {
        "schema_version": 1,
        "mode": "enforced",
        "failure_generation_counter": 9,
    }


async def test_bootstrap_recalibrates_counter_behind_persisted_rows(global_config):
    client = _make_client()
    client.get = AsyncMock(return_value=_ctrl_response(counter=3, seq_no=11))
    client.search = AsyncMock(
        return_value={"aggregations": {"max_failure_generation": {"value": 9.0}}}
    )
    storage = await _make_doc_status(global_config, client)

    kwargs = client.index.call_args.kwargs
    assert kwargs["index"] == storage._ctrl_index_name
    assert kwargs["body"]["failure_generation_counter"] == 9
    assert kwargs["if_seq_no"] == 11
