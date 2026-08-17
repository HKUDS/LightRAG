import asyncio
import itertools
import logging
from collections.abc import Callable, Generator
from typing import Any
from unittest.mock import patch, MagicMock

import grpc  # type: ignore
import numpy as np
import pytest
from pymilvus import MilvusException

from lightrag.kg.milvus_impl import MilvusVectorDBStorage, _escape_milvus_str

pytestmark = pytest.mark.offline

# the gRPC default of 4 MiB enforced by the gateway.
_GRPC_MAX_MESSAGE_BYTES = 4 * 1024 * 1024
_ROW_OVERHEAD_BYTES = 256


class MockEmbeddingFunc:
    """Mock embedding function that returns random vectors."""

    def __init__(self, dim: int = 8) -> None:
        self.embedding_dim = dim
        self.max_token_size = 512
        self.model_name = "mock-embed"

    async def __call__(self, texts: list[str], **kwargs: Any) -> np.ndarray:
        return np.random.rand(len(texts), self.embedding_dim).astype(np.float32)


@pytest.fixture(autouse=True)
def patch_namespace_lock() -> Generator[
    dict[tuple[str, str], asyncio.Lock], None, None
]:
    """Cache real asyncio.Locks per (namespace, workspace) for shared semantics.

    Two storage instances whose ``final_namespace`` matches must observe the
    same Lock instance — this fixture lets us assert that and also exercises
    real serialization between concurrent flush/upsert coroutines.
    """
    cache: dict[tuple[str, str], asyncio.Lock] = {}

    def factory(
        namespace: str, workspace: str | None = None, enable_logging: bool = False
    ) -> asyncio.Lock:
        key = (namespace, workspace or "")
        lock = cache.get(key)
        if lock is None:
            lock = asyncio.Lock()
            cache[key] = lock
        return lock

    with patch("lightrag.kg.milvus_impl.get_namespace_lock", side_effect=factory):
        yield cache


def _make_storage(
    embed_func: MockEmbeddingFunc,
    *,
    namespace: str = "entities",
    workspace: str = "test",
    meta_fields: set[str] | None = None,
) -> MilvusVectorDBStorage:
    """Build a MilvusVectorDBStorage skipping `initialize()` (no real client)."""
    if meta_fields is None:
        meta_fields = {"content", "entity_name", "src_id", "tgt_id"}
    storage = MilvusVectorDBStorage(
        namespace=namespace,
        workspace=workspace,
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        embedding_func=embed_func,
        meta_fields=meta_fields,
    )
    # Bypass real Milvus client; manually wire the bits initialize() would set.
    # The flush lock is already constructed in __post_init__ via the patched
    # get_namespace_lock factory, so no manual lock wiring is needed here.
    storage._client = MagicMock()
    storage._client.has_collection.return_value = True
    storage._client.upsert = MagicMock(return_value={"upsert_count": 0})
    storage._client.delete = MagicMock(return_value={"delete_count": 0})
    storage._client.query = MagicMock(return_value=[])
    storage._client.load_collection = MagicMock()
    storage._initialized = True
    return storage


def _requested_id_count(filter: str) -> int:
    """Number of ids packed into an ``id in [...]`` filter — a plain comma
    count, not escaping-aware parsing. We only need "how many", not "which"."""
    return filter.count(",") + 1


def _resource_exhausted_query_side_effect(
    embedding_dim: int,
) -> Callable[..., list[dict[str, Any]]]:
    row_bytes = embedding_dim * 4 + _ROW_OVERHEAD_BYTES
    row_ids = itertools.count()

    def _side_effect(*, filter: str, **kwargs: Any) -> list[dict[str, Any]]:
        count = _requested_id_count(filter)
        response_bytes = count * row_bytes
        if response_bytes > _GRPC_MAX_MESSAGE_BYTES:
            raise MilvusException(
                message=(
                    "grpc: received message larger than max "
                    f"({response_bytes} vs. {_GRPC_MAX_MESSAGE_BYTES})"
                )
            )
        return [
            {"id": f"row-{next(row_ids)}", "vector": [0.0] * embedding_dim}
            for _ in range(count)
        ]

    return _side_effect


def _echo_query_side_effect(
    embedding_dim: int, call_sizes: list[int]
) -> Callable[..., list[dict[str, Any]]]:
    """query() side_effect: records the id count of each call
    and echoes that many unique placeholder rows."""
    row_ids = itertools.count()

    def _side_effect(*, filter: str, **kwargs: Any) -> list[dict[str, Any]]:
        count = _requested_id_count(filter)
        call_sizes.append(count)
        return [
            {"id": f"row-{next(row_ids)}", "vector": [0.0] * embedding_dim}
            for _ in range(count)
        ]

    return _side_effect


async def _page_and_record(
    embedding_dim: int, ids: list[str]
) -> tuple[dict[str, list[float]], list[int]]:
    """Run get_vectors_by_ids and return (vectors, per-call id counts)."""
    call_sizes: list[int] = []
    s = _make_storage(MockEmbeddingFunc(dim=embedding_dim))
    s._client.query.side_effect = _echo_query_side_effect(embedding_dim, call_sizes)
    vectors = await s.get_vectors_by_ids(ids)
    return vectors, call_sizes


def _oversize_above_side_effect(
    max_ids_per_response: int, call_sizes: list[int]
) -> Callable[..., list[dict[str, Any]]]:
    """query() side_effect modelling a gateway that rejects any response
    carrying more than `max_ids_per_response` rows.

    Metadata rows have no client-estimable size, so this is the failure the
    record cap alone cannot prevent: the caller must shrink the page itself.
    """

    def _side_effect(*, filter: str, **kwargs: Any) -> list[dict[str, Any]]:
        count = _requested_id_count(filter)
        call_sizes.append(count)
        if count > max_ids_per_response:
            raise MilvusException(
                message=(
                    "<MilvusException: (code=1, message=grpc: received message "
                    f"larger than max ({count} rows vs. {max_ids_per_response}))>"
                )
            )
        return []

    return _side_effect


class _QuotaExhaustedRpcError(grpc.RpcError):
    """RESOURCE_EXHAUSTED carrying no size marker — the shape a per-user quota
    or rate-limit rejection takes. grpc reuses this status for both, so only
    the message text separates throttling from an oversized response."""

    def code(self) -> grpc.StatusCode:
        return grpc.StatusCode.RESOURCE_EXHAUSTED

    def __str__(self) -> str:
        return (
            "<_InactiveRpcError of RPC that terminated with: "
            "StatusCode.RESOURCE_EXHAUSTED, rate limit exceeded>"
        )


def _ordered_echo_side_effect(
    ordered_ids: list[str], captured_filters: list[str]
) -> Callable[..., list[dict[str, Any]]]:
    """query() side_effect: pages sequentially through `ordered_ids` (using
    the same count-only filter signal as _requested_id_count) and records
    every filter seen — lets a test assert both result identity/order and
    that specific ids never appeared in a server-bound filter."""
    cursor = list(ordered_ids)

    def _side_effect(*, filter: str, **kwargs: Any) -> list[dict[str, Any]]:
        captured_filters.append(filter)
        count = _requested_id_count(filter)
        page = cursor[:count]
        del cursor[:count]
        return [{"id": doc_id, "content": f"server-{doc_id}"} for doc_id in page]

    return _side_effect


@pytest.mark.offline
class TestGetVectorsByIdsPaging:
    @pytest.mark.asyncio
    async def test_returns_all_vectors_above_grpc_size_limit(self) -> None:
        """600 ids at dim=3072 would exceed the gRPC ceiling in one query() call;
        all 600 vectors must still come back."""
        embed = MockEmbeddingFunc(dim=3072)
        s = _make_storage(embed)
        s._client.query.side_effect = _resource_exhausted_query_side_effect(
            embedding_dim=3072
        )

        ids = [f"chunk-{i}" for i in range(600)]
        vectors = await s.get_vectors_by_ids(ids)

        assert len(vectors) == 600
        assert s._client.query.call_count > 1

    @pytest.mark.asyncio
    async def test_skips_id_missing_from_database(self) -> None:
        """A single id that is neither buffered nor pending-delete but simply
        doesn't exist in Milvus must not appear in the result and must not
        raise — the server matches nothing, not an error."""
        embed = MockEmbeddingFunc(dim=8)
        s = _make_storage(embed)
        s._client.query.return_value = []

        vectors = await s.get_vectors_by_ids(["missing-1"])

        assert vectors == {}

    @pytest.mark.asyncio
    async def test_page_size_shrinks_with_higher_dimension(self) -> None:
        """Heavier rows (higher embedding dim) fit fewer ids per call, so paging
        the same 1000 ids must take more query() calls at dim=3072 than at
        dim=8 — while every call, at either dimension, stays within its
        dimension's derived page-size bound."""
        ids = [f"chunk-{i}" for i in range(1000)]

        vectors_dim8, call_sizes_dim8 = await _page_and_record(8, ids)
        vectors_dim3072, call_sizes_dim3072 = await _page_and_record(3072, ids)

        assert len(vectors_dim8) == len(ids)
        assert len(vectors_dim3072) == len(ids)
        assert sum(call_sizes_dim8) == len(ids)
        assert sum(call_sizes_dim3072) == len(ids)

        # Smaller pages at the higher dimension mean more calls to cover the
        # same id list.
        assert len(call_sizes_dim3072) > len(call_sizes_dim8)

    @pytest.mark.asyncio
    async def test_record_cap_binds_at_small_dimension(self) -> None:
        """At a tiny embedding dim the byte budget alone would allow a huge
        page, so the record-count cap is the only thing left to bind it. Patch
        the module constant to a small value instead of relying on the shipped
        256."""
        ids = [f"chunk-{i}" for i in range(20)]

        with patch("lightrag.kg.milvus_impl.MILVUS_QUERY_MAX_RECORDS_PER_BATCH", 5):
            vectors, call_sizes = await _page_and_record(1, ids)

        assert len(vectors) == len(ids)
        assert sum(call_sizes) == len(ids)
        assert len(call_sizes) > 1
        assert all(size <= 5 for size in call_sizes)

    @pytest.mark.asyncio
    async def test_returns_buffered_portion_on_page_failure(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A page raising partway through server-side pagination must not crash
        get_vectors_by_ids: it falls back to whatever was already resolved from
        the buffer, logs the failure, and swallows the exception — and the
        first page's rows, which succeeded before the second page raised, must
        not leak into the result either (no partial pages)."""
        embed = MockEmbeddingFunc(dim=8)
        s = _make_storage(embed)

        await s.upsert({"buffered-1": {"content": "hello"}})

        remaining_ids = [f"srv-{i}" for i in range(10)]
        ids = ["buffered-1", *remaining_ids]

        call_count = 0

        def _side_effect(*, filter: str, **kwargs: Any) -> list[dict[str, Any]]:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise MilvusException(message="milvus down mid-page")
            count = _requested_id_count(filter)
            return [
                {"id": f"row-{call_count}-{i}", "vector": [0.0] * 8}
                for i in range(count)
            ]

        s._client.query.side_effect = _side_effect

        lightrag_logger = logging.getLogger("lightrag")
        previous_propagate = lightrag_logger.propagate
        lightrag_logger.propagate = True  # caplog hooks the root logger
        try:
            with caplog.at_level(logging.ERROR, logger="lightrag"):
                with patch(
                    "lightrag.kg.milvus_impl.MILVUS_QUERY_MAX_RECORDS_PER_BATCH", 2
                ):
                    vectors = await s.get_vectors_by_ids(ids)
        finally:
            lightrag_logger.propagate = previous_propagate

        # Buffered vector survives; nothing from the server (not even the
        # first, successful page) leaked into the result.
        assert vectors == {"buffered-1": s._pending_vector_docs["buffered-1"].vector}
        assert not any(doc_id.startswith("row-") for doc_id in vectors)
        assert "Error retrieving vectors by IDs" in caplog.text


@pytest.mark.offline
class TestGetByIdsPaging:
    @pytest.mark.asyncio
    async def test_preserves_order_and_excludes_buffered_and_deleted(self) -> None:
        """get_by_ids must preserve the caller's input order across a paginated
        server query, never send buffered/pending-delete ids to the server, and
        return None for an id that is pending delete."""
        embed = MockEmbeddingFunc(dim=8)
        s = _make_storage(embed, meta_fields={"content"})

        await s.upsert({"buffered-1": {"content": "buffered content"}})
        await s.delete(["deleted-1"])

        server_ids = [f"srv-{i}" for i in range(12)]
        # Interleave the buffered/deleted ids among the server ids so a
        # positional check actually exercises order preservation.
        ids = [
            server_ids[0],
            "buffered-1",
            server_ids[1],
            "deleted-1",
            *server_ids[2:],
        ]

        captured_filters: list[str] = []
        s._client.query.side_effect = _ordered_echo_side_effect(
            server_ids, captured_filters
        )

        with patch("lightrag.kg.milvus_impl.MILVUS_QUERY_MAX_RECORDS_PER_BATCH", 5):
            results = await s.get_by_ids(ids)

        assert len(results) == len(ids)
        for doc_id, row in zip(ids, results):
            if doc_id == "deleted-1":
                assert row is None
            elif doc_id == "buffered-1":
                assert row["id"] == "buffered-1"
                assert row["content"] == "buffered content"
            else:
                assert row == {"id": doc_id, "content": f"server-{doc_id}"}

        # The buffered upsert and the pending delete must never reach the server.
        for filter_expr in captured_filters:
            assert "buffered-1" not in filter_expr
            assert "deleted-1" not in filter_expr

        assert s._client.query.call_count > 1

    @pytest.mark.asyncio
    async def test_returns_none_for_ids_missing_from_the_database(self) -> None:
        """An id that is neither buffered nor pending-delete but simply doesn't
        exist in Milvus must come back as None at its own position — the missing
        row must not raise, and must not shift the other results out of place."""
        embed = MockEmbeddingFunc(dim=8)
        s = _make_storage(embed, meta_fields={"content"})

        ids = ["missing-1"]

        results = await s.get_by_ids(ids)

        assert results == [None]

    @pytest.mark.asyncio
    async def test_escapes_special_characters_across_pages(self) -> None:
        """Escaping must hold for every page's filter, not just a single
        unpaginated call — an id containing `"` or `\\` must still come out
        correctly escaped in whichever page it lands in."""
        embed = MockEmbeddingFunc(dim=8)
        s = _make_storage(embed, meta_fields={"content"})

        quote_id = 'doc"1'
        backslash_id = "doc\\2"
        plain_ids = [f"plain-{i}" for i in range(4)]
        # Interleaved so the two special ids land in different pages once the
        # record cap is patched down to 2.
        ids = [plain_ids[0], quote_id, plain_ids[1], backslash_id, *plain_ids[2:]]

        captured_filters: list[str] = []

        def _side_effect(*, filter: str, **kwargs: Any) -> list[dict[str, Any]]:
            captured_filters.append(filter)
            return []

        s._client.query.side_effect = _side_effect

        with patch("lightrag.kg.milvus_impl.MILVUS_QUERY_MAX_RECORDS_PER_BATCH", 2):
            await s.get_by_ids(ids)

        assert s._client.query.call_count > 1

        joined_filters = " ".join(captured_filters)
        assert f'"{_escape_milvus_str(quote_id)}"' in joined_filters
        assert f'"{_escape_milvus_str(backslash_id)}"' in joined_filters


@pytest.mark.offline
class TestOversizePageBisection:
    """The record cap bounds metadata pages by count, not by bytes: `content`
    and `source_id` are each capped only by MILVUS_MAX_VARCHAR_BYTES, so 256
    rows can still cross the gateway ceiling. Such a page must be halved and
    retried rather than failing the whole read."""

    @pytest.mark.asyncio
    async def test_oversize_metadata_page_is_bisected_and_returns_every_row(
        self,
    ) -> None:
        embed = MockEmbeddingFunc(dim=8)
        s = _make_storage(embed, meta_fields={"content"})

        server_ids = [f"srv-{i}" for i in range(12)]
        captured_filters: list[str] = []
        echo = _ordered_echo_side_effect(server_ids, captured_filters)

        # Reject any response above 4 rows, echoing the requested rows below it.
        def _side_effect(*, filter: str, **kwargs: Any) -> list[dict[str, Any]]:
            if _requested_id_count(filter) > 4:
                raise MilvusException(
                    message="grpc: received message larger than max (x vs. y)"
                )
            return echo(filter=filter, **kwargs)

        s._client.query.side_effect = _side_effect

        with patch("lightrag.kg.milvus_impl.MILVUS_QUERY_MAX_RECORDS_PER_BATCH", 8):
            results = await s.get_by_ids(server_ids)

        # Every row came back, in the caller's order, despite the first page
        # overflowing.
        assert results == [
            {"id": doc_id, "content": f"server-{doc_id}"} for doc_id in server_ids
        ]

    @pytest.mark.asyncio
    async def test_reduced_page_size_carries_to_the_remaining_pages(self) -> None:
        """After one page overflows, the smaller size must be kept for the rest
        of the id list — otherwise every subsequent page pays its own failure
        first, turning one overflow into one per page."""
        embed = MockEmbeddingFunc(dim=8)
        s = _make_storage(embed, meta_fields={"content"})

        call_sizes: list[int] = []
        s._client.query.side_effect = _oversize_above_side_effect(4, call_sizes)

        ids = [f"srv-{i}" for i in range(20)]
        with patch("lightrag.kg.milvus_impl.MILVUS_QUERY_MAX_RECORDS_PER_BATCH", 8):
            await s.get_by_ids(ids)

        # One rejected 8-id page, then five accepted 4-id pages. A page size
        # that reset to 8 each time would instead reject five times (10 calls).
        assert call_sizes == [8, 4, 4, 4, 4, 4]

    @pytest.mark.asyncio
    async def test_bare_resource_exhausted_status_is_not_bisected(self) -> None:
        """grpc reuses RESOURCE_EXHAUSTED for per-user quota, so classifying on
        the status alone would read a rate-limit rejection as an oversized
        response and re-issue it at every halved size with no backoff —
        hammering the gateway that is already throttling. Only the size marker
        in the message may trigger bisection; a genuine overflow always carries
        it (see issue #3584), so nothing real is lost."""
        embed = MockEmbeddingFunc(dim=8)
        s = _make_storage(embed, meta_fields={"content"})

        call_sizes: list[int] = []

        def _side_effect(*, filter: str, **kwargs: Any) -> list[dict[str, Any]]:
            call_sizes.append(_requested_id_count(filter))
            raise _QuotaExhaustedRpcError()

        s._client.query.side_effect = _side_effect

        ids = [f"srv-{i}" for i in range(8)]
        with patch("lightrag.kg.milvus_impl.MILVUS_QUERY_MAX_RECORDS_PER_BATCH", 8):
            results = await s.get_by_ids(ids)

        # One attempt only: no 8 -> 4 -> 2 -> 1 retry ladder.
        assert call_sizes == [8]
        assert results == []

    @pytest.mark.asyncio
    async def test_non_size_error_is_not_bisected(self) -> None:
        """A connection or schema failure must fail the read immediately. Retrying
        it at half the page size cannot help and would multiply the outage."""
        embed = MockEmbeddingFunc(dim=8)
        s = _make_storage(embed, meta_fields={"content"})

        s._client.query.side_effect = MilvusException(message="milvus down")

        ids = [f"srv-{i}" for i in range(8)]
        with patch("lightrag.kg.milvus_impl.MILVUS_QUERY_MAX_RECORDS_PER_BATCH", 4):
            results = await s.get_by_ids(ids)

        assert s._client.query.call_count == 1
        assert results == []

    @pytest.mark.asyncio
    async def test_overflow_surviving_to_a_single_id_returns_no_partial_rows(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Bisection bottoms out at one id. The read then fails whole: the pages
        that already succeeded must not be returned, since a short result is a
        storage-consistency signal to callers, not a partial one."""
        embed = MockEmbeddingFunc(dim=8)
        s = _make_storage(embed, meta_fields={"content"})

        call_sizes: list[int] = []
        # Rejects at every size, including a single id.
        s._client.query.side_effect = _oversize_above_side_effect(0, call_sizes)

        ids = [f"srv-{i}" for i in range(8)]
        lightrag_logger = logging.getLogger("lightrag")
        previous_propagate = lightrag_logger.propagate
        lightrag_logger.propagate = True  # caplog hooks the root logger
        try:
            with caplog.at_level(logging.ERROR, logger="lightrag"):
                with patch(
                    "lightrag.kg.milvus_impl.MILVUS_QUERY_MAX_RECORDS_PER_BATCH", 8
                ):
                    results = await s.get_by_ids(ids)
        finally:
            lightrag_logger.propagate = previous_propagate

        assert call_sizes == [8, 4, 2, 1]
        assert results == []
        assert "Error retrieving vector data" in caplog.text
        # The error logs a bounded 5-id sample, never the whole id list.
        assert "srv-7" not in caplog.text


@pytest.mark.offline
class TestPagingCooperativeYield:
    @pytest.mark.asyncio
    async def test_event_loop_runs_between_pages(self) -> None:
        """Each page is a blocking gRPC round-trip, so the loop must be released
        between pages. At `_cooperative_yield`'s default 64-iteration cadence a
        realistic id count (page size caps at 256) would never reach the first
        yield, starving every other coroutine for the whole read.
        """
        embed = MockEmbeddingFunc(dim=8)
        s = _make_storage(embed, meta_fields={"content"})
        s._client.query.side_effect = lambda **kwargs: []

        observer_ticks = 0

        async def observer() -> None:
            nonlocal observer_ticks
            while True:
                observer_ticks += 1
                await asyncio.sleep(0)

        ids = [f"srv-{i}" for i in range(20)]
        task = asyncio.create_task(observer())
        try:
            with patch("lightrag.kg.milvus_impl.MILVUS_QUERY_MAX_RECORDS_PER_BATCH", 2):
                await s.get_by_ids(ids)  # 10 pages
        finally:
            task.cancel()

        # 10 pages yield 9 times (the last page does not), so the observer must
        # have been scheduled at least that often. Without a per-page yield the
        # read never suspends and the observer never runs at all.
        assert observer_ticks >= 9
