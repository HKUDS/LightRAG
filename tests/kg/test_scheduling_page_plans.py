"""EXPLAIN / profile the scheduling page: no full-table sort (LR2 §13.1).

§13.1 asks for exactly this, and asking was justified — both assertions below
FAILED when first written. The page query is the sweep's hot-path read, and a
plan regression there is invisible from outside: results stay correct, the page
stays memory-bounded (a top-N sort keeps only ``limit`` rows), and only the I/O
quietly becomes O(table) per page, i.e. O(n²/page_size) for a full sweep. Nothing
but a plan assertion catches that.

What was found and fixed:

* **PostgreSQL** — one status was a bitmap scan plus a top-N sort of every row
  past the cursor; TWO statuses (the AUTO sweep's own shape) was a full
  ``Seq Scan`` per page. Two causes, both needed: the index has to be declared
  ``created_at NULLS FIRST`` (PG's ``ASC`` default is NULLS LAST, so the old
  index could not satisfy the ORDER BY at all), and the query has to use
  per-status equality branches (a single ``status = ANY(...)`` cannot produce
  ordered output).
* **MongoDB** — multi-status ``$in`` fell back to a blocking ``SORT`` examining
  the whole matching key range.

Live services only (``--run-integration``); see
``test_doc_status_scheduling_contract`` for how to start them.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from lightrag.base import CURSOR_START, CursorAfter, DocStatus

from .conftest_scheduling_backends import BACKEND_SPECS

pytestmark = [pytest.mark.integration, pytest.mark.requires_db]

# Enough rows that a planner has a real choice to make; small enough to seed fast.
_ROWS = 20000
_PAGE = 500
_BASE = datetime(2026, 1, 1)
_STATUS_CYCLE = (
    DocStatus.PENDING,
    DocStatus.PROCESSED,
    DocStatus.FAILED,
    DocStatus.PROCESSING,
)
# The keyset order every backend's page sorts by — and, for Mongo, the order the
# explain below is judged against.
_PAGE_SORT = [("created_at", 1), ("_id", 1)]


def _spec(name: str):
    return next(spec for spec in BACKEND_SPECS if spec.name == name)


async def _seed_at_scale(doc_status) -> None:
    """Cycle statuses so no single status is contiguous in storage order — a
    contiguous layout would let a scan look cheap for the wrong reason."""
    batch: dict[str, dict] = {}
    for i in range(_ROWS):
        stamp = (_BASE + timedelta(seconds=i)).isoformat()
        batch[f"doc-{i:07d}"] = {
            "status": _STATUS_CYCLE[i % len(_STATUS_CYCLE)].value,
            "created_at": stamp,
            "updated_at": stamp,
            "content_summary": "x" * 20,
            "content_length": 200,
            "file_path": f"f{i}.md",
            "chunks_list": [],
            "chunks_count": 0,
            "track_id": "t",
        }
        if len(batch) >= 5000:
            await doc_status.upsert(batch)
            batch = {}
    if batch:
        await doc_status.upsert(batch)
    await doc_status.index_done_callback()


async def _mid_sweep_cursor(doc_status, statuses) -> CursorAfter:
    """A cursor roughly halfway in, so the plan is judged where a scan-based plan
    hurts most rather than on the first page."""
    position: object = CURSOR_START
    for _ in range(_ROWS // (4 * _PAGE)):
        page = await doc_status.get_docs_by_statuses_page(
            statuses, limit=_PAGE, position=position, strict=True
        )
        position = page.next_position
        if not isinstance(position, CursorAfter):
            break
    assert isinstance(position, CursorAfter), "backlog too small to reach mid-sweep"
    return position


async def test_postgres_page_is_an_ordered_index_scan(tmp_path):
    """One test, three status-set widths: seeding 20k rows once and asserting
    three plans beats paying the seed three times, and the widths matter — a
    single status and a multi-status sweep took different (both wrong) plans.
    """
    spec = _spec("postgres")
    spec.skip_unless_available()
    doc_status = await spec.build(tmp_path)
    try:
        await _seed_at_scale(doc_status)

        for statuses in (
            [DocStatus.PENDING],
            [DocStatus.PENDING, DocStatus.FAILED],
            [DocStatus.PENDING, DocStatus.PROCESSING, DocStatus.FAILED],
        ):
            position = await _mid_sweep_cursor(doc_status, statuses)
            sql, params = await _capture_page_sql(doc_status, statuses, position)
            plan_rows = await doc_status.db.query(
                "EXPLAIN (ANALYZE) " + sql, params, multirows=True
            )
            plan = "\n".join(row["QUERY PLAN"] for row in plan_rows)
            label = f"{len(statuses)} status(es)"

            assert "Seq Scan" not in plan, f"{label}\n{plan}"
            # A Sort node means the index could not supply the requested order.
            assert "Sort Method" not in plan, f"{label}\n{plan}"
            assert "Index Scan" in plan or "Index Only Scan" in plan, f"{label}\n{plan}"
    finally:
        try:
            await doc_status.drop()
        finally:
            await doc_status.finalize()


async def _capture_page_sql(doc_status, statuses, position):
    """Capture the SQL the real page method builds rather than restating it here:
    a future rewrite must be explained as-changed, not as-remembered."""
    captured: dict = {}
    original_query = doc_status.db.query

    async def _capture(sql, params=None, multirows=False, **kwargs):
        captured.setdefault("sql", sql)
        captured.setdefault("params", params)
        return await original_query(sql, params, multirows=multirows, **kwargs)

    doc_status.db.query = _capture
    try:
        await doc_status.get_docs_by_statuses_page(
            statuses, limit=_PAGE, position=position, strict=True
        )
    finally:
        doc_status.db.query = original_query
    return captured["sql"], captured["params"]


async def test_mongodb_page_uses_the_index_order_not_a_blocking_sort(tmp_path):
    spec = _spec("mongodb")
    spec.skip_unless_available()
    storage = await spec.build(tmp_path)
    try:
        await _seed_at_scale(storage)
        statuses = [DocStatus.PENDING, DocStatus.FAILED]
        position = await _mid_sweep_cursor(storage, statuses)

        plan = await _mongo_page_explain(storage, statuses, position)
        winning = json.dumps(plan["queryPlanner"]["winningPlan"])
        stats = plan["executionStats"]

        assert "COLLSCAN" not in winning, winning
        assert '"stage": "SORT"' not in winning, winning
        # Index-ordered paging examines about a page worth of keys, not the whole
        # matching set (~10k here).
        assert stats["totalKeysExamined"] <= 6 * _PAGE, stats
    finally:
        try:
            await storage.drop()
        finally:
            await storage.finalize()


async def test_opensearch_pages_with_search_after_not_deep_from_size(tmp_path):
    """``search_after`` is keyset by construction, so what matters is that the
    implementation uses it instead of ``from``/``size``, which degrades with
    offset."""
    spec = _spec("opensearch")
    spec.skip_unless_available()
    storage = await spec.build(tmp_path)
    try:
        await _seed_at_scale(storage)
        statuses = [DocStatus.PENDING, DocStatus.FAILED]
        position = await _mid_sweep_cursor(storage, statuses)

        bodies: list[dict] = []
        original_search = storage.client.search

        async def _capture(*args, **kwargs):
            body = kwargs.get("body")
            if isinstance(body, dict):
                bodies.append(body)
            return await original_search(*args, **kwargs)

        storage.client.search = _capture
        try:
            await storage.get_docs_by_statuses_page(
                statuses, limit=_PAGE, position=position, strict=True
            )
        finally:
            storage.client.search = original_search

        assert bodies, "no search issued"
        page_body = bodies[0]
        assert "search_after" in page_body, page_body
        assert "from" not in page_body, page_body
        assert page_body["sort"], page_body
    finally:
        try:
            await storage.drop()
        finally:
            await storage.finalize()


class _CaptureFind:
    """Collection proxy that records the filter its ``find`` is called with.

    Wrapping the storage's ``_data`` attribute rather than patching a method on
    the driver's collection object keeps this independent of whatever motor lets
    us assign to."""

    def __init__(self, collection, captured: dict):
        self._collection = collection
        self._captured = captured

    def find(self, query, *args, **kwargs):
        self._captured.setdefault("query", query)
        return self._collection.find(query, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._collection, name)


async def _mongo_page_explain(storage, statuses, position):
    """Explain the filter the page really issues from ``position``.

    Captured, not restated, for the same reason ``_capture_page_sql`` captures
    the SQL: the previous version hand-rolled a flat ``$or`` and started from
    ``find_one``'s FIRST matching row, so it explained page one of a query shape
    the implementation does not use (it builds ``$and``, with a separate
    missing/null-bucket branch) — and it silently ignored the mid-sweep cursor
    the caller paid ten real pages to reach, which is exactly where a
    scan-based plan hurts and page one does not.

    ``_PAGE_SORT``/``_PAGE`` are the implementation's own literals; the sort is
    the property under test, so stating it here is deliberate.
    """
    collection = storage._data
    captured: dict = {}
    storage._data = _CaptureFind(collection, captured)
    try:
        await storage.get_docs_by_statuses_page(
            statuses, limit=_PAGE, position=position, strict=True
        )
    finally:
        storage._data = collection

    assert "query" in captured, "the page issued no find"
    cursor = collection.find(captured["query"]).sort(_PAGE_SORT).limit(_PAGE)
    return await cursor.explain()
