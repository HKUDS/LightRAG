"""The scheduling-index migration never leaves the table worse off (LR2 §13.1).

The migration replaces a wrongly-ordered index with a ``created_at NULLS FIRST``
one. It used to DROP the superseded index first and then swallow a failed
create — so a create that did not land left LIGHTRAG_DOC_STATUS with NO
scheduling index at all: strictly worse than before the upgrade, and invisible.
The service starts, pages stay correct and memory-bounded (a top-N sort keeps
only ``limit`` rows), and only the I/O silently degrades to a full scan plus a
sort per page — O(rows²/page_size) over a sweep.

So the order is create → verify from ``pg_indexes`` → drop, the verification
reads the catalog instead of inferring success from "no exception was raised",
and a missing index is reported with its consequence and its DDL rather than as
a bare "failed to create index".
"""

from __future__ import annotations

import logging

import pytest

from lightrag.kg.postgres_impl import PostgreSQLDB

pytestmark = pytest.mark.offline

_NEW = "idx_lightrag_doc_status_ws_status_created_nf_id"
_SUPERSEDED = "idx_lightrag_doc_status_ws_status_created_id"


class _FakeDB:
    """Catalog + statement log; only ``query``/``execute`` are exercised."""

    # The real helper, so its SQL and parameters are under test too.
    _doc_status_index_exists = PostgreSQLDB._doc_status_index_exists

    def __init__(self, *, existing: set[str] | None = None):
        self.existing: set[str] = set(existing or ())
        self.executed: list[str] = []
        # index name -> exception to raise instead of creating it
        self.create_fails: Exception | None = None
        # True: CREATE returns cleanly but the index does not appear (a pooler /
        # aborted transaction swallowing the DDL).
        self.create_is_a_lie = False

    async def query(self, sql, params=None, multirows=False, **kwargs):
        assert "pg_indexes" in sql, sql
        assert params and len(params) == 1, params
        return {"indexname": params[0]} if params[0] in self.existing else None

    async def execute(self, sql, data=None, **kwargs):
        self.executed.append(" ".join(sql.split()))
        if sql.strip().upper().startswith("CREATE INDEX"):
            if self.create_fails is not None:
                raise self.create_fails
            if not self.create_is_a_lie:
                self.existing.add(_NEW)
        elif "DROP INDEX" in sql.upper():
            self.existing.discard(_SUPERSEDED)


def _kinds(db: _FakeDB) -> list[str]:
    out = []
    for sql in db.executed:
        upper = sql.upper()
        if upper.startswith("CREATE INDEX"):
            out.append("create")
        elif "DROP INDEX" in upper:
            out.append("drop")
    return out


@pytest.mark.asyncio
async def test_creates_then_drops_the_superseded_index_in_that_order():
    db = _FakeDB(existing={_SUPERSEDED})
    await PostgreSQLDB._migrate_doc_status_add_scheduling_index(db)

    assert _kinds(db) == ["create", "drop"]
    assert _NEW in db.existing
    assert _SUPERSEDED not in db.existing
    # The new index carries the load-bearing NULLS FIRST declaration.
    assert any("CREATED_AT NULLS FIRST" in sql.upper() for sql in db.executed)


@pytest.mark.asyncio
async def test_a_failed_create_keeps_the_superseded_index(caplog):
    """Fix-proof: the drop used to run first, so this left the table with no
    scheduling index at all."""
    from lightrag.utils import logger

    db = _FakeDB(existing={_SUPERSEDED})
    db.create_fails = RuntimeError("canceling statement due to lock timeout")

    logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger=logger.name):
            await PostgreSQLDB._migrate_doc_status_add_scheduling_index(db)
    finally:
        logger.propagate = False

    assert "drop" not in _kinds(db)
    assert _SUPERSEDED in db.existing  # paging is no worse than before the upgrade
    # The operator is told what it costs and how to fix it, not just that a
    # statement failed.
    assert "is MISSING" in caplog.text
    assert "full scan" in caplog.text
    assert "NULLS FIRST" in caplog.text


@pytest.mark.asyncio
async def test_the_drop_waits_on_the_catalog_not_on_the_absence_of_an_error():
    """A CREATE that returns cleanly is not proof the index exists; only
    pg_indexes is."""
    db = _FakeDB(existing={_SUPERSEDED})
    db.create_is_a_lie = True

    await PostgreSQLDB._migrate_doc_status_add_scheduling_index(db)

    assert _kinds(db) == ["create"]
    assert _SUPERSEDED in db.existing


@pytest.mark.asyncio
async def test_a_second_startup_creates_nothing_and_stays_idempotent():
    db = _FakeDB(existing={_NEW})
    await PostgreSQLDB._migrate_doc_status_add_scheduling_index(db)

    # Nothing to create; the drop is still issued (IF EXISTS, so a no-op here).
    assert _kinds(db) == ["drop"]
    assert _NEW in db.existing


@pytest.mark.asyncio
async def test_a_failed_verification_read_also_holds_the_drop_back():
    """The catalog read itself can fail; that is not evidence of success."""
    db = _FakeDB(existing={_SUPERSEDED})
    calls = {"n": 0}
    original = db.query

    async def _second_read_fails(sql, params=None, multirows=False, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("connection reset")
        return await original(sql, params, multirows=multirows, **kwargs)

    db.query = _second_read_fails
    await PostgreSQLDB._migrate_doc_status_add_scheduling_index(db)

    assert "drop" not in _kinds(db)
    assert _SUPERSEDED in db.existing
