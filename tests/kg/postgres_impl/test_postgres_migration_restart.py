"""Legacy vector migration: segmented copy, in-progress marker, resume.

The fake below stands in for the database at the ``db.query`` / ``db.execute``
boundary and commits every statement on its own, which is the property the
migration has to survive: a failure after some segments leaves their rows in
place. It checks control flow and statement shape, not SQL semantics.
See docs/design/PostgresLegacyVectorMigration.md.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

import lightrag.kg.postgres_impl as pg_impl
from lightrag.exceptions import DataMigrationError
from lightrag.kg.postgres_impl import PGVectorStorage

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]

STATE_TABLE = pg_impl.LEGACY_VECTOR_MIGRATION_STATE_TABLE
COLUMNS = ["id", "workspace", "content_vector"]
SETUP_ARGS = dict(
    table_name="target",
    workspace="tenant",
    embedding_dim=3,
    legacy_table_name="legacy",
    base_table="LIGHTRAG_VDB_CHUNKS",
)


class FakeServer:
    """Committed database state, shared by every FakeDB facade."""

    def __init__(self, rows: int = 750):
        self.source = {f"row-{i:04}": [1.0, 0.0, float(i)] for i in range(rows)}
        self.target: dict[str, list[float]] = {}
        self.state_table_exists = False
        self.markers: set[tuple[str, str]] = set()
        self.fail_on_id: str | None = None
        self.after_segment = None
        self.events: list[str] = []
        self.segment_sql: list[str] = []


class FakeDB:
    vector_index_type = None

    def __init__(self, server: FakeServer):
        self.server = server
        self._create_vector_index = AsyncMock()

    async def check_table_exists(self, name: str) -> bool:
        if name == STATE_TABLE:
            return self.server.state_table_exists
        return True

    async def query(self, sql, params=None, multirows=False, **kwargs):
        server = self.server
        if "pg_attribute" in sql:
            assert params == ["legacy"]
            return [{"attname": name} for name in COLUMNS]
        if "in_progress" in sql:
            return {"in_progress": True} if tuple(params) in server.markers else None
        if sql.startswith("WITH guard"):
            return self._copy_segment(sql, params)
        if "AS cleared" in sql:
            return self._verify_and_clear(params)
        assert "COUNT(*)" in sql, sql
        if "FROM legacy" in sql:
            return {"count": len(server.source)}
        return {"count": len(server.target)}

    def _copy_segment(self, sql, params):
        server = self.server
        server.segment_sql.append(sql)
        if "id > $3" in sql:
            workspace, target, last_id, limit = params
        else:
            (workspace, target, limit), last_id = params, None
        assert workspace == "tenant"
        if (target, workspace) not in server.markers:
            # The guard found no marker: the statement copies nothing.
            server.events.append("segment:fenced")
            return {"batch_rows": 0, "last_id": None, "active": False}
        ids = sorted(i for i in server.source if last_id is None or i > last_id)
        batch = ids[:limit]
        if server.fail_on_id in batch:
            # One statement is one transaction: nothing of this segment lands.
            raise RuntimeError("injected segment failure")
        for row_id in batch:
            server.target.setdefault(row_id, server.source[row_id])
        server.events.append(f"segment:{len(batch)}")
        if server.after_segment:
            server.after_segment()
        return {
            "batch_rows": len(batch),
            "last_id": batch[-1] if batch else None,
            "active": True,
        }

    def _verify_and_clear(self, params):
        server = self.server
        workspace, target = params
        server.events.append("verify")
        missing = len(set(server.source) - set(server.target))
        cleared = int(missing == 0 and (target, workspace) in server.markers)
        if cleared:
            server.markers.discard((target, workspace))
            server.events.append("marker:clear")
        return {"missing": missing, "cleared": cleared}

    async def execute(self, sql, data=None, **kwargs):
        server = self.server
        if sql.startswith("CREATE TABLE"):
            assert STATE_TABLE in sql
            server.state_table_exists = True
        elif sql.startswith(f"INSERT INTO {STATE_TABLE}"):
            # Conditional on the target workspace still being empty.
            assert "WHERE NOT EXISTS (SELECT 1 FROM target WHERE workspace = $2)" in sql
            if not server.target:
                server.markers.add((data["target_table"], data["workspace"]))
                server.events.append("marker:set")
        else:
            raise AssertionError(f"unexpected statement: {sql}")


@pytest.fixture
def small_batches(monkeypatch):
    monkeypatch.setattr(pg_impl, "LEGACY_VECTOR_MIGRATION_BATCH_SIZE", 200)


async def test_interrupted_migration_is_resumed_not_accepted(small_batches):
    """The defect: a restart took a partly filled target for a finished one."""
    server = FakeServer()
    server.fail_on_id = "row-0600"
    with pytest.raises(DataMigrationError):
        await PGVectorStorage.setup_table(db=FakeDB(server), **SETUP_ARGS)
    assert len(server.target) == 600
    assert server.markers == {("target", "tenant")}

    # Same fault on the next startup: still refused, nothing lost.
    with pytest.raises(DataMigrationError):
        await PGVectorStorage.setup_table(db=FakeDB(server), **SETUP_ARGS)
    assert len(server.target) == 600

    server.fail_on_id = None
    await PGVectorStorage.setup_table(db=FakeDB(server), **SETUP_ARGS)
    assert server.target == server.source
    assert server.markers == set()

    # A finished migration is not repeated.
    server.events.clear()
    await PGVectorStorage.setup_table(db=FakeDB(server), **SETUP_ARGS)
    assert server.events == []


async def test_marker_brackets_the_copy_and_verification(small_batches):
    server = FakeServer()
    await PGVectorStorage.setup_table(db=FakeDB(server), **SETUP_ARGS)
    assert server.events == [
        "marker:set",
        "segment:200",
        "segment:200",
        "segment:200",
        "segment:150",
        "verify",
        "marker:clear",
    ]


async def test_segments_are_server_side_keyset_statements(small_batches):
    server = FakeServer()
    await PGVectorStorage.setup_table(db=FakeDB(server), **SETUP_ARGS)
    names = '"id", "workspace", "content_vector"'
    first, later = server.segment_sql[0], server.segment_sql[1]
    assert "id >" not in first
    assert "AND id > $3 ORDER BY id LIMIT $4" in later
    for sql in (first, later):
        # Every segment holds the marker row and copies only while it exists.
        assert f"FROM {STATE_TABLE} WHERE target_table = $2 AND workspace = $1" in sql
        assert "FOR SHARE" in sql
        assert "AND EXISTS (SELECT 1 FROM guard)" in sql
        assert f"INSERT INTO target ({names}) SELECT {names} FROM batch" in sql
        assert f"SELECT {names} FROM legacy WHERE workspace = $1" in sql
        assert "ON CONFLICT (workspace, id) DO NOTHING" in sql


async def test_exact_multiple_of_batch_size_terminates(small_batches):
    server = FakeServer(rows=400)
    await PGVectorStorage.setup_table(db=FakeDB(server), **SETUP_ARGS)
    assert server.target == server.source
    assert server.events.count("segment:200") == 2
    assert server.events[-3:] == ["segment:0", "verify", "marker:clear"]


async def test_failed_verification_keeps_the_marker(small_batches):
    server = FakeServer()
    # A row vanishes from the target while the copy is still running.
    server.after_segment = lambda: server.target.pop("row-0001", None)
    with pytest.raises(DataMigrationError, match="verification failed"):
        await PGVectorStorage.setup_table(db=FakeDB(server), **SETUP_ARGS)
    assert server.markers == {("target", "tenant")}

    server.after_segment = None
    await PGVectorStorage.setup_table(db=FakeDB(server), **SETUP_ARGS)
    assert server.target == server.source
    assert server.markers == set()


async def test_cancellation_leaves_a_resumable_marker(small_batches):
    server = FakeServer()

    def cancel():
        raise asyncio.CancelledError()

    server.after_segment = cancel
    with pytest.raises(asyncio.CancelledError):
        await PGVectorStorage.setup_table(db=FakeDB(server), **SETUP_ARGS)
    assert len(server.target) == 200
    assert server.markers == {("target", "tenant")}


async def test_resume_does_not_overwrite_rows_already_in_the_target(small_batches):
    server = FakeServer()
    server.state_table_exists = True
    server.markers.add(("target", "tenant"))
    server.target["row-0003"] = [9.0, 9.0, 9.0]
    await PGVectorStorage.setup_table(db=FakeDB(server), **SETUP_ARGS)
    assert len(server.target) == 750
    assert server.target["row-0003"] == [9.0, 9.0, 9.0]


async def test_follower_stops_when_another_instance_finishes(small_batches):
    """A second server instance that joined a running migration must not
    write back rows the first instance deleted after finishing it."""
    server = FakeServer()
    server.state_table_exists = True
    server.markers.add(("target", "tenant"))
    segments = 0

    def leader_finishes_and_serves_a_deletion():
        nonlocal segments
        segments += 1
        if segments == 1:
            server.target = dict(server.source)
            server.markers.clear()
            del server.target["row-0700"]

    server.after_segment = leader_finishes_and_serves_a_deletion
    await PGVectorStorage.setup_table(db=FakeDB(server), **SETUP_ARGS)
    assert "row-0700" not in server.target
    assert len(server.target) == 749
    assert server.events == ["segment:200", "segment:fenced"]


async def test_deletion_between_last_segment_and_verification_is_not_a_failure(
    small_batches,
):
    server = FakeServer(rows=150)
    server.state_table_exists = True
    server.markers.add(("target", "tenant"))

    def leader_finishes_and_serves_a_deletion():
        server.markers.clear()
        del server.target["row-0001"]

    server.after_segment = leader_finishes_and_serves_a_deletion
    await PGVectorStorage.setup_table(db=FakeDB(server), **SETUP_ARGS)
    assert "row-0001" not in server.target
    assert server.markers == set()


async def test_resume_never_recreates_the_marker(small_batches):
    """The marker vanished between the caller's check and the copy: another
    instance finished. Re-creating it would reopen the migration."""
    server = FakeServer()
    server.state_table_exists = True
    await PGVectorStorage._pg_migrate_workspace_data(
        FakeDB(server), "legacy", "target", "tenant", 750, resume=True
    )
    assert server.markers == set()
    assert server.target == {}
    assert server.events == ["segment:fenced"]


async def test_fresh_start_does_not_mark_a_target_populated_meanwhile(small_batches):
    server = FakeServer()
    server.target = {"row-0000": server.source["row-0000"]}
    await PGVectorStorage._pg_migrate_workspace_data(
        FakeDB(server), "legacy", "target", "tenant", 750
    )
    assert server.markers == set()
    assert len(server.target) == 1
    assert server.events == ["segment:fenced"]


async def test_populated_target_without_marker_is_left_alone(lightrag_caplog):
    """An unmarked target is a finished migration or an old partial one; the
    two cannot be told apart, so it is preserved and the shortfall reported."""
    server = FakeServer()
    server.target = {"row-0000": [9.0, 9.0, 9.0]}
    await PGVectorStorage.setup_table(db=FakeDB(server), **SETUP_ARGS)
    assert server.target == {"row-0000": [9.0, 9.0, 9.0]}
    assert server.events == []
    assert "fewer than the 750" in lightrag_caplog.text


async def test_progress_is_logged_per_segment(small_batches, lightrag_caplog):
    server = FakeServer()
    await PGVectorStorage.setup_table(db=FakeDB(server), **SETUP_ARGS)
    for done in (200, 400, 600, 750):
        assert f"{done}/750 records migrated" in lightrag_caplog.text


async def test_malformed_segment_result_is_not_read_as_end_of_data():
    server = FakeServer()
    db = FakeDB(server)
    db._copy_segment = lambda sql, params: {}
    with pytest.raises(DataMigrationError, match="segment result"):
        await PGVectorStorage.setup_table(db=db, **SETUP_ARGS)
    assert server.markers == {("target", "tenant")}


async def test_helper_rejects_empty_workspace_before_touching_the_database():
    server = FakeServer()
    with pytest.raises(ValueError, match="workspace must be provided"):
        await PGVectorStorage._pg_migrate_workspace_data(
            FakeDB(server), "legacy", "target", "", 750
        )
    assert server.events == [] and not server.state_table_exists
