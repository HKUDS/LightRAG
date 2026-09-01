"""The entity/relation migration must read doc_status COMPLETE-or-raise.

WHY: ``processed_docs`` is the ONLY input that decides which documents get
``full_entities`` / ``full_relations`` recovery anchors written.  A relaxed
read (``strict=False``) skips-and-logs an unparseable row, and the migration
then happily writes anchors for that row's siblings.  From the next startup
on, the "anchors already exist" sample check short-circuits before the
migration ever runs again, so the omitted document never gets anchors — and
per the purge recovery contract, every later purge of it fails closed (409)
with nothing able to repair it but an offline ``audit_kg_integrity``.

A malformed row must therefore abort startup, not be silently dropped.
Before this was pinned, four of the five doc-status backends read relaxed
here (their ``get_docs_by_status`` delegated to ``get_docs_by_statuses``
with the default ``strict=False``), so the hazard was live on all of them.
"""

import pytest

from lightrag.base import DocProcessingStatus, DocStatus
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.storage_migrations import _StorageMigrationMixin

pytestmark = pytest.mark.offline


class _CorruptRowDocStatus:
    """One healthy PROCESSED doc plus one row that cannot be parsed.

    Mirrors what every real backend does: log the bad row, then re-raise it
    only when the caller asked for the strict (complete-or-raise) contract.
    """

    def __init__(self):
        self.calls: list[bool] = []
        self.healthy = {
            "doc-ok": DocProcessingStatus(
                content_summary="ok",
                content_length=10,
                file_path="ok.pdf",
                status=DocStatus.PROCESSED,
                created_at="2024-01-01T00:00:00+00:00",
                updated_at="2024-01-01T00:00:00+00:00",
                chunks_list=["chunk-ok"],
                metadata={},
            )
        }

    async def get_docs_by_statuses(self, statuses, strict: bool = False):
        self.calls.append(strict)
        if strict:
            raise KeyError("content_summary")
        return dict(self.healthy)


class _FakeKV:
    def __init__(self):
        self.data: dict = {}

    async def get_by_id(self, key):
        return self.data.get(key)

    async def upsert(self, payload):
        self.data.update(payload)

    async def index_done_callback(self):
        return None


class _FakeGraph:
    """A legacy graph: one entity attributed to the CORRUPT document's chunk."""

    async def get_all_labels(self):
        return ["E1"]

    async def get_all_nodes(self):
        return [{"entity_id": "E1", "source_id": "chunk-corrupt"}]

    async def get_all_edges(self):
        return []


class _Migrator(_StorageMigrationMixin):
    def __init__(self, doc_status):
        self.doc_status = doc_status
        self.chunk_entity_relation_graph = _FakeGraph()
        self.full_entities = _FakeKV()
        self.full_relations = _FakeKV()
        self.entity_chunks = None
        self.relation_chunks = None

    async def _migrate_chunk_tracking_storage(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _shared_storage():
    initialize_share_data(workers=1)


@pytest.mark.asyncio
async def test_unparseable_doc_status_row_aborts_the_migration():
    """FIX-PROOF: with a relaxed read this completed and persisted nothing for
    the corrupt document, leaving it permanently anchor-less."""
    doc_status = _CorruptRowDocStatus()
    migrator = _Migrator(doc_status)

    with pytest.raises(KeyError):
        await migrator.check_and_migrate_data()

    # The read asked for the strict contract...
    assert doc_status.calls == [True]
    # ...and nothing was persisted, so a later run can still migrate the whole
    # corpus once the bad row is repaired. A relaxed read would have written
    # anchors here and made that repair unreachable.
    assert migrator.full_entities.data == {}
    assert migrator.full_relations.data == {}


@pytest.mark.asyncio
async def test_healthy_corpus_still_migrates():
    """STABILITY: the strict read must not break the normal migration path."""

    class _HealthyDocStatus(_CorruptRowDocStatus):
        async def get_docs_by_statuses(self, statuses, strict: bool = False):
            self.calls.append(strict)
            return dict(self.healthy)

    class _MatchingGraph(_FakeGraph):
        async def get_all_nodes(self):
            return [{"entity_id": "E1", "source_id": "chunk-ok"}]

    doc_status = _HealthyDocStatus()
    migrator = _Migrator(doc_status)
    migrator.chunk_entity_relation_graph = _MatchingGraph()

    await migrator.check_and_migrate_data()

    assert doc_status.calls == [True]
    assert migrator.full_entities.data["doc-ok"]["entity_names"] == ["E1"]
