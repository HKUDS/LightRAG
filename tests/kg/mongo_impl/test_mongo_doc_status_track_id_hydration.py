"""Coverage for MongoDocStatusStorage.get_docs_by_track_id hydration resilience.

WHY: the track_id path duplicated the hydration its sibling reads already share
via ``_mongo_doc_processing_status_from_doc`` and caught only ``KeyError``.
``DocProcessingStatus`` is a dataclass, so a row missing a required field — or
carrying one the running build does not declare — raises **TypeError**, which
escaped and crashed the listing for every document sharing that track_id.

No live MongoDB: the collection is a minimal fake recording the
``find(...).to_list()`` chain the method actually uses.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "pymongo",
    reason="pymongo is required for Mongo storage tests",
)

from lightrag.kg.mongo_impl import MongoDocStatusStorage  # noqa: E402

pytestmark = pytest.mark.offline


class _FakeFindCursor:
    def __init__(self, docs):
        self._docs = list(docs)

    async def to_list(self, length=None):
        return self._docs


class _FakeCollection:
    """Returns every seeded doc whose fields match the equality filter."""

    def __init__(self, docs):
        self._docs = list(docs)
        self.find_queries: list[dict] = []

    def find(self, query):
        self.find_queries.append(query)
        return _FakeFindCursor(
            [d for d in self._docs if all(d.get(k) == v for k, v in query.items())]
        )


def _storage(docs) -> MongoDocStatusStorage:
    storage = MongoDocStatusStorage.__new__(MongoDocStatusStorage)
    storage.workspace = "test"
    storage.namespace = "doc_status"
    storage._data = _FakeCollection(docs)
    return storage


def _valid_doc(doc_id: str, track_id: str = "track_123") -> dict:
    return {
        "_id": doc_id,
        "track_id": track_id,
        "status": "processed",
        "content_summary": "summary",
        "content_length": 10,
        "file_path": "report.pdf",
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00",
        "metadata": {},
        "error_msg": None,
    }


async def test_track_id_listing_survives_a_doc_missing_required_fields():
    storage = _storage(
        [
            _valid_doc("valid_doc"),
            # Only _id + track_id: DocProcessingStatus(**data) raises TypeError.
            {"_id": "invalid_doc", "track_id": "track_123"},
        ]
    )

    docs = await storage.get_docs_by_track_id("track_123")

    assert "valid_doc" in docs
    assert "invalid_doc" not in docs


async def test_track_id_listing_hydrates_a_doc_with_an_unknown_field():
    """Version-rollback shape: a field the running build does not declare.

    Undeclared fields are ignored at construction
    (``DocProcessingStatus.from_stored``), so the row stays visible instead
    of silently vanishing from the listing.
    """
    storage = _storage(
        [
            _valid_doc("old_doc"),
            {**_valid_doc("written_by_newer_build"), "field_from_the_future": "value"},
        ]
    )

    docs = await storage.get_docs_by_track_id("track_123")

    assert set(docs) == {"old_doc", "written_by_newer_build"}
    assert not hasattr(docs["written_by_newer_build"], "field_from_the_future")


async def test_one_bad_doc_does_not_hide_its_healthy_siblings():
    """track_id groups a whole upload batch — the point of skip-and-log is that
    the other documents in that batch stay visible."""
    storage = _storage(
        [_valid_doc(f"doc-{i}") for i in range(5)]
        + [{"_id": "doc-bad", "track_id": "track_123"}]
    )

    docs = await storage.get_docs_by_track_id("track_123")

    assert set(docs) == {f"doc-{i}" for i in range(5)}


async def test_hydrated_doc_keeps_its_fields_and_query_is_scoped_to_track_id():
    storage = _storage([_valid_doc("mine"), _valid_doc("theirs", "track_999")])

    docs = await storage.get_docs_by_track_id("track_123")

    assert set(docs) == {"mine"}
    assert docs["mine"].file_path == "report.pdf"
    assert docs["mine"].track_id == "track_123"
    assert storage._data.find_queries == [{"track_id": "track_123"}]
