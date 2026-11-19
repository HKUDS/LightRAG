"""Doc-status pagination must not skip or repeat documents across pages
when the sort field has duplicate values.

``get_docs_paginated`` sorts by a single, non-unique field (``updated_at``,
``created_at`` or ``file_path``) and then applies ``skip``/``limit``.
MongoDB does not guarantee any particular order among documents that tie on
the sort field, so the relative order of tied documents can differ between
two separate queries. Without a unique tie-breaker appended to the sort
criteria, that lets a document be returned on two pages or omitted from
every page.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pymongo", reason="pymongo is required for Mongo storage tests")

from lightrag.kg.mongo_impl import MongoDocStatusStorage  # noqa: E402

pytestmark = pytest.mark.offline


class _Cursor:
    def __init__(self, documents, collection):
        self.documents = list(documents)
        self.collection = collection
        self.offset = 0
        self.size = None

    def sort(self, criteria):
        self.collection.sorts.append(criteria)
        for field, direction in reversed(criteria):
            self.documents.sort(key=lambda doc: doc[field], reverse=direction == -1)
        return self

    def collation(self, value):
        return self

    def skip(self, value):
        self.offset = value
        return self

    def limit(self, value):
        self.size = value
        return self

    async def to_list(self, length=None):
        return self.documents[self.offset : self.offset + self.size]


class _Collection:
    """Returns documents in a different natural order on each ``find()``.

    This models a collection whose underlying natural order shifts between
    two paginated queries: MongoDB documents no ordering guarantee among
    tied sort keys, so successive queries may return them in different
    relative order.
    """

    def __init__(self, orders):
        self._orders = list(orders)
        self.sorts = []

    async def count_documents(self, query):
        return len(self._orders[0])

    def find(self, query):
        documents = self._orders.pop(0)
        return _Cursor(documents, self)


_TIED_COUNT = 14  # > the page_size floor of 10, so a real page boundary is crossed


def _tied_documents():
    # All documents share the same updated_at value, so the sort field alone
    # cannot order them deterministically.
    return [
        {
            "_id": f"doc-{index:02}",
            "status": "processed",
            "content_summary": "synthetic document",
            "content_length": 1,
            "file_path": f"file-{index:02}.txt",
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
            "metadata": {},
        }
        for index in range(_TIED_COUNT)
    ]


def _make_storage(collection):
    storage = MongoDocStatusStorage.__new__(MongoDocStatusStorage)
    storage.namespace = "doc_status"
    storage.workspace = "test"
    storage._data = collection
    return storage


@pytest.mark.asyncio
async def test_tied_updated_at_pagination_does_not_skip_or_repeat_documents():
    tied = _tied_documents()
    # Page 1 sees one natural order; page 2 sees a reversed one, simulating
    # the collection's natural order changing between the two separate
    # paginated queries. page_size is left at the default floor (10), so
    # with 14 tied documents the second page genuinely starts mid-tie
    # (skip=10) rather than being empty or a no-op, unlike a page_size
    # below the floor, which MongoDB would silently round up anyway.
    first_order = list(tied)
    second_order = list(reversed(tied))
    collection = _Collection([first_order, second_order])
    storage = _make_storage(collection)

    page1, _ = await storage.get_docs_paginated(
        page=1, page_size=10, sort_field="updated_at", sort_direction="asc"
    )
    page2, _ = await storage.get_docs_paginated(
        page=2, page_size=10, sort_field="updated_at", sort_direction="asc"
    )

    assert len(page1) == 10
    assert len(page2) == _TIED_COUNT - 10
    seen = [doc_id for doc_id, _ in page1] + [doc_id for doc_id, _ in page2]
    assert len(set(seen)) == _TIED_COUNT, (
        f"expected all {_TIED_COUNT} distinct documents across both pages, got {seen}"
    )

    # The sort criteria sent to MongoDB must include a unique tie-breaker
    # so that ties resolve deterministically regardless of natural order.
    for criteria in collection.sorts:
        fields = [field for field, _ in criteria]
        assert "_id" in fields, (
            f"sort criteria {criteria} has no unique tie-breaker field"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("sort_direction", ["asc", "desc"])
async def test_tie_breaker_direction_matches_primary_sort_direction(sort_direction):
    """The _id tie-breaker must sort in the same direction as the primary
    field, not a fixed ascending, so a compound index on
    (primary_field, _id) declared in one direction can serve the query
    (MongoDB can still scan such an index backwards for the opposite
    direction, but not when the two keys disagree on direction).
    """
    collection = _Collection([_tied_documents()])
    storage = _make_storage(collection)

    await storage.get_docs_paginated(
        page=1, page_size=10, sort_field="updated_at", sort_direction=sort_direction
    )

    expected_direction = 1 if sort_direction == "asc" else -1
    assert collection.sorts == [
        [("updated_at", expected_direction), ("_id", expected_direction)]
    ]
