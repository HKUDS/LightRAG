"""Doc-status pagination must not skip or repeat documents across pages
when the sort field has duplicate values.

``get_docs_paginated`` sorts by a single, non-unique field (``updated_at``,
``created_at`` or ``file_path``) and then applies ``skip``/``limit``.
MongoDB does not guarantee a stable order for tied sort keys, so the
relative order of tied documents can differ between two separate queries
(for example because the collection's natural order shifted between the
page-1 and page-2 calls). Without a unique tie-breaker appended to the
sort criteria, that lets a document be returned on two pages or omitted
from every page.
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
    two paginated queries (a concurrent write, an internal reorganization,
    or simply MongoDB's documented lack of an ordering guarantee for ties).
    """

    def __init__(self, orders):
        self._orders = list(orders)
        self.sorts = []

    async def count_documents(self, query):
        return len(self._orders[0])

    def find(self, query):
        documents = self._orders.pop(0)
        return _Cursor(documents, self)


def _tied_documents():
    # All four documents share the same updated_at value, so the sort field
    # alone cannot order them deterministically.
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
        for index in range(4)
    ]


@pytest.mark.asyncio
async def test_tied_updated_at_pagination_does_not_skip_or_repeat_documents():
    tied = _tied_documents()
    # Page 1 sees one natural order; page 2 sees a reshuffled one (the first
    # two tied documents moved to the back), simulating the collection
    # changing between the two separate paginated queries.
    first_order = [tied[0], tied[1], tied[2], tied[3]]
    second_order = [tied[2], tied[3], tied[0], tied[1]]
    collection = _Collection([first_order, second_order])
    storage = MongoDocStatusStorage.__new__(MongoDocStatusStorage)
    storage.namespace = "doc_status"
    storage.workspace = "test"
    storage._data = collection

    page1, _ = await storage.get_docs_paginated(
        page=1, page_size=2, sort_field="updated_at", sort_direction="asc"
    )
    page2, _ = await storage.get_docs_paginated(
        page=2, page_size=2, sort_field="updated_at", sort_direction="asc"
    )

    seen = [doc_id for doc_id, _ in page1] + [doc_id for doc_id, _ in page2]
    assert len(set(seen)) == 4, (
        f"expected all 4 distinct documents across both pages, got {seen}"
    )

    # The sort criteria sent to MongoDB must include a unique tie-breaker
    # so that ties resolve deterministically regardless of natural order.
    for criteria in collection.sorts:
        fields = [field for field, _ in criteria]
        assert "_id" in fields, (
            f"sort criteria {criteria} has no unique tie-breaker field"
        )
