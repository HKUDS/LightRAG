"""Document pagination honors the public ID sort while retaining native sorts."""

from __future__ import annotations

import pytest

pytest.importorskip("pymongo", reason="pymongo is required for Mongo storage tests")

from lightrag.base import DocStatus  # noqa: E402
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
        self.collection.collations.append(value)
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
    def __init__(self, documents):
        self.documents = documents
        self.sorts = []
        self.collations = []
        self.queries = []

    def _matching(self, query):
        if not query:
            return self.documents
        return [
            doc for doc in self.documents if doc["status"] in query["status"]["$in"]
        ]

    async def count_documents(self, query):
        return len(self._matching(query))

    def find(self, query):
        self.queries.append(query)
        return _Cursor(self._matching(query), self)


@pytest.fixture
def storage():
    documents = [
        {
            "_id": f"doc-{index:02}",
            "status": "processed" if index % 2 == 0 else "failed",
            "content_summary": "synthetic document",
            "content_length": 1,
            "file_path": f"file-{(index + 5) % 13:02}.txt",
            "created_at": f"2026-01-{index + 1:02}T00:00:00+00:00",
            "updated_at": f"2026-01-{13 - index:02}T00:00:00+00:00",
            "metadata": {},
        }
        for index in range(13)
    ]
    # Deliberately make insertion order differ from all requested sort orders.
    collection = _Collection(documents[4:] + documents[:4])
    instance = MongoDocStatusStorage.__new__(MongoDocStatusStorage)
    instance.namespace = "doc_status"
    instance.workspace = "test"
    instance._data = collection
    return instance


@pytest.mark.parametrize("direction", ["asc", "desc"])
async def test_public_id_sort_orders_documents_across_pages(storage, direction):
    first, first_total = await storage.get_docs_paginated(
        page=1, page_size=10, sort_field="id", sort_direction=direction
    )
    second, second_total = await storage.get_docs_paginated(
        page=2, page_size=10, sort_field="id", sort_direction=direction
    )

    expected = [f"doc-{index:02}" for index in range(13)]
    if direction == "desc":
        expected.reverse()
    assert [doc_id for doc_id, _ in first] == expected[:10]
    assert [doc_id for doc_id, _ in second] == expected[10:]
    assert first_total == second_total == 13
    mongo_direction = 1 if direction == "asc" else -1
    assert storage._data.sorts == [[("_id", mongo_direction)]] * 2


async def test_public_id_sort_respects_status_filter(storage):
    documents, total = await storage.get_docs_paginated(
        status_filter=DocStatus.PROCESSED, sort_field="id", sort_direction="asc"
    )

    assert [doc_id for doc_id, _ in documents] == [
        f"doc-{index:02}" for index in range(0, 13, 2)
    ]
    assert total == 7
    assert all(doc.status == DocStatus.PROCESSED for _, doc in documents)
    assert storage._data.queries == [{"status": {"$in": ["processed"]}}]


@pytest.mark.parametrize(
    ("options", "expected_indices", "mongo_field"),
    [
        ({"sort_field": "_id"}, list(range(12, -1, -1)), "_id"),
        (
            {"sort_field": "created_at", "sort_direction": "asc"},
            list(range(13)),
            "created_at",
        ),
        (
            {"sort_field": "updated_at", "sort_direction": "asc"},
            list(range(12, -1, -1)),
            "updated_at",
        ),
        (
            {"sort_field": "file_path", "sort_direction": "asc"},
            list(range(8, 13)) + list(range(8)),
            "file_path",
        ),
        ({}, list(range(13)), "updated_at"),
        ({"sort_field": "unknown"}, list(range(13)), "updated_at"),
    ],
)
async def test_existing_sort_options_remain_supported(
    storage, options, expected_indices, mongo_field
):
    documents, total = await storage.get_docs_paginated(page_size=50, **options)

    assert [doc_id for doc_id, _ in documents] == [
        f"doc-{index:02}" for index in expected_indices
    ]
    assert total == 13
    direction = 1 if options.get("sort_direction") == "asc" else -1
    assert storage._data.sorts == [[(mongo_field, direction)]]
    assert storage._data.collations == (
        [{"locale": "zh", "numericOrdering": True}]
        if mongo_field == "file_path"
        else []
    )
