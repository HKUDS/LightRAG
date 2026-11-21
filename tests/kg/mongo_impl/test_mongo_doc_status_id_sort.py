"""Document pagination: the public ID sort, the native sorts with their _id
tie-breaker, and the indexes that serve them."""

from __future__ import annotations

import pytest

pytest.importorskip("pymongo", reason="pymongo is required for Mongo storage tests")

from lightrag.base import DocStatus  # noqa: E402
from lightrag.kg.mongo_impl import MongoDocStatusStorage  # noqa: E402
from pymongo.errors import PyMongoError  # noqa: E402

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
        (
            {"sort_field": "created_at", "sort_direction": "desc"},
            list(range(12, -1, -1)),
            "created_at",
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
    expected_criteria = [(mongo_field, direction)]
    if mongo_field != "_id":
        # Non-unique fields get an _id tie-breaker in the SAME direction, so
        # one (field, _id) index serves both asc and desc.
        expected_criteria.append(("_id", direction))
    assert storage._data.sorts == [expected_criteria]
    assert storage._data.collations == (
        [{"locale": "zh", "numericOrdering": True}]
        if mongo_field == "file_path"
        else []
    )


async def test_tied_sort_values_do_not_skip_or_repeat_across_pages(storage):
    """Documents tied on the sort field come back in an unspecified order,
    which may differ between the page-1 and page-2 queries; the _id
    tie-breaker must make the pages partition the result anyway."""
    tied = [
        {**doc, "updated_at": "2026-01-01T00:00:00+00:00"}
        for doc in storage._data.documents
    ]
    storage._data.documents = tied
    first, _ = await storage.get_docs_paginated(
        page=1, page_size=10, sort_field="updated_at", sort_direction="asc"
    )
    # The natural order changes between the two queries.
    storage._data.documents = list(reversed(tied))
    second, _ = await storage.get_docs_paginated(
        page=2, page_size=10, sort_field="updated_at", sort_direction="asc"
    )

    seen = [doc_id for doc_id, _ in first + second]
    assert sorted(seen) == sorted(doc["_id"] for doc in tied)


class _IndexCollection:
    def __init__(self, existing, failing=()):
        self.names = set(existing)
        self.failing = set(failing)
        self.created = {}
        self.dropped = []

    async def list_indexes(self):
        names = sorted(self.names)

        class _IndexCursor:
            async def to_list(self, length=None):
                return [{"name": name} for name in names]

        return _IndexCursor()

    async def create_index(self, keys, name, **kwargs):
        if name in self.failing:
            raise PyMongoError(f"cannot build {name}")
        self.names.add(name)
        self.created[name] = (keys, kwargs.get("collation"))

    async def drop_index(self, name):
        self.names.discard(name)
        self.dropped.append(name)


_PREVIOUS_SORT_INDEXES = [
    "ws_status_updated_at",
    "ws_status_created_at",
    "ws_updated_at",
    "ws_created_at",
    "ws_file_path_zh_collation",
    "ws_status_file_path_zh_collation",
]


@pytest.mark.parametrize("failing", [(), ("ws_file_path_id_zh_collation",)])
async def test_sort_indexes_replace_superseded_ones_only_once_built(failing):
    """Every get_docs_paginated sort shape gets a (field, _id) index, and an
    index from an earlier version is dropped only after its replacement
    exists, so no sort is left unindexed when a build fails."""
    collection = _IndexCollection(
        ["_id_", "ws_id", "ws_status_created_at_id_asc", *_PREVIOUS_SORT_INDEXES],
        failing=failing,
    )
    instance = MongoDocStatusStorage.__new__(MongoDocStatusStorage)
    instance.workspace = "ws"
    instance._collection_name = "ws_doc_status"
    instance._data = collection

    await instance.create_and_migrate_indexes_if_not_exists()

    zh = {"locale": "zh", "numericOrdering": True}
    expected_created = {
        "ws_updated_at_id": ([("updated_at", -1), ("_id", -1)], None),
        "ws_status_updated_at_id": (
            [("status", 1), ("updated_at", -1), ("_id", -1)],
            None,
        ),
        "ws_created_at_id": ([("created_at", -1), ("_id", -1)], None),
        "ws_file_path_id_zh_collation": ([("file_path", 1), ("_id", 1)], zh),
        "ws_status_file_path_id_zh_collation": (
            [("status", 1), ("file_path", 1), ("_id", 1)],
            zh,
        ),
    }
    for name in failing:
        expected_created.pop(name)
    sort_indexes = {
        name: spec
        for name, spec in collection.created.items()
        if name in expected_created
    }
    assert sort_indexes == expected_created
    expected_dropped = set(_PREVIOUS_SORT_INDEXES)
    if failing:
        expected_dropped.discard("ws_file_path_zh_collation")
    assert set(collection.dropped) == expected_dropped
