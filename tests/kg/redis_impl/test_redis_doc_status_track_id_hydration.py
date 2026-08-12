"""Coverage for RedisDocStatusStorage.get_docs_by_track_id hydration resilience.

WHY: the track_id path built DocProcessingStatus inline and caught only
``(json.JSONDecodeError, KeyError)``. A dataclass raises **TypeError** on a
missing/unknown field, so the real failure escaped to the function's outer
``except Exception`` — which abandons the SCAN loop and returns a silently
TRUNCATED listing. That is worse than a crash: the caller sees a short, plausible
result with no indication documents are missing.

These tests do NOT require a live Redis — the client is the shared in-memory fake.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

pytest.importorskip(
    "redis",
    reason="redis is required for Redis storage tests",
)

from lightrag.namespace import NameSpace  # noqa: E402

from .fake_redis import FakeRedis  # noqa: E402

pytestmark = pytest.mark.offline


class _DummyEmbeddingFunc:
    embedding_dim = 1
    max_token_size = 1

    async def __call__(self, texts, **kwargs):
        return [[0.0] for _ in texts]


@pytest.fixture
def redis_doc_status(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(
        "lightrag.kg.redis_impl.RedisConnectionManager.get_pool",
        lambda redis_url: MagicMock(name="fake_pool"),
    )
    monkeypatch.setattr(
        "lightrag.kg.redis_impl.RedisConnectionManager.release_pool",
        lambda redis_url: None,
    )
    monkeypatch.setattr(
        "lightrag.kg.redis_impl.Redis", lambda connection_pool=None, **_: fake
    )

    from lightrag.kg.redis_impl import RedisDocStatusStorage

    storage = RedisDocStatusStorage(
        namespace=NameSpace.DOC_STATUS,
        global_config={},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    storage._initialized = True  # skip the real ping in initialize()
    return storage


def _valid_row(track_id: str = "track_123") -> dict:
    return {
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


def _store_raw(storage, doc_id: str, payload: dict) -> None:
    """Write straight into the fake backing store, bypassing ``upsert`` so we
    control the serialized shape (unhydratable rows can only arrive that way)."""
    storage._redis.store[f"{storage.final_namespace}:{doc_id}"] = json.dumps(payload)


async def test_track_id_listing_survives_a_row_missing_required_fields(
    redis_doc_status,
):
    _store_raw(redis_doc_status, "valid_doc", _valid_row())
    # Only track_id: DocProcessingStatus(**data) raises TypeError, not KeyError.
    _store_raw(redis_doc_status, "invalid_doc", {"track_id": "track_123"})

    docs = await redis_doc_status.get_docs_by_track_id("track_123")

    assert "valid_doc" in docs
    assert "invalid_doc" not in docs


async def test_track_id_listing_hydrates_a_row_with_an_unknown_field(redis_doc_status):
    """Version-rollback shape: a field the running build does not declare.

    Undeclared fields are ignored at construction
    (``DocProcessingStatus.from_stored``), so the row stays visible instead
    of silently vanishing from the listing.
    """
    _store_raw(
        redis_doc_status,
        "doc-a-written-by-newer-build",
        {**_valid_row(), "field_from_the_future": "value"},
    )
    _store_raw(redis_doc_status, "doc-b-old", _valid_row())

    docs = await redis_doc_status.get_docs_by_track_id("track_123")

    assert set(docs) == {"doc-a-written-by-newer-build", "doc-b-old"}
    assert not hasattr(docs["doc-a-written-by-newer-build"], "field_from_the_future")


async def test_a_bad_row_does_not_truncate_the_rest_of_the_scan(redis_doc_status):
    """The specific Redis failure mode: an escaping TypeError hit the outer
    ``except Exception``, which breaks the SCAN loop — so documents not yet
    reached vanished from the response without any error surfacing.

    Scan position is load-bearing, and it comes from the KEY NAME, not the
    seeding order: SCAN walks a sorted snapshot, so ``doc-000-corrupt`` is
    reached before every ``doc-1NN``. With the bad row scanned LAST the healthy
    rows are already collected, the truncated result is identical to the correct
    one, and the assertion would hold against the unfixed code — pinning nothing.
    """
    _store_raw(redis_doc_status, "doc-000-corrupt", {"track_id": "track_123"})
    for i in range(12):
        _store_raw(redis_doc_status, f"doc-1{i:02d}", _valid_row())

    docs = await redis_doc_status.get_docs_by_track_id("track_123")

    assert set(docs) == {f"doc-1{i:02d}" for i in range(12)}


async def test_rows_under_other_track_ids_are_not_returned(redis_doc_status):
    _store_raw(redis_doc_status, "mine", _valid_row("track_123"))
    _store_raw(redis_doc_status, "theirs", _valid_row("track_999"))

    docs = await redis_doc_status.get_docs_by_track_id("track_123")

    assert set(docs) == {"mine"}
