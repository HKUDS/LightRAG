"""Redis storage with workspace names that contain ":" or glob characters.

validate_workspace() allows these, and they reach Redis through
REDIS_WORKSPACE or LightRAG(workspace=...). A ":" used to end up inside the
parsed doc ids, and "*" / "?" / "[" made SCAN match other workspaces' keys.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from lightrag.base import DocStatus
from lightrag.namespace import NameSpace

from .fake_redis import FakeRedis

pytestmark = pytest.mark.offline


class _DummyEmbeddingFunc:
    embedding_dim = 1
    max_token_size = 1

    async def __call__(self, texts, **kwargs):
        return [[0.0] for _ in texts]


@pytest.fixture
def fake(monkeypatch):
    instance = FakeRedis()
    monkeypatch.setattr(
        "lightrag.kg.redis_impl.RedisConnectionManager.get_pool",
        lambda redis_url: MagicMock(name="fake_pool"),
    )
    monkeypatch.setattr(
        "lightrag.kg.redis_impl.Redis", lambda connection_pool=None, **_: instance
    )
    return instance


def _make(fake, kind: str, workspace: str):
    from lightrag.kg.redis_impl import RedisDocStatusStorage, RedisKVStorage

    if kind == "doc_status":
        storage = RedisDocStatusStorage(
            namespace=NameSpace.DOC_STATUS,
            global_config={},
            embedding_func=_DummyEmbeddingFunc(),
            workspace=workspace,
        )
    else:
        storage = RedisKVStorage(
            namespace=NameSpace.KV_STORE_FULL_DOCS,
            global_config={},
            embedding_func=_DummyEmbeddingFunc(),
            workspace=workspace,
        )
    # Stand in for initialize(): attach the fake, skip the real ping.
    storage._pool = MagicMock(name="fake_pool")
    storage._redis = fake
    storage._initialized = True
    return storage


def _doc_row() -> dict:
    return {
        "status": "processed",
        "content_summary": "s",
        "content_length": 1,
        "chunks_count": 1,
        "file_path": "report.pdf",
        "track_id": "track-1",
        "content_hash": "hash-1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "metadata": {},
        "error_msg": None,
    }


async def _paginated(storage):
    docs, _ = await storage.get_docs_paginated(page=1, page_size=10)
    return [doc_id for doc_id, _ in docs]


async def _by_statuses(storage):
    return list(await storage.get_docs_by_statuses([DocStatus.PROCESSED]))


async def _by_track_id(storage):
    return list(await storage.get_docs_by_track_id("track-1"))


async def _by_content_hash(storage):
    hit = await storage.get_doc_by_content_hash("hash-1")
    return [hit[0]] if hit else []


async def _sidecar_page(storage):
    await storage._rebuild_scheduling_sidecar()
    page = await storage.get_docs_by_statuses_page([DocStatus.PROCESSED], limit=10)
    return list(page.docs)


@pytest.mark.parametrize(
    "read",
    [_paginated, _by_statuses, _by_track_id, _by_content_hash, _sidecar_page],
    ids=lambda f: f.__name__.lstrip("_"),
)
async def test_doc_ids_survive_a_colon_in_the_workspace(fake, read):
    storage = _make(fake, "doc_status", "team:a")
    fake.store[f"{storage.final_namespace}:doc-1"] = json.dumps(_doc_row())

    ids = await read(storage)

    assert ids == ["doc-1"]
    # and the id works for a normal lookup
    assert await storage.get_by_id(ids[0]) is not None


# Each glob character goes through the shared _scan_match once; each drop()
# implementation (KV, and doc status with its sidecar patterns) once.
@pytest.mark.parametrize(
    ("kind", "workspace"),
    [
        ("kv", "a*"),
        ("doc_status", "a*"),
        ("doc_status", "a?c"),
        ("doc_status", "[a]bc"),
    ],
)
async def test_drop_leaves_a_glob_matched_workspace_alone(fake, kind, workspace):
    # Each workspace, read as an unescaped glob, matches "abc".
    victim = _make(fake, kind, "abc")
    dropped = _make(fake, kind, workspace)
    for storage in (victim, dropped):
        fake.store[f"{storage.final_namespace}:doc-1"] = json.dumps(_doc_row())
    if kind == "doc_status":
        # drop() clears the sidecar keys by pattern too
        await victim._rebuild_scheduling_sidecar()
        await dropped._rebuild_scheduling_sidecar()
    victim_keys = {
        k
        for k in list(fake.store) + list(fake.zsets) + list(fake.sets)
        if k.startswith(victim.final_namespace)
    }
    assert victim_keys  # make sure abc actually has something to lose

    result = await dropped.drop()

    assert result["status"] == "success"
    # is_empty() uses the same pattern: unescaped, it would see abc's keys.
    assert await dropped.is_empty()
    remaining = set(fake.store) | set(fake.zsets) | set(fake.sets)
    assert victim_keys <= remaining
    assert not any(k.startswith(dropped.final_namespace) for k in remaining)
