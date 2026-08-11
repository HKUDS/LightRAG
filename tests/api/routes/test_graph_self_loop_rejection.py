"""Manual relation creation must not put a self-loop into the graph.

LightRAG's extraction pipeline never emits ``src == tgt`` (dropped in both
record parsers and again in ``_merge_edges_then_upsert``) and
``amerge_entities`` skips any endpoint pair that would collapse into one.
``acreate_relation`` was the remaining path that accepted one, so these tests
pin the guard and — crucially — pin that it refuses *before* touching storage.
"""

import pytest

from lightrag import utils_graph


pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


async def test_acreate_relation_rejects_self_loop():
    """Passing ``None`` storages is the assertion: the guard must run before
    the keyed lock, which dereferences ``relationships_vdb.global_config``.
    An ``AttributeError`` here would mean the refusal came too late."""
    with pytest.raises(ValueError, match="self-loop relation on 'A'"):
        await utils_graph.acreate_relation(
            chunk_entity_relation_graph=None,
            entities_vdb=None,
            relationships_vdb=None,
            source_entity="A",
            target_entity="A",
            relation_data={"description": "A refers to itself"},
        )


async def test_acreate_relation_still_accepts_distinct_endpoints(monkeypatch):
    """The guard must not disturb an ordinary two-entity relation."""

    class _NoopLock:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        utils_graph,
        "get_storage_keyed_lock",
        lambda *args, **kwargs: _NoopLock(),
    )

    upserted_edges = []

    class _Graph:
        async def has_node(self, entity_name):
            return True

        async def has_edge(self, source_entity, target_entity):
            return False

        async def get_edge(self, source_entity, target_entity):
            return {"description": "A relates to B", "keywords": "k", "weight": 1.0}

        async def upsert_edge(self, source_entity, target_entity, edge_data):
            upserted_edges.append((source_entity, target_entity))

        async def index_done_callback(self):
            return None

    class _VectorStorage:
        def __init__(self):
            self.global_config = {"workspace": ""}
            self.records = {}

        async def upsert(self, data):
            self.records.update(data)

        async def index_done_callback(self):
            return None

    await utils_graph.acreate_relation(
        _Graph(),
        _VectorStorage(),
        _VectorStorage(),
        "A",
        "B",
        {"description": "A relates to B", "keywords": "k"},
    )

    assert upserted_edges == [("A", "B")]
