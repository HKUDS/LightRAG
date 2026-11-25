"""Regression tests (#3609 follow-up): entity merge must keep chunk-tracking
row PRESENCE intact and decide it PER INPUT ENTITY.

Pinned defects, all in ``_merge_entities_impl`` (``lightrag/utils_graph.py``):

1. **Row presence is per input entity.** One entity's curated-empty row
   asserts only that THAT entity tracks no chunks — never that another
   input's unknown (absent/malformed) row is empty too. An input without a
   usable row falls back to its own pre-merge graph ``source_id``, exactly as
   a single-entity edit would. Folding it into an authoritative empty target
   row would silently discard its attribution with no recovery path: the
   empty row is authoritative from then on, so later edits never reseed it.

2. **Curated-empty rows survive the merge.** Merging entities whose rows are
   all present-but-empty must leave the target with a present empty row —
   an absent target row would let the next edit reseed tracking from the
   merged node's possibly-stale graph ``source_id`` (``join_unique`` keeps
   every input's IDs) through the HTTP-reachable merge endpoint.

3. **Unknown stays unknown.** With no input row present at all, the target
   row stays absent; fabricating one would promote UNKNOWN to authoritative.

4. **Every incident relation is redirected**, whichever endpoint
   ``get_node_edges`` lists first. MongoDB, Memgraph and OpenSearch return
   each edge in its stored direction, so a relation stored *towards* a source
   entity must still move to the target, with its tracking row migrated.

The write ordering of the migrated rows (upsert before delete, old keys
retired only after the graph commit, a key that is both old and new never
deleted) is pinned by ``tests/pipeline/test_graph_merge_rename_tracking_order.py``.
"""

import pytest

from lightrag import utils_graph
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.utils import make_relation_chunk_key

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


class _MergeGraphStorage:
    """Entities A and B, each with an edge to a shared node C."""

    def __init__(self):
        self.nodes = {
            name: {
                "entity_id": name,
                "description": f"Entity {name}",
                "entity_type": "ORG",
                "source_id": f"chunk-{name.lower()}-node",
                "file_path": f"doc{name}.txt",
            }
            for name in ("A", "B", "C")
        }
        self.edges = {
            ("A", "C"): {
                "description": "A relates to C",
                "keywords": "k1",
                "source_id": GRAPH_FIELD_SEP.join(["chunk-a1"]),
                "weight": 1.0,
                "file_path": "docA.txt",
            },
            ("B", "C"): {
                "description": "B relates to C",
                "keywords": "k2",
                "source_id": GRAPH_FIELD_SEP.join(["chunk-b1"]),
                "weight": 1.0,
                "file_path": "docB.txt",
            },
        }

    async def has_node(self, node_id):
        return node_id in self.nodes

    async def get_node(self, node_id):
        return self.nodes[node_id]

    async def upsert_node(self, node_id, node_data):
        self.nodes[node_id] = dict(node_data)

    async def get_node_edges(self, node_id):
        return [
            (src, tgt) for (src, tgt) in self.edges if src == node_id or tgt == node_id
        ]

    async def get_edge(self, src, tgt):
        return self.edges.get((src, tgt)) or self.edges.get((tgt, src))

    async def upsert_edge(self, src, tgt, edge_data):
        self.edges[(src, tgt)] = dict(edge_data)

    async def delete_node(self, node_id):
        self.nodes.pop(node_id, None)
        self.edges = {
            (src, tgt): data
            for (src, tgt), data in self.edges.items()
            if src != node_id and tgt != node_id
        }

    async def index_done_callback(self):
        return True


class _DummyVectorStorage:
    def __init__(self):
        self.global_config = {"workspace": "test"}

    async def upsert(self, data):
        return None

    async def delete(self, ids):
        return None

    async def get_by_id(self, id_):
        return None

    async def index_done_callback(self):
        return True


class _KV:
    def __init__(self, data: dict | None = None):
        self.data = dict(data or {})

    async def get_by_id(self, key):
        return self.data.get(key)

    async def upsert(self, data):
        self.data.update(data)

    async def delete(self, keys):
        for key in keys:
            self.data.pop(key, None)

    async def index_done_callback(self):
        return True


async def _run_merge(
    ecs: _KV,
    *,
    source_entities: list[str],
    target_entity: str,
    rcs: _KV | None = None,
) -> _MergeGraphStorage:
    graph = _MergeGraphStorage()
    await utils_graph._merge_entities_impl(
        chunk_entity_relation_graph=graph,
        entities_vdb=_DummyVectorStorage(),
        relationships_vdb=_DummyVectorStorage(),
        source_entities=source_entities,
        target_entity=target_entity,
        entity_chunks_storage=ecs,
        relation_chunks_storage=rcs,
    )
    return graph


# The graph nodes carry source_ids chunk-a-node / chunk-b-node, so every case
# also tells a present row (used as stored) from the graph fallback. The whole
# store is compared, so each case also pins that the source rows are retired.
@pytest.mark.parametrize(
    ("rows", "sources", "target", "expected"),
    [
        pytest.param(
            {"A": {"chunk_ids": [], "count": 0}, "B": {"chunk_ids": [], "count": 0}},
            ["A", "B"],
            "AB",
            {"AB": {"chunk_ids": [], "count": 0}},
            id="all-curated-empty-stays-present-and-empty",
        ),
        pytest.param(
            {
                "A": {"chunk_ids": ["c-a"], "count": 1},
                "B": {"chunk_ids": ["c-b"], "count": 1},
            },
            ["A", "B"],
            "AB",
            {"AB": {"chunk_ids": ["c-a", "c-b"], "count": 2}},
            id="every-present-row-joins-the-union",
        ),
        pytest.param(
            {},
            ["A", "B"],
            "AB",
            {},
            id="no-row-present-stays-absent",
        ),
        pytest.param(
            {"A": {"chunk_ids": [], "count": 0}},
            ["A", "B"],
            "AB",
            {"AB": {"chunk_ids": ["chunk-b-node"], "count": 1}},
            id="empty-row-does-not-silence-an-absent-row",
        ),
        pytest.param(
            {"A": {"chunk_ids": ["c-a"], "count": 1}, "B": {"chunk_ids": None}},
            ["A", "B"],
            "AB",
            {"AB": {"chunk_ids": ["c-a", "chunk-b-node"], "count": 2}},
            id="malformed-row-falls-back-like-an-absent-one",
        ),
        # The existing target's fallback is its PRE-merge source_id, not the
        # merged node's, which already unions every input's chunks.
        pytest.param(
            {"A": {"chunk_ids": ["c-a"], "count": 1}},
            ["A"],
            "B",
            {"B": {"chunk_ids": ["c-a", "chunk-b-node"], "count": 2}},
            id="existing-target-falls-back-to-its-pre-merge-source-id",
        ),
    ],
)
async def test_merge_decides_entity_row_presence_per_input(
    rows, sources, target, expected
):
    ecs = _KV(rows)

    await _run_merge(ecs, source_entities=sources, target_entity=target)

    assert ecs.data == expected


async def test_merge_unions_relation_rows_that_land_on_one_key():
    """Merging A into the existing B maps (A,C) onto B's own key (B,C): the
    row under that key must carry both relations' chunks, and only the
    replaced key (A,C) is retired."""
    rcs = _KV(
        {
            make_relation_chunk_key("A", "C"): {"chunk_ids": ["c-ac"], "count": 1},
            make_relation_chunk_key("B", "C"): {"chunk_ids": ["c-bc"], "count": 1},
        }
    )

    await _run_merge(_KV(), source_entities=["A"], target_entity="B", rcs=rcs)

    assert rcs.data == {
        make_relation_chunk_key("B", "C"): {"chunk_ids": ["c-ac", "c-bc"], "count": 2}
    }


async def test_merge_redirects_relations_stored_towards_the_source():
    """The stub lists edges in stored direction, like MongoDB, Memgraph and
    OpenSearch. C is the TARGET of both (A,C) and (B,C); a merge that kept
    only tuples whose first element is the source dropped both relations
    together with C and left their tracking rows behind."""
    rcs = _KV(
        {
            make_relation_chunk_key("A", "C"): {"chunk_ids": ["c-ac"], "count": 1},
            make_relation_chunk_key("B", "C"): {"chunk_ids": ["c-bc"], "count": 1},
        }
    )

    graph = await _run_merge(_KV(), source_entities=["C"], target_entity="D", rcs=rcs)

    assert {pair: data["description"] for pair, data in graph.edges.items()} == {
        ("A", "D"): "A relates to C",
        ("B", "D"): "B relates to C",
    }
    assert rcs.data == {
        make_relation_chunk_key("A", "D"): {"chunk_ids": ["c-ac"], "count": 1},
        make_relation_chunk_key("B", "D"): {"chunk_ids": ["c-bc"], "count": 1},
    }
