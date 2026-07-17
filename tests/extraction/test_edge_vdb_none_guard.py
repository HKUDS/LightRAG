"""Regression test: an edge merge that adds a new endpoint entity must not
crash when ``entity_vdb=None`` (issue #3367 sibling).

In ``_merge_edges_then_upsert`` the entity-vdb upsert call for the
new-endpoint path used to sit OUTSIDE the ``if entity_vdb is not None:`` guard
while ``vdb_data`` is assigned only inside it. So with ``entity_vdb=None`` a new
endpoint source raised ``UnboundLocalError: vdb_data`` (and would have called
``None.upsert``). The fix moves the call inside the guard.

This drives the real merge round-trip against an in-memory graph, no DB or LLM.
"""

import pytest

from lightrag.operate import _merge_edges_then_upsert
import lightrag.operate as operate


class _FakeTokenizer:
    def encode(self, s: str):  # token count == char count; thresholds kept slack
        return list(range(len(s)))


class _MemGraph:
    """Minimal in-memory graph: real get_node/get_edge -> merge -> upsert round-trip."""

    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: dict = {}

    async def get_node(self, name):
        return self.nodes.get(name)

    async def has_node(self, name):
        return name in self.nodes

    async def upsert_node(self, name, node_data):
        self.nodes[name] = dict(node_data)

    async def has_edge(self, s, t):
        return (s, t) in self.edges or (t, s) in self.edges

    async def get_edge(self, s, t):
        return self.edges.get((s, t)) or self.edges.get((t, s))

    async def upsert_edge(self, s, t, edge_data):
        self.edges[(s, t)] = dict(edge_data)


def _cfg() -> dict:
    return {
        "tokenizer": _FakeTokenizer(),
        "summary_context_size": 1_000_000,
        "summary_max_tokens": 1_000_000,
        "force_llm_summary_on_merge": 6,
        "source_ids_limit_method": operate.SOURCE_IDS_LIMIT_METHOD_KEEP,
        "max_source_ids_per_entity": 10_000,
        "max_source_ids_per_relation": 10_000,
        "max_file_paths": 100,
        "file_path_more_placeholder": "...",
    }


@pytest.mark.offline
@pytest.mark.asyncio
async def test_edge_merge_with_entity_vdb_none_does_not_crash():
    """entity_vdb=None must not raise UnboundLocalError when an edge merge adds a
    new endpoint entity. The vdb upsert call must stay inside its `entity_vdb is
    not None` guard (it references vdb_data, which is assigned only inside it)."""
    g = _MemGraph()
    cfg = _cfg()
    for name in ("A", "B"):
        await g.upsert_node(
            name,
            {
                "entity_id": name,
                "description": name,
                "source_id": "c1",
                "entity_type": "X",
                "file_path": "f",
            },
        )

    def _edge(src: str) -> dict:
        return {
            "weight": 1.0,
            "source_id": src,
            "description": "rel",
            "keywords": "k",
            "file_path": "f",
        }

    # entity_vdb=None + genuinely new sources is the path that used to crash.
    for src in ("c1", "c2", "c3"):
        await _merge_edges_then_upsert("A", "B", [_edge(src)], g, None, None, cfg)

    assert await g.get_edge("A", "B") is not None
