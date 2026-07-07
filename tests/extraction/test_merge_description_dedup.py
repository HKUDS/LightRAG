"""Regression tests: re-merging an entity/relation that already exists must
not re-append descriptions already stored on it.

``_merge_nodes_then_upsert``/``_merge_edges_then_upsert`` combined
``already_description`` (read back from the graph) with the newly extracted
descriptions via plain concatenation, deduping only *within* the new batch.
Any reprocess or resume of a document whose entities/relations already exist
would therefore re-append the same description every time: a single
failed-then-retried entity went 1 -> 2 -> 3 fragments instead of staying
idempotent at 1.

See https://github.com/HKUDS/LightRAG/issues/3367.
"""

import pytest

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.operate import _merge_edges_then_upsert, _merge_nodes_then_upsert
from lightrag.utils import Tokenizer, TokenizerInterface


class DummyTokenizer(TokenizerInterface):
    """1:1 character-to-token mapping, matching other operate.py tests."""

    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


def _make_global_config() -> dict:
    tokenizer = Tokenizer("dummy", DummyTokenizer())
    return {
        "tokenizer": tokenizer,
        "summary_context_size": 1_000_000,
        "summary_max_tokens": 1_000_000,
        "force_llm_summary_on_merge": 6,
        "source_ids_limit_method": "KEEP",
        "max_source_ids_per_entity": 10_000,
        "max_source_ids_per_relation": 10_000,
        "max_file_paths": 100,
        "file_path_more_placeholder": "...",
    }


class MemGraph:
    """Minimal in-memory graph storage exercising the real get/upsert round-trip."""

    def __init__(self):
        self.nodes = {}
        self.edges = {}

    async def get_node(self, name):
        return self.nodes.get(name)

    async def upsert_node(self, name, node_data):
        self.nodes[name] = dict(node_data)

    async def has_edge(self, src, tgt):
        return (src, tgt) in self.edges

    async def get_edge(self, src, tgt):
        return self.edges.get((src, tgt))

    async def upsert_edge(self, src, tgt, edge_data):
        self.edges[(src, tgt)] = dict(edge_data)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_reprocessing_same_entity_description_is_idempotent():
    """Merging the same single entity description 3 times must leave the
    stored description at 1 fragment, not grow by one per reprocess."""
    graph = MemGraph()
    cfg = _make_global_config()
    description = "Alice is a software engineer at Acme."
    batch = [
        {
            "entity_type": "person",
            "description": description,
            "source_id": "chunk-1",
            "file_path": "doc1.txt",
            "timestamp": 1,
        }
    ]

    fragment_counts = []
    for _ in range(3):
        await _merge_nodes_then_upsert("ALICE", batch, graph, None, cfg)
        stored = graph.nodes["ALICE"]["description"]
        fragment_counts.append(len(stored.split(GRAPH_FIELD_SEP)))

    assert fragment_counts == [1, 1, 1]
    assert graph.nodes["ALICE"]["description"] == description


@pytest.mark.offline
@pytest.mark.asyncio
async def test_reprocessing_adds_only_genuinely_new_descriptions():
    """A second, different description must still be appended once — only
    exact repeats of what's already stored should be skipped."""
    graph = MemGraph()
    cfg = _make_global_config()
    first = "Alice is a software engineer at Acme."
    second = "Alice also leads the platform team."

    await _merge_nodes_then_upsert(
        "ALICE",
        [
            {
                "entity_type": "person",
                "description": first,
                "source_id": "chunk-1",
                "file_path": "doc1.txt",
                "timestamp": 1,
            }
        ],
        graph,
        None,
        cfg,
    )
    # Reprocess with the same first chunk again, plus a genuinely new one.
    await _merge_nodes_then_upsert(
        "ALICE",
        [
            {
                "entity_type": "person",
                "description": first,
                "source_id": "chunk-1",
                "file_path": "doc1.txt",
                "timestamp": 1,
            },
            {
                "entity_type": "person",
                "description": second,
                "source_id": "chunk-2",
                "file_path": "doc1.txt",
                "timestamp": 2,
            },
        ],
        graph,
        None,
        cfg,
    )

    stored = graph.nodes["ALICE"]["description"]
    assert stored.split(GRAPH_FIELD_SEP) == [first, second]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_reprocessing_same_relation_description_is_idempotent():
    """Same idempotency guarantee for edges (_merge_edges_then_upsert)."""
    graph = MemGraph()
    cfg = _make_global_config()
    description = "Alice works at Acme as a software engineer."
    batch = [
        {
            "description": description,
            "source_id": "chunk-1",
            "file_path": "doc1.txt",
            "weight": 1.0,
            "timestamp": 1,
        }
    ]

    fragment_counts = []
    for _ in range(3):
        await _merge_edges_then_upsert("ALICE", "ACME", batch, graph, None, None, cfg)
        stored = graph.edges[("ALICE", "ACME")]["description"]
        fragment_counts.append(len(stored.split(GRAPH_FIELD_SEP)))

    assert fragment_counts == [1, 1, 1]
    assert graph.edges[("ALICE", "ACME")]["description"] == description
