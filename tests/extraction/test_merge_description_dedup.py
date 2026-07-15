"""Regression tests: merge must not accumulate duplicate description fragments
across reprocess/resume (issue #3367).

``_merge_nodes_then_upsert`` / ``_merge_edges_then_upsert`` combine the stored
node/edge description (read back from the graph) with the newly extracted
descriptions. Before the fix this was ``already_description + sorted_descriptions``,
which deduplicated only *within* the new batch, not *between* stored and new.
So re-running merge for an entity/relation that already exists (any reprocess or
resume) re-appended the same description, growing the stored fragment count by
one copy per reprocess (N -> N+1).

These drive the real merge round-trip (``get_node``/``get_edge`` -> merge ->
``upsert_node``/``upsert_edge``) against an in-memory graph, no DB or LLM.
"""

import pytest

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.operate import (
    SOURCE_IDS_LIMIT_METHOD_KEEP,
    _combine_descriptions_dedup,
    _merge_edges_then_upsert,
    _merge_nodes_then_upsert,
)
from lightrag.utils import Tokenizer, TokenizerInterface


class _DummyTokenizer(TokenizerInterface):
    """1:1 char-to-token mapping; keeps summary thresholds easy to reason about."""

    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


class _MemGraph:
    """Minimal in-memory graph: real read-back + write round-trip."""

    def __init__(self):
        self.nodes = {}
        self.edges = {}

    async def get_node(self, name):
        return self.nodes.get(name)

    async def upsert_node(self, name, node_data):
        self.nodes[name] = dict(node_data)

    async def has_node(self, name):
        return name in self.nodes

    async def has_edge(self, src, tgt):
        return (src, tgt) in self.edges

    async def get_edge(self, src, tgt):
        return self.edges.get((src, tgt))

    async def upsert_edge(self, src, tgt, edge_data):
        self.edges[(src, tgt)] = dict(edge_data)


def _config():
    # Summary limits kept slack so the no-LLM join path is always taken:
    # the accumulation bug lives in fragment assembly, not summarization.
    return {
        "tokenizer": Tokenizer("dummy", _DummyTokenizer()),
        "summary_context_size": 1_000_000,
        "summary_max_tokens": 1_000_000,
        "force_llm_summary_on_merge": 6,
        "source_ids_limit_method": SOURCE_IDS_LIMIT_METHOD_KEEP,
        "max_source_ids_per_entity": 10_000,
        "max_source_ids_per_relation": 10_000,
        "max_file_paths": 100,
        "file_path_more_placeholder": "...",
    }


def _node_fragments(graph, name):
    return graph.nodes[name]["description"].split(GRAPH_FIELD_SEP)


def _edge_fragments(graph, src, tgt):
    return graph.edges[(src, tgt)]["description"].split(GRAPH_FIELD_SEP)


# --- direct unit tests of the dedup helper ---------------------------------


def test_combine_descriptions_dedup_cross_boundary():
    combined, already = _combine_descriptions_dedup(["A", "B"], ["B", "C"])
    # Stored first, then only genuinely new fragments; "B" is not re-appended.
    assert combined == ["A", "B", "C"]
    assert already == 2


def test_combine_descriptions_dedup_collapses_legacy_stored_duplicates():
    # Stored data that already accumulated duplicates self-heals on next merge.
    combined, already = _combine_descriptions_dedup(["A", "A", "A"], ["A"])
    assert combined == ["A"]
    assert already == 1


def test_combine_descriptions_dedup_preserves_distinct():
    combined, already = _combine_descriptions_dedup(["A"], ["B", "C"])
    assert combined == ["A", "B", "C"]
    assert already == 1


def test_combine_descriptions_dedup_sanitizes_before_compare():
    # A re-extracted description carrying an XML-illegal control char (\x08)
    # sanitizes to a fragment already stored, so it must NOT accumulate on
    # reprocess (issue #3367 P3c; aligns with #3373's dirty-char case). A raw
    # comparison would treat the two as distinct and grow the fragment count.
    stored = ["Alice is a software engineer at Acme."]
    dirty_new = ["Alice\x08 is a software engineer at Acme."]
    combined, already = _combine_descriptions_dedup(stored, dirty_new)
    assert combined == ["Alice is a software engineer at Acme."]
    assert already == 1


def test_combine_descriptions_dedup_collapses_dirty_stored_fragments():
    # Legacy dirty stored fragments (written before sanitization) collapse
    # against a clean re-extraction; the surviving stored count reflects the
    # sanitized fragment, not the raw duplicate.
    combined, already = _combine_descriptions_dedup(
        ["Bob\x08 builds pipelines.", "Bob builds pipelines."],
        ["Bob builds pipelines."],
    )
    assert combined == ["Bob builds pipelines."]
    assert already == 1


def test_combine_descriptions_dedup_drops_empty_after_sanitize():
    # A fragment that is only control chars sanitizes to "" and is dropped, so
    # it neither inflates already_fragment nor emits a <sep><sep> artifact.
    combined, already = _combine_descriptions_dedup(["A", "\x08"], ["\x00", "B"])
    assert combined == ["A", "B"]
    assert already == 1


# --- node merge round-trip --------------------------------------------------


@pytest.mark.offline
@pytest.mark.asyncio
async def test_node_reprocess_with_dirty_char_does_not_accumulate():
    """Round-trip twin of the helper test (aligns with #3373 test #2): a
    re-extracted description with an XML-illegal control char must dedup against
    the sanitized stored copy instead of accumulating on reprocess (#3367)."""
    graph = _MemGraph()
    cfg = _config()
    clean = "Alice is a software engineer at Acme."
    dirty = "Alice\x08 is a software engineer at Acme."
    base = {
        "entity_name": "ALICE",
        "entity_type": "person",
        "source_id": "chunk-1",
        "file_path": "doc1.txt",
        "timestamp": 1,
    }

    await _merge_nodes_then_upsert(
        "ALICE", [dict(base, description=clean)], graph, None, cfg
    )
    # Reprocess with a dirty variant of the SAME description.
    await _merge_nodes_then_upsert(
        "ALICE", [dict(base, description=dirty, timestamp=2)], graph, None, cfg
    )

    assert _node_fragments(graph, "ALICE") == [clean]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_node_reprocess_does_not_accumulate_description():
    graph = _MemGraph()
    cfg = _config()
    batch = {
        "entity_name": "ALICE",
        "entity_type": "person",
        "description": "Alice is a software engineer at Acme.",
        "source_id": "chunk-1",
        "file_path": "doc1.txt",
        "timestamp": 1,
    }

    counts = []
    for _ in range(3):  # 3 reprocesses of the SAME description, no purge between
        await _merge_nodes_then_upsert("ALICE", [dict(batch)], graph, None, cfg)
        counts.append(len(_node_fragments(graph, "ALICE")))

    # Before the fix this was [1, 2, 3]; the stored description must stay single.
    assert counts == [1, 1, 1]
    assert _node_fragments(graph, "ALICE") == ["Alice is a software engineer at Acme."]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_node_merge_keeps_distinct_descriptions():
    """Negative twin: genuinely different descriptions are still accumulated."""
    graph = _MemGraph()
    cfg = _config()
    base = {
        "entity_name": "ALICE",
        "entity_type": "person",
        "source_id": "chunk-1",
        "file_path": "doc1.txt",
        "timestamp": 1,
    }

    await _merge_nodes_then_upsert(
        "ALICE", [dict(base, description="Alice is an engineer.")], graph, None, cfg
    )
    await _merge_nodes_then_upsert(
        "ALICE",
        [dict(base, description="Alice leads the platform team.", timestamp=2)],
        graph,
        None,
        cfg,
    )

    frags = _node_fragments(graph, "ALICE")
    assert set(frags) == {
        "Alice is an engineer.",
        "Alice leads the platform team.",
    }
    assert len(frags) == 2


# --- edge merge round-trip --------------------------------------------------


@pytest.mark.offline
@pytest.mark.asyncio
async def test_edge_reprocess_does_not_accumulate_description():
    graph = _MemGraph()
    # Endpoints already exist so the edge path does not synthesize new nodes.
    graph.nodes["ALICE"] = {
        "entity_type": "person",
        "description": "x",
        "source_id": "chunk-1",
        "file_path": "doc1.txt",
    }
    graph.nodes["ACME"] = {
        "entity_type": "organization",
        "description": "y",
        "source_id": "chunk-1",
        "file_path": "doc1.txt",
    }
    cfg = _config()
    batch = {
        "src_id": "ALICE",
        "tgt_id": "ACME",
        "description": "Alice works at Acme.",
        "keywords": "employment",
        "weight": 1.0,
        "source_id": "chunk-1",
        "file_path": "doc1.txt",
        "timestamp": 1,
    }

    counts = []
    for _ in range(3):
        await _merge_edges_then_upsert(
            "ALICE", "ACME", [dict(batch)], graph, None, None, cfg
        )
        counts.append(len(_edge_fragments(graph, "ALICE", "ACME")))

    assert counts == [1, 1, 1]
    assert _edge_fragments(graph, "ALICE", "ACME") == ["Alice works at Acme."]
