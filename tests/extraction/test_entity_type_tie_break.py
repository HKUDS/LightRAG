"""_merge_nodes_then_upsert only ever carries the CURRENT stored entity_type
forward as a single vote (``already_entity_types`` has at most one entry, no
matter how many earlier merges established it). One new, differently-typed
extraction in the next merge is therefore a tie, not a minority -- and a
stable sort that put the new vote first let ties silently flip an
established entity's type to whatever the latest extraction happened to say.
"""

import pytest

from lightrag.operate import SOURCE_IDS_LIMIT_METHOD_KEEP, _merge_nodes_then_upsert
from lightrag.utils import Tokenizer, TokenizerInterface


class _DummyTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


class _MemGraph:
    def __init__(self):
        self.nodes = {}

    async def get_node(self, name):
        return self.nodes.get(name)

    async def upsert_node(self, name, node_data):
        self.nodes[name] = dict(node_data)


def _config():
    return {
        "tokenizer": Tokenizer("dummy", _DummyTokenizer()),
        "summary_context_size": 1_000_000,
        "summary_max_tokens": 1_000_000,
        "force_llm_summary_on_merge": 1_000_000,
        "source_ids_limit_method": SOURCE_IDS_LIMIT_METHOD_KEEP,
        "max_source_ids_per_entity": 10_000,
        "max_source_ids_per_relation": 10_000,
        "max_file_paths": 100,
        "file_path_more_placeholder": "...",
    }


@pytest.mark.offline
@pytest.mark.asyncio
async def test_established_type_survives_a_single_conflicting_extraction():
    """An entity_type established over several prior merges must not flip on
    a tie against one new, differently-typed extraction."""
    graph = _MemGraph()
    cfg = _config()
    base = {
        "entity_name": "ALICE",
        "source_id": "chunk-1",
        "file_path": "doc1.txt",
        "timestamp": 1,
        "description": "Alice works here.",
    }

    for i in range(5):
        await _merge_nodes_then_upsert(
            "ALICE",
            [dict(base, entity_type="PERSON", timestamp=i)],
            graph,
            None,
            cfg,
        )
    assert graph.nodes["ALICE"]["entity_type"] == "PERSON"

    await _merge_nodes_then_upsert(
        "ALICE",
        [dict(base, entity_type="ORGANIZATION", timestamp=99)],
        graph,
        None,
        cfg,
    )

    assert graph.nodes["ALICE"]["entity_type"] == "PERSON"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_genuine_majority_still_changes_the_type():
    """Negative twin: a real majority within one merge batch must still win,
    not just be swallowed by the incumbent-favoring tiebreak."""
    graph = _MemGraph()
    cfg = _config()
    base = {
        "entity_name": "ALICE",
        "source_id": "chunk-1",
        "file_path": "doc1.txt",
        "timestamp": 1,
    }

    await _merge_nodes_then_upsert(
        "ALICE",
        [dict(base, entity_type="PERSON", description="d1")],
        graph,
        None,
        cfg,
    )
    assert graph.nodes["ALICE"]["entity_type"] == "PERSON"

    await _merge_nodes_then_upsert(
        "ALICE",
        [
            dict(base, entity_type="ORGANIZATION", description="d2", timestamp=2),
            dict(base, entity_type="ORGANIZATION", description="d3", timestamp=3),
        ],
        graph,
        None,
        cfg,
    )

    assert graph.nodes["ALICE"]["entity_type"] == "ORGANIZATION"
