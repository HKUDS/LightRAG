"""The deletion rebuild sees a multimodal chunk's core-generated records.

A drawing / table / equation chunk's entity and its association edges are
built from the chunk's ``sidecar`` block, never from LLM output, so its cached
extraction row cannot contain them. The rebuild must synthesize them exactly as
``extract_entities`` does at ingestion; otherwise the multimodal entity counts
as unattested and is rebuilt without the chunk it was made from.
"""

import pytest

from lightrag.kg.shared_storage import initialize_share_data
from lightrag.operate import rebuild_knowledge_from_chunks
from lightrag.utils import Tokenizer

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _shared_data():
    initialize_share_data()


class _Characters:
    def encode(self, content):
        return [ord(c) for c in content]

    def decode(self, tokens):
        return "".join(chr(c) for c in tokens)


class _KV:
    def __init__(self, data=None):
        self.data = dict(data or {})

    async def get_by_id(self, key):
        return self.data.get(key)

    async def get_by_ids(self, keys):
        return [self.data.get(k) for k in keys]

    async def upsert(self, data):
        self.data.update(data)

    async def delete(self, keys):
        for key in keys:
            self.data.pop(key, None)


class _Graph:
    def __init__(self, nodes, edges):
        self.nodes = dict(nodes)
        self.edges = dict(edges)

    async def get_node(self, name):
        return self.nodes.get(name)

    async def has_node(self, name):
        return name in self.nodes

    async def upsert_node(self, name, node_data):
        self.nodes[name] = node_data

    async def get_edge(self, src, tgt):
        return self.edges.get((src, tgt)) or self.edges.get((tgt, src))

    async def upsert_edge(self, src, tgt, edge_data):
        self.edges[(src, tgt)] = edge_data

    async def get_node_edges(self, name):
        return [key for key in self.edges if name in key]


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["best_effort", "rollback"])
async def test_multimodal_entity_is_rebuilt_from_its_sidecar_chunk(policy):
    content = "Table T1: quarterly revenue"
    text_chunks = _KV(
        {
            "c1": {
                "content": content,
                "file_path": "report.pdf",
                "sidecar": {"type": "table", "id": "tbl-1"},
                "llm_cache_list": ["default:extract:k1"],
            }
        }
    )
    llm_cache = _KV(
        {
            "default:extract:k1": {
                "cache_type": "extract",
                "return": "entity<|#|>Revenue<|#|>concept<|#|>Quarterly revenue\n"
                "<|COMPLETE|>",
                "original_prompt": "user prompt\nsystem prompt",
            }
        }
    )
    stale = {"source_id": "c1<SEP>c-deleted", "file_path": "old.pdf"}
    graph = _Graph(
        nodes={
            "tbl-1": {
                "description": "stale table text",
                "entity_type": "table",
                **stale,
            },
        },
        edges={
            ("tbl-1", "Revenue"): {
                "description": "stale edge text",
                "keywords": "old",
                "weight": 2.0,
                **stale,
            }
        },
    )
    global_config = {
        "llm_model_max_async": 1,
        "max_source_ids_per_entity": 10,
        "max_source_ids_per_relation": 10,
        "max_file_paths": 10,
        "source_ids_limit_method": "KEEP",
        "tokenizer": Tokenizer("characters", _Characters()),
        "summary_context_size": 1_000_000,
        "summary_max_tokens": 1_000_000,
        "force_llm_summary_on_merge": 6,
    }

    report = await rebuild_knowledge_from_chunks(
        entities_to_rebuild={"tbl-1": ["c1"]},
        relationships_to_rebuild={("tbl-1", "Revenue"): ["c1"]},
        knowledge_graph_inst=graph,
        entities_vdb=_KV(),
        relationships_vdb=_KV(),
        text_chunks_storage=text_chunks,
        llm_response_cache=llm_cache,
        global_config=global_config,
        rebuild_policy=policy,
    )

    assert report.degraded_entities == {}
    assert report.degraded_relationships == {}
    assert report.reextracted_chunk_ids == set()
    node = graph.nodes["tbl-1"]
    assert node["description"] == content
    assert (node["source_id"], node["file_path"]) == ("c1", "report.pdf")
    edge = graph.edges[("tbl-1", "Revenue")]
    assert edge["description"].startswith("Revenue is associated with table")
    assert (edge["source_id"], edge["file_path"]) == ("c1", "report.pdf")
