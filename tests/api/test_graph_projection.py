from types import SimpleNamespace
import sys

import pytest

# The API package lazily parses command-line configuration at import time.
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
from lightrag.api.assets import WorkspaceAssetCatalog  # noqa: E402
from lightrag.api.graph_projection import build_graph_projection  # noqa: E402
sys.argv = _original_argv


class _Graph:
    async def get_all_nodes(self):
        return [
            {"entity_id": "A", "description": "A node", "source_id": "chunk-a"},
            {"entity_id": "B", "description": "B node", "source_id": "doc-b"},
            {"entity_id": "C", "description": "C node", "source_id": "doc-c"},
        ]

    async def get_all_edges(self):
        return [
            {"source": "A", "target": "B", "keywords": "relates,to", "weight": 2},
            {"source": "B", "target": "C", "keywords": ["supports"], "weight": 1},
        ]


class _Chunks:
    async def get_by_ids(self, ids):
        assert ids == ["chunk-a"]
        return [{"full_doc_id": "doc-a"}]


class _Rag:
    chunk_entity_relation_graph = _Graph()
    text_chunks = _Chunks()


@pytest.mark.asyncio
async def test_projection_normalizes_and_resolves_sources(tmp_path):
    manager = SimpleNamespace(input_dir=tmp_path, workspace="storage-key")
    result = await build_graph_projection(
        _Rag(),
        WorkspaceAssetCatalog(manager),
        "workspace-id",
        max_nodes=2,
        max_edges=10,
    )

    assert result["workspace_id"] == "workspace-id"
    assert {node["id"] for node in result["nodes"]} == {"A", "B"}
    assert result["nodes"][0]["metadata"]["degree"] >= 1
    assert result["edges"][0]["keywords"] == ["relates", "to"]
    assert result["stats"]["total_nodes"] == 3
    assert result["page"]["next_cursor"] is not None


@pytest.mark.asyncio
async def test_projection_seed_limits_by_depth(tmp_path):
    manager = SimpleNamespace(input_dir=tmp_path, workspace="storage-key")
    result = await build_graph_projection(
        _Rag(),
        WorkspaceAssetCatalog(manager),
        "workspace-id",
        seed="A",
        max_depth=0,
    )
    assert [node["id"] for node in result["nodes"]] == ["A"]
    assert result["edges"] == []
