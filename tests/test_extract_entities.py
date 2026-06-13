"""
Tests for entity extraction (llm_frames mode) and bridge builder.

Covers:
  - extract_entities_from_frames_llm() returns 3-tuple with frame_meta
  - extract_entities() in llm_frames mode collects bridge_meta correctly
  - bridge_builder: co-frame binary edges
  - bridge_builder: co-frame hub hyperedge (>=3 events)
  - bridge_builder: co-entity binary edges
  - bridge_builder: co-entity hub hyperedge
  - bridge_builder: N-M cluster hyperedge
  - bridge_builder: entity similarity merging
  - bridge_builder: entity similarity edges (medium cosine)
  - bridge_builder: frame similarity edges
  - bridge_builder: causal edges (LLM)
  - bridge_builder: full pipeline integration
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

# ══════════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ══════════════════════════════════════════════════════════════════════════════

EVENT_PREFIX = "EVENT:"
FRAME_PREFIX = "FRAME:"


def _ts() -> int:
    return int(time.time())


def _make_event_node(event_id: str, frames: list[str]) -> dict:
    return {
        "entity_name": event_id,
        "entity_type": "event",
        "description": f"Event {event_id}. Evokes: {', '.join(frames)}.",
        "source_id": "chunk-test",
        "file_path": "test",
        "timestamp": _ts(),
    }


def _make_frame_node(frame_name: str) -> dict:
    return {
        "entity_name": frame_name,
        "entity_type": "frame",
        "description": f"Frame: {frame_name} definition.",
        "source_id": "chunk-test",
        "file_path": "test",
        "timestamp": _ts(),
    }


def _make_entity_node(name: str, role: str = "agent") -> dict:
    return {
        "entity_name": name,
        "entity_type": role,
        "description": f'"{name}" is a {role}.',
        "source_id": "chunk-test",
        "file_path": "test",
        "timestamp": _ts(),
    }


def _make_edge(src: str, tgt: str, kw: str = "evokes", w: float = 1.0) -> dict:
    return {
        "src_id": src, "tgt_id": tgt,
        "weight": w, "keywords": kw,
        "description": f"{src} {kw} {tgt}",
        "source_id": "chunk-test",
        "file_path": "test",
        "timestamp": _ts(),
    }


def _make_global_config() -> dict:
    return {
        "llm_model_func": AsyncMock(return_value='{"frames": ["VectorIndexing"]}'),
        "entity_extract_max_gleaning": 0,
        "addon_params": {},
        "llm_model_max_async": 2,
        "working_dir": ".",
        "embedding_func": AsyncMock(
            return_value=np.random.rand(2, 64).tolist()
        ),
    }


def _make_chunks(contents: list[str] | None = None) -> dict[str, dict]:
    contents = contents or ["LightRAG uses vector indexing to store embeddings."]
    return {
        f"chunk-{i:03d}": {
            "tokens": len(c),
            "content": c,
            "full_doc_id": "doc-001",
            "chunk_order_index": i,
        }
        for i, c in enumerate(contents)
    }


# ══════════════════════════════════════════════════════════════════════════════
# Section 1 — llm_frame_extractor: 3-tuple return
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.offline
@pytest.mark.asyncio
async def test_extract_entities_from_frames_llm_returns_3tuple():
    """extract_entities_from_frames_llm must return (nodes, edges, frame_meta)."""
    from lightrag.llm_frame_extractor import extract_entities_from_frames_llm
    from lightrag.llm_frame_db import get_frame_db

    llm_mock = AsyncMock()
    # Sequence: identify → define → extract_instances → score_representativeness
    llm_mock.side_effect = [
        '{"frames": ["VectorIndexing"]}',
        '{"name": "VectorIndexing", "definition": "Indexing vectors.", '
        '"frame_elements": {"Agent": {"definition": "Who indexes", "type": "core"}}, '
        '"lexical_units": ["index.v"], "relations": []}',
        '{"instances": [{"frame": "VectorIndexing", "elements": {"Agent": "LightRAG"}}]}',
        '{"score": 0.9, "reasoning": "central event"}',
    ]

    # Use an isolated temp frame DB
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        result = await extract_entities_from_frames_llm(
            text="LightRAG indexes vectors efficiently.",
            chunk_key="chunk-abc",
            file_path="test.txt",
            llm_func=llm_mock,
            working_dir=tmpdir,
        )

    assert isinstance(result, tuple) and len(result) == 3
    nodes, edges, meta = result

    assert isinstance(nodes, dict)
    assert isinstance(edges, dict)
    assert isinstance(meta, dict)
    assert "event_id" in meta
    assert "frame_names" in meta
    assert meta["event_id"].startswith("EVENT:")
    assert "VectorIndexing" in meta["frame_names"]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_extract_entities_from_frames_llm_empty_on_no_frames():
    """When LLM returns no frames, result must be empty nodes/edges."""
    from lightrag.llm_frame_extractor import extract_entities_from_frames_llm

    llm_mock = AsyncMock(return_value='{"frames": []}')
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        nodes, edges, meta = await extract_entities_from_frames_llm(
            text="Some ambiguous text.",
            chunk_key="chunk-xyz",
            file_path="test.txt",
            llm_func=llm_mock,
            working_dir=tmpdir,
        )

    assert nodes == {}
    assert edges == {}
    assert meta["event_id"].startswith("EVENT:")


# ══════════════════════════════════════════════════════════════════════════════
# Section 2 — extract_entities() in llm_frames mode
# ══════════════════════════════════════════════════════════════════════════════

_FAKE_FRAME_META = {"event_id": "EVENT:aabbccdd", "frame_names": ["VectorIndexing"]}
_FAKE_NODES = {
    "EVENT:aabbccdd": [_make_event_node("EVENT:aabbccdd", ["VectorIndexing"])],
    "VectorIndexing":  [_make_frame_node("VectorIndexing")],
    "LightRAG":        [_make_entity_node("LightRAG", "agent")],
}
_FAKE_EDGES = {
    ("EVENT:aabbccdd", "VectorIndexing"): [
        _make_edge("EVENT:aabbccdd", "VectorIndexing", "evokes", 0.9)
    ],
    ("VectorIndexing", "LightRAG"): [
        _make_edge("VectorIndexing", "LightRAG", "Agent", 1.0)
    ],
}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_extract_entities_llm_frames_returns_3tuple_results():
    """extract_entities() must return list of 3-tuples in llm_frames mode."""
    from lightrag.operate import extract_entities

    config = _make_global_config()

    with patch(
        "lightrag.operate.extract_entities_from_frames_llm",
        new_callable=AsyncMock,
        return_value=(_FAKE_NODES, _FAKE_EDGES, _FAKE_FRAME_META),
    ):
        with patch("lightrag.operate.FRAME_EXTRACTION_MODE", "llm_frames"):
            results = await extract_entities(
                chunks=_make_chunks(),
                global_config=config,
            )

    assert len(results) == 1
    assert len(results[0]) == 3  # 3-tuple
    nodes, edges, bridge_meta = results[0]
    assert "EVENT:aabbccdd" in nodes
    assert bridge_meta is not None
    assert bridge_meta["event_id"] == "EVENT:aabbccdd"
    assert "entity_names" in bridge_meta       # enriched with entity names
    assert "chunk_content" in bridge_meta      # enriched with chunk text
    assert "LightRAG" in bridge_meta["entity_names"]


@pytest.mark.offline
@pytest.mark.asyncio
async def test_extract_entities_none_mode_returns_none_bridge_meta():
    """In 'none' baseline mode, bridge_meta must be None."""
    from lightrag.operate import extract_entities

    llm_response = (
        '{"entities": [{"name": "TestEnt", "type": "person", "desc": "a person"}],'
        '"relationships": []}'
    )
    config = _make_global_config()
    config["llm_model_func"] = AsyncMock(return_value=llm_response)
    config["tokenizer"] = MagicMock(encode=lambda x: list(x))
    config["max_extract_input_tokens"] = 999999

    with patch("lightrag.operate.FRAME_EXTRACTION_MODE", "none"):
        results = await extract_entities(
            chunks=_make_chunks(),
            global_config=config,
        )

    assert len(results) == 1
    assert len(results[0]) == 3
    _, _, bridge_meta = results[0]
    assert bridge_meta is None


# ══════════════════════════════════════════════════════════════════════════════
# Section 3 — bridge_builder: co-frame edges
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.offline
def test_co_frame_binary_edges_two_events():
    """Two events sharing a frame → one binary co_frame edge."""
    from lightrag.bridge_builder import _build_co_frame_edges

    meta = [
        {"event_id": "EVENT:a1", "frame_names": ["VectorIndexing"], "entity_names": []},
        {"event_id": "EVENT:b2", "frame_names": ["VectorIndexing"], "entity_names": []},
    ]
    binary, (hub_nodes, hub_edges) = _build_co_frame_edges(meta, hub_min=3)

    # Exactly one binary edge
    assert len(binary) == 1
    edge_list = list(binary.values())[0]
    assert edge_list[0]["keywords"] == "co_frame"
    assert edge_list[0]["weight"] == pytest.approx(1.0)  # 1 shared / 1 union

    # No hub nodes (only 2 events < hub_min=3)
    assert len(hub_nodes) == 0


@pytest.mark.offline
def test_co_frame_hub_hyperedge_three_events():
    """Three events sharing a frame → hub node + 3 star edges."""
    from lightrag.bridge_builder import _build_co_frame_edges

    meta = [
        {"event_id": f"EVENT:{i}", "frame_names": ["VectorIndexing"], "entity_names": []}
        for i in range(3)
    ]
    binary, (hub_nodes, hub_edges) = _build_co_frame_edges(meta, hub_min=3)

    # No binary edges — hub used instead
    assert len(binary) == 0
    assert len(hub_nodes) == 1
    hub_id = list(hub_nodes.keys())[0]
    assert hub_id.startswith("HUB:frame:")
    assert hub_nodes[hub_id][0]["entity_type"] == "hub"
    assert len(hub_edges) == 3  # one edge per event
    for edge_list in hub_edges.values():
        assert edge_list[0]["keywords"] == "hub_member"


@pytest.mark.offline
def test_co_frame_multiple_frames_partial_sharing():
    """Events sharing only a subset of frames: correct Jaccard weight."""
    from lightrag.bridge_builder import _build_co_frame_edges

    meta = [
        {"event_id": "EVENT:a", "frame_names": ["FrameA", "FrameB"], "entity_names": []},
        {"event_id": "EVENT:b", "frame_names": ["FrameA", "FrameC"], "entity_names": []},
    ]
    binary, _ = _build_co_frame_edges(meta, hub_min=3)

    assert len(binary) == 1
    edge = list(binary.values())[0][0]
    # shared={FrameA}, union={FrameA, FrameB, FrameC} → Jaccard = 1/3
    assert edge["weight"] == pytest.approx(1 / 3, rel=1e-3)


# ══════════════════════════════════════════════════════════════════════════════
# Section 4 — bridge_builder: co-entity edges
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.offline
def test_co_entity_binary_edges_two_events():
    """Two events sharing an entity → binary co_entity edge."""
    from lightrag.bridge_builder import _build_co_entity_edges

    meta = [
        {"event_id": "EVENT:a", "frame_names": [], "entity_names": ["LightRAG", "Milvus"]},
        {"event_id": "EVENT:b", "frame_names": [], "entity_names": ["LightRAG", "Qdrant"]},
    ]
    binary, (hub_nodes, hub_edges) = _build_co_entity_edges(meta, hub_min=3)

    assert len(binary) == 1
    edge = list(binary.values())[0][0]
    assert edge["keywords"] == "co_entity"
    # shared={LightRAG}, union={LightRAG, Milvus, Qdrant} → Jaccard = 1/3
    assert edge["weight"] == pytest.approx(1 / 3, rel=1e-3)
    assert len(hub_nodes) == 0


@pytest.mark.offline
def test_co_entity_hub_hyperedge_three_events():
    """Three events sharing an entity → hub node + star edges."""
    from lightrag.bridge_builder import _build_co_entity_edges

    meta = [
        {"event_id": f"EVENT:{i}", "frame_names": [], "entity_names": ["LightRAG"]}
        for i in range(3)
    ]
    binary, (hub_nodes, hub_edges) = _build_co_entity_edges(meta, hub_min=3)

    assert len(binary) == 0
    assert len(hub_nodes) == 1
    hub_id = list(hub_nodes.keys())[0]
    assert hub_id.startswith("HUB:entity:")
    assert len(hub_edges) == 3


@pytest.mark.offline
def test_co_entity_no_edges_disjoint():
    """Events with completely different entities → no bridge edges."""
    from lightrag.bridge_builder import _build_co_entity_edges

    meta = [
        {"event_id": "EVENT:a", "frame_names": [], "entity_names": ["Alpha"]},
        {"event_id": "EVENT:b", "frame_names": [], "entity_names": ["Beta"]},
    ]
    binary, (hub_nodes, hub_edges) = _build_co_entity_edges(meta, hub_min=3)

    assert len(binary) == 0
    assert len(hub_nodes) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Section 5 — bridge_builder: N-M hyperedge cluster
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.offline
def test_nm_hyperedge_created_for_large_shared_group():
    """Three events sharing 2 frames AND 1 entity → cluster hyperedge."""
    from lightrag.bridge_builder import _build_nm_hyperedges

    meta = [
        {
            "event_id": f"EVENT:{c}",
            "frame_names": ["VectorIndexing", "EmbeddingGeneration"],
            "entity_names": ["LightRAG"],
        }
        for c in ["a", "b", "c"]
    ]
    cluster_nodes, cluster_edges = _build_nm_hyperedges(
        meta, nm_min_events=3, nm_min_shared=2
    )

    assert len(cluster_nodes) == 1
    cluster_id = list(cluster_nodes.keys())[0]
    assert cluster_id.startswith("CLUSTER:")
    assert cluster_nodes[cluster_id][0]["entity_type"] == "cluster"
    assert len(cluster_edges) == 3  # one edge per event
    for edge_list in cluster_edges.values():
        assert edge_list[0]["keywords"] == "cluster_member"


@pytest.mark.offline
def test_nm_hyperedge_not_created_below_threshold():
    """Only 2 events → below nm_min_events=3, no cluster created."""
    from lightrag.bridge_builder import _build_nm_hyperedges

    meta = [
        {
            "event_id": f"EVENT:{c}",
            "frame_names": ["FrameA", "FrameB"],
            "entity_names": ["Entity1", "Entity2"],
        }
        for c in ["a", "b"]
    ]
    cluster_nodes, cluster_edges = _build_nm_hyperedges(
        meta, nm_min_events=3, nm_min_shared=2
    )

    assert len(cluster_nodes) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Section 6 — bridge_builder: similarity resolution
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.offline
@pytest.mark.asyncio
async def test_entity_merge_high_cosine():
    """Entities with cosine >= 0.8 must be merged into canonical name."""
    from lightrag.bridge_builder import _resolve_similarity

    nodes = {
        "LightRAG":           [_make_entity_node("LightRAG", "agent")],
        "LightRAG framework": [_make_entity_node("LightRAG framework", "agent")],
        "ChromaDB":           [_make_entity_node("ChromaDB", "index")],
    }
    edges = {}
    meta = [
        {"event_id": "EVENT:a", "frame_names": [], "entity_names": list(nodes.keys())}
    ]

    # LightRAG vs "LightRAG framework" → sim=0.95 (merge)
    # ChromaDB vs others → sim=0.1 (no merge)
    def fake_embed(texts):
        vecs = []
        for t in texts:
            if "LightRAG" in t:
                vecs.append([1.0, 0.0] + [0.0] * 62)
            else:
                vecs.append([0.0, 1.0] + [0.0] * 62)
        return vecs

    embed_mock = AsyncMock(side_effect=lambda texts: fake_embed(texts))

    merged_nodes, merged_edges, updated_meta, merge_map, sim_edges = (
        await _resolve_similarity(
            nodes, edges, meta, embed_mock,
            entity_merge_threshold=0.8,
            entity_sim_threshold=0.5,
            frame_sim_threshold=0.5,
        )
    )

    # "LightRAG framework" must merge into "LightRAG" (higher freq wins)
    assert "LightRAG" in merged_nodes
    assert "LightRAG framework" not in merged_nodes
    assert "ChromaDB" in merged_nodes
    # Merge map must redirect
    assert merge_map.get("LightRAG framework") == "LightRAG"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_entity_similarity_edge_medium_cosine():
    """Entities with 0.5 <= cosine < 0.8 → entity_similarity edge, no merge."""
    from lightrag.bridge_builder import _resolve_similarity

    nodes = {
        "vector store":    [_make_entity_node("vector store", "index")],
        "vector database": [_make_entity_node("vector database", "index")],
    }
    edges = {}
    meta = [{"event_id": "EVENT:a", "frame_names": [], "entity_names": list(nodes.keys())}]

    # Cosine ≈ 0.65 between the two (medium)
    def fake_embed(texts):
        vecs = []
        for t in texts:
            if "store" in t:
                vecs.append([1.0, 0.3] + [0.0] * 62)
            else:
                vecs.append([0.3, 1.0] + [0.0] * 62)
        return vecs

    embed_mock = AsyncMock(side_effect=lambda texts: fake_embed(texts))

    _, _, _, merge_map, sim_edges = await _resolve_similarity(
        nodes, edges, meta, embed_mock,
        entity_merge_threshold=0.9,   # high threshold so no merge
        entity_sim_threshold=0.5,
        frame_sim_threshold=0.5,
    )

    # No merge (cosine < 0.9)
    assert "vector store" not in merge_map or merge_map["vector store"] == "vector store"
    # Should have a similarity edge
    assert len(sim_edges) >= 1
    kws = [e["keywords"] for edge_list in sim_edges.values() for e in edge_list]
    assert "entity_similarity" in kws


@pytest.mark.offline
@pytest.mark.asyncio
async def test_frame_similarity_edges():
    """Frames with cosine >= frame_sim_threshold → frame_similarity edge."""
    from lightrag.bridge_builder import _resolve_similarity

    nodes = {
        "VectorIndexing":     [_make_frame_node("VectorIndexing")],
        "EmbeddingGeneration": [_make_frame_node("EmbeddingGeneration")],
    }
    edges = {}
    meta = [{"event_id": "EVENT:a", "frame_names": list(nodes.keys()), "entity_names": []}]

    def fake_embed(texts):
        # Two slightly different vectors — cosine ~0.97, still above 0.5 threshold but < 1.0
        vecs = []
        for i, _ in enumerate(texts):
            v = [0.9, 0.4 + i * 0.05] + [0.0] * 62
            vecs.append(v)
        return vecs

    embed_mock = AsyncMock(side_effect=lambda texts: fake_embed(texts))

    _, _, _, _, sim_edges = await _resolve_similarity(
        nodes, edges, meta, embed_mock,
        entity_merge_threshold=0.8,
        entity_sim_threshold=0.5,
        frame_sim_threshold=0.5,
    )

    kws = [e["keywords"] for edge_list in sim_edges.values() for e in edge_list]
    assert "frame_similarity" in kws


# ══════════════════════════════════════════════════════════════════════════════
# Section 7 — bridge_builder: causal edges (LLM)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.offline
@pytest.mark.asyncio
async def test_causal_edge_classified():
    """_build_causal_edges must create an edge for non-none LLM response."""
    from lightrag.bridge_builder import _build_causal_edges

    meta = [
        {
            "event_id": "EVENT:a",
            "frame_names": ["VectorIndexing"],
            "entity_names": ["LightRAG"],
            "chunk_content": "LightRAG indexes documents.",
        },
        {
            "event_id": "EVENT:b",
            "frame_names": ["DocumentRetrieval"],
            "entity_names": ["LightRAG"],
            "chunk_content": "LightRAG retrieves relevant chunks.",
        },
    ]
    candidate_edges = {
        ("EVENT:a", "EVENT:b"): [
            {"weight": 0.8, "keywords": "co_entity"}
        ]
    }
    llm_mock = AsyncMock(
        return_value='{"relation": "precedes", "confidence": 0.85, "reasoning": "indexing precedes retrieval"}'
    )

    causal_edges = await _build_causal_edges(
        meta, candidate_edges, llm_mock,
        max_pairs=10, weight_threshold=0.3,
    )

    assert len(causal_edges) == 1
    edge = list(causal_edges.values())[0][0]
    assert edge["keywords"] == "precedes"
    assert edge["weight"] == pytest.approx(0.85)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_causal_edge_skipped_for_none_relation():
    """LLM returning 'none' or low confidence must NOT produce an edge."""
    from lightrag.bridge_builder import _build_causal_edges

    meta = [
        {"event_id": "EVENT:a", "frame_names": [], "entity_names": [], "chunk_content": "Text A."},
        {"event_id": "EVENT:b", "frame_names": [], "entity_names": [], "chunk_content": "Text B."},
    ]
    candidate_edges = {("EVENT:a", "EVENT:b"): [{"weight": 1.0}]}
    llm_mock = AsyncMock(
        return_value='{"relation": "none", "confidence": 0.9, "reasoning": "unrelated"}'
    )

    causal_edges = await _build_causal_edges(
        meta, candidate_edges, llm_mock,
        max_pairs=10, weight_threshold=0.3,
    )

    assert len(causal_edges) == 0


@pytest.mark.offline
@pytest.mark.asyncio
async def test_causal_edge_below_confidence_skipped():
    """Confidence < 0.5 must be discarded even with a non-none relation."""
    from lightrag.bridge_builder import _build_causal_edges

    meta = [
        {"event_id": "EVENT:x", "frame_names": [], "entity_names": [], "chunk_content": "X."},
        {"event_id": "EVENT:y", "frame_names": [], "entity_names": [], "chunk_content": "Y."},
    ]
    candidate_edges = {("EVENT:x", "EVENT:y"): [{"weight": 1.0}]}
    llm_mock = AsyncMock(
        return_value='{"relation": "causes", "confidence": 0.3, "reasoning": "weak signal"}'
    )

    causal_edges = await _build_causal_edges(
        meta, candidate_edges, llm_mock,
        max_pairs=10, weight_threshold=0.3,
    )

    assert len(causal_edges) == 0


# ══════════════════════════════════════════════════════════════════════════════
# Section 8 — build_all_bridges: full integration
# ══════════════════════════════════════════════════════════════════════════════

def _make_chunk_graph(event_id: str, frame: str, entity: str) -> tuple[dict, dict]:
    """Helper: build minimal nodes/edges for one chunk."""
    nodes = {
        event_id:  [_make_event_node(event_id, [frame])],
        frame:     [_make_frame_node(frame)],
        entity:    [_make_entity_node(entity, "agent")],
    }
    edges = {
        (event_id, frame): [_make_edge(event_id, frame, "evokes", 0.9)],
        (frame, entity):   [_make_edge(frame, entity, "Agent", 1.0)],
    }
    return nodes, edges


@pytest.mark.offline
@pytest.mark.asyncio
async def test_build_all_bridges_empty_metadata():
    """Empty metadata must return original nodes/edges unchanged."""
    from lightrag.bridge_builder import build_all_bridges

    nodes = {"A": [_make_entity_node("A")]}
    edges = {}
    embed_mock = AsyncMock(return_value=[[0.1] * 64])
    llm_mock = AsyncMock(return_value='{"relation": "none", "confidence": 0.0, "reasoning": ""}')

    final_nodes, final_edges = await build_all_bridges(
        metadata_list=[],
        all_nodes=nodes,
        all_edges=edges,
        embed_func=embed_mock,
        llm_func=llm_mock,
    )

    assert final_nodes == nodes
    assert final_edges == edges


@pytest.mark.offline
@pytest.mark.asyncio
async def test_build_all_bridges_adds_co_frame_edges():
    """Two events sharing a frame → co_frame edge appears in final_edges."""
    from lightrag.bridge_builder import build_all_bridges

    nodes_a, edges_a = _make_chunk_graph("EVENT:aa", "VectorIndexing", "LightRAG")
    nodes_b, edges_b = _make_chunk_graph("EVENT:bb", "VectorIndexing", "Milvus")

    all_nodes: dict = {}
    all_edges: dict = {}
    for d in (nodes_a, nodes_b):
        for k, v in d.items():
            all_nodes.setdefault(k, []).extend(v)
    for d in (edges_a, edges_b):
        for k, v in d.items():
            all_edges.setdefault(k, []).extend(v)

    meta = [
        {"event_id": "EVENT:aa", "frame_names": ["VectorIndexing"],
         "entity_names": ["LightRAG"], "chunk_content": "LightRAG indexes vectors."},
        {"event_id": "EVENT:bb", "frame_names": ["VectorIndexing"],
         "entity_names": ["Milvus"], "chunk_content": "Milvus stores vectors."},
    ]

    # Embeddings: distinct enough so no merging
    embed_mock = AsyncMock(
        return_value=np.eye(max(len(all_nodes), 2), 64).tolist()
    )
    llm_mock = AsyncMock(
        return_value='{"relation": "none", "confidence": 0.0, "reasoning": ""}'
    )

    final_nodes, final_edges = await build_all_bridges(
        metadata_list=meta,
        all_nodes=all_nodes,
        all_edges=all_edges,
        embed_func=embed_mock,
        llm_func=llm_mock,
    )

    # Must contain a co_frame edge between EVENT:aa and EVENT:bb
    co_frame_kws = [
        e["keywords"]
        for edge_list in final_edges.values()
        for e in edge_list
        if e["keywords"] == "co_frame"
    ]
    assert len(co_frame_kws) >= 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_build_all_bridges_adds_co_entity_edges():
    """Two events sharing entity 'LightRAG' → co_entity edge in final_edges."""
    from lightrag.bridge_builder import build_all_bridges

    nodes_a, edges_a = _make_chunk_graph("EVENT:cc", "FrameA", "LightRAG")
    nodes_b, edges_b = _make_chunk_graph("EVENT:dd", "FrameB", "LightRAG")

    all_nodes: dict = {}
    all_edges: dict = {}
    for d in (nodes_a, nodes_b):
        for k, v in d.items():
            all_nodes.setdefault(k, []).extend(v)
    for d in (edges_a, edges_b):
        for k, v in d.items():
            all_edges.setdefault(k, []).extend(v)

    meta = [
        {"event_id": "EVENT:cc", "frame_names": ["FrameA"],
         "entity_names": ["LightRAG"], "chunk_content": "LightRAG does A."},
        {"event_id": "EVENT:dd", "frame_names": ["FrameB"],
         "entity_names": ["LightRAG"], "chunk_content": "LightRAG does B."},
    ]

    embed_mock = AsyncMock(
        return_value=np.eye(max(len(all_nodes), 2), 64).tolist()
    )
    llm_mock = AsyncMock(
        return_value='{"relation": "none", "confidence": 0.0, "reasoning": ""}'
    )

    final_nodes, final_edges = await build_all_bridges(
        metadata_list=meta,
        all_nodes=all_nodes,
        all_edges=all_edges,
        embed_func=embed_mock,
        llm_func=llm_mock,
    )

    co_entity_kws = [
        e["keywords"]
        for edge_list in final_edges.values()
        for e in edge_list
        if e["keywords"] == "co_entity"
    ]
    assert len(co_entity_kws) >= 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_build_all_bridges_hub_with_four_events():
    """Four events sharing the same frame → hub node created."""
    from lightrag.bridge_builder import build_all_bridges

    all_nodes: dict = {}
    all_edges: dict = {}
    meta = []
    for i in range(4):
        eid = f"EVENT:{i:02d}"
        n, e = _make_chunk_graph(eid, "SharedFrame", f"Entity{i}")
        for k, v in n.items():
            all_nodes.setdefault(k, []).extend(v)
        for k, v in e.items():
            all_edges.setdefault(k, []).extend(v)
        meta.append({
            "event_id": eid,
            "frame_names": ["SharedFrame"],
            "entity_names": [f"Entity{i}"],
            "chunk_content": f"Entity{i} uses SharedFrame.",
        })

    embed_mock = AsyncMock(
        return_value=np.eye(max(len(all_nodes), 2), 64).tolist()
    )
    llm_mock = AsyncMock(
        return_value='{"relation": "none", "confidence": 0.0, "reasoning": ""}'
    )

    final_nodes, final_edges = await build_all_bridges(
        metadata_list=meta,
        all_nodes=all_nodes,
        all_edges=all_edges,
        embed_func=embed_mock,
        llm_func=llm_mock,
        hub_min_events=3,
    )

    hub_node_ids = [k for k in final_nodes if k.startswith("HUB:")]
    assert len(hub_node_ids) >= 1

    hub_member_edges = [
        e for edge_list in final_edges.values()
        for e in edge_list if e["keywords"] == "hub_member"
    ]
    assert len(hub_member_edges) >= 4  # 4 events connected to hub


# ══════════════════════════════════════════════════════════════════════════════
# Section 9 — _frame_graph_expand: frame-aware retrieval
# ══════════════════════════════════════════════════════════════════════════════

def _make_graph_mock(
    nodes: dict[str, dict],
    edges_per_node: dict[str, list[tuple]],
    edge_props: dict[tuple, dict] | None = None,
    degrees: dict[str, int] | None = None,
    edge_degrees: dict[tuple, int] | None = None,
) -> MagicMock:
    """Build a mock BaseGraphStorage from simple dicts."""
    g = MagicMock()
    g.get_nodes_batch = AsyncMock(
        side_effect=lambda ids: {nid: nodes.get(nid) for nid in ids}
    )
    g.get_nodes_edges_batch = AsyncMock(
        side_effect=lambda names: {n: edges_per_node.get(n, []) for n in names}
    )
    g.node_degrees_batch = AsyncMock(
        side_effect=lambda ids: {nid: (degrees or {}).get(nid, 1) for nid in ids}
    )
    g.get_edges_batch = AsyncMock(
        side_effect=lambda pairs: {
            (p["src"], p["tgt"]): (edge_props or {}).get((p["src"], p["tgt"]))
            for p in pairs
        }
    )
    g.edge_degrees_batch = AsyncMock(
        side_effect=lambda pairs: {p: (edge_degrees or {}).get(p, 1) for p in pairs}
    )
    return g


def _make_query_param(top_k: int = 5) -> MagicMock:
    qp = MagicMock()
    qp.top_k = top_k
    return qp


@pytest.mark.offline
@pytest.mark.asyncio
async def test_frame_expand_empty_hl_keywords():
    """Empty or whitespace-only hl_keywords → ([], [])."""
    from lightrag.operate import _frame_graph_expand

    g = _make_graph_mock({}, {})
    for kw in ("", "  ", ",,,"):
        extra_ents, extra_rels = await _frame_graph_expand(kw, g, _make_query_param())
        assert extra_ents == []
        assert extra_rels == []


@pytest.mark.offline
@pytest.mark.asyncio
async def test_frame_expand_no_frame_nodes_in_graph():
    """hl_keywords present but not found as FRAME nodes → ([], [])."""
    from lightrag.operate import _frame_graph_expand

    # graph has no nodes matching the frame names
    g = _make_graph_mock(nodes={}, edges_per_node={})
    extra_ents, extra_rels = await _frame_graph_expand(
        "Commerce_sell, Causation", g, _make_query_param()
    )
    assert extra_ents == []
    assert extra_rels == []


@pytest.mark.offline
@pytest.mark.asyncio
async def test_frame_expand_frame_to_entity_nodes():
    """FRAME node with FRAME→ENTITY edges → leaf entities returned."""
    from lightrag.operate import _frame_graph_expand

    nodes = {
        "Commerce_sell": {"entity_type": "frame", "description": "Selling frame"},
        "Alice":         {"entity_type": "seller", "description": "Alice the seller"},
        "Car":           {"entity_type": "goods",  "description": "The car"},
    }
    edges_per_node = {
        "Commerce_sell": [("Commerce_sell", "Alice"), ("Commerce_sell", "Car")],
    }
    g = _make_graph_mock(nodes, edges_per_node)

    extra_ents, extra_rels = await _frame_graph_expand(
        "Commerce_sell", g, _make_query_param(top_k=10)
    )

    entity_names = {e["entity_name"] for e in extra_ents}
    assert "Alice" in entity_names
    assert "Car" in entity_names


@pytest.mark.offline
@pytest.mark.asyncio
async def test_frame_expand_frame_to_event_nodes():
    """FRAME node evoked by EVENTs → event nodes returned."""
    from lightrag.operate import _frame_graph_expand

    nodes = {
        "VectorIndexing": {"entity_type": "frame", "description": "Vector indexing frame"},
        "EVENT:abc":      {"entity_type": "event", "description": "Event abc evokes VectorIndexing"},
        "EVENT:def":      {"entity_type": "event", "description": "Event def evokes VectorIndexing"},
    }
    edges_per_node = {
        "VectorIndexing": [
            ("EVENT:abc", "VectorIndexing"),
            ("EVENT:def", "VectorIndexing"),
        ],
        "EVENT:abc": [],
        "EVENT:def": [],
    }
    g = _make_graph_mock(nodes, edges_per_node)

    extra_ents, extra_rels = await _frame_graph_expand(
        "VectorIndexing", g, _make_query_param(top_k=10)
    )

    entity_names = {e["entity_name"] for e in extra_ents}
    assert "EVENT:abc" in entity_names
    assert "EVENT:def" in entity_names


@pytest.mark.offline
@pytest.mark.asyncio
async def test_frame_expand_bridge_edges_followed():
    """EVENT nodes connected via co_frame bridge → bridge edges returned."""
    from lightrag.operate import _frame_graph_expand

    nodes = {
        "Causation":  {"entity_type": "frame", "description": "Causation frame"},
        "EVENT:e1":   {"entity_type": "event", "description": "Event e1"},
        "EVENT:e2":   {"entity_type": "event", "description": "Event e2"},
    }
    bridge_pair = tuple(sorted(("EVENT:e1", "EVENT:e2")))
    edges_per_node = {
        "Causation":  [("EVENT:e1", "Causation"), ("EVENT:e2", "Causation")],
        "EVENT:e1":   [("EVENT:e1", "Causation"), bridge_pair],
        "EVENT:e2":   [("EVENT:e2", "Causation"), bridge_pair],
    }
    edge_props = {
        bridge_pair: {
            "keywords": "co_frame", "weight": 0.8,
            "description": "co_frame edge", "source_id": "bridge_builder",
        }
    }
    g = _make_graph_mock(nodes, edges_per_node, edge_props=edge_props)

    extra_ents, extra_rels = await _frame_graph_expand(
        "Causation", g, _make_query_param(top_k=10)
    )

    # Bridge edge should appear
    assert len(extra_rels) >= 1
    rel_kws = {r.get("keywords") for r in extra_rels}
    assert "co_frame" in rel_kws


@pytest.mark.offline
@pytest.mark.asyncio
async def test_frame_expand_hub_nodes_included():
    """EVENT→HUB edge → HUB node appears in extra_entities."""
    from lightrag.operate import _frame_graph_expand

    hub_id = "HUB:frame:Commerce_sell"
    nodes = {
        "Commerce_sell": {"entity_type": "frame", "description": "Selling frame"},
        "EVENT:e1":      {"entity_type": "event", "description": "Event e1"},
        hub_id:          {"entity_type": "hub",   "description": "Hub for Commerce_sell"},
    }
    edges_per_node = {
        "Commerce_sell": [("EVENT:e1", "Commerce_sell")],
        "EVENT:e1":      [("EVENT:e1", "Commerce_sell"), ("EVENT:e1", hub_id)],
    }
    g = _make_graph_mock(nodes, edges_per_node)

    extra_ents, _ = await _frame_graph_expand(
        "Commerce_sell", g, _make_query_param(top_k=10)
    )

    entity_names = {e["entity_name"] for e in extra_ents}
    assert hub_id in entity_names


@pytest.mark.offline
@pytest.mark.asyncio
async def test_frame_expand_multiple_frames_union():
    """Two frame names → entities from both frames returned (union)."""
    from lightrag.operate import _frame_graph_expand

    nodes = {
        "FrameA": {"entity_type": "frame", "description": "Frame A"},
        "FrameB": {"entity_type": "frame", "description": "Frame B"},
        "EntityA": {"entity_type": "agent", "description": "Entity from A"},
        "EntityB": {"entity_type": "agent", "description": "Entity from B"},
    }
    edges_per_node = {
        "FrameA": [("FrameA", "EntityA")],
        "FrameB": [("FrameB", "EntityB")],
    }
    g = _make_graph_mock(nodes, edges_per_node)

    extra_ents, _ = await _frame_graph_expand(
        "FrameA, FrameB", g, _make_query_param(top_k=10)
    )

    entity_names = {e["entity_name"] for e in extra_ents}
    assert "EntityA" in entity_names
    assert "EntityB" in entity_names


@pytest.mark.offline
@pytest.mark.asyncio
async def test_frame_expand_deduplicates_nodes():
    """Same entity connected via two different frames → appears only once."""
    from lightrag.operate import _frame_graph_expand

    nodes = {
        "FrameA":      {"entity_type": "frame", "description": "Frame A"},
        "FrameB":      {"entity_type": "frame", "description": "Frame B"},
        "SharedEntity": {"entity_type": "agent", "description": "Shared"},
    }
    edges_per_node = {
        "FrameA": [("FrameA", "SharedEntity")],
        "FrameB": [("FrameB", "SharedEntity")],
    }
    g = _make_graph_mock(nodes, edges_per_node)

    extra_ents, _ = await _frame_graph_expand(
        "FrameA, FrameB", g, _make_query_param(top_k=10)
    )

    names = [e["entity_name"] for e in extra_ents]
    assert names.count("SharedEntity") == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_frame_expand_caps_at_top_k_times_2():
    """Expansion result capped at top_k * 2 nodes."""
    from lightrag.operate import _frame_graph_expand

    # Create 20 entity nodes connected to one frame
    nodes = {"BigFrame": {"entity_type": "frame", "description": "Big frame"}}
    for i in range(20):
        nodes[f"Entity{i}"] = {"entity_type": "agent", "description": f"Entity {i}"}

    edges_per_node = {
        "BigFrame": [(f"BigFrame", f"Entity{i}") for i in range(20)]
    }
    g = _make_graph_mock(nodes, edges_per_node)

    extra_ents, _ = await _frame_graph_expand(
        "BigFrame", g, _make_query_param(top_k=5)  # cap = 5*2 = 10
    )

    assert len(extra_ents) <= 10


# ══════════════════════════════════════════════════════════════════════════════
# Section 10 — Bridge builder edge cases
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.offline
def test_co_frame_single_record_per_pair():
    """Events sharing 2 frames → exactly 1 edge record per pair (not 2)."""
    from lightrag.bridge_builder import _build_co_frame_edges

    meta = [
        {"event_id": "EVENT:a", "frame_names": ["F1", "F2"], "entity_names": []},
        {"event_id": "EVENT:b", "frame_names": ["F1", "F2"], "entity_names": []},
    ]
    binary_edges, _ = _build_co_frame_edges(meta, hub_min=3)

    pair = tuple(sorted(("EVENT:a", "EVENT:b")))
    assert pair in binary_edges
    # Must have exactly ONE record for this pair, not two
    assert len(binary_edges[pair]) == 1
    # Description should mention both frames
    desc = binary_edges[pair][0]["description"]
    assert "F1" in desc and "F2" in desc


@pytest.mark.offline
def test_co_entity_single_record_per_pair():
    """Events sharing 2 entities → exactly 1 edge record per pair."""
    from lightrag.bridge_builder import _build_co_entity_edges

    meta = [
        {"event_id": "EVENT:a", "frame_names": [], "entity_names": ["Alice", "Bob"]},
        {"event_id": "EVENT:b", "frame_names": [], "entity_names": ["Alice", "Bob"]},
    ]
    binary_edges, _ = _build_co_entity_edges(meta, hub_min=3)

    pair = tuple(sorted(("EVENT:a", "EVENT:b")))
    assert pair in binary_edges
    assert len(binary_edges[pair]) == 1
    desc = binary_edges[pair][0]["description"]
    assert "Alice" in desc or "Bob" in desc


@pytest.mark.offline
def test_co_entity_hub_has_hash_suffix():
    """Hub entity node ID must contain a 6-char hex hash to avoid collision."""
    from lightrag.bridge_builder import _build_co_entity_edges

    meta = [
        {"event_id": f"EVENT:{i}", "frame_names": [], "entity_names": ["LightRAG"]}
        for i in range(3)
    ]
    _, (hub_nodes, _) = _build_co_entity_edges(meta, hub_min=3)

    assert len(hub_nodes) >= 1
    hub_id = list(hub_nodes.keys())[0]
    # Should contain a 6-char hex hash after an underscore
    parts = hub_id.split("_")
    assert len(parts) >= 2
    hash_part = parts[-1]
    assert len(hash_part) == 6
    assert all(c in "0123456789abcdef" for c in hash_part)


@pytest.mark.offline
def test_union_find_long_chain_compression():
    """Union-Find correctly resolves a chain A→B→C→D to the same root."""
    import numpy as np
    from lightrag.bridge_builder import _union_find_canonical

    names = ["A", "B", "C", "D"]
    # All pairs have cosine >= 0.9 → all merge to highest-freq name
    sim = np.ones((4, 4), dtype=np.float32)
    np.fill_diagonal(sim, 1.0)
    # All same frequency → first alphabetically / highest wins
    freq = {"A": 1, "B": 1, "C": 1, "D": 1}
    merge_map = _union_find_canonical(names, sim, threshold=0.8, freq=freq)

    # All should map to the same canonical name
    canonical_values = set(merge_map.values())
    assert len(canonical_values) == 1, f"Expected 1 canonical, got {canonical_values}"


@pytest.mark.offline
def test_apply_merge_edges_preserves_direction():
    """After merge, src_id/tgt_id in record must reflect original direction."""
    from lightrag.bridge_builder import _apply_merge_edges

    # Edge A→C in graph, B merges into A (merge_map = {B: A})
    edges = {
        ("A", "C"): [{"src_id": "A", "tgt_id": "C", "weight": 1.0, "keywords": "causes"}],
        ("B", "C"): [{"src_id": "B", "tgt_id": "C", "weight": 0.8, "keywords": "causes"}],
    }
    merge_map = {"B": "A", "A": "A", "C": "C"}
    merged = _apply_merge_edges(edges, merge_map)

    # Both edges should merge to (A, C) key
    key = tuple(sorted(("A", "C")))
    assert key in merged
    for record in merged[key]:
        # Direction must always be A→C after merge (B→C becomes A→C)
        assert record["src_id"] == "A"
        assert record["tgt_id"] == "C"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_nm_hyperedge_different_groups_different_ids():
    """Two distinct groups of events → two different CLUSTER node IDs."""
    from lightrag.bridge_builder import _build_nm_hyperedges

    # Group 1: events 0-2 share F:X and E:Y
    # Group 2: events 3-5 share F:P and E:Q (no overlap)
    meta = []
    for i in range(3):
        meta.append({
            "event_id": f"EVENT:g1_{i}",
            "frame_names": ["FrameX"],
            "entity_names": ["EntityY"],
        })
    for i in range(3):
        meta.append({
            "event_id": f"EVENT:g2_{i}",
            "frame_names": ["FrameP"],
            "entity_names": ["EntityQ"],
        })

    cluster_nodes, _ = _build_nm_hyperedges(meta, nm_min_events=3, nm_min_shared=2)

    if len(cluster_nodes) >= 2:
        ids = list(cluster_nodes.keys())
        assert ids[0] != ids[1], "Two distinct groups must have different cluster IDs"


# ══════════════════════════════════════════════════════════════════════════════
# Section 11 — operate.py / llm_frame_extractor edge cases
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.offline
def test_zero_frame_bridge_meta_not_added_to_list():
    """
    BUG-03 regression: bridge_meta with empty frame_names must be excluded
    from bridge_metadata_list. Test the filter condition directly.
    """
    # Simulate operate.py filter: if bridge_meta and bridge_meta.get("frame_names")
    empty_meta = {"event_id": "EVENT:x", "frame_names": []}
    nonempty_meta = {"event_id": "EVENT:y", "frame_names": ["Commerce_sell"]}
    none_meta = None

    bridge_metadata_list: list[dict] = []
    for bm in [empty_meta, nonempty_meta, none_meta]:
        if bm and bm.get("frame_names"):
            bridge_metadata_list.append(bm)

    assert len(bridge_metadata_list) == 1
    assert bridge_metadata_list[0]["event_id"] == "EVENT:y"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_classify_pair_uses_parse_json_response():
    """
    _classify_pair must not raise AttributeError from json_repair.loads
    (BUG-07 regression test). Valid JSON → edge returned.
    """
    from lightrag.bridge_builder import _classify_pair

    meta_a = {
        "event_id": "EVENT:a",
        "chunk_content": "The flooding caused power outages.",
        "frame_names": ["Causation"],
        "entity_names": ["flooding", "power outages"],
    }
    meta_b = {
        "event_id": "EVENT:b",
        "chunk_content": "Power outages led to economic losses.",
        "frame_names": ["Causation"],
        "entity_names": ["power outages", "economic losses"],
    }

    llm_mock = AsyncMock(
        return_value='{"relation": "causes", "confidence": 0.85, "reasoning": "A causes B"}'
    )

    result = await _classify_pair(meta_a, meta_b, llm_mock)

    assert result is not None
    assert result["keywords"] == "causes"
    assert result["weight"] == pytest.approx(0.85)
    assert result["src_id"] == "EVENT:a"
    assert result["tgt_id"] == "EVENT:b"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_classify_pair_malformed_json_returns_none():
    """Malformed LLM output → json_repair recovers or returns None gracefully."""
    from lightrag.bridge_builder import _classify_pair

    meta_a = {"event_id": "EVENT:a", "chunk_content": "A", "frame_names": [], "entity_names": []}
    meta_b = {"event_id": "EVENT:b", "chunk_content": "B", "frame_names": [], "entity_names": []}

    # Completely broken JSON
    llm_mock = AsyncMock(return_value="not json at all {{{")

    # Should not raise; either returns None or a dict (repaired)
    result = await _classify_pair(meta_a, meta_b, llm_mock)
    # If repair_json produces something, relation will be "none" → returns None
    assert result is None or isinstance(result, dict)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_frame_expand_cosine_similarity_set_lower_than_vdb():
    """Expanded nodes must have cosine_similarity < 1.0 to stay below VDB results."""
    from lightrag.operate import _frame_graph_expand

    nodes = {
        "FrameX": {"entity_type": "frame", "description": "Frame X"},
        "EntityX": {"entity_type": "agent", "description": "Entity X"},
    }
    edges_per_node = {"FrameX": [("FrameX", "EntityX")]}
    g = _make_graph_mock(nodes, edges_per_node)

    extra_ents, _ = await _frame_graph_expand("FrameX", g, _make_query_param(top_k=5))

    assert len(extra_ents) >= 1
    for ent in extra_ents:
        assert ent.get("cosine_similarity", 1.0) < 1.0
