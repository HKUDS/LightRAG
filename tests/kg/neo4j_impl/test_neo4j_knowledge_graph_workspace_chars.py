#!/usr/bin/env python
"""
get_knowledge_graph must return the whole subgraph for any workspace name.

APOC ``labelFilter`` parses its value with its own syntax (``|`` and ``,``
separate labels, ``+`` ``-`` ``>`` ``/`` are prefixes), so passing the raw
workspace there made a workspace such as ``tenant:1`` filter out its own
nodes: the graph view showed only the start node. These tests run the real
APOC path against a live Neo4j and fail if it falls back to the plain Cypher
search, which never had the bug.
"""

import os

import numpy as np
import pytest

from lightrag.kg.shared_storage import initialize_share_data


async def mock_embedding_func(texts):
    return np.random.rand(len(texts), 10)


async def _make_storage(workspace):
    from lightrag.kg.neo4j_impl import Neo4JStorage

    storage = Neo4JStorage(
        namespace="test_kg_workspace_chars",
        workspace=workspace,
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.5},
            "working_dir": os.environ.get("WORKING_DIR", "./rag_storage"),
        },
        embedding_func=mock_embedding_func,
    )
    await storage.initialize()
    await storage.drop()
    return storage


async def _has_apoc(storage) -> bool:
    try:
        async with storage._driver.session(database=storage._DATABASE) as session:
            result = await session.run("RETURN apoc.version() AS v")
            await result.consume()
        return True
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.requires_db
@pytest.mark.parametrize(
    "workspace",
    ["tenant_1", "tenant:1", "acme,eu", "a|b", "+team", ">x"],
)
async def test_get_knowledge_graph_returns_full_subgraph(workspace):
    if not os.getenv("NEO4J_URI"):
        pytest.skip("Neo4j not configured (NEO4J_URI not set)")

    initialize_share_data()
    storage = await _make_storage(workspace)
    try:
        if not await _has_apoc(storage):
            pytest.skip("APOC is not installed on this Neo4j instance")

        async def _no_fallback(*args, **kwargs):
            raise AssertionError("APOC path failed and fell back to plain Cypher")

        storage._robust_fallback = _no_fallback

        for entity in ("A", "B", "C"):
            await storage.upsert_node(
                entity, {"entity_id": entity, "entity_type": "concept"}
            )
        await storage.upsert_edge("A", "B", {"weight": 1.0})
        await storage.upsert_edge("B", "C", {"weight": 1.0})

        kg = await storage.get_knowledge_graph("A", max_depth=3)

        assert sorted(n.labels[0] for n in kg.nodes) == ["A", "B", "C"]
        assert len(kg.edges) == 2
        assert kg.is_truncated is False
    finally:
        await storage.drop()
        await storage.finalize()
