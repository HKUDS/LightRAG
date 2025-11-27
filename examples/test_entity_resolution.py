"""
Quick test for Entity Resolution feature.

Tests that:
1. "FDA" and "US Food and Drug Administration" resolve to the same entity
2. "Dupixant" (typo) matches "Dupixent" via fuzzy matching
"""

import asyncio
import os
import shutil

from lightrag import LightRAG
from lightrag.entity_resolution import EntityResolutionConfig
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import logger
import logging

WORKING_DIR = "./test_entity_resolution"

# Test document with entities that should be deduplicated
TEST_DOC = """
The FDA approved Dupixent for treating eczema in 2017.
The US Food and Drug Administration later expanded the drug's indications.
Dupixant (sometimes misspelled) has shown good results in clinical trials.
The FDA continues to monitor the safety of Dupixent.
"""


async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable")
        return

    # Clean up previous test
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR)

    # Set up logging to see resolution messages
    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)

    print("\n" + "=" * 60)
    print("Entity Resolution Test")
    print("=" * 60)

    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
        entity_resolution_config=EntityResolutionConfig(
            enabled=True,
            fuzzy_threshold=0.85,
            vector_threshold=0.5,
            max_candidates=3,
        ),
    )

    await rag.initialize_storages()

    print("\nInserting test document...")
    print(f"Document: {TEST_DOC.strip()}")
    print("\n" + "-" * 60)

    await rag.ainsert(TEST_DOC)

    print("\n" + "-" * 60)
    print("Checking extracted entities...")

    # Read the graph to see what entities were created
    graph_file = os.path.join(WORKING_DIR, "graph_chunk_entity_relation.graphml")
    if os.path.exists(graph_file):
        import networkx as nx

        G = nx.read_graphml(graph_file)
        print(f"\nEntities in graph ({len(G.nodes())} total):")
        for node in sorted(G.nodes()):
            print(f"  - {node}")

        print(f"\nRelationships: {len(G.edges())}")
    else:
        print("Graph file not found")

    await rag.finalize_storages()

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
