#!/usr/bin/env python
"""
Example of using LightRAG with FalkorDB - Updated Version
=========================================================
Fixed imports and modern LightRAG syntax.

Prerequisites:
1. FalkorDB running: docker run -p 6379:6379 falkordb/falkordb:latest
2. OpenAI API key in .env file
3. Required packages: pip install lightrag falkordb openai python-dotenv
"""

import asyncio
import os
from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

# Load environment variables
load_dotenv()


async def main():
    """Example usage of LightRAG with FalkorDB"""

    # Set up environment for FalkorDB
    os.environ.setdefault("FALKORDB_HOST", "localhost")
    os.environ.setdefault("FALKORDB_PORT", "6379")
    os.environ.setdefault("FALKORDB_GRAPH_NAME", "lightrag_example")
    os.environ.setdefault("FALKORDB_WORKSPACE", "example_workspace")

    # Initialize LightRAG with FalkorDB
    rag = LightRAG(
        working_dir="./falkordb_example",
        llm_model_func=gpt_4o_mini_complete,  # Updated function name
        embedding_func=openai_embed,  # Updated function name
        graph_storage="FalkorDBStorage",  # Specify FalkorDB backend
    )

    # Initialize storage connections
    await rag.initialize_storages()
    await initialize_pipeline_status()

    # Example text to process
    sample_text = """
    FalkorDB is a high-performance graph database built on Redis.
    It supports OpenCypher queries and provides excellent performance for graph operations.
    LightRAG can now use FalkorDB as its graph storage backend, enabling scalable
    knowledge graph operations with Redis-based persistence. This integration
    allows developers to leverage both the speed of Redis and the power of
    graph databases for advanced AI applications.
    """

    print("Inserting text into LightRAG with FalkorDB backend...")
    await rag.ainsert(sample_text)

    # Check what was created
    storage = rag.chunk_entity_relation_graph
    nodes = await storage.get_all_nodes()
    edges = await storage.get_all_edges()
    print(f"Knowledge graph created: {len(nodes)} nodes, {len(edges)} edges")

    print("\nQuerying the knowledge graph...")

    # Test different query modes
    questions = [
        "What is FalkorDB and how does it relate to LightRAG?",
        "What are the benefits of using Redis with graph databases?",
        "How does FalkorDB support OpenCypher queries?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n--- Question {i} ---")
        print(f"Q: {question}")

        try:
            response = await rag.aquery(
                question, param=QueryParam(mode="hybrid", top_k=3)
            )
            print(f"A: {response}")
        except Exception as e:
            print(f"Error querying: {e}")

    # Show some graph statistics
    print("\n--- Graph Statistics ---")
    try:
        all_labels = await storage.get_all_labels()
        print(f"Unique entities: {len(all_labels)}")

        if nodes:
            print("Sample entities:")
            for i, node in enumerate(nodes[:3]):
                entity_id = node.get("entity_id", "Unknown")
                entity_type = node.get("entity_type", "Unknown")
                print(f"  {i+1}. {entity_id} ({entity_type})")

        if edges:
            print("Sample relationships:")
            for i, edge in enumerate(edges[:2]):
                source = edge.get("source", "Unknown")
                target = edge.get("target", "Unknown")
                print(f"  {i+1}. {source} → {target}")

    except Exception as e:
        print(f"Error getting statistics: {e}")


if __name__ == "__main__":
    print("LightRAG with FalkorDB Example")
    print("==============================")
    print("Note: This requires FalkorDB running on localhost:6379")
    print(
        "You can start FalkorDB with: docker run -p 6379:6379 falkordb/falkordb:latest"
    )
    print()

    # Check OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set your OpenAI API key in .env file!")
        print("   Create a .env file with: OPENAI_API_KEY=your-actual-api-key")
        exit(1)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Example interrupted. Goodbye!")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print("🔧 Make sure FalkorDB is running and your .env file is configured")
