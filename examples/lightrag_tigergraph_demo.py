from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger
import os
import asyncio
import json
from pathlib import Path

WORKING_DIR = "./tigergraph_test_dir"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

setup_logger("lightrag", level="INFO")


def load_json_texts(json_path: str | Path) -> list[str]:
    """
    Load texts from a plain JSON file.

    Expects JSON array format: [{"text": "..."}, {"text": "..."}]

    Args:
        json_path: Path to JSON file

    Returns:
        List of text strings extracted from "text" field
    """
    json_path = Path(json_path)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")

    texts = []
    for item in data:
        if isinstance(item, dict) and "text" in item:
            texts.append(item["text"])
        else:
            raise ValueError(
                f"Expected object with 'text' field, got {type(item).__name__}"
            )

    return texts


async def initialize_rag():
    """Initialize LightRAG with TigerGraph implementation."""
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,  # Use gpt_4o_mini_complete LLM model
        embedding_func=openai_embed,  # Use OpenAI embedding function
        graph_storage="TigerGraphStorage",
    )

    # Initialize database connections
    await rag.initialize_storages()
    # Initialize pipeline status for document processing
    await initialize_pipeline_status()

    return rag


async def test_ingestion():
    """Test document ingestion into TigerGraph"""
    print("=" * 60)
    print("Initializing LightRAG with TigerGraph...")
    print("=" * 60)

    rag = await initialize_rag()
    print(f"✓ LightRAG initialized: {type(rag)}")

    # Test documents for ingestion
    test_documents = [
        "TigerGraph is a graph database platform designed for enterprise-scale graph analytics. It supports distributed graph processing and real-time queries.",
        "LightRAG is a framework that combines retrieval-augmented generation with knowledge graphs. It uses graph storage backends like TigerGraph, Neo4j, and Memgraph.",
        "Graph databases store data as nodes and edges, making them ideal for relationship-heavy data. They excel at traversing complex connections between entities.",
    ]

    print("\n" + "=" * 60)
    print("Ingesting test documents...")
    print("=" * 60)

    # Insert documents
    for i, doc in enumerate(test_documents, 1):
        print(f"\n[{i}/{len(test_documents)}] Inserting document...")
        track_id = await rag.ainsert(input=doc, file_paths=f"test_doc_{i}.txt")
        print(f"  ✓ Document inserted with track_id: {track_id}")

    # Test JSON ingestion if JSON file exists
    json_test_file = Path("test_data.json")
    if json_test_file.exists():
        print("\n" + "=" * 60)
        print("Ingesting JSON file...")
        print("=" * 60)

        try:
            texts = load_json_texts(json_test_file)
            print(f"✓ Loaded {len(texts)} texts from {json_test_file}")

            for i, text in enumerate(texts, 1):
                print(f"\n[{i}/{len(texts)}] Inserting from JSON...")
                track_id = await rag.ainsert(input=text, file_paths=str(json_test_file))
                print(f"  ✓ Text inserted with track_id: {track_id}")
        except Exception as e:
            print(f"✗ Error loading JSON file: {e}")
            import traceback

            traceback.print_exc()
    else:
        print(
            f"\nℹ No JSON file found at {json_test_file} (skipping JSON ingestion test)"
        )
        print("  Create a test_data.json file with format:")
        print('  [{"text": "Your text here"}, {"text": "Another text"}]')

    print("\n" + "=" * 60)
    print("Verifying ingestion...")
    print("=" * 60)

    # Verify by checking graph stats
    try:
        # Get all labels (entity IDs) from the graph
        all_labels = await rag.chunk_entity_relation_graph.get_all_labels()
        print(f"\n✓ Found {len(all_labels)} entities in the graph")
        if all_labels:
            print(f"  Sample entities: {all_labels[:5]}")

        # Get all nodes
        all_nodes = await rag.chunk_entity_relation_graph.get_all_nodes()
        print(f"✓ Found {len(all_nodes)} nodes in the graph")

        # Get all edges
        all_edges = await rag.chunk_entity_relation_graph.get_all_edges()
        print(f"✓ Found {len(all_edges)} edges in the graph")

        # Test a simple query
        print("\n" + "=" * 60)
        print("Testing query...")
        print("=" * 60)
        response = await rag.aquery("What is TigerGraph?")
        print("\nQuery: 'What is TigerGraph?'")
        print(f"Response: {response}")

    except Exception as e:
        print(f"\n✗ Error during verification: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 60)
    print("Ingestion test completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_ingestion())
