#!/usr/bin/env python
"""
Example script demonstrating LightRAG export/import functionality

This script shows how to:
1. Export data from a LightRAG instance
2. Import data to another instance, potentially with different storage backends
"""

import os
import sys
import json
import asyncio
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lightrag import LightRAG
from lightrag.tools.exporter import LightRAGExporter
from lightrag.utils import setup_logger, EmbeddingFunc

# Setup logging - using the correct parameter 'level' instead of 'log_level'
setup_logger("lightrag", level="INFO")

# Define a mock embedding function that implements the required interface
EMBEDDING_DIM = 128
async def mock_embedding_func(texts, **kwargs):
    return [np.array([0.1] * EMBEDDING_DIM) for _ in range(len(texts))]

# Define a mock LLM function
async def mock_llm_func(text, **kwargs):
    return "This is a mock LLM response"

async def create_sample_data(lightrag_instance):
    """Add some sample data to a LightRAG instance for demonstration purposes"""
    print("Creating sample data...")
    
    # Create entities directly (without document processing)
    await lightrag_instance.acreate_entity(
        "LightRAG", 
        {
            "entity_type": "SYSTEM",
            "description": "Lightweight Retrieval Augmented Generation framework"
        }
    )
    
    await lightrag_instance.acreate_entity(
        "PostgreSQL", 
        {
            "entity_type": "DATABASE",
            "description": "Open source relational database"
        }
    )
    
    await lightrag_instance.acreate_entity(
        "MongoDB", 
        {
            "entity_type": "DATABASE",
            "description": "NoSQL document database"
        }
    )
    
    # Create relationships
    await lightrag_instance.acreate_relation(
        "LightRAG", 
        "PostgreSQL", 
        {
            "description": "LightRAG can use PostgreSQL as a storage backend",
            "relation_id": "USES"
        }
    )
    
    await lightrag_instance.acreate_relation(
        "LightRAG", 
        "MongoDB", 
        {
            "description": "LightRAG can use MongoDB as a storage backend",
            "relation_id": "USES"
        }
    )
    
    # Add a text chunk directly to demonstrate KV storage
    chunk_id = "sample_chunk_001"
    await lightrag_instance.text_chunks.upsert({
        chunk_id: {
            "content": "LightRAG is a powerful RAG framework with multiple storage backends.",
            "tokens": 12,
            "full_doc_id": "sample_doc_001",
            "chunk_order_index": 0
        }
    })
    
    # Wait for all storage operations to complete
    await lightrag_instance.finalize_storages()
    
    print("Sample data created")

async def main_async():
    # Create timestamp-based directories for the example
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"./lightrag_demo_{timestamp}"
    source_dir = os.path.join(base_dir, "source_instance")
    target_dir = os.path.join(base_dir, "target_instance")
    export_dir = os.path.join(base_dir, "exports")
    
    # Create directories
    for directory in [source_dir, target_dir, export_dir]:
        os.makedirs(directory, exist_ok=True)
    
    print(f"Created demo directories in {base_dir}")
    
    # Create embedding function with proper attributes
    embedding_func = EmbeddingFunc(
        embedding_dim=EMBEDDING_DIM,
        max_token_size=512,
        func=mock_embedding_func
    )
    
    # Step 1: Create source LightRAG instance with in-memory storage
    print("\n=== Creating source LightRAG instance ===")
    source_instance = LightRAG(
        working_dir=source_dir,
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="NetworkXStorage",
        embedding_func=embedding_func,  # Properly configured embedding function
        llm_model_func=mock_llm_func,   # Async mock LLM function
    )
    
    # Initialize storages
    await source_instance.initialize_storages()
    
    # Step 2: Add sample data to source instance
    await create_sample_data(source_instance)
    
    # Step 3: Export data from the source instance - using async function directly
    print("\n=== Exporting data from source instance ===")
    export_path = await LightRAGExporter.export_data(
        lightrag_instance=source_instance,
        output_dir=export_dir,
        include_cache=False
    )
    
    # Step 4: Create target LightRAG instance (potentially with different storage)
    print("\n=== Creating target LightRAG instance ===")
    target_instance = LightRAG(
        working_dir=target_dir,
        # Could use different backends in a real scenario
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="NetworkXStorage",
        embedding_func=embedding_func,  # Use the same embedding function for compatibility
        llm_model_func=mock_llm_func    # Use the same LLM function
    )
    
    # Initialize target storages
    await target_instance.initialize_storages()
    
    # Step 5: Import data from export directory to target instance - using async function directly
    print("\n=== Importing data into target instance ===")
    await LightRAGExporter.import_data(
        lightrag_instance=target_instance,
        import_dir=export_path,
        include_cache=False
    )
    
    # Step 6: Force write to disk by making a small modification to the imported data
    print("\n=== Making a small modification to trigger persistence ===")
    
    # Add a new entity to trigger Vector DB save
    await target_instance.acreate_entity(
        "Neo4j", 
        {
            "entity_type": "DATABASE",
            "description": "Graph database (added to trigger persistence)"
        }
    )
    
    # Add a new text chunk to trigger KV storage save
    await target_instance.text_chunks.upsert({
        "sample_chunk_002": {
            "content": "This is a test chunk to trigger persistence to disk.",
            "tokens": 10,
            "full_doc_id": "sample_doc_001",
            "chunk_order_index": 1
        }
    })
    
    # Step 7: Verify data was transferred successfully
    print("\n=== Verifying data transfer ===")
    
    # Get knowledge graph
    kg = await target_instance.get_knowledge_graph("*")
    print(f"Knowledge graph imported with {len(kg.nodes)} nodes and {len(kg.edges)} edges")
    
    # Print node names
    node_names = [node.id for node in kg.nodes]
    print(f"Imported entities: {', '.join(node_names)}")
    
    # Verify KV storage (text chunks)
    chunks = await target_instance.text_chunks.get_all()
    print(f"Imported chunks: {len(chunks)}")
    if chunks:
        print(f"Sample chunk content: {next(iter(chunks.values()))['content']}")
    
    print("\n=== Export/Import complete ===")
    print(f"Source instance: {source_dir}")
    print(f"Export data: {export_path}")
    print(f"Target instance: {target_dir}")
    
    # Explicitly finalize both instances to ensure data is written to disk
    print("\n=== Finalizing storages ===")
    await source_instance.finalize_storages()
    await target_instance.finalize_storages()
    print("Storage finalization complete")
    
    # Verify the files are actually created in the target directory
    print("\n=== Checking target directory contents ===")
    target_files = os.listdir(target_dir)
    print(f"Files in target directory: {', '.join(target_files)}")

def main():
    """Run the async main function"""
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 