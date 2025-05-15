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
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lightrag import LightRAG
from lightrag.tools.exporter import export_lightrag_data, import_lightrag_data
from lightrag.utils import setup_logger

# Setup logging
setup_logger(log_level="INFO")

def create_sample_data(lightrag_instance):
    """Add some sample data to a LightRAG instance for demonstration purposes"""
    print("Creating sample data...")
    
    # Insert a sample document
    sample_text = """
    LightRAG is a powerful and flexible Retrieval Augmented Generation framework.
    It supports multiple storage backends including JSON, PostgreSQL, MongoDB, and others.
    The system can extract entities and relationships from text automatically.
    """
    
    lightrag_instance.insert(sample_text, file_paths="sample.txt")
    
    # Create some sample entities and relationships manually
    lightrag_instance.create_entity(
        "LightRAG", 
        {
            "entity_type": "SYSTEM",
            "description": "Lightweight Retrieval Augmented Generation framework"
        }
    )
    
    lightrag_instance.create_entity(
        "PostgreSQL", 
        {
            "entity_type": "DATABASE",
            "description": "Open source relational database"
        }
    )
    
    lightrag_instance.create_relation(
        "LightRAG", 
        "PostgreSQL", 
        {
            "description": "LightRAG can use PostgreSQL as a storage backend",
            "relation_id": "USES"
        }
    )
    
    print("Sample data created")

def main():
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
    
    # Step 1: Create source LightRAG instance with in-memory storage
    print("\n=== Creating source LightRAG instance ===")
    source_instance = LightRAG(
        working_dir=source_dir,
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="NetworkXStorage",
        embedding_func=lambda texts, **kwargs: [[0.1] * 128 for _ in range(len(texts))],  # Mock embedding function
        llm_model_func=lambda text, **kwargs: "This is a mock LLM response",  # Mock LLM function
    )
    
    # Step 2: Add sample data to source instance
    create_sample_data(source_instance)
    
    # Step 3: Export data from the source instance
    print("\n=== Exporting data from source instance ===")
    export_path = export_lightrag_data(
        lightrag_instance=source_instance,
        output_dir=export_dir,
        include_cache=False  # Usually you don't need to transfer cache between instances
    )
    
    # Step 4: Create target LightRAG instance (potentially with different storage)
    print("\n=== Creating target LightRAG instance ===")
    target_instance = LightRAG(
        working_dir=target_dir,
        # Could use different backends in a real scenario
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="NetworkXStorage",
        embedding_func=lambda texts, **kwargs: [[0.1] * 128 for _ in range(len(texts))],  # Must match dimensions
        llm_model_func=lambda text, **kwargs: "This is a mock LLM response in the target instance"
    )
    
    # Step 5: Import data from export directory to target instance
    print("\n=== Importing data into target instance ===")
    import_lightrag_data(
        lightrag_instance=target_instance,
        import_dir=export_path,
        include_cache=False
    )
    
    # Step 6: Verify data was transferred successfully
    print("\n=== Verifying data transfer ===")
    # Get a knowledge graph from the target instance to verify
    import asyncio
    
    async def verify_data():
        # Get knowledge graph
        kg = await target_instance.get_knowledge_graph("*")
        print(f"Knowledge graph imported with {len(kg.nodes)} nodes and {len(kg.edges)} edges")
        
        # Print node names
        node_names = [node.id for node in kg.nodes]
        print(f"Imported entities: {', '.join(node_names)}")
    
    loop = asyncio.get_event_loop()
    loop.run_until_complete(verify_data())
    
    print("\n=== Export/Import complete ===")
    print(f"Source instance: {source_dir}")
    print(f"Export data: {export_path}")
    print(f"Target instance: {target_dir}")

if __name__ == "__main__":
    main() 