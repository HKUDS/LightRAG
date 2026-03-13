import os
import asyncio

from rag_texts import text_to_rag
from lightrag import LightRAG
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import setup_logger

setup_logger("lightrag", level="INFO")

# WORKING_DIR = "./rag_storage"
# os.makedirs(WORKING_DIR, exist_ok=True)

# Neo4j connection details (set as environment variables or directly here)
os.environ["NEO4J_URI"] = "bolt://localhost:7687"  # Replace with your Neo4j URI
os.environ["NEO4J_USERNAME"] = "neo4j"                # Default username
os.environ["NEO4J_PASSWORD"] = "your_password"        # Replace with your password
os.environ["NEO4J_DATABASE"] = "neo4j"        # Optional: specify a database name

async def main():
    # Initialize LightRAG with Neo4j as graph storage
    rag = LightRAG(
        # working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
        graph_storage="Neo4JStorage",  # Use Neo4j for graph storage
    )
    await rag.initialize_storages()  # Connects to Neo4j and initializes storage

    # Insert text (if not already indexed)
    await rag.ainsert(text_to_rag)

    await rag.finalize_storages()

asyncio.run(main())
