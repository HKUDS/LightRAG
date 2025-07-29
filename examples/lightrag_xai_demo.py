"""
Example demonstrating LightRAG integration with xAI Grok models.

This script shows how to use xAI's Grok models (like Grok 3 Mini) with LightRAG
for retrieval-augmented generation tasks.

Requirements:
- Set XAI_API_KEY environment variable with your xAI API key
- Install LightRAG: pip install lightrag-hku
- Have the demo document (book.txt) in the current directory

Usage:
    export XAI_API_KEY="your-xai-api-key"
    python examples/lightrag_xai_demo.py
"""

import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.xai import xai_complete_if_cache
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger, EmbeddingFunc
from lightrag.llm.ollama import ollama_embed

# Setup logging
setup_logger("lightrag", level="INFO")

# Configuration
WORKING_DIR = "./dickens_xai"
XAI_API_KEY = os.environ.get("XAI_API_KEY")

if not XAI_API_KEY:
    raise ValueError("XAI_API_KEY environment variable is required")

# Available xAI models
XAI_MODELS = {
    "grok-3-mini": "Fast and efficient model for general tasks",
    "grok-2-1212": "More capable model with better reasoning",
    "grok-2-vision-1212": "Supports vision capabilities (multimodal)",
}

print("Available xAI Models:")
for model, description in XAI_MODELS.items():
    print(f"  - {model}: {description}")

# Use Grok 3 Mini by default
SELECTED_MODEL = "grok-3-mini"
print(f"\nUsing model: {SELECTED_MODEL}")

async def xai_model_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    """Custom wrapper for xAI model completion."""
    result = await xai_complete_if_cache(
        model=SELECTED_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=XAI_API_KEY,
        base_url="https://api.x.ai/v1",
        **kwargs,
    )
    # Since stream=False, result should always be a string
    return result


async def initialize_rag():
    """Initialize LightRAG with xAI Grok model."""
    # Clean slate - remove existing working dir to avoid embedding dimension conflicts
    import shutil
    if os.path.exists(WORKING_DIR):
        print(f"🧹 Cleaning existing working directory: {WORKING_DIR}")
        shutil.rmtree(WORKING_DIR)
    
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Use consistent embedding model to avoid dimension mismatches
    embedding_model = os.getenv("EMBEDDING_MODEL", "bge-m3:latest")
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "1024"))
    
    print(f"📊 Using embedding model: {embedding_model} (dim: {embedding_dim})")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=xai_model_complete,
        llm_model_name=SELECTED_MODEL,
        # Note: xAI doesn't have dedicated embedding models yet,
        # so we use Ollama embeddings as fallback
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
            func=lambda texts: ollama_embed(
                texts,
                embed_model=embedding_model,
                host=os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434"),
            ),
        ),
        chunk_token_size=1200,  # Grok models handle larger contexts well
        chunk_overlap_token_size=100,
        summary_max_tokens=32000,  # Grok models have large context windows
        llm_model_max_async=4,
        enable_llm_cache=True,
    )

    # IMPORTANT: Both initialization calls are required!
    await rag.initialize_storages()  # Initialize storage backends
    await initialize_pipeline_status()  # Initialize processing pipeline

    return rag


async def demo_queries(rag):
    """Run demonstration queries with different modes."""

    queries = [
        "What are the main themes in this story?",
        "Who are the main characters and what are their relationships?",
        "How does Scrooge's character change throughout the story?",
        "What is the significance of the three spirits?",
    ]

    # Available modes for demonstration
    # modes = ["naive", "local", "global", "hybrid", "mix"]

    print("\n" + "="*60)
    print("RUNNING DEMONSTRATION QUERIES")
    print("="*60)

    for i, query in enumerate(queries):
        print(f"\n📝 Query {i+1}: {query}")
        print("-" * 50)

        # Use hybrid mode for demonstration (good balance)
        mode = "hybrid"

        try:
            print(f"🤖 Using xAI Grok ({SELECTED_MODEL}) in {mode} mode...")
            response = await rag.aquery(
                query,
                param=QueryParam(mode=mode)
            )
            print(f"💡 Response:\n{response}\n")

        except Exception as e:
            print(f"❌ Error: {str(e)}\n")

    # Demonstrate different query modes with one question
    print("\n" + "="*60)
    print("COMPARING DIFFERENT QUERY MODES")
    print("="*60)

    comparison_query = "What is the main message of A Christmas Carol?"
    print(f"\n📝 Query: {comparison_query}")

    for mode in ["local", "global", "hybrid"]:
        print(f"\n--- {mode.upper()} MODE ---")
        try:
            response = await rag.aquery(
                comparison_query,
                param=QueryParam(mode=mode)  # type: ignore
            )
            print(response[:300] + "..." if len(response) > 300 else response)
        except Exception as e:
            print(f"❌ Error in {mode} mode: {str(e)}")


async def main():
    """Main demonstration function."""
    rag = None
    try:
        print("🚀 Initializing LightRAG with xAI Grok...")
        rag = await initialize_rag()

        # Check if we have the demo document
        demo_file = "./book.txt"
        if not os.path.exists(demo_file):
            print(f"\n📥 Demo document not found. Downloading...")
            import urllib.request
            url = "https://raw.githubusercontent.com/gusye1234/nano-graphrag/main/tests/mock_data.txt"
            urllib.request.urlretrieve(url, demo_file)
            print(f"✅ Downloaded demo document to {demo_file}")

        # Insert the document into LightRAG
        print(f"\n📚 Reading and processing document: {demo_file}")
        with open(demo_file, "r", encoding="utf-8") as f:
            content = f.read()

        print("🔄 Inserting document into knowledge graph...")
        await rag.ainsert(content)
        print("✅ Document processed successfully!")

        # Run demonstration queries
        await demo_queries(rag)

        print(f"\n🎉 Demo completed successfully!")
        print(f"💾 Knowledge graph stored in: {WORKING_DIR}")
        print(f"🤖 Used xAI model: {SELECTED_MODEL}")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rag is not None:
            await rag.finalize_storages()


if __name__ == "__main__":
    print("🔮 LightRAG + xAI Grok Demo")
    print("="*40)
    print("This demo shows how to use xAI's Grok models with LightRAG")
    print("for intelligent document processing and querying.\n")

    # Check for API key
    if not XAI_API_KEY:
        print("❌ Error: XAI_API_KEY environment variable not set")
        print("Please set your xAI API key:")
        print("  export XAI_API_KEY='your-xai-api-key'")
        exit(1)

    print(f"✅ xAI API key configured")

    # Run the demo
    asyncio.run(main())