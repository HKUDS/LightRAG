"""
LightRAG + xAI Demo with Timeout and Concurrency Fixes

This version addresses timeout issues during embedding operations
by increasing timeouts and reducing concurrency.

Requirements:
- Set XAI_API_KEY environment variable with your xAI API key
- Have Ollama running with bge-m3:latest model installed
- Install LightRAG: pip install lightrag-hku

Usage:
    export XAI_API_KEY="your-xai-api-key"
    python examples/lightrag_xai_demo_timeout_fix.py
"""

import os
import asyncio
import shutil
from pathlib import Path
from lightrag import LightRAG, QueryParam
from lightrag.llm.xai import xai_complete_if_cache
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger, EmbeddingFunc
from lightrag.llm.ollama import ollama_embed

# Setup logging
setup_logger("lightrag", level="INFO")

# Configuration
WORKING_DIR = "./dickens_xai_timeout_fix"
XAI_API_KEY = os.environ.get("XAI_API_KEY")

if not XAI_API_KEY:
    print("❌ Error: XAI_API_KEY environment variable not set")
    print("Please set your xAI API key:")
    print("  export XAI_API_KEY='your-xai-api-key'")
    exit(1)

# Use Grok 3 Mini by default
SELECTED_MODEL = "grok-3-mini"

print("🔮 LightRAG + xAI Demo (Timeout Fix Version)")
print("=" * 50)
print(f"🤖 Using model: {SELECTED_MODEL}")


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
    return result


async def robust_ollama_embed(texts, embed_model="bge-m3:latest", host="http://localhost:11434"):
    """Ollama embedding with increased timeout and retry logic."""
    max_retries = 3
    timeout = 120  # Increased timeout to 2 minutes

    for attempt in range(max_retries):
        try:
            print(f"🔄 Embedding attempt {attempt + 1}/{max_retries} for {len(texts)} texts...")
            result = await ollama_embed(
                texts,
                embed_model=embed_model,
                host=host,
                timeout=timeout
            )
            print("✅ Embedding successful")
            return result
        except Exception as e:
            print(f"⚠️  Embedding attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 5  # Exponential backoff
                print(f"🕐 Waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)
            else:
                print(f"❌ All embedding attempts failed")
                raise


async def test_embedding_dimension():
    """Test the actual embedding dimension with robust error handling."""
    try:
        print("🔍 Testing embedding model dimensions...")
        embedding_model = "bge-m3:latest"
        host = "http://localhost:11434"

        result = await robust_ollama_embed(
            ["test"],
            embed_model=embedding_model,
            host=host
        )
        if result is None:
            raise Exception("Embedding function returned None")
        actual_dim = result.shape[1]
        print(f"✅ {embedding_model} produces {actual_dim}-dimensional embeddings")
        return embedding_model, actual_dim, host

    except Exception as e:
        print(f"❌ Error testing embedding model: {e}")
        print("💡 Make sure Ollama is running and bge-m3:latest is installed:")
        print("   ollama pull bge-m3:latest")
        raise


async def initialize_rag():
    """Initialize LightRAG with xAI Grok model and robust timeout handling."""

    # Clean slate - completely remove working directory to avoid any conflicts
    working_path = Path(WORKING_DIR)
    if working_path.exists():
        print(f"🧹 Removing existing working directory: {WORKING_DIR}")
        shutil.rmtree(WORKING_DIR)

    print(f"📁 Creating fresh working directory: {WORKING_DIR}")
    working_path.mkdir(parents=True, exist_ok=True)

    # Test embedding model and get actual dimensions
    embedding_model, embedding_dim, host = await test_embedding_dimension()

    print(f"🚀 Initializing LightRAG with:")
    print(f"   - LLM: xAI {SELECTED_MODEL}")
    print(f"   - Embedding: {embedding_model} ({embedding_dim}D)")
    print(f"   - Working Dir: {WORKING_DIR}")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=xai_model_complete,
        llm_model_name=SELECTED_MODEL,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=lambda texts: robust_ollama_embed(
                texts,
                embed_model=embedding_model,
                host=host,
            ),
        ),
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        summary_max_tokens=32000,
        llm_model_max_async=2,  # Reduced from 4 to decrease load
        enable_llm_cache=True,
    )

    # IMPORTANT: Both initialization calls are required!
    print("⚙️  Initializing storage backends...")
    await rag.initialize_storages()
    await initialize_pipeline_status()
    print("✅ LightRAG initialized successfully!")

    return rag


async def demo_queries(rag):
    """Run demonstration queries with different modes."""

    queries = [
        "What are the main themes in this story?",
        "Who are the main characters and what are their relationships?",
        "How does Scrooge's character change throughout the story?",
    ]

    print("\\n" + "="*60)
    print("RUNNING DEMONSTRATION QUERIES")
    print("="*60)

    for i, query in enumerate(queries):
        print(f"\\n📝 Query {i+1}: {query}")
        print("-" * 50)

        try:
            print(f"🤖 Using xAI Grok ({SELECTED_MODEL}) in hybrid mode...")
            response = await rag.aquery(
                query,
                param=QueryParam(mode="hybrid")
            )
            print(f"💡 Response:\\n{response}\\n")

        except Exception as e:
            print(f"❌ Error: {str(e)}\\n")


async def main():
    """Main demonstration function."""
    rag = None
    try:
        # Initialize RAG system
        rag = await initialize_rag()

        # Download demo document if needed
        demo_file = "./book.txt"
        if not os.path.exists(demo_file):
            print(f"📥 Demo document not found. Downloading...")
            import urllib.request
            url = "https://raw.githubusercontent.com/gusye1234/nano-graphrag/main/tests/mock_data.txt"
            urllib.request.urlretrieve(url, demo_file)
            print(f"✅ Downloaded demo document to {demo_file}")

        # Process the document
        print(f"📚 Reading and processing document: {demo_file}")
        with open(demo_file, "r", encoding="utf-8") as f:
            content = f.read()

        print("🔄 Inserting document into knowledge graph...")
        print("   This may take several minutes due to reduced concurrency...")
        print("   The system will automatically retry on timeout errors.")

        try:
            await rag.ainsert(content)
            print("✅ Document processed successfully!")
        except Exception as e:
            print(f"❌ Error processing document: {e}")
            if "timeout" in str(e).lower() or "connect" in str(e).lower():
                print("💡 This is a connection timeout issue.")
                print("   - Check that Ollama is running: systemctl status ollama")
                print("   - Try restarting Ollama: sudo systemctl restart ollama")
                print("   - Wait a moment and try again")
            raise

        # Run demonstration queries
        await demo_queries(rag)

        print(f"\\n🎉 Demo completed successfully!")
        print(f"💾 Knowledge graph stored in: {WORKING_DIR}")
        print(f"🤖 Used xAI model: {SELECTED_MODEL}")
        print(f"📊 Embedding model: bge-m3:latest (1024D)")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if rag is not None:
            print("🧹 Cleaning up...")
            await rag.finalize_storages()


if __name__ == "__main__":
    # Check prerequisites
    print("🔍 Checking prerequisites...")

    if not XAI_API_KEY:
        print("❌ XAI_API_KEY not set")
        exit(1)
    else:
        print("✅ xAI API key configured")

    # Check if Ollama is accessible
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=10)
        if response.status_code == 200:
            print("✅ Ollama is accessible")
        else:
            print("⚠️  Ollama may not be running properly")
    except Exception:
        print("❌ Cannot connect to Ollama at http://localhost:11434")
        print("   Please start Ollama: systemctl start ollama")
        exit(1)

    print("🚀 Starting timeout-resistant LightRAG + xAI demo...")
    asyncio.run(main())