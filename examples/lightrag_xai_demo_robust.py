"""
Robust LightRAG + xAI Demo with Dimension Conflict Resolution

This script automatically detects and resolves embedding dimension conflicts
and provides a clean, working demonstration of xAI integration.

Requirements:
- Set XAI_API_KEY environment variable with your xAI API key
- Have Ollama running with bge-m3:latest model installed
- Install LightRAG: pip install lightrag-hku

Usage:
    export XAI_API_KEY="your-xai-api-key"
    python examples/lightrag_xai_demo_robust.py
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
WORKING_DIR = "./dickens_xai_robust"
XAI_API_KEY = os.environ.get("XAI_API_KEY")

if not XAI_API_KEY:
    print("❌ Error: XAI_API_KEY environment variable not set")
    print("Please set your xAI API key:")
    print("  export XAI_API_KEY='your-xai-api-key'")
    exit(1)

# Available xAI models
XAI_MODELS = {
    "grok-3-mini": "Fast and efficient model for general tasks",
    "grok-2-1212": "More capable model with better reasoning",
    "grok-2-vision-1212": "Supports vision capabilities (multimodal)",
}

print("🔮 LightRAG + xAI Grok Demo (Robust Version)")
print("="*50)
print("Available xAI Models:")
for model, description in XAI_MODELS.items():
    print(f"  - {model}: {description}")

# Use Grok 3 Mini by default
SELECTED_MODEL = "grok-3-mini"
print(f"\n🤖 Using model: {SELECTED_MODEL}")


async def test_embedding_dimension():
    """Test the actual embedding dimension to avoid conflicts."""
    try:
        print("🔍 Testing embedding model dimensions...")
        embedding_model = "bge-m3:latest"
        host = "http://localhost:11434"
        
        result = await ollama_embed(
            ["test"], 
            embed_model=embedding_model,
            host=host
        )
        actual_dim = result.shape[1]
        print(f"✅ {embedding_model} produces {actual_dim}-dimensional embeddings")
        return embedding_model, actual_dim, host
        
    except Exception as e:
        print(f"❌ Error testing embedding model: {e}")
        print("💡 Make sure Ollama is running and bge-m3:latest is installed:")
        print("   ollama pull bge-m3:latest")
        raise


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
    """Initialize LightRAG with xAI Grok model and robust dimension handling."""
    
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
            embedding_dim=embedding_dim,  # Use actual tested dimension
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts,
                embed_model=embedding_model,
                host=host,
            ),
        ),
        chunk_token_size=1200,  # Grok models handle larger contexts well
        chunk_overlap_token_size=100,
        summary_max_tokens=32000,  # Grok models have large context windows
        llm_model_max_async=2,  # Reduced concurrency to prevent timeouts
        enable_llm_cache=True,
    )

    # IMPORTANT: Both initialization calls are required!
    print("⚙️  Initializing storage backends...")
    await rag.initialize_storages()  # Initialize storage backends
    await initialize_pipeline_status()  # Initialize processing pipeline
    print("✅ LightRAG initialized successfully!")

    return rag


async def demo_queries(rag):
    """Run demonstration queries with different modes."""
    
    queries = [
        "What are the main themes in this story?",
        "Who are the main characters and what are their relationships?", 
        "How does Scrooge's character change throughout the story?",
        "What is the significance of the three spirits?",
    ]

    print("\n" + "="*60)
    print("RUNNING DEMONSTRATION QUERIES")
    print("="*60)

    for i, query in enumerate(queries):
        print(f"\n📝 Query {i+1}: {query}")
        print("-" * 50)
        
        try:
            print(f"🤖 Using xAI Grok ({SELECTED_MODEL}) in hybrid mode...")
            response = await rag.aquery(
                query,
                param=QueryParam(mode="hybrid")
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
        # Initialize RAG system
        rag = await initialize_rag()
        
        # Download demo document if needed
        demo_file = "./book.txt"
        if not os.path.exists(demo_file):
            print(f"\n📥 Demo document not found. Downloading...")
            import urllib.request
            url = "https://raw.githubusercontent.com/gusye1234/nano-graphrag/main/tests/mock_data.txt"
            urllib.request.urlretrieve(url, demo_file)
            print(f"✅ Downloaded demo document to {demo_file}")
        
        # Process the document
        print(f"\n📚 Reading and processing document: {demo_file}")
        with open(demo_file, "r", encoding="utf-8") as f:
            content = f.read()
        
        print("🔄 Inserting document into knowledge graph...")
        print("   This may take a few moments as xAI processes the content...")
        
        try:
            await rag.ainsert(content)
            print("✅ Document processed successfully!")
        except Exception as e:
            print(f"❌ Error processing document: {e}")
            if "dimension" in str(e).lower():
                print("💡 This looks like an embedding dimension issue.")
                print("   The working directory has been cleaned, so this shouldn't happen.")
                print("   Please check your Ollama installation and try again.")
            elif "timeout" in str(e).lower() or "connect" in str(e).lower():
                print("💡 This is a connection timeout issue.")
                print("   - Check that Ollama is running and responsive")
                print("   - Try restarting the demo - it may work on retry")
                print("   - Consider using the timeout-fix version:")
                print("     python examples/lightrag_xai_demo_timeout_fix.py")
            raise
        
        # Run demonstration queries
        await demo_queries(rag)
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"💾 Knowledge graph stored in: {WORKING_DIR}")
        print(f"🤖 Used xAI model: {SELECTED_MODEL}")
        print(f"📊 Embedding model: bge-m3:latest (1024D)")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Provide helpful error messages
        if "dimension" in str(e).lower():
            print("\n💡 Dimension Error Troubleshooting:")
            print("   1. Make sure Ollama is running: systemctl status ollama")
            print("   2. Verify bge-m3 is installed: ollama list | grep bge-m3")
            print("   3. Try manually cleaning: rm -rf ./dickens_xai_robust")
        elif "api" in str(e).lower() or "key" in str(e).lower():
            print("\n💡 API Error Troubleshooting:")
            print("   1. Check your xAI API key: echo $XAI_API_KEY")
            print("   2. Verify the key is valid on https://console.x.ai")
            print("   3. Make sure you have API credits available")
            
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
        response = httpx.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is accessible")
        else:
            print("⚠️  Ollama may not be running properly")
    except Exception:
        print("❌ Cannot connect to Ollama at http://localhost:11434")
        print("   Please start Ollama: systemctl start ollama")
        exit(1)
    
    print("\n🚀 Starting robust LightRAG + xAI demo...")
    asyncio.run(main())