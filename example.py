"""
LightRAG with Neo4j Graph Storage and Qdrant Vector Storage
This script uses structured Pydantic outputs with Ollama.
It can process PDF and EPUB files.
"""

import asyncio
import os
import sys
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama_structured import ollama_model_complete_structured
from lightrag.llm.ollama import ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from dotenv import load_dotenv
from rerank_ollama import create_ollama_rerank_func

# For PDF and EPUB processing
from pypdf import PdfReader
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file."""
    print(f"📄 Reading PDF file: {file_path}")
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_epub(file_path: str) -> str:
    """Extracts text from an EPUB file."""
    print(f"📖 Reading EPUB file: {file_path}")
    book = epub.read_epub(file_path)
    text = ""
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), 'html.parser')
        text += soup.get_text() + "\n\n"
    return text

async def test(file_path: str):
    print("🚀 Initializing LightRAG with Neo4j and Qdrant...\n")

    rerank_func = create_ollama_rerank_func(
        embed_model="embeddinggemma:300m",
        host="http://localhost:11434"
    )

    # Initialize LightRAG with Neo4j and Qdrant
    rag = LightRAG(
        # Directory for local files (KV storage, document status, etc.)
        working_dir="./lightrag_neo4j_qdrant",

        # Storage Configuration
        graph_storage="Neo4JStorage",        # Use Neo4j for graph
        vector_storage="QdrantVectorDBStorage",  # Use Qdrant for vectors
        kv_storage="JsonKVStorage",          # Keep JSON for key-value (or use RedisKVStorage)
        doc_status_storage="JsonDocStatusStorage",  # Keep JSON for doc status

        # LLM Configuration with Structured Output
        llm_model_func=ollama_model_complete_structured,
        llm_model_name=os.getenv("LLM_MODEL", "qwen2.5:7b"),
        llm_model_kwargs={
            "host": os.getenv("LLM_BINDING_HOST", "http://localhost:11434"),
            "max_retries": 3,  # Retry on validation failures
            "options": {
                "temperature": float(os.getenv("TEMPERATURE", "0.5")),
                "num_ctx": 4096,
            }
        },

        # Embedding Configuration
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "768")),
            max_token_size=512,
            func=lambda texts: ollama_embed(
                texts,
                embed_model=os.getenv("EMBEDDING_MODEL", "embeddinggemma:300m"),
                host=os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434")
            )
        ),

        # Chunk Configuration
        chunk_token_size=int(os.getenv("CHUNK_TOKEN_SIZE", "1200")),
        chunk_overlap_token_size=int(os.getenv("CHUNK_OVERLAP_SIZE", "100")),

        # Extraction Configuration
        entity_extract_max_gleaning=int(os.getenv("MAX_GLEANING", "1")),

        # Performance Settings
        llm_model_max_async=int(os.getenv("MAX_ASYNC", "4")),
        embedding_func_max_async=int(os.getenv("EMBEDDING_FUNC_MAX_ASYNC", "8")),
        embedding_batch_num=int(os.getenv("EMBEDDING_BATCH_NUM", "10")),

        rerank_model_func=rerank_func,
        min_rerank_score=0.3,
    )

    print("📦 Initializing storages...")
    # Initialize storages (connects to Neo4j and Qdrant)
    await rag.initialize_storages()

    print("🔧 Initializing pipeline status...")
    await initialize_pipeline_status()

    print("✅ Initialization complete!\n")

    # ========================================================================
    # Test 1: Insert Document from file
    # ========================================================================
    print("=" * 70)
    print("TEST 1: Document Insertion from File")
    print("=" * 70)

    try:
        if file_path.lower().endswith('.pdf'):
            document_text = extract_text_from_pdf(file_path)
        elif file_path.lower().endswith('.epub'):
            document_text = extract_text_from_epub(file_path)
        else:
            print(f"Unsupported file type: {file_path}")
            return

        print(f"📄 Inserting document:\n{document_text[:500]}...\n") # Print first 500 chars
        await rag.ainsert(document_text)
        print("✅ Document inserted successfully!\n")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
        return


    # ========================================================================
    # Test 2: Local Query (Entity-focused)
    # ========================================================================
    print("=" * 70)
    print("TEST 2: Local Query (Entity-focused)")
    print("=" * 70)

    query1 = "Who founded Apple?"
    print(f"❓ Query: {query1}\n")

    result1 = await rag.aquery(
        query1,
        param=QueryParam(
            mode="mix",  # Focus on local entities
            top_k=20,
        )
    )
    print(f"💡 Answer:\n{result1}\n")

    # ========================================================================
    # Test 3: Global Query (Relationship patterns)
    # ========================================================================
    print("=" * 70)
    print("TEST 3: Global Query (Relationship patterns)")
    print("=" * 70)

    query2 = "What is Apple known for?"
    print(f"❓ Query: {query2}\n")

    result2 = await rag.aquery(
        query2,
        param=QueryParam(
            mode="global",  # Focus on relationship patterns
            top_k=20,
        )
    )
    print(f"💡 Answer:\n{result2}\n")

    # ========================================================================
    # Test 4: Hybrid Query (Both entities and relationships)
    # ========================================================================
    print("=" * 70)
    print("TEST 4: Hybrid Query")
    print("=" * 70)

    query3 = "Tell me about Steve Jobs and Apple"
    print(f"❓ Query: {query3}\n")

    result3 = await rag.aquery(
        query3,
        param=QueryParam(
            mode="hybrid",  # Combine local and global
            top_k=20,
        )
    )
    print(f"💡 Answer:\n{result3}\n")

    # ========================================================================
    # Test 5: Check Storage Statistics
    # ========================================================================
    print("=" * 70)
    print("Storage Statistics")
    print("=" * 70)

    # Get entity count
    try:
        # This is a rough estimate - Neo4j/Qdrant don't have direct count methods
        print("📊 Data stored in:")
        print(f"   - Neo4j Graph: {rag.graph_storage}")
        print(f"   - Qdrant Vectors: {rag.vector_storage}")
        print(f"   - Working Directory: {rag.working_dir}")
    except Exception as e:
        print(f"Could not retrieve statistics: {e}")

    print("\n")

    # ========================================================================
    # Cleanup
    # ========================================================================
    print("🧹 Finalizing storages...")
    await rag.finalize_storages()
    print("✅ All done!\n")

    print("=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("1. 🌐 Check Neo4j Browser: http://localhost:7474")
    print("   - Run query: MATCH (n) RETURN n LIMIT 25")
    print("   - See your knowledge graph visually!")
    print("")
    print("2. 📊 Check Qdrant Dashboard: http://localhost:6333/dashboard")
    print("   - View your vector collections")
    print("   - Check embedding statistics")
    print("")
    print("3. 💾 Your data is now persisted in:")
    print("   - Graph relationships: Neo4j")
    print("   - Vector embeddings: Qdrant")
    print("   - Metadata: ./lightrag_neo4j_qdrant/")
    print("=" * 70)


if __name__ == "__main__":
    # Check if a file path is provided
    if len(sys.argv) < 2:
        print("Usage: python your_script_name.py <path_to_your_file.pdf_or_epub>")
    else:
        file_path = sys.argv[1]
        # Run the async test function with the provided file path
        asyncio.run(test(file_path))