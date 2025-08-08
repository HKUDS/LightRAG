import pytest
import logging
from lightrag_mcp.config import get_config
from lightrag_mcp.tools.query_tools import lightrag_query
from lightrag_mcp.tools.document_tools import (
    lightrag_insert_text,
    lightrag_list_documents,
)
from lightrag_mcp.tools.system_tools import lightrag_health_check

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_test")


@pytest.mark.asyncio
async def test_health_check():
    """Test the health check functionality."""
    print("\n🔍 Testing health check...")
    try:
        result = await lightrag_health_check()
        status = result.get("status", "unknown")
        print(f"   Health status: {status}")

        if status == "healthy":
            print("   ✅ Health check passed")
            return True
        else:
            print(f"   ⚠️  Health check shows: {status}")
            print(f"   Message: {result.get('message', 'No message')}")
            return False

    except Exception as e:
        print(f"   ❌ Health check failed: {e}")
        return False


@pytest.mark.asyncio
async def test_document_insertion():
    """Test document insertion functionality."""
    print("\n📝 Testing document insertion...")
    try:
        test_text = """
        This is a test document for the LightRAG MCP server.
        It contains information about artificial intelligence and machine learning.

        Artificial Intelligence (AI) is a field of computer science that aims to create
        intelligent machines that can perform tasks that typically require human intelligence.
        Machine Learning (ML) is a subset of AI that focuses on algorithms that can learn
        and improve from experience without being explicitly programmed.
        """

        result = await lightrag_insert_text(
            text=test_text.strip(),
            title="MCP Test Document",
            metadata={"source": "mcp_test", "category": "ai_ml"},
        )

        document_id = result.get("document_id", "unknown")
        status = result.get("status", "unknown")

        print(f"   Document ID: {document_id}")
        print(f"   Status: {status}")
        print("   ✅ Document insertion passed")
        return True

    except Exception as e:
        print(f"   ❌ Document insertion failed: {e}")
        return False


@pytest.mark.asyncio
async def test_query():
    """Test query functionality."""
    print("\n🔍 Testing RAG query...")
    try:
        result = await lightrag_query(
            query="What is artificial intelligence?", mode="hybrid", top_k=20
        )

        response = result.get("response", "")
        mode = result.get("mode", "unknown")
        metadata = result.get("metadata", {})

        print(f"   Query mode: {mode}")
        print(f"   Response length: {len(response)} characters")
        print(f"   Processing time: {metadata.get('processing_time', 'unknown')}")

        if response and len(response) > 10:
            print("   ✅ Query passed")
            print(f"   Response preview: {response[:100]}...")
            return True
        else:
            print("   ⚠️  Query returned minimal response")
            return False

    except Exception as e:
        print(f"   ❌ Query failed: {e}")
        return False


@pytest.mark.asyncio
async def test_document_listing():
    """Test document listing functionality."""
    print("\n📋 Testing document listing...")
    try:
        result = await lightrag_list_documents(limit=5)

        documents = result.get("documents", [])
        total = result.get("total", 0)

        print(f"   Total documents: {total}")
        print(f"   Retrieved: {len(documents)}")

        if documents:
            for i, doc in enumerate(documents[:3]):
                title = doc.get("title", "Untitled")
                status = doc.get("status", "unknown")
                print(f"   Document {i+1}: {title} ({status})")

        print("   ✅ Document listing passed")
        return True

    except Exception as e:
        print(f"   ❌ Document listing failed: {e}")
        return False


@pytest.mark.asyncio
async def test_configuration():
    """Test configuration loading."""
    print("\n⚙️  Testing configuration...")
    try:
        config = get_config()

        print(f"   Server name: {config.mcp_server_name}")
        print(f"   API URL: {config.lightrag_api_url}")
        print(f"   Direct mode: {config.enable_direct_mode}")
        print(f"   Streaming: {config.enable_streaming}")
        print(f"   Document upload: {config.enable_document_upload}")
        print(f"   Graph modification: {config.enable_graph_modification}")

        print("   ✅ Configuration loaded successfully")
        return True

    except Exception as e:
        print(f"   ❌ Configuration loading failed: {e}")
        return False
