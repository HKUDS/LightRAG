#!/usr/bin/env python3
"""
LightRAG MCP Usage Example

This example demonstrates how to use the LightRAG MCP server
for typical RAG operations through the MCP interface.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lightrag_mcp.tools.query_tools import lightrag_query
from lightrag_mcp.tools.document_tools import lightrag_insert_text, lightrag_list_documents
from lightrag_mcp.tools.system_tools import lightrag_health_check
from lightrag_mcp.tools.graph_tools import lightrag_search_entities, lightrag_get_graph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_usage_example")


async def demonstrate_document_workflow():
    """Demonstrate a complete document processing workflow."""
    print("\n📚 Document Processing Workflow")
    print("-" * 40)
    
    # 1. Insert sample documents
    print("1. Inserting sample documents...")
    
    documents = [
        {
            "title": "Introduction to AI",
            "content": """
            Artificial Intelligence (AI) represents one of the most transformative 
            technologies of our time. AI systems can perform tasks that typically 
            require human intelligence, such as visual perception, speech recognition, 
            decision-making, and language translation.
            
            Machine Learning is a key subset of AI that enables systems to 
            automatically learn and improve from experience without being 
            explicitly programmed.
            """,
            "metadata": {"category": "technology", "difficulty": "beginner"}
        },
        {
            "title": "Neural Networks",
            "content": """
            Neural networks are computing systems inspired by biological neural 
            networks. They consist of interconnected nodes (neurons) that process 
            information through weighted connections.
            
            Deep learning uses neural networks with multiple layers to model 
            high-level abstractions in data. This approach has revolutionized 
            fields like computer vision and natural language processing.
            """,
            "metadata": {"category": "technology", "difficulty": "intermediate"}
        },
        {
            "title": "Climate Change",
            "content": """
            Climate change refers to long-term shifts in global temperatures and 
            weather patterns. While climate variations are natural, scientific 
            evidence shows that human activities have been the main driver of 
            climate change since the mid-20th century.
            
            The primary cause is the emission of greenhouse gases, particularly 
            carbon dioxide from burning fossil fuels.
            """,
            "metadata": {"category": "environment", "difficulty": "beginner"}
        }
    ]
    
    doc_ids = []
    for doc in documents:
        try:
            result = await lightrag_insert_text(
                text=doc["content"].strip(),
                title=doc["title"],
                metadata=doc["metadata"]
            )
            doc_id = result.get("document_id", "unknown")
            doc_ids.append(doc_id)
            print(f"   ✅ Inserted: {doc['title']} (ID: {doc_id})")
        except Exception as e:
            print(f"   ❌ Failed to insert {doc['title']}: {e}")
    
    # 2. List documents
    print("\n2. Listing documents...")
    try:
        result = await lightrag_list_documents(limit=10)
        documents = result.get("documents", [])
        total = result.get("total", 0)
        
        print(f"   Total documents in system: {total}")
        print("   Recent documents:")
        for i, doc in enumerate(documents[:5]):
            title = doc.get("title", "Untitled")
            status = doc.get("status", "unknown")
            print(f"   {i+1}. {title} ({status})")
    except Exception as e:
        print(f"   ❌ Failed to list documents: {e}")
    
    return doc_ids


async def demonstrate_query_workflow():
    """Demonstrate different query modes and capabilities."""
    print("\n🔍 Query Workflow")
    print("-" * 40)
    
    # Test different query modes
    queries = [
        ("What is artificial intelligence?", "hybrid"),
        ("How do neural networks work?", "local"),
        ("Tell me about climate change causes", "global"),
        ("Compare AI and climate science", "mix")
    ]
    
    for query_text, mode in queries:
        print(f"\n3. Querying: '{query_text}' (mode: {mode})")
        try:
            result = await lightrag_query(
                query=query_text,
                mode=mode,
                top_k=20,
                chunk_top_k=5
            )
            
            response = result.get("response", "")
            metadata = result.get("metadata", {})
            processing_time = metadata.get("processing_time", 0)
            
            print(f"   Mode: {mode}")
            print(f"   Processing time: {processing_time:.2f}s")
            print(f"   Response length: {len(response)} characters")
            print(f"   Response preview: {response[:150]}...")
            
        except Exception as e:
            print(f"   ❌ Query failed: {e}")


async def demonstrate_graph_exploration():
    """Demonstrate knowledge graph exploration capabilities."""
    print("\n🕸️  Knowledge Graph Exploration")
    print("-" * 40)
    
    # 1. Search for entities
    print("4. Searching for AI-related entities...")
    try:
        result = await lightrag_search_entities(
            query="artificial intelligence",
            search_type="fuzzy",
            limit=5
        )
        
        entities = result.get("entities", [])
        total_matches = result.get("total_matches", 0)
        
        print(f"   Found {total_matches} entities")
        for i, entity_match in enumerate(entities):
            entity = entity_match.get("entity", {})
            score = entity_match.get("relevance_score", 0)
            name = entity.get("name", "Unknown")
            print(f"   {i+1}. {name} (score: {score:.3f})")
            
    except Exception as e:
        print(f"   ❌ Entity search failed: {e}")
    
    # 2. Get graph structure
    print("\n5. Extracting knowledge graph structure...")
    try:
        result = await lightrag_get_graph(
            max_nodes=20,
            max_edges=30,
            output_format="json",
            include_properties=True
        )
        
        nodes = result.get("nodes", [])
        edges = result.get("edges", [])
        statistics = result.get("statistics", {})
        
        print(f"   Nodes: {len(nodes)}")
        print(f"   Edges: {len(edges)}")
        print(f"   Statistics: {statistics}")
        
        if nodes:
            print("   Sample nodes:")
            for i, node in enumerate(nodes[:3]):
                name = node.get("properties", {}).get("name", "Unknown")
                labels = node.get("labels", [])
                print(f"   {i+1}. {name} (labels: {labels})")
                
    except Exception as e:
        print(f"   ❌ Graph extraction failed: {e}")


async def demonstrate_system_monitoring():
    """Demonstrate system monitoring and health checks."""
    print("\n🏥 System Monitoring")
    print("-" * 40)
    
    # Health check
    print("6. Performing health check...")
    try:
        result = await lightrag_health_check(include_detailed=True)
        
        status = result.get("status", "unknown")
        version = result.get("version", "unknown")
        uptime = result.get("uptime", "unknown")
        
        print(f"   Status: {status}")
        print(f"   Version: {version}")
        print(f"   Uptime: {uptime}")
        
        # Configuration info
        config = result.get("configuration", {})
        if config:
            print("   Configuration highlights:")
            for key, value in list(config.items())[:3]:
                print(f"   - {key}: {value}")
                
    except Exception as e:
        print(f"   ❌ Health check failed: {e}")


async def main():
    """Run the complete usage demonstration."""
    print("🚀 LightRAG MCP Usage Demonstration")
    print("=" * 50)
    
    try:
        # Run all demonstrations
        await demonstrate_document_workflow()
        await demonstrate_query_workflow()
        await demonstrate_graph_exploration()
        await demonstrate_system_monitoring()
        
        print("\n" + "=" * 50)
        print("✅ Usage demonstration completed successfully!")
        print("\nNext steps:")
        print("1. Set up Claude CLI with MCP configuration")
        print("2. Use 'claude mcp lightrag_query \"your question\"'")
        print("3. Try document upload with 'claude mcp lightrag_insert_file'")
        print("4. Explore the knowledge graph with graph tools")
        
    except Exception as e:
        print(f"\n💥 Demonstration failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️  Demonstration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Demo runner failed: {e}")
        sys.exit(1)