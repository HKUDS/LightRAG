"""
Enhanced LightRAG Demo with Document Labeling System

This example demonstrates how to use LightRAG with the new document labeling features:
1. Insert documents with labels
2. Query documents filtered by labels
3. Manage labels and document organization

Requirements:
- OpenAI API key set in environment variable OPENAI_API_KEY
- or modify the code to use other LLM providers
"""

import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import setup_logger

# Setup logger
setup_logger("lightrag", level="INFO")

# Working directory for this demo
WORKING_DIR = "./rag_storage_with_labels"

async def initialize_rag():
    """Initialize LightRAG with OpenAI models"""
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )
    
    # IMPORTANT: Both initialization calls are required!
    await rag.initialize_storages()
    await initialize_pipeline_status()
    
    return rag

async def demo_basic_labeling():
    """Demonstrate basic labeling functionality"""
    print("\n=== Basic Labeling Demo ===")
    
    rag = await initialize_rag()
    
    try:
        # Create some labels
        await rag.create_label("research", "Research papers and articles", "#FF6B6B")
        await rag.create_label("technology", "Technology-related content", "#4ECDC4")
        await rag.create_label("ai", "Artificial Intelligence content", "#45B7D1")
        await rag.create_label("health", "Health and medical content", "#96CEB4")
        
        print("✅ Created labels: research, technology, ai, health")
        
        # Insert documents with labels
        documents = [
            "Artificial Intelligence has revolutionized healthcare by enabling faster diagnosis and personalized treatment plans.",
            "Machine learning algorithms are being used to analyze medical images and detect diseases early.",
            "The latest research in neural networks shows promising results for natural language processing.",
            "Cloud computing technologies are transforming how businesses operate and store data.",
            "Recent studies show that regular exercise can significantly improve mental health outcomes."
        ]
        
        labels_for_docs = [
            ["ai", "health", "research"],
            ["ai", "health", "technology"],
            ["ai", "research", "technology"],
            ["technology"],
            ["health", "research"]
        ]
        
        # Insert documents with their respective labels
        for i, (doc, labels) in enumerate(zip(documents, labels_for_docs)):
            track_id = rag.insert_with_labels(
                input=doc,
                labels=labels,
                ids=[f"doc_{i+1}"]
            )
            print(f"📄 Inserted document {i+1} with labels {labels}")
        
        print("\n✅ All documents inserted with labels")
        
        # Display label statistics
        stats = rag.get_label_statistics()
        print(f"\n📊 Label Statistics:")
        print(f"Total labels: {stats['total_labels']}")
        print(f"Total labeled documents: {stats['total_labeled_documents']}")
        for label, count in stats['labels_with_counts'].items():
            print(f"  - {label}: {count} documents")
            
        return rag
        
    except Exception as e:
        print(f"❌ Error in basic labeling demo: {e}")
        raise

async def demo_label_filtering():
    """Demonstrate label-based query filtering"""
    print("\n=== Label Filtering Demo ===")
    
    rag = await initialize_rag()
    
    try:
        # Query all documents (no filtering)
        print("🔍 Query: 'What are the benefits of AI?' (no label filtering)")
        response = await rag.aquery("What are the benefits of AI?")
        print(f"Response length: {len(response)} characters")
        print(f"Response preview: {response[:200]}...\n")
        
        # Query with single label filter
        print("🏷️ Query: 'What are the benefits of AI?' (filtered by 'health' label)")
        response = rag.query_with_labels(
            query="What are the benefits of AI?",
            labels=["health"],
            mode="hybrid"
        )
        print(f"Response length: {len(response)} characters")
        print(f"Response preview: {response[:200]}...\n")
        
        # Query with multiple labels (ANY match)
        print("🏷️ Query: 'Tell me about technology' (filtered by 'technology' OR 'research' labels)")
        response = rag.query_with_labels(
            query="Tell me about technology",
            labels=["technology", "research"],
            label_match_all=False,  # ANY of the labels
            mode="hybrid"
        )
        print(f"Response length: {len(response)} characters")
        print(f"Response preview: {response[:200]}...\n")
        
        # Query with multiple labels (ALL match)
        print("🏷️ Query: 'AI and health research' (filtered by 'ai' AND 'health' labels)")
        response = rag.query_with_labels(
            query="AI and health research",
            labels=["ai", "health"],
            label_match_all=True,  # ALL labels required
            mode="hybrid"
        )
        print(f"Response length: {len(response)} characters")
        print(f"Response preview: {response[:200]}...\n")
        
    except Exception as e:
        print(f"❌ Error in label filtering demo: {e}")
        raise

async def demo_label_management():
    """Demonstrate label management features"""
    print("\n=== Label Management Demo ===")
    
    rag = await initialize_rag()
    
    try:
        # Show all labels
        all_labels = rag.get_all_labels()
        print("📋 All Labels:")
        for name, label_obj in all_labels.items():
            print(f"  - {name}: {label_obj.description} (color: {label_obj.color})")
        
        # Show documents for a specific label
        print("\n📄 Documents with 'ai' label:")
        ai_docs = rag.get_documents_by_label("ai")
        for doc_id in ai_docs:
            labels = rag.get_document_labels(doc_id)
            print(f"  - {doc_id}: labels = {labels}")
        
        # Add a new label to existing document
        print("\n🏷️ Adding 'machine-learning' label to doc_1...")
        await rag.assign_labels_to_document("doc_1", ["ai", "health", "research", "machine-learning"])
        updated_labels = rag.get_document_labels("doc_1")
        print(f"Updated labels for doc_1: {updated_labels}")
        
        # Show updated statistics
        stats = rag.get_label_statistics()
        print(f"\n📊 Updated Label Statistics:")
        for label, count in stats['labels_with_counts'].items():
            print(f"  - {label}: {count} documents")
            
    except Exception as e:
        print(f"❌ Error in label management demo: {e}")
        raise

async def demo_advanced_usage():
    """Demonstrate advanced labeling features"""
    print("\n=== Advanced Usage Demo ===")
    
    rag = await initialize_rag()
    
    try:
        # Using QueryParam directly for more control
        print("🔧 Using QueryParam for advanced filtering...")
        
        param = QueryParam(
            mode="hybrid",
            labels=["research", "technology"],
            label_match_all=False,
            top_k=5,
            chunk_top_k=10,
            response_type="Bullet Points"
        )
        
        response = await rag.aquery("What research has been done?", param)
        print(f"Advanced query response:\n{response[:300]}...\n")
        
        # Check which documents were filtered
        if hasattr(param, 'filtered_doc_ids'):
            print(f"Documents matching filter criteria: {param.filtered_doc_ids}")
        
    except Exception as e:
        print(f"❌ Error in advanced usage demo: {e}")
        raise

async def main():
    """Main demo function"""
    print("🚀 Enhanced LightRAG with Labels Demo")
    print("=====================================")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment")
        print("Please set your OpenAI API key or modify the code to use other LLM providers")
        return
    
    try:
        # Run all demos
        await demo_basic_labeling()
        await demo_label_filtering()
        await demo_label_management()
        await demo_advanced_usage()
        
        print("\n🎉 All demos completed successfully!")
        print("\nFeatures demonstrated:")
        print("✅ Document insertion with labels")
        print("✅ Label-based query filtering")
        print("✅ Label management and statistics")
        print("✅ Advanced parameter configuration")
        print("\nYou can now:")
        print("- Organize documents by topic using labels")
        print("- Filter search results by specific labels")
        print("- Manage labels and track document organization")
        print("- Use both simple and advanced APIs")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())