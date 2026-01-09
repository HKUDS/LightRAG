"""
Company Relationship Inference Demo

This example demonstrates how to use LightRAG's advanced company relationship
inference to automatically detect:
1. Competitor relationships between organizations
2. Partnership/collaboration relationships
3. Supply chain relationships

The system analyzes document structure and contextual indicators to infer
implicit business relationships that weren't explicitly stated in the text.
"""

import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, openai_embedding
from lightrag.utils import EmbeddingFunc


async def main():
    print("=" * 80)
    print("Company Relationship Inference Demo")
    print("=" * 80)

    # Initialize LightRAG with company relationship inference enabled
    print("\n1. Initializing LightRAG with company relationship inference...")

    rag = LightRAG(
        working_dir="./company_rag_demo",

        # === LLM Configuration ===
        llm_model_func=openai_complete_if_cache,
        llm_model_name="gpt-4o-mini",

        # === Embedding Configuration ===
        embedding_func=EmbeddingFunc(
            embedding_dim=1536,
            max_token_size=8192,
            func=lambda texts: openai_embedding(
                texts,
                model="text-embedding-3-small"
            )
        ),

        # === Chunking Strategy ===
        chunk_token_size=1200,
        chunk_overlap_token_size=400,  # 33% overlap for better entity co-occurrence

        # === Extraction Quality ===
        entity_extract_max_gleaning=2,

        # === Advanced Relationship Inference ===
        addon_params={
            # Standard co-occurrence inference
            "enable_cooccurrence_inference": True,
            "min_cooccurrence": 3,

            # Advanced company relationship inference
            "enable_company_relationship_inference": True,
            "company_inference_min_cooccurrence": 2,  # Lower threshold for companies
            "company_inference_confidence_threshold": 0.5,  # 50% confidence minimum

            "language": "English",
        },

        enable_llm_cache=True,
    )

    print("✓ LightRAG initialized with company relationship inference enabled")

    # === Example Documents ===
    # These documents demonstrate different company relationship scenarios

    print("\n2. Inserting example documents...")

    # Document 1: Competitors
    competitor_doc = """
    Summary: TechFlow Solutions is a leading software development company.
    Entity: TechFlow Solutions

    TechFlow Solutions specializes in enterprise software development and competes
    directly with CodeCraft Inc. and DevSphere in the same market segment.
    Both TechFlow and CodeCraft are market leaders in enterprise solutions,
    with similar product offerings and target customers.

    Industry analysts often compare TechFlow Solutions to CodeCraft Inc. when
    evaluating enterprise software vendors. These competitors vie for the same
    customer base in the Fortune 500 market.
    """

    # Document 2: Partnership
    partnership_doc = """
    Summary: DataCloud Systems provides cloud infrastructure solutions.
    Entity: DataCloud Systems

    DataCloud Systems has announced a strategic partnership with TechFlow Solutions
    to integrate their cloud infrastructure with TechFlow's enterprise software.
    The two companies will collaborate on joint projects and co-develop new products.

    This partnership between DataCloud and TechFlow creates a powerful ecosystem
    for enterprise customers, combining infrastructure and software expertise.
    """

    # Document 3: Supply Chain
    supply_chain_doc = """
    Summary: SecureAuth Corp specializes in authentication and security.
    Entity: SecureAuth Corp

    SecureAuth Corp supplies authentication modules to TechFlow Solutions,
    acting as a key vendor in TechFlow's product stack. TechFlow procures
    security components from SecureAuth for integration into their enterprise
    software platform.

    The customer-supplier relationship between TechFlow and SecureAuth has been
    stable for over 5 years.
    """

    # Document 4: Main company mentioned with others (ambiguous)
    ambiguous_doc = """
    Summary: InnovateTech is an emerging technology company.
    Entity: InnovateTech

    InnovateTech operates in the same technology sector as TechFlow Solutions
    and CodeCraft Inc. The company offers similar services and targets enterprise
    clients. Industry reports mention InnovateTech alongside TechFlow and CodeCraft
    when discussing the enterprise software landscape.
    """

    # Insert documents
    documents = {
        "competitors_doc": competitor_doc,
        "partnership_doc": partnership_doc,
        "supply_chain_doc": supply_chain_doc,
        "ambiguous_doc": ambiguous_doc,
    }

    for doc_name, doc_content in documents.items():
        print(f"   Inserting {doc_name}...")
        await rag.ainsert(doc_content)

    print("✓ All documents inserted")

    # === Query the Knowledge Graph ===
    print("\n3. Querying inferred relationships...")

    # Query for TechFlow's relationships
    query = "What are the relationships between TechFlow Solutions and other companies?"

    print(f"\nQuery: {query}")
    result = await rag.aquery(
        query,
        param=QueryParam(mode="hybrid", top_k=10)
    )

    print(f"\nResult:\n{result}")

    # === Inspect the Knowledge Graph ===
    print("\n" + "=" * 80)
    print("4. Knowledge Graph Analysis")
    print("=" * 80)

    # Access the graph storage directly
    graph = rag.chunk_entity_relation_graph

    # Get all relationships
    print("\nInferred Company Relationships:")
    print("-" * 80)

    # This would require accessing the actual graph storage
    # For demo purposes, we'll show what types of relationships were created

    relationship_types = {
        "competitor": "competes_with, competitor, market_rival",
        "partnership": "partners_with, collaborates_with, strategic_alliance",
        "supply_chain": "business_relationship, supply_chain, vendor_customer",
    }

    print("\nExpected Relationship Types:")
    for rel_type, keywords in relationship_types.items():
        print(f"  • {rel_type.upper()}: {keywords}")

    print("\nRelationships that should be inferred:")
    print("  1. TechFlow Solutions ←→ CodeCraft Inc. [COMPETITOR]")
    print("     Evidence: 'competes directly', 'market leaders', 'similar product offerings'")
    print()
    print("  2. TechFlow Solutions ←→ DevSphere [COMPETITOR]")
    print("     Evidence: 'competes', 'same market segment'")
    print()
    print("  3. DataCloud Systems ←→ TechFlow Solutions [PARTNERSHIP]")
    print("     Evidence: 'strategic partnership', 'collaborate', 'co-develop'")
    print()
    print("  4. SecureAuth Corp ←→ TechFlow Solutions [SUPPLY_CHAIN]")
    print("     Evidence: 'supplies', 'vendor', 'customer-supplier relationship'")
    print()
    print("  5. InnovateTech ←→ TechFlow Solutions [COMPETITOR - lower confidence]")
    print("     Evidence: 'same sector', 'similar services', 'mentioned alongside'")

    print("\n" + "=" * 80)
    print("5. Configuration Summary")
    print("=" * 80)
    print("""
Key Configuration Parameters:

1. enable_company_relationship_inference: True
   - Enables advanced company relationship detection

2. company_inference_min_cooccurrence: 2
   - Companies must appear together in at least 2 chunks

3. company_inference_confidence_threshold: 0.5
   - Requires 50% confidence based on contextual indicators

4. chunk_overlap_token_size: 400 (33%)
   - Ensures entities appear together in multiple chunks

Relationship Detection Logic:

COMPETITOR Indicators:
- Explicit: "competitor", "rival", "versus", "competes with"
- Implicit: "similar to", "alternative", "same market"

PARTNERSHIP Indicators:
- Explicit: "partner", "collaboration", "joint venture"
- Implicit: "ecosystem", "alliance", "integrates with"

SUPPLY_CHAIN Indicators:
- "supplier", "vendor", "customer", "provides to", "procures"

The system analyzes sentences where both companies are mentioned
and counts contextual indicators to determine relationship type
and confidence level.
    """)

    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    # Set up OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        print("Usage: export OPENAI_API_KEY='your-api-key'")
        exit(1)

    # Run the demo
    asyncio.run(main())
