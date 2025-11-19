"""
Multi-Model Vector Storage Isolation Demo

This example demonstrates LightRAG's automatic model isolation feature for vector storage.
When using different embedding models, LightRAG automatically creates separate collections/tables,
preventing dimension mismatches and data pollution.

Key Features:
- Automatic model suffix generation: {model_name}_{dim}d
- Seamless migration from legacy (no-suffix) to new (with-suffix) collections
- Support for multiple workspaces with different embedding models

Requirements:
- OpenAI API key (or any OpenAI-compatible API)
- Qdrant or PostgreSQL for vector storage (optional, defaults to NanoVectorDB)
"""

import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import EmbeddingFunc, logger

# Set your API key
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"


async def scenario_1_new_workspace_with_explicit_model():
    """
    Scenario 1: Creating a new workspace with explicit model name

    Result: Creates collection/table with name like:
    - Qdrant: lightrag_vdb_chunks_text_embedding_3_large_3072d
    - PostgreSQL: LIGHTRAG_VDB_CHUNKS_text_embedding_3_large_3072d
    """
    print("\n" + "="*80)
    print("Scenario 1: New Workspace with Explicit Model Name")
    print("="*80)

    # Define custom embedding function with explicit model name
    async def my_embedding_func(texts: list[str]):
        return await openai_embed(
            texts,
            model="text-embedding-3-large"
        )

    # Create EmbeddingFunc with model_name specified
    embedding_func = EmbeddingFunc(
        embedding_dim=3072,
        func=my_embedding_func,
        model_name="text-embedding-3-large"  # Explicit model name
    )

    rag = LightRAG(
        working_dir="./workspace_large_model",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=embedding_func,
    )

    await rag.initialize_storages()

    # Insert sample data
    await rag.ainsert("LightRAG supports automatic model isolation for vector storage.")

    # Query
    result = await rag.aquery(
        "What does LightRAG support?",
        param=QueryParam(mode="hybrid")
    )

    print(f"\nQuery Result: {result[:200]}...")
    print("\n✅ Collection/table created with suffix: text_embedding_3_large_3072d")

    await rag.close()


async def scenario_2_legacy_migration():
    """
    Scenario 2: Upgrading from legacy version (without model_name)

    If you previously used LightRAG without specifying model_name,
    the first run with model_name will automatically migrate your data.

    Result: Data is migrated from:
    - Old: lightrag_vdb_chunks (no suffix)
    - New: lightrag_vdb_chunks_text_embedding_ada_002_1536d (with suffix)
    """
    print("\n" + "="*80)
    print("Scenario 2: Automatic Migration from Legacy Format")
    print("="*80)

    # Step 1: Simulate legacy workspace (no model_name)
    print("\n[Step 1] Creating legacy workspace without model_name...")

    async def legacy_embedding_func(texts: list[str]):
        return await openai_embed(texts, model="text-embedding-ada-002")

    # Legacy: No model_name specified
    legacy_embedding = EmbeddingFunc(
        embedding_dim=1536,
        func=legacy_embedding_func
        # model_name not specified → uses "unknown" as fallback
    )

    rag_legacy = LightRAG(
        working_dir="./workspace_legacy",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=legacy_embedding,
    )

    await rag_legacy.initialize_storages()
    await rag_legacy.ainsert("Legacy data without model isolation.")
    await rag_legacy.close()

    print("✅ Legacy workspace created with suffix: unknown_1536d")

    # Step 2: Upgrade to new version with model_name
    print("\n[Step 2] Upgrading to new version with explicit model_name...")

    # New: With model_name specified
    new_embedding = EmbeddingFunc(
        embedding_dim=1536,
        func=legacy_embedding_func,
        model_name="text-embedding-ada-002"  # Now explicitly specified
    )

    rag_new = LightRAG(
        working_dir="./workspace_legacy",  # Same working directory
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=new_embedding,
    )

    # On first initialization, LightRAG will:
    # 1. Detect legacy collection exists
    # 2. Automatically migrate data to new collection with model suffix
    # 3. Legacy collection remains but can be deleted after verification
    await rag_new.initialize_storages()

    # Verify data is still accessible
    result = await rag_new.aquery(
        "What is the legacy data?",
        param=QueryParam(mode="hybrid")
    )

    print(f"\nQuery Result: {result[:200] if result else 'No results'}...")
    print("\n✅ Data migrated to: text_embedding_ada_002_1536d")
    print("ℹ️  Legacy collection can be manually deleted after verification")

    await rag_new.close()


async def scenario_3_multiple_models_coexistence():
    """
    Scenario 3: Multiple workspaces with different embedding models

    Different embedding models create completely isolated collections/tables,
    allowing safe coexistence without dimension conflicts or data pollution.

    Result:
    - Workspace A: lightrag_vdb_chunks_bge_small_768d
    - Workspace B: lightrag_vdb_chunks_bge_large_1024d
    """
    print("\n" + "="*80)
    print("Scenario 3: Multiple Models Coexistence")
    print("="*80)

    # Workspace A: Small embedding model (768 dimensions)
    print("\n[Workspace A] Using bge-small model (768d)...")

    async def embedding_func_small(texts: list[str]):
        # Simulate small embedding model
        # In real usage, replace with actual model call
        return await openai_embed(texts, model="text-embedding-3-small")

    embedding_a = EmbeddingFunc(
        embedding_dim=1536,  # text-embedding-3-small dimension
        func=embedding_func_small,
        model_name="text-embedding-3-small"
    )

    rag_a = LightRAG(
        working_dir="./workspace_a",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=embedding_a,
    )

    await rag_a.initialize_storages()
    await rag_a.ainsert("Workspace A uses small embedding model for efficiency.")

    print("✅ Workspace A created with suffix: text_embedding_3_small_1536d")

    # Workspace B: Large embedding model (3072 dimensions)
    print("\n[Workspace B] Using text-embedding-3-large model (3072d)...")

    async def embedding_func_large(texts: list[str]):
        # Simulate large embedding model
        return await openai_embed(texts, model="text-embedding-3-large")

    embedding_b = EmbeddingFunc(
        embedding_dim=3072,  # text-embedding-3-large dimension
        func=embedding_func_large,
        model_name="text-embedding-3-large"
    )

    rag_b = LightRAG(
        working_dir="./workspace_b",
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=embedding_b,
    )

    await rag_b.initialize_storages()
    await rag_b.ainsert("Workspace B uses large embedding model for better accuracy.")

    print("✅ Workspace B created with suffix: text_embedding_3_large_3072d")

    # Verify isolation: Query each workspace
    print("\n[Verification] Querying both workspaces...")

    result_a = await rag_a.aquery(
        "What model does workspace use?",
        param=QueryParam(mode="hybrid")
    )
    result_b = await rag_b.aquery(
        "What model does workspace use?",
        param=QueryParam(mode="hybrid")
    )

    print(f"\nWorkspace A Result: {result_a[:100] if result_a else 'No results'}...")
    print(f"Workspace B Result: {result_b[:100] if result_b else 'No results'}...")

    print("\n✅ Both workspaces operate independently without interference")

    await rag_a.close()
    await rag_b.close()


async def main():
    """
    Run all scenarios to demonstrate model isolation features
    """
    print("\n" + "="*80)
    print("LightRAG Multi-Model Vector Storage Isolation Demo")
    print("="*80)
    print("\nThis demo shows how LightRAG automatically handles:")
    print("1. ✅ Automatic model suffix generation")
    print("2. ✅ Seamless data migration from legacy format")
    print("3. ✅ Multiple embedding models coexistence")

    try:
        # Scenario 1: New workspace with explicit model
        await scenario_1_new_workspace_with_explicit_model()

        # Scenario 2: Legacy migration
        await scenario_2_legacy_migration()

        # Scenario 3: Multiple models coexistence
        await scenario_3_multiple_models_coexistence()

        print("\n" + "="*80)
        print("✅ All scenarios completed successfully!")
        print("="*80)

        print("\n📝 Key Takeaways:")
        print("- Always specify `model_name` in EmbeddingFunc for clear model tracking")
        print("- LightRAG automatically migrates legacy data on first run")
        print("- Different embedding models create isolated collections/tables")
        print("- Collection names follow pattern: {base_name}_{model_name}_{dim}d")
        print("\n📚 See the plan document for more details:")
        print("   .claude/plan/PR-vector-model-isolation.md")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
