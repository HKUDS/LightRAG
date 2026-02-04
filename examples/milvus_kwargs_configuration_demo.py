"""
Example: Configuring Milvus Index Parameters via vector_db_storage_cls_kwargs

This example demonstrates how to configure Milvus indexing parameters through
vector_db_storage_cls_kwargs, which is the recommended approach when using
frameworks that build on top of LightRAG (like RAGAnything).

This approach allows configuration to be passed through framework layers without
requiring environment variable changes or direct code modifications.
"""

import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed


async def main():
    # Configure Milvus connection
    os.environ["MILVUS_URI"] = "http://localhost:19530"
    # os.environ["MILVUS_USER"] = "root"
    # os.environ["MILVUS_PASSWORD"] = "your_password"
    # os.environ["MILVUS_DB_NAME"] = "lightrag"

    # Initialize LightRAG with Milvus index configuration via vector_db_storage_cls_kwargs
    # This is the recommended approach for framework integration (e.g., RAGAnything)
    rag = LightRAG(
        working_dir="./demo_index",
        llm_model_func=openai_complete_if_cache,
        embedding_func=openai_embed,
        # Specify Milvus as the vector storage backend
        vector_storage="MilvusVectorDBStorage",
        # Configure Milvus indexing parameters via vector_db_storage_cls_kwargs
        # These parameters are extracted and passed to MilvusIndexConfig
        vector_db_storage_cls_kwargs={
            # Required parameter for all vector storage backends
            "cosine_better_than_threshold": 0.2,
            
            # Milvus index configuration parameters
            # All of these can be configured via vector_db_storage_cls_kwargs
            
            # Index type (AUTOINDEX, HNSW, HNSW_SQ, IVF_FLAT, etc.)
            "index_type": "HNSW",
            
            # Distance metric (COSINE, L2, IP)
            "metric_type": "COSINE",
            
            # HNSW parameters
            "hnsw_m": 32,                    # Number of connections per layer (2-2048)
            "hnsw_ef_construction": 256,     # Size of dynamic candidate list during construction
            "hnsw_ef": 150,                  # Size of dynamic candidate list during search
            
            # IVF parameters (used when index_type is IVF_FLAT, IVF_SQ8, IVF_PQ)
            # "ivf_nlist": 2048,              # Number of cluster units
            # "ivf_nprobe": 32,               # Number of units to query
            
            # HNSW_SQ parameters (requires Milvus 2.6.8+)
            # "sq_type": "SQ8",               # Quantization type (SQ4U, SQ6, SQ8, BF16, FP16)
            # "sq_refine": True,              # Enable refinement
            # "sq_refine_type": "FP32",       # Refinement type
            # "sq_refine_k": 20,              # Number of candidates to refine
        },
    )

    # Initialize storage backends
    await rag.initialize_storages()

    print("✅ LightRAG initialized with Milvus index configuration via vector_db_storage_cls_kwargs")
    print(f"   Index Type: {rag.vector_db_storages['entities'].index_config.index_type}")
    print(f"   Metric Type: {rag.vector_db_storages['entities'].index_config.metric_type}")
    print(f"   HNSW M: {rag.vector_db_storages['entities'].index_config.hnsw_m}")
    print(f"   HNSW EF Construction: {rag.vector_db_storages['entities'].index_config.hnsw_ef_construction}")
    print(f"   HNSW EF: {rag.vector_db_storages['entities'].index_config.hnsw_ef}")

    # Example: Insert some text
    sample_text = """
    LightRAG is a Retrieval-Augmented Generation framework that uses graph-based
    knowledge representation for enhanced information retrieval. It supports multiple
    vector storage backends including Milvus, which offers advanced indexing options
    for optimal performance.
    """
    
    await rag.ainsert(sample_text)
    print("\n✅ Sample text inserted")

    # Example: Query with different modes
    result = await rag.aquery(
        "What is LightRAG?",
        param=QueryParam(mode="hybrid")
    )
    print(f"\n✅ Query result: {result[:200]}...")

    # Cleanup
    await rag.finalize_storages()


if __name__ == "__main__":
    print("=" * 80)
    print("Milvus Configuration via vector_db_storage_cls_kwargs Example")
    print("=" * 80)
    print()
    print("This example shows how to configure Milvus indexing parameters through")
    print("vector_db_storage_cls_kwargs, which is ideal for framework integration.")
    print()
    print("Key Benefits:")
    print("  • No environment variable changes required")
    print("  • Configuration can be passed through framework layers")
    print("  • Perfect for RAGAnything and similar frameworks")
    print("  • All 11 index parameters are supported")
    print()
    print("=" * 80)
    print()
    
    asyncio.run(main())
