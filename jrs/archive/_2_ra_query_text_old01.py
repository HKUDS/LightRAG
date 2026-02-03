#!/usr/bin/env python
import os
import argparse
import asyncio
import sys
from pathlib import Path

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

async def run_query(query_text, api_key, base_url, working_dir, mode):
    try:
        # 1. Setup Config
        config = RAGAnythingConfig(
            working_dir=working_dir,
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        # 2. Setup LLM Function
        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return openai_complete_if_cache(
                "gpt-4o-mini", prompt, system_prompt=system_prompt,
                history_messages=history_messages, api_key=api_key,
                base_url=base_url, **kwargs
            )

        # 3. Setup Embedding Function (must match indexing)
        embedding_func = EmbeddingFunc(
            embedding_dim=3072, max_token_size=8192,
            func=lambda texts: openai_embed(
                texts, model="text-embedding-3-large", api_key=api_key, base_url=base_url
            ),
        )

        # 4. Initialize RAGAnything Wrapper
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            embedding_func=embedding_func
        )

        # --- THE FIXES ---
        # 1. Use the correct internal initialization method with 'await'
        print("INFO: Connecting to existing index...")
        await rag._ensure_lightrag_initialized()

        # 2. Verify the 'lightrag' attribute is now populated
        if not rag.lightrag:
            raise RuntimeError(f"Failed to load LightRAG from {working_dir}. Check if files exist there.")

        print(f"\n--- Querying [{mode} mode]: {query_text} ---")
        
        # 3. Perform the actual query
        result = await rag.aquery(query_text, mode=mode)
        print(f"\nANSWER:\n{result}\n")

        # 4. Safely finalize storages (handling both sync and async possibilities)
        if hasattr(rag, 'finalize_storages'):
            res = rag.finalize_storages()
            if asyncio.iscoroutine(res):
                await res
        print("INFO: Storage finalized successfully.")

    except Exception as e:
        print(f"Query Error: {e}")
        # print(traceback.format_exc()) # Uncomment if you need deeper debugging

def main():
    parser = argparse.ArgumentParser(description="RAG Query Script")
    parser.add_argument("query", help="The question you want to ask")
    parser.add_argument("--working_dir", "-w", 
                        default="/home/js/LightRAG/jrs/work/seheult/_ra/nir_through_fabrics/_ra_seheult_work_dir")
    parser.add_argument("--mode", "-m", default="hybrid", choices=["naive", "local", "global", "hybrid", "mix"])
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--base-url", "-b", default=os.getenv("OPENAI_BASE_URL"))

    args = parser.parse_args()

    # Verify directory exists before starting
    if not Path(args.working_dir).exists():
        print(f"FATAL: Working directory {args.working_dir} not found.")
        sys.exit(1)

    # Execute the async run
    asyncio.run(run_query(args.query, args.api_key, args.base_url, args.working_dir, args.mode))

if __name__ == "__main__":
    main()