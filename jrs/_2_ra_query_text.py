#!/usr/bin/env python

# For usage instructions enter the following command:
# python3 path_to_this_script/_2_ra_query_text.py --help

import os
import argparse
import asyncio
import sys
from pathlib import Path

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

async def run_text_query(query_text, api_key, base_url, working_dir, modes, output_file):
    try:
        # 1. Setup Config
        config = RAGAnythingConfig(
            working_dir=working_dir,
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        # 2. Setup LLM Function (Standard text completion)
        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return openai_complete_if_cache(
                "gpt-4o-mini", prompt, system_prompt=system_prompt,
                history_messages=history_messages, api_key=api_key,
                base_url=base_url, **kwargs
            )

        # 3. Setup Embedding Function (Must match the indexing phase)
        embedding_func = EmbeddingFunc(
            embedding_dim=3072, max_token_size=8192,
            func=lambda texts: openai_embed(
                texts, model="text-embedding-3-large", api_key=api_key, base_url=base_url
            ),
        )

        # 4. Initialize RAGAnything
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            embedding_func=embedding_func
        )

        # --- INITIALIZATION ---
        print("INFO: Connecting to existing index...")
        await rag._ensure_lightrag_initialized()

        if not rag.lightrag:
            raise RuntimeError(f"Failed to load LightRAG from {working_dir}.")

        # Prepare Markdown File Entry
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"\n# Text Query: {query_text}\n")
            f.write(f"**Working Dir:** `{working_dir}`\n\n")

        # --- MULTI-MODE QUERY LOOP ---
        for current_mode in modes:
            print(f"\n>>> Executing [ {current_mode.upper()} ] mode...")
            
            try:
                # Standard text query (aquery)
                result = await rag.aquery(query_text, mode=current_mode)
                
                # Console Output
                print(f"\n[ {current_mode.upper()} ANSWER ]:")
                print(f"{result}")
                
                # File Output
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(f"## Mode: {current_mode.upper()}\n")
                    f.write(f"{result}\n\n")
                    f.write("---\n")
                
            except Exception as e:
                error_msg = f"Error in {current_mode} mode: {e}"
                print(error_msg)
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(f"### Mode: {current_mode.upper()} (FAILED)\n")
                    f.write(f"Error: {error_msg}\n\n")

        # --- CLEANUP ---
        if hasattr(rag, 'finalize_storages'):
            res = rag.finalize_storages()
            if asyncio.iscoroutine(res): await res
        
        if hasattr(rag, 'lightrag') and rag.lightrag:
            if hasattr(rag.lightrag, 'storage') and hasattr(rag.lightrag.storage, 'close'):
                await rag.lightrag.storage.close()
        
        del rag

    except Exception as e:
        print(f"Query Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Multi-Mode Text Query Script")
    parser.add_argument("query", help="The question you want to ask about the text")
    
    # Modes parameter: Split by comma to allow multiple (e.g., -m naive,hybrid)
    parser.add_argument("--modes", "-m", default="hybrid", 
                        help="Comma-separated list of modes: naive,local,global,hybrid,mix")
    
    # File parameter: Defaulting to LightRAG directory
    parser.add_argument("--file", "-f", default="/home/js/LightRAG/text_query_output.md", 
                        help="Path to the output markdown file")
    
    parser.add_argument("--working_dir", "-w", 
                        default="/home/js/LightRAG/jrs/work/seheult/_ra/nir_through_fabrics/_ra_seheult_work_dir",
                        help="Path to directory where index of knowledge is stored")
    
    args = parser.parse_args()

    # Process the mode string into a list
    mode_list = [m.strip().lower() for m in args.modes.split(",")]

    asyncio.run(run_text_query(
        args.query, 
        os.getenv("OPENAI_API_KEY"), 
        os.getenv("OPENAI_BASE_URL"), 
        args.working_dir,
        mode_list,
        args.file
    ))

if __name__ == "__main__":
    main()