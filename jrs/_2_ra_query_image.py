#!/usr/bin/env python

# For usage instructions enter the following command:
# python3 path_to_this_script/_2_ra_query_image.py --help

import os
import argparse
import asyncio
import sys
import base64
from pathlib import Path

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

async def run_image_query(query_text, api_key, base_url, working_dir, modes, output_file):
    try:
        config = RAGAnythingConfig(
            working_dir=working_dir,
            enable_image_processing=True,
            enable_table_processing=True,
        )

        # 1. Text LLM
        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return openai_complete_if_cache(
                "gpt-4o-mini", prompt, system_prompt=system_prompt,
                history_messages=history_messages, api_key=api_key,
                base_url=base_url, **kwargs
            )

        # 2. Vision LLM (Crucial for Image Queries)
        def vision_model_func(prompt, system_prompt=None, history_messages=[], image_data=None, **kwargs):
            if image_data:
                return openai_complete_if_cache(
                    "gpt-4o", "", system_prompt=None, history_messages=[],
                    messages=[
                        {"role": "system", "content": system_prompt} if system_prompt else None,
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                        ]}
                    ],
                    api_key=api_key, base_url=base_url, **kwargs
                )
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        embedding_func = EmbeddingFunc(
            embedding_dim=3072, max_token_size=8192,
            func=lambda texts: openai_embed(
                texts, model="text-embedding-3-large", api_key=api_key, base_url=base_url
            ),
        )

        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func
        )

        print("INFO: Initializing Multimodal Engine...")
        await rag._ensure_lightrag_initialized()

        # Prepare Markdown File
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"\n# Query: {query_text}\n")
            f.write(f"**Working Dir:** `{working_dir}`\n\n")

        # --- MULTI-MODE QUERY LOOP ---
        for current_mode in modes:
            print(f"\n>>> Executing [ {current_mode.upper()} ] mode...")
            
            try:
                # query_with_multimodal is the method for reasoning over indexed visuals
                result = await rag.aquery_with_multimodal(query_text, mode=current_mode)
                
                # Output to Console
                print(f"\n[ {current_mode.upper()} ANSWER ]:")
                print(f"{result}")
                
                # Output to Markdown File
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

        # Cleanup attempts (preserving existing logic)
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
    parser = argparse.ArgumentParser(description="Multimodal Image Query Script")
    parser.add_argument("query", help="Your question about the images/charts")
    
    # Modes parameter: Split by comma to allow multiple (e.g., -m naive,hybrid)
    parser.add_argument("--modes", "-m", default="hybrid", 
                        help="Comma-separated list of modes: naive,local,global,hybrid,mix")
    
    # File parameter: Defaulting to LightRAG directory
    parser.add_argument("--file", "-f", default="/home/js/LightRAG/mm_query_output.md", 
                        help="Path to the output markdown file")
    
    parser.add_argument("--working_dir", "-w", 
                        default="/home/js/LightRAG/jrs/work/seheult/_ra/nir_through_fabrics/_ra_seheult_work_dir",
                        help="Path to directory where index of knowledge is stored")
    
    args = parser.parse_args()

    # Convert the comma-separated string into a clean Python list
    mode_list = [m.strip().lower() for m in args.modes.split(",")]

    asyncio.run(run_image_query(
        args.query, 
        os.getenv("OPENAI_API_KEY"), 
        os.getenv("OPENAI_BASE_URL"), 
        args.working_dir,
        mode_list,
        args.file
    ))

if __name__ == "__main__":
    main()