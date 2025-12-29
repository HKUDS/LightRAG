#!/usr/bin/env python
import os
import argparse
import asyncio
import sys
from pathlib import Path

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

async def run_image_query(query_text, api_key, base_url, working_dir):
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
            # If the RAG engine finds a relevant image in the index, it passes it here
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

        # query_with_multimodal is the method for reasoning over indexed visuals
        print(f"\n--- Multimodal Analysis: {query_text} ---")
        
        # Note: We use 'hybrid' mode to check both the graph relationships 
        # (visuals) and the vector similarity (text captions).
        result = await rag.aquery_with_multimodal(query_text, mode="hybrid")
        
        print(f"\nVISUAL ANALYSIS ANSWER:\n{result}\n")

        if hasattr(rag, 'finalize_storages'):
            res = rag.finalize_storages()
            if asyncio.iscoroutine(res): await res

    except Exception as e:
        print(f"Query Error: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Your question about the images/charts")
    parser.add_argument("-w", "--working_dir", 
                        default="/home/js/LightRAG/jrs/work/seheult/_ra/nir_through_fabrics/_ra_seheult_work_dir")
    args = parser.parse_args()

    asyncio.run(run_image_query(args.query, os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_BASE_URL"), args.working_dir))

if __name__ == "__main__":
    main()


# test question:
# python3 /home/js/LightRAG/jrs/_1_ra_image_query.py "Explain what is happening in the Figure that shows how various fabrics and the amount of fabric layers attenuate NIR intensity."    