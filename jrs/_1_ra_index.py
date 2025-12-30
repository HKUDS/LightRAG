#!/usr/bin/env python
import os
import argparse
import asyncio
import sys
from pathlib import Path

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger

async def run_indexing(file_path, output_dir, api_key, base_url, working_dir):
    try:
        config = RAGAnythingConfig(
            working_dir=working_dir,
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return openai_complete_if_cache(
                "gpt-4o-mini", prompt, system_prompt=system_prompt,
                history_messages=history_messages, api_key=api_key,
                base_url=base_url, **kwargs
            )

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
            func=lambda texts: openai_embed(texts, model="text-embedding-3-large", api_key=api_key, base_url=base_url)
        )

        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func
        )

        print(f"--- Starting Indexing: {file_path} ---")
        await rag.process_document_complete(file_path=file_path, output_dir=output_dir, parse_method="auto")
        
        if hasattr(rag, 'close'):
            res = rag.close()
            if asyncio.iscoroutine(res): await res
        print("--- Indexing Success ---")

    except Exception as e:
        print(f"Indexing Error: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path", default="/home/js/LightRAG/jrs/work/seheult/_ra/nir_through_fabrics/_ra_seheult_docs/nir_through_fabrics.pdf")
    parser.add_argument("-w", "--working_dir", default="/home/js/LightRAG/jrs/work/seheult/_ra/nir_through_fabrics/_ra_seheult_work_dir")
    parser.add_argument("-o", "--output", default="/home/js/LightRAG/jrs/work/seheult/_ra/nir_through_fabrics/_ra_seheult_output_dir")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.working_dir, exist_ok=True)
    asyncio.run(run_indexing(args.file_path, args.output, args.api_key, args.base_url, args.working_dir))

if __name__ == "__main__":
    main()