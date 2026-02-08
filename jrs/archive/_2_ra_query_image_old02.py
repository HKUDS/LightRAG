#!/usr/bin/env python

# For usage instructions enter the following command:
# python3 ~/LightRAG/jrs/_2_ra_query_image.py --help

import os
import argparse
import asyncio
from datetime import datetime

from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc


async def run_image_query(
    query_text, api_key, base_url, working_dir, modes, output_file, query_params
):
    try:
        config = RAGAnythingConfig(
            working_dir=working_dir,
            enable_image_processing=True,
            enable_table_processing=True,
        )

        def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            return openai_complete_if_cache(
                "gpt-4o-mini",
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )

        def vision_model_func(
            prompt, system_prompt=None, history_messages=[], image_data=None, **kwargs
        ):
            if image_data:
                return openai_complete_if_cache(
                    "gpt-4o",
                    "",
                    system_prompt=None,
                    history_messages=[],
                    messages=[
                        (
                            {"role": "system", "content": system_prompt}
                            if system_prompt
                            else None
                        ),
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data}"
                                    },
                                },
                            ],
                        },
                    ],
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
            return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        embedding_func = EmbeddingFunc(
            embedding_dim=3072,
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts,
                model="text-embedding-3-large",
                api_key=api_key,
                base_url=base_url,
            ),
        )

        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

        print("INFO: Initializing Multimodal Engine...")
        await rag._ensure_lightrag_initialized()

        # Timestamp for the log
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Start the Markdown Log entry
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"\n# Query Session: {timestamp}\n")
            f.write(f"**Query:** `{query_text}`\n")
            f.write(f"**Working Directory:** `{working_dir}`\n\n")

            # Create a Markdown Table for Parameters
            f.write("### Session Parameters\n")
            f.write("| Parameter | Value |\n")
            f.write("| :--- | :--- |\n")
            for param, value in query_params.items():
                f.write(f"| {param} | {value} |\n")
            f.write(f"| modes_tested | {', '.join(modes)} |\n\n")

        # --- MULTI-MODE QUERY LOOP ---
        for current_mode in modes:
            print(f"\n>>> Executing [ {current_mode.upper()} ] mode...")

            # Local update for the current mode
            query_params["mode"] = current_mode

            try:
                result = await rag.aquery_with_multimodal(query_text, **query_params)

                # Output to Console
                print(f"\n[ {current_mode.upper()} ANSWER ]:")
                print(f"{result}")

                # Output to Markdown File
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(f"## Analysis Mode: {current_mode.upper()}\n")
                    f.write(f"{result}\n\n")
                    f.write("---\n")

            except Exception as e:
                error_msg = f"Error in {current_mode} mode: {e}"
                print(error_msg)
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(f"### Analysis Mode: {current_mode.upper()} (FAILED)\n")
                    f.write(f"**Error:** {error_msg}\n\n")

        # --- CLEANUP ---
        if hasattr(rag, "finalize_storages"):
            res = rag.finalize_storages()
            if asyncio.iscoroutine(res):
                await res

        if hasattr(rag, "lightrag") and rag.lightrag:
            if hasattr(rag.lightrag, "storage") and hasattr(
                rag.lightrag.storage, "close"
            ):
                await rag.lightrag.storage.close()

        del rag

    except Exception as e:
        print(f"Query Error: {e}")


def main():
    description = """
LightRAG Multimodal Query Tool
------------------------------
This script performs advanced retrieval-augmented generation on indexed documents,
focusing on multimodal content. It logs all results and parameters to a Markdown file.
"""
    epilog = (
        "For more detailed documentation, visit: file:///home/js/LightRAG/jrs/_notes"
    )

    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("query", help="Your question about the images/charts")
    parser.add_argument(
        "--modes",
        "-m",
        default="hybrid",
        help="Comma-separated list of modes: naive,local,global,hybrid,mix,bypass",
    )
    parser.add_argument(
        "--output_file",
        "-o",
        default="/home/js/LightRAG/mm_query_output.md",
        help="Output MD file where LLM response is found",
    )
    parser.add_argument(
        "--working_dir",
        "-w",
        default="/home/js/LightRAG/jrs/work/seheult/_ra/nir_through_fabrics/_ra_seheult_work_dir",
        help="Index path",
    )

    # LightRAG Parameters
    parser.add_argument(
        "--response_type", default="Multiple Paragraphs", help="Response format"
    )
    parser.add_argument("--top_k", type=int, default=60, help="Top items to retrieve")
    parser.add_argument(
        "--chunk_top_k", type=int, default=20, help="Initial text chunks"
    )
    parser.add_argument(
        "--max_entity_tokens", type=int, default=6000, help="Max entity tokens"
    )
    parser.add_argument(
        "--max_relation_tokens", type=int, default=8000, help="Max relation tokens"
    )
    parser.add_argument(
        "--max_total_tokens", type=int, default=30000, help="Total token budget"
    )

    # Flags
    parser.add_argument(
        "--only_context", action="store_true", help="Only return context"
    )
    parser.add_argument("--only_prompt", action="store_true", help="Only return prompt")
    parser.add_argument("--stream", action="store_true", help="Enable streaming")
    parser.add_argument(
        "--disable_rerank",
        action="store_false",
        dest="enable_rerank",
        help="Disable reranking",
    )

    parser.add_argument("--user_prompt", help="Custom instructions for LLM")

    args = parser.parse_args()

    mode_list = [m.strip().lower() for m in args.modes.split(",")]

    # Dictionary to be unpacked as **kwargs
    query_params = {
        "only_need_context": args.only_context,
        "only_need_prompt": args.only_prompt,
        "response_type": args.response_type,
        "stream": args.stream,
        "top_k": args.top_k,
        "chunk_top_k": args.chunk_top_k,
        "max_entity_tokens": args.max_entity_tokens,
        "max_relation_tokens": args.max_relation_tokens,
        "max_total_tokens": args.max_total_tokens,
        "user_prompt": args.user_prompt,
        "enable_rerank": args.enable_rerank,
    }

    asyncio.run(
        run_image_query(
            args.query,
            os.getenv("OPENAI_API_KEY"),
            os.getenv("OPENAI_BASE_URL"),
            args.working_dir,
            mode_list,
            args.output_file,
            query_params,
        )
    )


if __name__ == "__main__":
    main()
