#!/usr/bin/env python
"""
Example script demonstrating the integration of MinerU parser with RAGAnything

This example shows how to:
1. Process parsed documents with RAGAnything
2. Perform multimodal queries on the processed documents
3. Handle different types of content (text, images, tables)
"""

import os
import argparse
import asyncio
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.raganything import RAGAnything


async def process_with_rag(
    file_path: str,
    output_dir: str,
    api_key: str,
    base_url: str = None,
    working_dir: str = None,
):
    """
    Process document with RAGAnything

    Args:
        file_path: Path to the document
        output_dir: Output directory for RAG results
        api_key: OpenAI API key
        base_url: Optional base URL for API
    """
    try:
        # Initialize RAGAnything
        rag = RAGAnything(
            working_dir=working_dir,
            llm_model_func=lambda prompt,
            system_prompt=None,
            history_messages=[],
            **kwargs: openai_complete_if_cache(
                "gpt-4o-mini",
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            ),
            vision_model_func=lambda prompt,
            system_prompt=None,
            history_messages=[],
            image_data=None,
            **kwargs: openai_complete_if_cache(
                "gpt-4o",
                "",
                system_prompt=None,
                history_messages=[],
                messages=[
                    {"role": "system", "content": system_prompt}
                    if system_prompt
                    else None,
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
                    }
                    if image_data
                    else {"role": "user", "content": prompt},
                ],
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )
            if image_data
            else openai_complete_if_cache(
                "gpt-4o-mini",
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            ),
            embedding_func=lambda texts: openai_embed(
                texts,
                model="text-embedding-3-large",
                api_key=api_key,
                base_url=base_url,
            ),
            embedding_dim=3072,
            max_token_size=8192,
        )

        # Process document
        await rag.process_document_complete(
            file_path=file_path, output_dir=output_dir, parse_method="auto"
        )

        # Example queries
        queries = [
            "What is the main content of the document?",
            "Describe the images and figures in the document",
            "Tell me about the experimental results and data tables",
        ]

        print("\nQuerying processed document:")
        for query in queries:
            print(f"\nQuery: {query}")
            result = await rag.query_with_multimodal(query, mode="hybrid")
            print(f"Answer: {result}")

    except Exception as e:
        print(f"Error processing with RAG: {str(e)}")


def main():
    """Main function to run the example"""
    parser = argparse.ArgumentParser(description="MinerU RAG Example")
    parser.add_argument("file_path", help="Path to the document to process")
    parser.add_argument(
        "--working_dir", "-w", default="./rag_storage", help="Working directory path"
    )
    parser.add_argument(
        "--output", "-o", default="./output", help="Output directory path"
    )
    parser.add_argument(
        "--api-key", required=True, help="OpenAI API key for RAG processing"
    )
    parser.add_argument("--base-url", help="Optional base URL for API")

    args = parser.parse_args()

    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Process with RAG
    asyncio.run(
        process_with_rag(
            args.file_path, args.output, args.api_key, args.base_url, args.working_dir
        )
    )


if __name__ == "__main__":
    main()
