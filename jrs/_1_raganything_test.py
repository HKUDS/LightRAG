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
import logging
import logging.config
from pathlib import Path

# Add project root directory to Python path
import sys

sys.path.append(str(Path(__file__).parent.parent))

# Ensures the script can find local LightRAG modules if they aren't in site-packages
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from raganything import RAGAnything, RAGAnythingConfig


def configure_logging():
    """Configure logging for the application"""
    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "raganything_example.log"))

    print(f"\nRAGAnything example log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE", "false").lower() == "true")


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
        working_dir: Working directory for RAG storage
    """
    try:
        # Create RAGAnything configuration
        config = RAGAnythingConfig(
            working_dir=working_dir or "./rag_storage",
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        # Define LLM model function
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

        # Define vision model function for image processing
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
            else:
                return llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        # Define embedding function
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

        # Initialize RAGAnything with new dataclass structure
        rag = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            vision_model_func=vision_model_func,
            embedding_func=embedding_func,
        )

        # Process document
        await rag.process_document_complete(
            file_path=file_path, output_dir=output_dir, parse_method="auto"
        )

        # Example queries - demonstrating different query approaches
        logger.info("\nQuerying processed document:")

        # 1. Pure text queries using aquery()
        text_queries = [
            "What is the main content of the document?",
            "What are the key topics discussed?",
        ]

        for query in text_queries:
            logger.info(f"\n[Text Query]: {query}")
            result = await rag.aquery(query, mode="hybrid")
            logger.info(f"Answer: {result}")

        
        # 2. Multimodal query with specific multimodal content using aquery_with_multimodal()
        # logger.info(
        #     "\n[Multimodal Query]: Analyzing performance data in context of document"
        # )
        # multimodal_result = await rag.aquery_with_multimodal(
        #     "Compare this performance data with any similar results mentioned in the document",
        #     multimodal_content=[
        #         {
        #             "type": "table",
        #             "table_data": """Method,Accuracy,Processing_Time
        #                         RAGAnything,95.2%,120ms
        #                         Traditional_RAG,87.3%,180ms
        #                         Baseline,82.1%,200ms""",
        #             "table_caption": "Performance comparison results",
        #         }
        #     ],
        #     mode="hybrid",
        # )
        # logger.info(f"Answer: {multimodal_result}")
        

        
        # 3. Another multimodal query with equation content
        # logger.info("\n[Multimodal Query]: Mathematical formula analysis")
        # equation_result = await rag.aquery_with_multimodal(
        #     "Explain this formula and relate it to any mathematical concepts in the document",
        #     multimodal_content=[
        #         {
        #             "type": "equation",
        #             "latex": "F1 = 2 \\cdot \\frac{precision \\cdot recall}{precision + recall}",
        #             "equation_caption": "F1-score calculation formula",
        #         }
        #     ],
        #     mode="hybrid",
        # )
        # logger.info(f"Answer: {equation_result}")
        

        # Finalize RAGAnything storages.
        if hasattr(rag, 'close'):
            # Try to call it normally; if it's a coroutine, it will be handled
            result = rag.close()
            if asyncio.iscoroutine(result):
                await result
        
        logger.info("RAG processing and cleanup completed successfully.")


    except Exception as e:
        logger.error(f"Error processing with RAG: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())


def main():
    """Main function to run the example with flexible arguments"""
    parser = argparse.ArgumentParser(description="MinerU RAG Example")
    
    # 1. Input Argument (The file to be indexed)
    parser.add_argument(
        "--file_path", "-f",
        default="/home/js/LightRAG/jrs/work/seheult/_ra/nir_through_fabrics/_ra_seheult_docs/nir_through_fabrics.pdf",
        help="Path to the document to process"
    )
    
    # 2. Working Directory Arguments
    parser.add_argument(                   
        "--working_dir", "-w", 
        default=os.getenv("RAG_WORKING_DIR", "/home/js/LightRAG/jrs/work/seheult/_ra/nir_through_fabrics/_ra_seheult_work_dir"), 
        help="Working directory path"
    )
    
    # 3. Output Files Directory Arguments
    parser.add_argument(
        "--output", "-o", 
        default=os.getenv("RAG_OUTPUT_DIR", "/home/js/LightRAG/jrs/work/seheult/_ra/nir_through_fabrics/_ra_seheult_output_dir"), 
        help="Output directory path"
    )
    
    # 4. API Key Argument
    parser.add_argument(
        "--api-key", 
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key (defaults to OPENAI_API_KEY env var)"
    )

    # 5. Base URL Argument (Optional, for proxy or local LLM endpoints)
    parser.add_argument(
        "--base-url", "-b",
        default=os.getenv("OPENAI_BASE_URL"),
        help="Optional base URL for API (e.g., https://api.openai.com/v1)"
    )    

    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Print configuration and exit without processing"
    )    

    args = parser.parse_args()


    # Check if the input file exists
    # We use .resolve() to handle relative paths like './my_input_file.pdf'
    input_path = Path(args.file_path).resolve()
    file_exists = input_path.exists()


    if args.dry_run:
            print("\n=== DRY RUN MODE ===")
            print(f"File to process: {args.file_path}")
            print(f"Output Dir:      {args.output}")
            print(f"Working Dir:     {args.working_dir}")
            print(f"API Key:         {'LOADED' if args.api_key else 'MISSING'}")
            print(f"Base URL:        {args.base_url}")
            print("====================\n")
            print("Configuration looks good. Remove --dry-run to start processing.")
            file_exists = os.path.exists(args.file_path)
            print(f"Input File Exists?:     {'YES, The input file exists.' if file_exists else 'NO (Check your file path!)'}")
            return


    # 3. Guard Clause: Stop the script if not a dry run and file is missing
    if not file_exists:
        print(f"\nFATAL ERROR: The file '{args.file_path}' does not exist.")
        print(f"Resolved path: {input_path}")
        print("Please provide a valid path using -f or --file_path.\n")
        sys.exit(1) # Terminates the program immediately            

    # Create directories (only if not a dry run)
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.working_dir, exist_ok=True)


    # Priority Logic for file_path: Command line > Env Var > Error
    file_to_process = args.file_path or os.getenv("RAG_FILE_PATH")
    
    if not file_to_process:
        logger.error("Error: No file path provided via argument or RAG_FILE_PATH env var.")
        return

    if not args.api_key:
        logger.error("Error: OpenAI API key is required.")
        return

    # Run the RAG process
    asyncio.run(
        process_with_rag(
            args.file_path, args.output, args.api_key, args.base_url, args.working_dir
        )
    )


if __name__ == "__main__":
    # Configure logging first
    configure_logging()

    print("RAGAnything Example")
    print("=" * 30)
    print("Processing document with multimodal RAG pipeline")
    print("=" * 30)

    main()