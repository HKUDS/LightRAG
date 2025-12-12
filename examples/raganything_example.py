#!/usr/bin/env python
"""
Example script demonstrating the integration of MinerU parser with RAGAnything

This example shows how to:
1. Process parsed documents with RAGAnything
2. Perform multimodal queries on the processed documents
3. Handle different types of content (text, images, tables)
"""

import argparse
import asyncio
import logging
import logging.config
import os
from pathlib import Path

from raganything import RAGAnything, RAGAnythingConfig

from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug


def configure_logging():
    """Configure logging for the application"""
    # Get log directory path from environment variable or use current directory
    log_dir = Path(os.getenv('LOG_DIR', os.getcwd()))
    log_file_path = log_dir / 'raganything_example.log'

    logger.info('RAGAnything example log file: %s', log_file_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv('LOG_MAX_BYTES', 10485760))  # Default 10MB
    log_backup_count = int(os.getenv('LOG_BACKUP_COUNT', 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'default': {
                    'format': '%(levelname)s: %(message)s',
                },
                'detailed': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                },
            },
            'handlers': {
                'console': {
                    'formatter': 'default',
                    'class': 'logging.StreamHandler',
                    'stream': 'ext://sys.stderr',
                },
                'file': {
                    'formatter': 'detailed',
                    'class': 'logging.handlers.RotatingFileHandler',
                    'filename': log_file_path,
                    'maxBytes': log_max_bytes,
                    'backupCount': log_backup_count,
                    'encoding': 'utf-8',
                },
            },
            'loggers': {
                'lightrag': {
                    'handlers': ['console', 'file'],
                    'level': 'INFO',
                    'propagate': False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv('VERBOSE', 'false').lower() == 'true')


async def process_with_rag(
    file_path: str,
    output_dir: str,
    api_key: str,
    base_url: str | None = None,
    working_dir: str | None = None,
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
            working_dir=working_dir or './rag_storage',
            mineru_parse_method='auto',
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )

        # Define LLM model function
        def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
            if history_messages is None:
                history_messages = []
            return openai_complete_if_cache(
                'gpt-4o-mini',
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                **kwargs,
            )

        # Define vision model function for image processing
        def vision_model_func(prompt, system_prompt=None, history_messages=None, image_data=None, **kwargs):
            if history_messages is None:
                history_messages = []
            if image_data:
                return openai_complete_if_cache(
                    'gpt-4o',
                    '',
                    system_prompt=None,
                    history_messages=[],
                    messages=[
                        {'role': 'system', 'content': system_prompt} if system_prompt else None,
                        {
                            'role': 'user',
                            'content': [
                                {'type': 'text', 'text': prompt},
                                {
                                    'type': 'image_url',
                                    'image_url': {'url': f'data:image/jpeg;base64,{image_data}'},
                                },
                            ],
                        }
                        if image_data
                        else {'role': 'user', 'content': prompt},
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
                model='text-embedding-3-large',
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
        await rag.process_document_complete(file_path=file_path, output_dir=output_dir, parse_method='auto')

        # Example queries - demonstrating different query approaches
        logger.info('\nQuerying processed document:')

        # 1. Pure text queries using aquery()
        text_queries = [
            'What is the main content of the document?',
            'What are the key topics discussed?',
        ]

        for query in text_queries:
            logger.info(f'\n[Text Query]: {query}')
            result = await rag.aquery(query, mode='hybrid')
            logger.info(f'Answer: {result}')

        # 2. Multimodal query with specific multimodal content using aquery_with_multimodal()
        logger.info('\n[Multimodal Query]: Analyzing performance data in context of document')
        multimodal_result = await rag.aquery_with_multimodal(
            'Compare this performance data with any similar results mentioned in the document',
            multimodal_content=[
                {
                    'type': 'table',
                    'table_data': """Method,Accuracy,Processing_Time
                                RAGAnything,95.2%,120ms
                                Traditional_RAG,87.3%,180ms
                                Baseline,82.1%,200ms""",
                    'table_caption': 'Performance comparison results',
                }
            ],
            mode='hybrid',
        )
        logger.info(f'Answer: {multimodal_result}')

        # 3. Another multimodal query with equation content
        logger.info('\n[Multimodal Query]: Mathematical formula analysis')
        equation_result = await rag.aquery_with_multimodal(
            'Explain this formula and relate it to any mathematical concepts in the document',
            multimodal_content=[
                {
                    'type': 'equation',
                    'latex': 'F1 = 2 \\cdot \\frac{precision \\cdot recall}{precision + recall}',
                    'equation_caption': 'F1-score calculation formula',
                }
            ],
            mode='hybrid',
        )
        logger.info(f'Answer: {equation_result}')

    except Exception as e:
        logger.error(f'Error processing with RAG: {e!s}')
        import traceback

        logger.error(traceback.format_exc())


def main():
    """Main function to run the example"""
    parser = argparse.ArgumentParser(description='MinerU RAG Example')
    parser.add_argument('file_path', help='Path to the document to process')
    parser.add_argument('--working_dir', '-w', default='./rag_storage', help='Working directory path')
    parser.add_argument('--output', '-o', default='./output', help='Output directory path')
    parser.add_argument(
        '--api-key',
        default=os.getenv('OPENAI_API_KEY'),
        help='OpenAI API key (defaults to OPENAI_API_KEY env var)',
    )
    parser.add_argument('--base-url', help='Optional base URL for API')

    args = parser.parse_args()

    # Check if API key is provided
    if not args.api_key:
        logger.error('Error: OpenAI API key is required')
        logger.error('Set OPENAI_API_KEY environment variable or use --api-key option')
        return

    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    # Process with RAG
    asyncio.run(process_with_rag(args.file_path, args.output, args.api_key, args.base_url, args.working_dir))


if __name__ == '__main__':
    # Configure logging first
    configure_logging()

    print('RAGAnything Example')
    print('=' * 30)
    print('Processing document with multimodal RAG pipeline')
    print('=' * 30)

    main()
