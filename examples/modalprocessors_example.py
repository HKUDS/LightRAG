"""
Example of directly using modal processors

This example demonstrates how to use LightRAG's modal processors directly without going through MinerU.
"""

import argparse
import asyncio

from raganything.modalprocessors import (
    EquationModalProcessor,
    ImageModalProcessor,
    TableModalProcessor,
)

from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

WORKING_DIR = './rag_storage'


def get_llm_model_func(api_key: str, base_url: str | None = None):
    return lambda prompt, system_prompt=None, history_messages=[], **kwargs: openai_complete_if_cache(
        'gpt-4o-mini',
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=api_key,
        base_url=base_url,
        **kwargs,
    )


def get_vision_model_func(api_key: str, base_url: str | None = None):
    return (
        lambda prompt, system_prompt=None, history_messages=[], image_data=None, **kwargs: openai_complete_if_cache(
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
        if image_data
        else openai_complete_if_cache(
            'gpt-4o-mini',
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
    )


async def process_image_example(lightrag: LightRAG, vision_model_func):
    """Example of processing an image"""
    # Create image processor
    image_processor = ImageModalProcessor(lightrag=lightrag, modal_caption_func=vision_model_func)

    # Prepare image content
    image_content = {
        'img_path': 'image.jpg',
        'img_caption': ['Example image caption'],
        'img_footnote': ['Example image footnote'],
    }

    # Process image
    description, entity_info = await image_processor.process_multimodal_content(
        modal_content=image_content,
        content_type='image',
        file_path='image_example.jpg',
        entity_name='Example Image',
    )

    print('Image Processing Results:')
    print(f'Description: {description}')
    print(f'Entity Info: {entity_info}')


async def process_table_example(lightrag: LightRAG, llm_model_func):
    """Example of processing a table"""
    # Create table processor
    table_processor = TableModalProcessor(lightrag=lightrag, modal_caption_func=llm_model_func)

    # Prepare table content
    table_content = {
        'table_body': """
        | Name | Age | Occupation |
        |------|-----|------------|
        | John | 25  | Engineer   |
        | Mary | 30  | Designer   |
        """,
        'table_caption': ['Employee Information Table'],
        'table_footnote': ['Data updated as of 2024'],
    }

    # Process table
    description, entity_info = await table_processor.process_multimodal_content(
        modal_content=table_content,
        content_type='table',
        file_path='table_example.md',
        entity_name='Employee Table',
    )

    print('\nTable Processing Results:')
    print(f'Description: {description}')
    print(f'Entity Info: {entity_info}')


async def process_equation_example(lightrag: LightRAG, llm_model_func):
    """Example of processing a mathematical equation"""
    # Create equation processor
    equation_processor = EquationModalProcessor(lightrag=lightrag, modal_caption_func=llm_model_func)

    # Prepare equation content
    equation_content = {'text': 'E = mc^2', 'text_format': 'LaTeX'}

    # Process equation
    description, entity_info = await equation_processor.process_multimodal_content(
        modal_content=equation_content,
        content_type='equation',
        file_path='equation_example.txt',
        entity_name='Mass-Energy Equivalence',
    )

    print('\nEquation Processing Results:')
    print(f'Description: {description}')
    print(f'Entity Info: {entity_info}')


async def initialize_rag(api_key: str, base_url: str | None = None):
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=EmbeddingFunc(
            embedding_dim=3072,
            max_token_size=8192,
            func=lambda texts: openai_embed(
                texts,
                model='text-embedding-3-large',
                api_key=api_key,
                base_url=base_url,
            ),
        ),
        llm_model_func=lambda prompt, system_prompt=None, history_messages=[], **kwargs: openai_complete_if_cache(
            'gpt-4o-mini',
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        ),
    )

    await rag.initialize_storages()  # Auto-initializes pipeline_status
    return rag


def main():
    """Main function to run the example"""
    parser = argparse.ArgumentParser(description='Modal Processors Example')
    parser.add_argument('--api-key', required=True, help='OpenAI API key')
    parser.add_argument('--base-url', help='Optional base URL for API')
    parser.add_argument('--working-dir', '-w', default=WORKING_DIR, help='Working directory path')

    args = parser.parse_args()

    # Run examples
    asyncio.run(main_async(args.api_key, args.base_url))


async def main_async(api_key: str, base_url: str | None = None):
    # Initialize LightRAG
    lightrag = await initialize_rag(api_key, base_url)

    # Get model functions
    llm_model_func = get_llm_model_func(api_key, base_url)
    vision_model_func = get_vision_model_func(api_key, base_url)

    # Run examples
    await process_image_example(lightrag, vision_model_func)
    await process_table_example(lightrag, llm_model_func)
    await process_equation_example(lightrag, llm_model_func)


if __name__ == '__main__':
    main()
