#!/usr/bin/env python3
"""
Diagnostic tool to check LightRAG initialization status.

This tool helps developers verify that their LightRAG instance is properly
initialized and ready to use. It should be called AFTER initialize_storages()
to validate that all components are correctly set up.

Usage:
    # Basic usage in your code:
    rag = LightRAG(...)
    await rag.initialize_storages()
    await check_lightrag_setup(rag, verbose=True)

    # Run demo from command line:
    python -m lightrag.tools.check_initialization --demo
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lightrag import LightRAG
from lightrag.base import StoragesStatus
from lightrag.utils import logger


async def check_lightrag_setup(rag_instance: LightRAG, verbose: bool = False) -> bool:
    """
    Check if a LightRAG instance is properly initialized.

    Args:
        rag_instance: The LightRAG instance to check
        verbose: If True, print detailed diagnostic information

    Returns:
        True if properly initialized, False otherwise
    """
    issues = []
    warnings = []

    logger.info('🔍 Checking LightRAG initialization status...')

    # Check storage initialization status
    if not hasattr(rag_instance, '_storages_status'):
        issues.append('LightRAG instance missing _storages_status attribute')
    elif rag_instance._storages_status != StoragesStatus.INITIALIZED:
        issues.append(f'Storages not initialized (status: {rag_instance._storages_status.name})')
    else:
        logger.info('✅ Storage status: INITIALIZED')

    # Check individual storage components
    storage_components = [
        ('full_docs', 'Document storage'),
        ('text_chunks', 'Text chunks storage'),
        ('entities_vdb', 'Entity vector database'),
        ('relationships_vdb', 'Relationship vector database'),
        ('chunks_vdb', 'Chunks vector database'),
        ('doc_status', 'Document status tracker'),
        ('llm_response_cache', 'LLM response cache'),
        ('full_entities', 'Entity storage'),
        ('full_relations', 'Relation storage'),
        ('chunk_entity_relation_graph', 'Graph storage'),
    ]

    if verbose:
        logger.debug('📦 Storage Components:')

    for component, description in storage_components:
        if not hasattr(rag_instance, component):
            issues.append(f'Missing storage component: {component} ({description})')
        else:
            storage = getattr(rag_instance, component)
            if storage is None:
                warnings.append(f'Storage {component} is None (might be optional)')
            elif hasattr(storage, '_storage_lock'):
                if storage._storage_lock is None:
                    issues.append(f'Storage {component} not initialized (lock is None)')
                elif verbose:
                    logger.debug('  ✅ %s: Ready', description)
            elif verbose:
                logger.debug('  ✅ %s: Ready', description)

        # Check pipeline status
    try:
        from lightrag.kg.shared_storage import get_namespace_data

        await get_namespace_data('pipeline_status', workspace=rag_instance.workspace)
        logger.info('✅ Pipeline status: INITIALIZED')
    except KeyError:
        issues.append('Pipeline status not initialized - call rag.initialize_storages() first')
    except Exception as e:
        issues.append(f'Error checking pipeline status: {e!s}')

    # Print results
    logger.info('=' * 50)

    if issues:
        logger.error('❌ Issues found:')
        for issue in issues:
            logger.error('  • %s', issue)

        logger.info('📝 To fix, run this initialization sequence:')
        logger.info('  await rag.initialize_storages()')
        logger.info('📚 Documentation: https://github.com/HKUDS/LightRAG#important-initialization-requirements')

        if warnings and verbose:
            logger.warning('⚠️  Warnings (might be normal):')
            for warning in warnings:
                logger.warning('  • %s', warning)

        return False
    else:
        logger.info('✅ LightRAG is properly initialized and ready to use!')

        if warnings and verbose:
            logger.warning('⚠️  Warnings (might be normal):')
            for warning in warnings:
                logger.warning('  • %s', warning)

        return True


async def demo():
    """Demonstrate the diagnostic tool with a test instance."""
    from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed

    logger.info('=' * 50)
    logger.info('LightRAG Initialization Diagnostic Tool')
    logger.info('=' * 50)

    # Create test instance
    rag = LightRAG(
        working_dir='./test_diagnostic',
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )

    logger.info('🔄 Initializing storages...')
    await rag.initialize_storages()  # Auto-initializes pipeline_status

    logger.info('🔍 Checking initialization status...')
    await check_lightrag_setup(rag, verbose=True)

    # Cleanup
    import shutil

    try:
        shutil.rmtree('./test_diagnostic')
    except Exception as e:
        logger.warning('Failed to clean demo directory ./test_diagnostic: %s', e)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Check LightRAG initialization status')
    parser.add_argument('--demo', action='store_true', help='Run a demonstration with a test instance')
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Show detailed diagnostic information',
    )

    args = parser.parse_args()

    if args.demo:
        asyncio.run(demo())
    else:
        logger.info('Run with --demo to see the diagnostic tool in action')
        logger.info('Or import this module and use check_lightrag_setup() with your instance')
