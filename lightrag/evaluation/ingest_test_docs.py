#!/usr/bin/env python3
"""
Ingest test documents into LightRAG for testing.

This script reads text files from a directory and batch-uploads them to
LightRAG via the /documents/texts API endpoint, then polls for completion.

Usage:
    python lightrag/evaluation/ingest_test_docs.py
    python lightrag/evaluation/ingest_test_docs.py --input wiki_documents/ --rag-url http://localhost:9622
"""

import argparse
import asyncio
import os
import time
from pathlib import Path

import httpx

DEFAULT_RAG_URL = 'http://localhost:9622'


async def ingest_documents(
    input_dir: Path,
    rag_url: str,
) -> dict:
    """Ingest all text files from directory into LightRAG.

    Args:
        input_dir: Directory containing .txt or .md files
        rag_url: LightRAG API base URL

    Returns:
        Dict with ingestion statistics
    """
    timeout = httpx.Timeout(120.0, connect=30.0)
    api_key = os.getenv('LIGHTRAG_API_KEY')
    headers = {'X-API-Key': api_key} if api_key else {}

    async with httpx.AsyncClient(timeout=timeout) as client:
        # Check health
        try:
            health = await client.get(f'{rag_url}/health')
            if health.status_code != 200:
                raise ConnectionError(f'LightRAG not healthy: {health.status_code}')
        except httpx.ConnectError as e:
            raise ConnectionError(f'Cannot connect to LightRAG at {rag_url}') from e

        print(f'✓ Connected to LightRAG at {rag_url}')

        # Collect all text files
        files = list(input_dir.glob('*.txt')) + list(input_dir.glob('*.md'))
        if not files:
            print(f'✗ No .txt or .md files found in {input_dir}')
            return {'documents': 0, 'elapsed_seconds': 0}

        print(f'  Found {len(files)} documents to ingest')

        # Read all texts
        texts = []
        sources = []
        for file in sorted(files):
            content = file.read_text()
            texts.append(content)
            sources.append(file.name)
            word_count = len(content.split())
            print(f'    {file.name}: {word_count:,} words')

        # Batch ingest via /documents/texts
        print(f'\n  Uploading {len(texts)} documents...')
        start = time.time()

        response = await client.post(
            f'{rag_url}/documents/texts',
            json={'texts': texts, 'file_sources': sources},
            headers=headers,
        )
        response.raise_for_status()
        result = response.json()

        track_id = result.get('track_id', '')
        print(f'  Track ID: {track_id}')

        # Poll for completion - wait for processing to start first
        print('  Waiting for processing to start...')
        await asyncio.sleep(2)  # Give server time to queue documents

        last_status = ''
        processed_count = 0
        initial_check = True

        while True:
            status_response = await client.get(f'{rag_url}/documents')
            docs = status_response.json()
            statuses = docs.get('statuses', {})

            processing = len(statuses.get('processing', []))
            pending = len(statuses.get('pending', []))
            processed = len(statuses.get('processed', []))

            current_status = f'Pending: {pending}, Processing: {processing}, Processed: {processed}'
            if current_status != last_status:
                print(f'    {current_status}')
                last_status = current_status
                processed_count = processed

            # Wait until we see at least some of our docs in the queue
            if initial_check and (pending > 0 or processing > 0):
                initial_check = False
                print('  Processing started!')

            # Only exit when processing is done AND we've processed something new
            if processing == 0 and pending == 0 and not initial_check:
                break

            await asyncio.sleep(5)

        elapsed = time.time() - start
        print(f'\n✓ Ingestion complete in {elapsed:.1f}s')
        print(f'  Documents processed: {processed_count}')
        print(f'  Average: {elapsed / len(texts):.1f}s per document')

        return {
            'documents': len(texts),
            'processed': processed_count,
            'elapsed_seconds': elapsed,
            'track_id': track_id,
        }


async def main():
    parser = argparse.ArgumentParser(description='Ingest test documents into LightRAG')
    parser.add_argument(
        '--input',
        '-i',
        type=str,
        default='lightrag/evaluation/wiki_documents',
        help='Input directory with text files',
    )
    parser.add_argument(
        '--rag-url',
        '-r',
        type=str,
        default=None,
        help=f'LightRAG API URL (default: {DEFAULT_RAG_URL})',
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    rag_url = args.rag_url or os.getenv('LIGHTRAG_API_URL', DEFAULT_RAG_URL)

    print('=== LightRAG Document Ingestion ===')
    print(f'Input: {input_dir}/')
    print(f'RAG URL: {rag_url}')
    print()

    if not input_dir.exists():
        print(f'✗ Input directory not found: {input_dir}')
        print('  Run download_wikipedia.py first:')
        print('    python lightrag/evaluation/download_wikipedia.py')
        return

    await ingest_documents(input_dir, rag_url)


if __name__ == '__main__':
    asyncio.run(main())
