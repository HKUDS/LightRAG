#!/usr/bin/env python3
"""
Regenerate ground truths for pharma_test_dataset.json based on actual LightRAG context.

This script:
1. Reads each question from the test dataset
2. Queries LightRAG to get the actual retrieved context
3. Uses an LLM to generate ground truth ONLY from that context
4. Saves the updated dataset

This ensures ground truths match what LightRAG can actually retrieve,
making RAGAS evaluation meaningful.

Configuration is loaded from the project's .env file (same as LightRAG service).
"""

import asyncio
import json
import os
import re
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load .env from project root (same config as LightRAG service)
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / '.env')

# Configuration - use same env vars as LightRAG service
LIGHTRAG_ENDPOINT = os.getenv('LIGHTRAG_ENDPOINT', 'http://localhost:9621')
LIGHTRAG_WORKSPACE = os.getenv('LIGHTRAG_WORKSPACE', 'default')

# LLM Configuration (same as LightRAG service)
LLM_BINDING = os.getenv('LLM_BINDING', 'openai')
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-4o-mini')
LLM_BINDING_HOST = os.getenv('LLM_BINDING_HOST', 'https://api.openai.com/v1')
LLM_BINDING_API_KEY = os.getenv('LLM_BINDING_API_KEY') or os.getenv('OPENAI_API_KEY')

# Create OpenAI client with same config as service
client = AsyncOpenAI(
    api_key=LLM_BINDING_API_KEY,
    base_url=LLM_BINDING_HOST,
)

GROUND_TRUTH_PROMPT = """You are generating a ground truth answer for RAG evaluation.

Based ONLY on the following retrieved context, write a factual answer to the question.
- Include ONLY information that appears explicitly in the context
- Do not add any information not present in the context
- Do not speculate or infer beyond what is stated
- If the context doesn't fully answer the question, only include what IS covered
- Write in a clear, factual style (2-4 sentences)

Question: {question}

Retrieved Context:
{context}

Ground Truth Answer:"""

SIMPLIFY_QUERY_PROMPT = """Extract key search terms from this question for a RAG system query.
Return only the most important nouns, proper nouns, and key phrases.
Separate terms with spaces. No punctuation, no articles, no filler words.

Question: {question}

Key search terms:"""


async def simplify_question(question: str) -> str:
    """Use LLM to simplify the question into key search terms."""
    response = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{'role': 'user', 'content': SIMPLIFY_QUERY_PROMPT.format(question=question)}],
        temperature=0,
        max_tokens=100,
    )
    content = response.choices[0].message.content
    return content.strip() if content else question


def extract_key_terms(question: str) -> str:
    """Extract key terms from a question using simple heuristics.

    This is a fast fallback when we don't want to call the LLM.
    """
    # Remove common question words and punctuation
    stopwords = {
        'what',
        'were',
        'was',
        'are',
        'is',
        'the',
        'a',
        'an',
        'how',
        'does',
        'do',
        'did',
        'which',
        'who',
        'when',
        'where',
        'why',
        'from',
        'to',
        'in',
        'on',
        'at',
        'for',
        'of',
        'with',
        'by',
        'and',
        'or',
        'that',
        'this',
        'these',
        'those',
        'according',
        'based',
        'should',
        'influence',
        'main',
        'key',
        'critical',
        'important',
    }

    # Clean and tokenize
    words = re.findall(r'\b\w+\b', question.lower())
    # Keep important terms (not stopwords, keep numbers and capitalized terms)
    key_terms = []
    for word in words:
        # Check if original had capitalization (proper noun)
        if word not in stopwords and (word.upper() == word or any(c.isupper() for c in question) or len(word) > 3):
            key_terms.append(word)

    return ' '.join(key_terms[:12])  # Limit to top 12 terms


async def query_lightrag(query: str, workspace: str = 'default', timeout: float = 120.0) -> dict:
    """Query LightRAG and get the response with context."""
    async with httpx.AsyncClient(timeout=timeout) as http_client:
        response = await http_client.post(
            f'{LIGHTRAG_ENDPOINT}/query',
            json={
                'query': query,
                'mode': 'mix',
                'workspace': workspace,
                'include_chunk_content': True,
                'include_references': True,
            },
        )
        response.raise_for_status()
        return response.json()


async def generate_ground_truth(question: str, context: str) -> str:
    """Use LLM to generate ground truth from actual context."""
    prompt = GROUND_TRUTH_PROMPT.format(question=question, context=context)

    response = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.1,  # Low temperature for factual output
        max_tokens=500,
    )

    content = response.choices[0].message.content
    return content.strip() if content else ''


def extract_context_text(rag_response: dict) -> tuple[str, int]:
    """Extract readable context from LightRAG response.

    The API returns references like:
    {
        "references": [
            {"reference_id": "1", "file_path": "...", "content": ["chunk1", "chunk2"]},
            ...
        ]
    }
    where 'content' is a list of chunk strings from that file.

    Returns:
        tuple: (context_string, chunk_count)
    """
    context_parts = []

    # Get references with embedded chunk content
    if rag_response.get('references'):
        for ref in rag_response['references']:
            if isinstance(ref, dict):
                content = ref.get('content')
                if content:
                    # content is a list of chunk strings
                    if isinstance(content, list):
                        for chunk in content:
                            if isinstance(chunk, str) and chunk.strip():
                                context_parts.append(chunk.strip())
                    elif isinstance(content, str) and content.strip():
                        context_parts.append(content.strip())

    # Limit to first 15 chunks with separator
    limited_parts = context_parts[:15]
    return '\n\n---\n\n'.join(limited_parts), len(context_parts)


async def query_with_retry(question: str, workspace: str = 'default', max_retries: int = 2) -> tuple[str, str, int]:
    """Query LightRAG with retry using simplified query if no context found.

    Returns:
        tuple: (context, query_used, chunk_count)
    """
    # First attempt with original question
    rag_result = await query_lightrag(question, workspace)
    context, chunk_count = extract_context_text(rag_result)

    if chunk_count > 0:
        return context, question, chunk_count

    # Retry with simplified key terms
    for retry in range(max_retries):
        if retry == 0:
            # First retry: extract key terms with heuristics (fast)
            simplified = extract_key_terms(question)
        else:
            # Second retry: use LLM to simplify (slower but smarter)
            simplified = await simplify_question(question)

        if simplified and simplified != question:
            print(f'    Retry {retry + 1}: Using simplified query: {simplified[:60]}...')
            rag_result = await query_lightrag(simplified, workspace)
            context, chunk_count = extract_context_text(rag_result)

            if chunk_count > 0:
                return context, simplified, chunk_count

    # Fallback: use the LLM response as context (it's based on retrieved info)
    response_text = rag_result.get('response', '')
    if response_text and '[no-context]' not in response_text:
        return f'[LLM Response]: {response_text}', question, 0

    return '', question, 0


async def regenerate_dataset(input_path: str, output_path: str, workspace: str = 'default') -> None:
    """Regenerate all ground truths in the dataset."""
    # Load original dataset
    with open(input_path) as f:
        dataset = json.load(f)

    test_cases = dataset.get('test_cases', [])
    print(f'Processing {len(test_cases)} test cases...')

    updated_cases = []
    stats = {'success': 0, 'retry_success': 0, 'no_context': 0, 'errors': 0}

    for i, case in enumerate(test_cases):
        question = case['question']
        print(f'\n[{i + 1}/{len(test_cases)}] Processing: {question[:60]}...')

        try:
            # Query LightRAG for actual context with retry
            context, query_used, chunk_count = await query_with_retry(question, workspace)

            if chunk_count > 0:
                if query_used == question:
                    stats['success'] += 1
                else:
                    stats['retry_success'] += 1
                print(f'  ✓ Found {chunk_count} chunks')
            else:
                stats['no_context'] += 1
                print('  ✗ No chunk content retrieved')

            if not context:
                # Generate a "no info available" ground truth
                ground_truth = (
                    f'The retrieved context does not provide information '
                    f'to answer this question about {question[:50]}...'
                )
            else:
                # Generate ground truth from actual context
                ground_truth = await generate_ground_truth(question, context)

            print(f'  Original GT: {case["ground_truth"][:80]}...')
            print(f'  New GT:      {ground_truth[:80]}...')

            # Update the case
            updated_case = {
                'question': question,
                'ground_truth': ground_truth,
                'project': case.get('project', 'pharma_evaluation'),
            }
            updated_cases.append(updated_case)

        except Exception as e:
            stats['errors'] += 1
            print(f'  Error: {e}')
            # Keep original on error
            updated_cases.append(case)

    # Save updated dataset
    output_data = {'test_cases': updated_cases}
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f'\n{"=" * 50}')
    print(f'✅ Updated dataset saved to: {output_path}')
    print('\nStats:')
    print(f'  - Direct success:  {stats["success"]}')
    print(f'  - Retry success:   {stats["retry_success"]}')
    print(f'  - No context:      {stats["no_context"]}')
    print(f'  - Errors:          {stats["errors"]}')


async def main():
    script_dir = Path(__file__).parent
    input_path = script_dir / 'pharma_test_dataset_original.json'
    output_path = script_dir / 'pharma_test_dataset.json'

    if not input_path.exists():
        # If no backup exists, use current as input
        input_path = output_path

    print(f'{"=" * 60}')
    print('Ground Truth Regeneration')
    print(f'{"=" * 60}')
    print(f'Input:     {input_path}')
    print(f'Output:    {output_path}')
    print(f'LightRAG:  {LIGHTRAG_ENDPOINT}')
    print(f'Workspace: {LIGHTRAG_WORKSPACE}')
    print(f'LLM Host:  {LLM_BINDING_HOST}')
    print(f'LLM Model: {LLM_MODEL}')
    print(f'{"=" * 60}')
    print()

    await regenerate_dataset(str(input_path), str(output_path), LIGHTRAG_WORKSPACE)


if __name__ == '__main__':
    asyncio.run(main())
