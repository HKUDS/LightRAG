#!/usr/bin/env python3
"""
Debug prompt responses - see exactly what the LLM produces.

This helps identify WHY certain prompts fail on faithfulness or relevance.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from openai import AsyncOpenAI
import httpx


# Import variants from test script
from test_prompt_variants import PROMPT_VARIANTS


async def get_context(query: str, server_url: str = 'http://localhost:9621') -> str:
    """Get context from LightRAG server."""
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f'{server_url}/query',
            json={'query': query, 'mode': 'mix', 'only_need_context': True}
        )
        return response.json().get('response', '')


async def call_llm(prompt: str, client: AsyncOpenAI) -> str:
    """Call LLM and return response."""
    response = await client.chat.completions.create(
        model=os.getenv('LLM_MODEL', 'gpt-4o-mini'),
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.1,
        max_tokens=2000,
    )
    return response.choices[0].message.content


async def main():
    # Load dataset
    dataset_path = Path(__file__).parent / 'pharma_test_dataset.json'
    with open(dataset_path) as f:
        data = json.load(f)

    if isinstance(data, dict) and 'test_cases' in data:
        dataset = data['test_cases']
    else:
        dataset = data

    # Create client
    client = AsyncOpenAI(
        api_key=os.getenv('LLM_BINDING_API_KEY') or os.getenv('OPENAI_API_KEY'),
        base_url=os.getenv('LLM_BINDING_HOST', 'https://api.openai.com/v1'),
    )

    # Test specific queries that had issues
    test_queries = [
        dataset[0],  # Q1 - Isatuximab (0.0 relevance issue)
        dataset[1],  # Q2 - PKU IND
        dataset[5],  # Q6 - Japanese iCMC (low relevance in full eval)
        dataset[8],  # Q9 - Risk management (low in full eval)
    ]

    variants_to_test = ['baseline', 'mipro']

    for i, q in enumerate(test_queries):
        question = q['question']
        ground_truth = q['ground_truth']

        print(f"\n{'='*80}")
        print(f"QUERY {i+1}: {question[:70]}...")
        print(f"{'='*80}")

        # Get context once
        context = await get_context(question)

        if not context or 'No relevant context' in context:
            print("⚠️ NO CONTEXT RETRIEVED")
            continue

        print(f"\n📄 CONTEXT LENGTH: {len(context)} chars, {len(context.split())} words")
        print(f"📋 GROUND TRUTH (first 200 chars):\n{ground_truth[:200]}...")

        for variant_name in variants_to_test:
            template = PROMPT_VARIANTS[variant_name]
            prompt = template.format(context_data=context, user_prompt=question)

            print(f"\n--- {variant_name.upper()} RESPONSE ---")

            response = await call_llm(prompt, client)

            # Show response
            print(f"Length: {len(response)} chars, {len(response.split())} words")
            print(f"\nFull response:\n{response[:1000]}...")

            # Quick analysis
            print(f"\n📊 Quick Analysis:")

            # Check if response uses question terms
            q_terms = set(question.lower().split())
            r_terms = set(response.lower().split())
            term_overlap = len(q_terms & r_terms)
            print(f"  - Question term overlap: {term_overlap}/{len(q_terms)}")

            # Check response structure
            has_numbers = any(c.isdigit() for c in response[:200])
            has_bullets = any(x in response for x in ['•', '-', '1.', '(1)', '1)'])
            print(f"  - Has structure (numbers/bullets): {has_numbers or has_bullets}")

            # Check for hedging
            hedging = any(x in response.lower() for x in [
                "i'm sorry", "cannot answer", "no information", "not specified",
                "does not contain", "unable to"
            ])
            print(f"  - Contains hedging/refusal: {hedging}")


if __name__ == '__main__':
    asyncio.run(main())
