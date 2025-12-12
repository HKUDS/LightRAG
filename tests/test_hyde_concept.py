#!/usr/bin/env python3
"""
Test HyDE (Hypothetical Document Embeddings) concept in isolation.

Compares retrieval quality between:
1. Standard: embed query directly
2. HyDE: embed hypothetical answers, average them

Uses the same embedding model and vector DB as LightRAG.
"""

import asyncio
import json
import os

import numpy as np
from openai import AsyncOpenAI

# Config
EMBEDDING_MODEL = 'text-embedding-3-large'
LLM_MODEL = 'gpt-4o-mini'
PG_HOST = os.getenv('POSTGRES_HOST', 'localhost')
PG_PORT = os.getenv('POSTGRES_PORT', '5433')
PG_USER = os.getenv('POSTGRES_USER', 'lightrag')
PG_PASS = os.getenv('POSTGRES_PASSWORD', 'lightrag_pass')
PG_DB = os.getenv('POSTGRES_DATABASE', 'lightrag')

client = AsyncOpenAI()


async def get_embedding(text: str) -> list[float]:
    """Get embedding for a single text."""
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
        dimensions=1536,  # Match DB dimension (text-embedding-3-large at 1536)
    )
    return response.data[0].embedding


async def generate_hypothetical_answers(query: str, num_docs: int = 3) -> list[str]:
    """Generate hypothetical answers using LLM."""
    prompt = f"""Generate {num_docs} brief hypothetical answers to this question.
Each answer should be 2-3 sentences of factual-sounding content that would answer the question.
Write as if you have the information - no hedging or "I don't know".

Question: {query}

Output valid JSON:
{{"hypothetical_answers": ["answer1", "answer2", "answer3"]}}"""

    response = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.7,
    )

    content = response.choices[0].message.content
    # Parse JSON from response
    try:
        # Handle markdown code blocks
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0]
        elif '```' in content:
            content = content.split('```')[1].split('```')[0]
        data = json.loads(content.strip())
        return data.get('hypothetical_answers', [])
    except json.JSONDecodeError as e:
        print(f'Failed to parse JSON: {e}')
        print(f'Raw content: {content}')
        return []


def average_embeddings(embeddings: list[list[float]]) -> list[float]:
    """Average multiple embedding vectors."""
    arr = np.array(embeddings)
    return arr.mean(axis=0).tolist()


async def search_chunks(embedding: list[float], top_k: int = 5) -> list[dict]:
    """Search chunks in PostgreSQL using the embedding."""
    import asyncpg

    conn = await asyncpg.connect(
        host=PG_HOST,
        port=int(PG_PORT),
        user=PG_USER,
        password=PG_PASS,
        database=PG_DB,
    )

    embedding_str = ','.join(map(str, embedding))

    query = f"""
        SELECT c.id,
               LEFT(c.content, 200) as content_preview,
               c.content_vector <=> '[{embedding_str}]'::vector as distance
        FROM lightrag_vdb_chunks c
        WHERE c.workspace = 'default'
        ORDER BY c.content_vector <=> '[{embedding_str}]'::vector
        LIMIT $1;
    """

    rows = await conn.fetch(query, top_k)
    await conn.close()

    return [{'id': r['id'], 'preview': r['content_preview'], 'distance': float(r['distance'])} for r in rows]


async def compare_retrieval(query: str):
    """Compare standard vs HyDE retrieval for a query."""
    print(f'\n{"=" * 80}')
    print(f'QUERY: {query}')
    print('=' * 80)

    # Standard retrieval
    print('\n📌 STANDARD (embed query directly):')
    query_embedding = await get_embedding(query)
    standard_results = await search_chunks(query_embedding, top_k=5)

    for i, r in enumerate(standard_results, 1):
        print(f'  {i}. [dist={r["distance"]:.4f}] {r["preview"][:100]}...')

    avg_standard_dist = np.mean([r['distance'] for r in standard_results])
    print(f'  → Avg distance: {avg_standard_dist:.4f}')

    # HyDE retrieval
    print('\n🔮 HYDE (embed hypothetical answers):')
    hypotheticals = await generate_hypothetical_answers(query, num_docs=3)

    print('  Generated hypotheticals:')
    for i, h in enumerate(hypotheticals, 1):
        print(f'    {i}. {h[:100]}...')

    # Embed hypotheticals and average
    hyde_embeddings = []
    for h in hypotheticals:
        emb = await get_embedding(h)
        hyde_embeddings.append(emb)

    hyde_embedding = average_embeddings(hyde_embeddings)
    hyde_results = await search_chunks(hyde_embedding, top_k=5)

    print('\n  Results:')
    for i, r in enumerate(hyde_results, 1):
        print(f'  {i}. [dist={r["distance"]:.4f}] {r["preview"][:100]}...')

    avg_hyde_dist = np.mean([r['distance'] for r in hyde_results])
    print(f'  → Avg distance: {avg_hyde_dist:.4f}')

    # Compare
    print('\n📊 COMPARISON:')
    improvement = avg_standard_dist - avg_hyde_dist
    pct = (improvement / avg_standard_dist) * 100 if avg_standard_dist > 0 else 0

    if improvement > 0:
        print(f'  ✅ HyDE is BETTER by {improvement:.4f} ({pct:.1f}% closer)')
    else:
        print(f'  ❌ Standard is BETTER by {-improvement:.4f} ({-pct:.1f}% closer)')

    # Check overlap
    standard_ids = {r['id'] for r in standard_results}
    hyde_ids = {r['id'] for r in hyde_results}
    overlap = len(standard_ids & hyde_ids)
    print(f'  📎 Overlap: {overlap}/5 chunks in common')

    return {
        'query': query,
        'standard_avg_dist': avg_standard_dist,
        'hyde_avg_dist': avg_hyde_dist,
        'improvement': improvement,
        'overlap': overlap,
    }


async def main():
    # Test queries from pharma dataset
    test_queries = [
        'What were the key lessons learned from the Isatuximab monoclonal antibody drug development program in April 2020?',
        'What CMC dossier lessons were learned from the PKU IND submission in 2023?',
        'What risk management strategies were discussed in the 2017 Risk Review CIR for CMC development programs?',
        'What biopharmacy considerations were discussed in the February 2022 CMC Cross Sharing session?',
        'What were the main challenges and lessons learned from the COVID-19 mRNA vaccine development?',
    ]

    results = []
    for query in test_queries:
        result = await compare_retrieval(query)
        results.append(result)

    # Summary
    print('\n' + '=' * 80)
    print('SUMMARY')
    print('=' * 80)

    hyde_wins = sum(1 for r in results if r['improvement'] > 0)
    avg_improvement = np.mean([r['improvement'] for r in results])
    avg_overlap = np.mean([r['overlap'] for r in results])

    print(f'HyDE wins: {hyde_wins}/{len(results)} queries')
    print(f'Avg distance improvement: {avg_improvement:.4f}')
    print(f'Avg overlap with standard: {avg_overlap:.1f}/5 chunks')

    if hyde_wins >= len(results) / 2:
        print('\n✅ HyDE shows promise - worth implementing!')
    else:
        print("\n⚠️ HyDE doesn't help much for these queries")


if __name__ == '__main__':
    asyncio.run(main())
