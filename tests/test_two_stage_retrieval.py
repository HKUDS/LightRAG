#!/usr/bin/env python3
"""
Test two-stage retrieval concept in isolation.

Current LightRAG:
- Entity search: top_k=60, no reranking
- Chunk search: top_k=30, then rerank (but threshold=0.0)

Test concept:
- Stage 1: Retrieve 5x more candidates (top_k=150 for chunks)
- Stage 2: Rerank and keep top 30

Compare retrieval quality between:
1. Standard: top_k=30 chunks
2. Two-stage: top_k=150 → rerank → top 30
"""

import asyncio
import os

import numpy as np
from openai import AsyncOpenAI
from sentence_transformers import CrossEncoder

# Config
EMBEDDING_MODEL = 'text-embedding-3-large'
PG_HOST = os.getenv('POSTGRES_HOST', 'localhost')
PG_PORT = os.getenv('POSTGRES_PORT', '5433')
PG_USER = os.getenv('POSTGRES_USER', 'lightrag')
PG_PASS = os.getenv('POSTGRES_PASSWORD', 'lightrag_pass')
PG_DB = os.getenv('POSTGRES_DATABASE', 'lightrag')

client = AsyncOpenAI()

# Load reranker model (same as LightRAG uses)
print('Loading reranker model...')
reranker = CrossEncoder('mixedbread-ai/mxbai-rerank-xsmall-v1')
print('Reranker loaded!')


async def get_embedding(text: str) -> list[float]:
    """Get embedding for a single text."""
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
        dimensions=1536,
    )
    return response.data[0].embedding


async def search_chunks(embedding: list[float], top_k: int = 30) -> list[dict]:
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
               c.content,
               c.content_vector <=> '[{embedding_str}]'::vector as distance
        FROM lightrag_vdb_chunks c
        WHERE c.workspace = 'default'
        ORDER BY c.content_vector <=> '[{embedding_str}]'::vector
        LIMIT $1;
    """

    rows = await conn.fetch(query, top_k)
    await conn.close()

    return [{'id': r['id'], 'content': r['content'], 'distance': float(r['distance'])} for r in rows]


def rerank_chunks(query: str, chunks: list[dict], top_n: int = 30) -> list[dict]:
    """Rerank chunks using CrossEncoder."""
    if not chunks:
        return []

    # Prepare pairs for reranking
    pairs = [(query, chunk['content']) for chunk in chunks]

    # Get scores
    scores = reranker.predict(pairs)

    # Add scores to chunks
    for i, chunk in enumerate(chunks):
        chunk['rerank_score'] = float(scores[i])

    # Sort by rerank score (higher is better)
    sorted_chunks = sorted(chunks, key=lambda x: x['rerank_score'], reverse=True)

    return sorted_chunks[:top_n]


async def compare_retrieval(query: str):
    """Compare standard vs two-stage retrieval."""
    print(f'\n{"=" * 80}')
    print(f'QUERY: {query}')
    print('=' * 80)

    # Get query embedding
    query_embedding = await get_embedding(query)

    # Standard retrieval (top_k=30)
    print('\n📌 STANDARD (top_k=30, no reranking):')
    standard_chunks = await search_chunks(query_embedding, top_k=30)

    print(f'  Retrieved {len(standard_chunks)} chunks')
    print('  Top 5 by vector distance:')
    for i, c in enumerate(standard_chunks[:5], 1):
        preview = c['content'][:100].replace('\n', ' ')
        print(f'    {i}. [dist={c["distance"]:.4f}] {preview}...')

    np.mean([c['distance'] for c in standard_chunks[:10]])

    # Two-stage retrieval (top_k=150 → rerank → top 30)
    print('\n🔮 TWO-STAGE (top_k=150 → rerank → top 30):')

    # Stage 1: Get more candidates
    all_candidates = await search_chunks(query_embedding, top_k=150)
    print(f'  Stage 1: Retrieved {len(all_candidates)} candidates')

    # Stage 2: Rerank
    reranked_chunks = rerank_chunks(query, all_candidates, top_n=30)
    print(f'  Stage 2: Reranked to top {len(reranked_chunks)}')

    print('  Top 5 by rerank score:')
    for i, c in enumerate(reranked_chunks[:5], 1):
        preview = c['content'][:100].replace('\n', ' ')
        print(f'    {i}. [rerank={c["rerank_score"]:.4f}, dist={c["distance"]:.4f}] {preview}...')

    # Check score distribution
    scores = [c['rerank_score'] for c in reranked_chunks]
    print('\n  Rerank score distribution:')
    print(f'    min={min(scores):.4f}, max={max(scores):.4f}, avg={np.mean(scores):.4f}')

    # Compare: How many chunks changed?
    standard_ids = [c['id'] for c in standard_chunks[:10]]
    reranked_ids = [c['id'] for c in reranked_chunks[:10]]

    overlap = len(set(standard_ids) & set(reranked_ids))
    new_in_top10 = len(set(reranked_ids) - set(standard_ids))

    print('\n📊 COMPARISON (top 10):')
    print(f'  Overlap: {overlap}/10 chunks in common')
    print(f'  New chunks surfaced by reranking: {new_in_top10}')

    # Check if reranking brought up chunks from outside original top-30
    original_top30_ids = {c['id'] for c in standard_chunks}
    reranked_top10_ids = {c['id'] for c in reranked_chunks[:10]}
    surfaced_from_beyond = len(reranked_top10_ids - original_top30_ids)

    if surfaced_from_beyond > 0:
        print(f'  ✨ {surfaced_from_beyond} chunks from positions 31-150 made it to top 10!')

    # Position changes
    print('\n  Position changes (original → reranked):')
    for i, chunk in enumerate(reranked_chunks[:10], 1):
        chunk_id = chunk['id']
        try:
            orig_pos = standard_ids.index(chunk_id) + 1
            if orig_pos != i:
                print(f'    {chunk_id[:20]}...: pos {orig_pos} → {i} (Δ{orig_pos - i:+d})')
        except ValueError:
            # Find original position in all 150 candidates
            orig_pos = next((j + 1 for j, c in enumerate(all_candidates) if c['id'] == chunk_id), '?')
            print(f'    {chunk_id[:20]}...: pos {orig_pos} → {i} (SURFACED from beyond top-30!)')

    return {
        'query': query,
        'overlap_top10': overlap,
        'surfaced_from_beyond': surfaced_from_beyond,
        'avg_rerank_score': np.mean(scores),
    }


async def main():
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

    avg_overlap = np.mean([r['overlap_top10'] for r in results])
    total_surfaced = sum(r['surfaced_from_beyond'] for r in results)
    avg_rerank_score = np.mean([r['avg_rerank_score'] for r in results])

    print(f'Average overlap in top-10: {avg_overlap:.1f}/10')
    print(f'Total chunks surfaced from beyond top-30: {total_surfaced}')
    print(f'Average rerank score: {avg_rerank_score:.4f}')

    if total_surfaced > 0:
        print('\n✅ Two-stage retrieval surfaces hidden relevant chunks!')
        print(f'   {total_surfaced} chunks from positions 31-150 made it to top-10 across {len(results)} queries')
    else:
        print("\n⚠️ Two-stage retrieval didn't surface new chunks")
        print('   The top-30 from vector search already contains the best chunks')


if __name__ == '__main__':
    asyncio.run(main())
