import asyncio

import numpy as np

from lightrag.citation import CitationResult, extract_citations_from_response


# Mock embedding function
async def mock_embedding_func(texts: list[str]) -> list[list[float]]:
    embeddings = []
    for text in texts:
        text = text.lower()
        vec = [0.0, 0.0, 0.0]
        if 'sky' in text:
            vec[0] = 1.0
        if 'grass' in text:
            vec[1] = 1.0
        if 'water' in text:
            vec[2] = 1.0

        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = [v / norm for v in vec]
        embeddings.append(vec)
    return embeddings


async def main():
    # 1. Setup Chunks (The knowledge base)
    chunks = [
        {'id': 'chunk_1', 'content': 'The sky is usually blue during the day.', 'file_path': 'nature.txt'},
        {'id': 'chunk_2', 'content': 'The grass is green because of chlorophyll.', 'file_path': 'biology.txt'},
        {'id': 'chunk_3', 'content': 'Water is wet and essential for life.', 'file_path': 'chemistry.txt'},
    ]

    # 2. Setup References (Map files/chunks to IDs)
    references = [
        {'reference_id': '1', 'file_path': 'nature.txt'},
        {'reference_id': '2', 'file_path': 'biology.txt'},
        {'reference_id': '3', 'file_path': 'chemistry.txt'},
    ]

    # 3. Mock LLM Response
    # Sentence 1: Matches chunk 1 (sky)
    # Sentence 2: Matches chunk 2 (grass)
    # Sentence 3: Matches chunk 3 (water) partially?
    # Sentence 4: No match
    response = 'The sky is blue. The grass is green. Water is wet. Computers are silicon.'

    print('--- Running Citation Extraction ---')
    result: CitationResult = await extract_citations_from_response(
        response=response, chunks=chunks, references=references, embedding_func=mock_embedding_func, min_similarity=0.5
    )

    print(f'Original Response: {result.original_response}')
    print(f'Annotated Response: {result.annotated_response}')
    print('\nCitations Found:')
    for cit in result.citations:
        print(f"  - Text: '{cit.text}'")
        print(f'    Refs: {cit.reference_ids} (Conf: {cit.confidence:.2f})')
        print(f'    Pos: {cit.start_char}-{cit.end_char}')

    print('\nFootnotes:')
    for fn in result.footnotes:
        print(f'  {fn}')

    print('\nUncited Claims:')
    for claim in result.uncited_claims:
        print(f"  '{claim}'")


if __name__ == '__main__':
    asyncio.run(main())
