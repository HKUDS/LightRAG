import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete
from lightrag.kg.shared_storage import initialize_pipeline_status

from typing import List, Any
import functools
from lightrag.utils import cosine_similarity, EmbeddingFunc

#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio
# nest_asyncio.apply()
#########

WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

    def mmr_reranking(
        query: str,
        chunks: List[Any],
        embedding_func: EmbeddingFunc,
        lambda_param: float = 0.5,
        top_n: int = 10,
    ) -> List[Any]:
        if not chunks:
            # no chunks to rerank
            return []
        if len(chunks) <= top_n:
            # fewer or equal top_n elements (set higher top_k)
            return chunks[:]
        embedded_query = embedding_func.func(query)
        embedded_chunks = embedding_func.func(chunks)
        selected_indices = []
        remaining_indices = list(range(len(chunks)))
        while len(selected_indices) < top_n and remaining_indices:
            mmr_scores = {}
            query_similarities = {
                idx: cosine_similarity(embedded_query, embedded_chunks[idx])
                for idx in remaining_indices
            }

            # Find the highest query similarity for the first pick
            if not selected_indices:
                best_idx = max(query_similarities, key=query_similarities.get)
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
                continue  # Move to the next iteration

            # Calculate MMR for remaining items
            for idx in remaining_indices:
                relevance_score = query_similarities[idx]

                # Calculate max similarity to already selected items
                diversity_penalty = 0.0
                if (
                    selected_indices
                ):  # Should always be true here except first iteration
                    similarities_to_selected = [
                        cosine_similarity(
                            embedded_chunks[idx], embedded_chunks[sel_idx]
                        )
                        for sel_idx in selected_indices
                    ]
                    diversity_penalty = (
                        max(similarities_to_selected)
                        if similarities_to_selected
                        else 0.0
                    )

                mmr_score = (
                    lambda_param * relevance_score
                    - (1 - lambda_param) * diversity_penalty
                )
                mmr_scores[idx] = mmr_score

            if not mmr_scores:  # Should not happen if remaining_indices is not empty
                break

            # Select the item with the highest MMR score
            next_idx = max(mmr_scores, key=mmr_scores.get)
            print(f"Next pick: Index {next_idx} (MMR: {mmr_scores[next_idx]:.4f})")
            selected_indices.append(next_idx)
            remaining_indices.remove(next_idx)
        # Return the selected chunks in the order they were selected
        reranked_chunks = [chunks[i] for i in selected_indices]
        return reranked_chunks

    async def async_mmr_wrapper(
        query: str,
        chunks: List[Any],
        embedding_func: EmbeddingFunc,
        lambda_param: float = 0.5,
        top_n: int = 3,
    ) -> List[Any]:
        return mmr_reranking(
            query=query,
            chunks=chunks,
            embedding_func=embedding_func,
            lambda_param=lambda_param,
            top_n=top_n,
        )

    async_reranker_func = functools.partial(
        async_mmr_wrapper,
        embedding_func=EmbeddingFunc(model_name="all-MiniLM-L6-v2-sim"),
        top_n=3,
    )


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,  # Use gpt_4o_mini_complete LLM model
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())

    with open("./book.txt", "r", encoding="utf-8") as f:
        rag.insert(f.read())

    # Perform naive search
    print(
        rag.query(
            "What are the top themes in this story?",
            param=QueryParam(mode="naive", reranker_func=async_reranker_func),
        )
    )

    # Perform mix search
    print(
        rag.query(
            "What are the top themes in this story?",
            param=QueryParam(mode="mix", reranker_func=async_reranker_func),
        )
    )


if __name__ == "__main__":
    main()
