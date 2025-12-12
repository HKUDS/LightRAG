"""
Local reranker using sentence-transformers CrossEncoder.

Uses cross-encoder/ms-marco-MiniLM-L-6-v2 by default - a 22M param model with
excellent accuracy and clean score separation (-11 to +10 range).
Runs entirely locally without API calls.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol, SupportsFloat, TypedDict, runtime_checkable

from .utils import logger

# Global model cache to avoid reloading on every call
_reranker_model: RerankerModel | None = None
_reranker_model_name: str | None = None

# Default model - mxbai-rerank-xsmall-v1 performs best on domain-specific content
# Used for ordering only (no score filtering) - see constants.py DEFAULT_MIN_RERANK_SCORE
DEFAULT_RERANK_MODEL = 'mixedbread-ai/mxbai-rerank-xsmall-v1'


class RerankResult(TypedDict):
    index: int
    relevance_score: float


@runtime_checkable
class SupportsToList(Protocol):
    def tolist(self) -> list[float]: ...


ScoreLike = Sequence[SupportsFloat] | SupportsToList


@runtime_checkable
class RerankerModel(Protocol):
    def predict(
        self,
        sentences: list[list[str]],
        batch_size: int = ...,
    ) -> ScoreLike: ...


def get_reranker_model(model_name: str | None = None):
    """
    Get or initialize the reranker model (cached).

    Args:
        model_name: HuggingFace model name. Defaults to mxbai-rerank-xsmall-v1

    Returns:
        CrossEncoder-like model instance implementing predict(pairs)->list[float]
    """
    global _reranker_model, _reranker_model_name

    if model_name is None:
        model_name = os.getenv('RERANK_MODEL', DEFAULT_RERANK_MODEL)

    # Return cached model if same name
    if _reranker_model is not None and _reranker_model_name == model_name:
        return _reranker_model

    try:
        from sentence_transformers import CrossEncoder

        logger.info(f'Loading reranker model: {model_name}')
        _reranker_model = CrossEncoder(model_name, trust_remote_code=True)
        _reranker_model_name = model_name
        logger.info(f'Reranker model loaded: {model_name}')
        return _reranker_model

    except ImportError as err:
        raise ImportError(
            'sentence-transformers is required for local reranking. Install with: pip install sentence-transformers'
        ) from err
    except Exception as e:
        logger.error(f'Failed to load reranker model {model_name}: {e}')
        raise


async def local_rerank(
    query: str,
    documents: list[str],
    top_n: int | None = None,
    model_name: str | None = None,
) -> list[RerankResult]:
    """
    Rerank documents using local CrossEncoder model.

    Args:
        query: The search query
        documents: List of document strings to rerank
        top_n: Number of top results to return (None = all)
        model_name: HuggingFace model name (default: mxbai-rerank-xsmall-v1)

    Returns:
        List of dicts with 'index' and 'relevance_score', sorted by score descending

    Example:
        >>> results = await local_rerank(
        ...     query="What is machine learning?",
        ...     documents=["ML is a subset of AI...", "The weather is nice..."],
        ...     top_n=5
        ... )
        >>> print(results[0])
        {'index': 0, 'relevance_score': 0.95}
    """
    if not documents:
        return []

    model = get_reranker_model(model_name)

    # Create query-document pairs
    pairs = [[query, doc] for doc in documents]

    # Get scores from model
    # CrossEncoder.predict returns a list[float]; guard None for type checkers
    if model is None:
        raise RuntimeError('Reranker model failed to load')
    raw_scores = model.predict(pairs)

    # Normalize to a list[float] regardless of backend (list, numpy array, tensor)
    if isinstance(raw_scores, SupportsToList):
        raw_scores = raw_scores.tolist()

    scores = [float(score) for score in raw_scores]

    # Build results with index and score
    results: list[RerankResult] = [
        RerankResult(index=i, relevance_score=float(score)) for i, score in enumerate(scores)
    ]

    # Sort by score descending
    results.sort(key=lambda x: x['relevance_score'], reverse=True)

    # Apply top_n limit if specified
    if top_n is not None and top_n < len(results):
        results = results[:top_n]

    return results


def create_local_rerank_func(
    model_name: str | None = None,
) -> Callable[..., Awaitable[list[RerankResult]]]:
    """
    Create a rerank function with pre-configured model.

    This is used by lightrag_server to create a rerank function
    that can be passed to LightRAG initialization.

    Args:
        model_name: HuggingFace model name (default: mxbai-rerank-xsmall-v1)

    Returns:
        Async rerank function
    """
    # Pre-load model to fail fast if there's an issue
    get_reranker_model(model_name)

    async def rerank_func(
        query: str,
        documents: list[str],
        top_n: int | None = None,
        **kwargs,
    ) -> list[RerankResult]:
        return await local_rerank(
            query=query,
            documents=documents,
            top_n=top_n,
            model_name=model_name,
        )

    return rerank_func


# For backwards compatibility - alias to local_rerank
rerank = local_rerank


if __name__ == '__main__':
    import asyncio

    async def main():
        docs = [
            'The capital of France is Paris.',
            'Tokyo is the capital of Japan.',
            'London is the capital of England.',
            'Python is a programming language.',
        ]

        query = 'What is the capital of France?'

        print('=== Local Reranker Test ===')
        print(f'Model: {os.getenv("RERANK_MODEL", DEFAULT_RERANK_MODEL)}')
        print(f'Query: {query}')
        print()

        results = await local_rerank(query=query, documents=docs, top_n=3)

        print('Results (top 3):')
        for item in results:
            idx = item['index']
            score = item['relevance_score']
            print(f'  [{idx}] Score: {score:.4f} - {docs[idx]}')

    asyncio.run(main())
