"""
Reranking implementation using Ollama embeddings.
This provides a rerank_model_func for LightRAG that uses embedding similarity.
"""

import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger("lightrag")


async def ollama_embedding_rerank(
    query: str,
    documents: List[str],
    top_n: int = 5,
    embed_model: str = "embeddinggemma:300m",
    host: str = "http://localhost:11434",
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Rerank documents using embedding similarity.
    
    This function computes embeddings for the query and all documents,
    then reranks documents by cosine similarity to the query.
    
    Args:
        query: The search query
        documents: List of document texts to rerank
        top_n: Number of top documents to return
        embed_model: Ollama embedding model name
        host: Ollama server host
        **kwargs: Additional arguments (ignored)
        
    Returns:
        List of dicts with 'index' and 'relevance_score' for top_n documents
    """
    if not documents:
        logger.warning("No documents provided for reranking")
        return []
    
    try:
        # Import here to avoid dependency issues
        from lightrag.llm.ollama import ollama_embed
        
        logger.debug(f"Reranking {len(documents)} documents for query: {query[:50]}...")
        
        # Get embeddings for query and documents
        all_texts = [query] + documents
        
        # Use ollama_embed function
        embeddings = await ollama_embed(
            all_texts,
            embed_model=embed_model,
            host=host
        )
        
        # Split query and document embeddings
        query_embedding = embeddings[0]
        doc_embeddings = embeddings[1:]
        
        # Compute cosine similarity
        scores = []
        query_norm = np.linalg.norm(query_embedding)
        
        for idx, doc_embedding in enumerate(doc_embeddings):
            doc_norm = np.linalg.norm(doc_embedding)
            
            if query_norm == 0 or doc_norm == 0:
                similarity = 0.0
            else:
                similarity = np.dot(query_embedding, doc_embedding) / (query_norm * doc_norm)
            
            scores.append({
                'index': idx,
                'relevance_score': float(similarity)
            })
        
        # Sort by score (descending) and take top_n
        scores.sort(key=lambda x: x['relevance_score'], reverse=True)
        top_scores = scores[:top_n]
        
        logger.debug(
            f"Reranking complete. Top score: {top_scores[0]['relevance_score']:.4f}, "
            f"Bottom score: {top_scores[-1]['relevance_score']:.4f}"
        )
        
        return top_scores
        
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        # Return original order if reranking fails
        return [{'index': i, 'relevance_score': 1.0} for i in range(min(top_n, len(documents)))]


def create_ollama_rerank_func(
    embed_model: str = "embeddinggemma:300m",
    host: str = "http://localhost:11434"
):
    """
    Create a rerank function with preset configuration.
    
    Args:
        embed_model: Ollama embedding model to use
        host: Ollama server host
        
    Returns:
        Async function configured for reranking
    """
    async def rerank_func(
        query: str,
        documents: List[str],
        top_n: int = 5,
        **kwargs
    ) -> List[Dict[str, Any]]:
        return await ollama_embedding_rerank(
            query=query,
            documents=documents,
            top_n=top_n,
            embed_model=embed_model,
            host=host,
            **kwargs
        )
    
    return rerank_func