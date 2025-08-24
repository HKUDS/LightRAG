"""
Semantic Retriever using embedding-based similarity search.

Implements semantic search using vector embeddings and cosine similarity
as described in the AI News & Strategy Daily video.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime

from ..embeddings.base import EmbeddingModel
from .base import Retriever, RetrievalResult


class SemanticRetriever(Retriever):
    """
    Semantic retriever using embedding-based similarity search.
    
    Implements the core semantic search functionality described in the video:
    - Converts text to embeddings (vectors in high-dimensional space)
    - Uses cosine similarity to find "nearest neighbors"
    - Similar meanings cluster together mathematically
    """
    
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        similarity_threshold: float = 0.0,
        max_chunks: int = 10000,
    ):
        """
        Initialize semantic retriever.
        
        Args:
            embedding_model: Model for generating embeddings
            similarity_threshold: Minimum similarity score to include in results
            max_chunks: Maximum number of chunks to store (for memory management)
        """
        super().__init__(name=f"SemanticRetriever({embedding_model.model_name})")
        
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_chunks = max_chunks
        
        # Storage for chunks and embeddings
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.chunk_id_to_index: Dict[str, int] = {}
        
        # Performance tracking
        self.last_embedding_time: Optional[float] = None
        self.last_search_time: Optional[float] = None
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add chunks to the semantic index.
        
        Generates embeddings for each chunk and stores them for similarity search.
        """
        if not chunks:
            return
        
        start_time = datetime.now()
        
        # Extract text content for embedding
        texts = []
        chunk_data = []
        
        for chunk in chunks:
            # Get content - support different formats
            content = chunk.get('content', '')
            if not content and 'text' in chunk:
                content = chunk['text']
            
            if not content or not content.strip():
                continue  # Skip empty chunks
            
            texts.append(content)
            
            # Prepare chunk data with required fields
            chunk_data.append({
                'content': content,
                'chunk_id': chunk.get('chunk_id', f"chunk_{len(self.chunks)}"),
                'source_id': chunk.get('source_id', chunk.get('source', 'unknown')),
                'metadata': chunk.get('metadata', {}),
                'source_file': chunk.get('source_file'),
                'source_section': chunk.get('source_section'),
                'source_page': chunk.get('source_page'),
            })
        
        if not texts:
            return  # No valid content to add
        
        # Generate embeddings
        try:
            embedding_result = self.embedding_model.embed_batch(texts)
            new_embeddings = embedding_result.embeddings
            
            if new_embeddings.size == 0:
                print("Warning: No embeddings generated")
                return
            
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return
        
        # Add to storage
        current_index = len(self.chunks)
        
        for i, chunk in enumerate(chunk_data):
            chunk_index = current_index + i
            
            # Add to chunks list
            self.chunks.append(chunk)
            
            # Add to index mapping
            self.chunk_id_to_index[chunk['chunk_id']] = chunk_index
        
        # Update embeddings matrix
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        # Memory management - remove oldest chunks if exceeding limit
        if len(self.chunks) > self.max_chunks:
            excess = len(self.chunks) - self.max_chunks
            self._remove_oldest_chunks(excess)
        
        # Track timing
        end_time = datetime.now()
        self.last_embedding_time = (end_time - start_time).total_seconds() * 1000
        
        print(f"Added {len(chunk_data)} chunks to semantic index in {self.last_embedding_time:.1f}ms")
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve semantically similar chunks using cosine similarity.
        
        Implements the core semantic search process:
        1. Embed the query
        2. Compute cosine similarity with all chunk embeddings
        3. Return top-k most similar chunks
        """
        if not self.chunks or self.embeddings is None:
            return []
        
        start_time = datetime.now()
        
        try:
            # Embed the query
            query_result = self.embedding_model.embed_text(query)
            query_embedding = query_result.embeddings
            
            # Compute similarities with all chunks
            similarities = self._compute_similarities(query_embedding)
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            # Filter by similarity threshold
            valid_indices = [
                idx for idx in top_indices 
                if similarities[idx] >= self.similarity_threshold
            ]
            
            # Apply additional filters if provided
            if filters:
                valid_indices = self._apply_filters(valid_indices, filters)
            
            # Create results
            results = []
            for rank, idx in enumerate(valid_indices):
                chunk = self.chunks[idx]
                similarity_score = float(similarities[idx])
                
                result = RetrievalResult(
                    content=chunk['content'],
                    score=similarity_score,
                    source_id=chunk['source_id'],
                    metadata=chunk['metadata'].copy(),
                    retrieval_method="semantic_similarity",
                    chunk_id=chunk['chunk_id'],
                    original_rank=rank,
                    source_file=chunk.get('source_file'),
                    source_section=chunk.get('source_section'),
                    source_page=chunk.get('source_page'),
                )
                
                # Add retrieval-specific metadata
                result.add_metadata("similarity_score", similarity_score)
                result.add_metadata("embedding_model", self.embedding_model.model_name)
                result.add_metadata("query_embedding_dims", len(query_embedding))
                
                results.append(result)
            
            # Track timing
            end_time = datetime.now()
            self.last_search_time = (end_time - start_time).total_seconds() * 1000
            
            return results
            
        except Exception as e:
            print(f"Error in semantic retrieval: {e}")
            return []
    
    def _compute_similarities(self, query_embedding: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarities between query and all chunk embeddings.
        
        Implements efficient vectorized cosine similarity computation.
        """
        # Normalize embeddings for cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # Normalize chunk embeddings
        chunk_norms = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        
        # Compute cosine similarities
        similarities = np.dot(chunk_norms, query_norm)
        
        return similarities
    
    def _apply_filters(self, indices: List[int], filters: Dict[str, Any]) -> List[int]:
        """
        Apply metadata filters to retrieved indices.
        
        Supports filtering by source, date, section, etc.
        """
        filtered_indices = []
        
        for idx in indices:
            chunk = self.chunks[idx]
            
            # Check each filter condition
            include_chunk = True
            
            for filter_key, filter_value in filters.items():
                if filter_key == 'source' or filter_key == 'source_id':
                    if chunk['source_id'] != filter_value:
                        include_chunk = False
                        break
                
                elif filter_key == 'source_file':
                    if chunk.get('source_file') != filter_value:
                        include_chunk = False
                        break
                
                elif filter_key == 'source_section':
                    if chunk.get('source_section') != filter_value:
                        include_chunk = False
                        break
                
                elif filter_key in chunk['metadata']:
                    if chunk['metadata'][filter_key] != filter_value:
                        include_chunk = False
                        break
                
                # Date range filtering
                elif filter_key == 'date_after':
                    doc_date = chunk['metadata'].get('document_date')
                    if doc_date and doc_date < filter_value:
                        include_chunk = False
                        break
                
                elif filter_key == 'date_before':
                    doc_date = chunk['metadata'].get('document_date')
                    if doc_date and doc_date > filter_value:
                        include_chunk = False
                        break
            
            if include_chunk:
                filtered_indices.append(idx)
        
        return filtered_indices
    
    def _remove_oldest_chunks(self, count: int) -> None:
        """
        Remove oldest chunks to manage memory usage.
        
        Uses FIFO (First In, First Out) strategy.
        """
        if count <= 0 or count >= len(self.chunks):
            return
        
        # Remove from chunks list
        removed_chunks = self.chunks[:count]
        self.chunks = self.chunks[count:]
        
        # Remove from embeddings
        self.embeddings = self.embeddings[count:]
        
        # Update index mapping
        removed_chunk_ids = {chunk['chunk_id'] for chunk in removed_chunks}
        
        # Rebuild index mapping
        new_mapping = {}
        for new_idx, chunk in enumerate(self.chunks):
            chunk_id = chunk['chunk_id']
            if chunk_id not in removed_chunk_ids:
                new_mapping[chunk_id] = new_idx
        
        self.chunk_id_to_index = new_mapping
        
        print(f"Removed {count} oldest chunks from semantic index")
    
    def get_chunk_count(self) -> int:
        """Get number of indexed chunks."""
        return len(self.chunks)
    
    def clear_index(self) -> None:
        """Clear the entire semantic index."""
        self.chunks.clear()
        self.embeddings = None
        self.chunk_id_to_index.clear()
        print("Cleared semantic index")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the semantic retriever."""
        base_stats = super().get_stats()
        
        stats = {
            **base_stats,
            "embedding_model": self.embedding_model.model_name,
            "embedding_dimensions": self.embedding_model.dimensions,
            "similarity_threshold": self.similarity_threshold,
            "max_chunks": self.max_chunks,
            "embeddings_shape": self.embeddings.shape if self.embeddings is not None else None,
            "last_embedding_time_ms": self.last_embedding_time,
            "last_search_time_ms": self.last_search_time,
        }
        
        # Memory usage estimate
        if self.embeddings is not None:
            memory_bytes = self.embeddings.nbytes
            stats["estimated_memory_mb"] = memory_bytes / (1024 * 1024)
        
        return stats
    
    def _explain_retrieval(self, query: str, results: List[RetrievalResult]) -> Dict[str, Any]:
        """
        Explain semantic retrieval process.
        """
        explanation = super()._explain_retrieval(query, results)
        
        explanation.update({
            "method": "semantic_similarity",
            "embedding_model": self.embedding_model.model_name,
            "similarity_metric": "cosine",
            "similarity_threshold": self.similarity_threshold,
            "total_indexed_chunks": len(self.chunks),
        })
        
        if results:
            similarities = [r.get_metadata("similarity_score", 0) for r in results]
            explanation["similarity_stats"] = {
                "min": min(similarities),
                "max": max(similarities),
                "avg": sum(similarities) / len(similarities),
            }
        
        return explanation