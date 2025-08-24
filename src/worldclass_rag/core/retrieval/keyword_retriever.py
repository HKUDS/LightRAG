"""
Keyword Retriever using traditional text search methods.

Implements keyword-based search using TF-IDF and BM25 scoring,
complementing semantic search in hybrid retrieval approaches.
"""

import re
import math
from typing import List, Dict, Any, Optional, Set
from collections import defaultdict, Counter
from datetime import datetime

from .base import Retriever, RetrievalResult


class KeywordRetriever(Retriever):
    """
    Keyword-based retriever using TF-IDF and BM25 scoring.
    
    Provides traditional text search capabilities that complement
    semantic search in hybrid approaches. Particularly effective for:
    - Exact term matching
    - Named entity searches
    - Technical terms and acronyms
    - Code and identifier searches
    """
    
    def __init__(
        self,
        scoring_method: str = "bm25",  # "tfidf" or "bm25"
        language: str = "en",
        min_term_freq: int = 1,
        max_term_freq_ratio: float = 0.8,
        k1: float = 1.2,  # BM25 term frequency saturation parameter
        b: float = 0.75,  # BM25 field length normalization parameter
    ):
        """
        Initialize keyword retriever.
        
        Args:
            scoring_method: Scoring method ("tfidf" or "bm25")
            language: Language for text processing
            min_term_freq: Minimum term frequency to include in index
            max_term_freq_ratio: Maximum term frequency ratio (remove too common terms)
            k1: BM25 term frequency saturation parameter
            b: BM25 field length normalization parameter
        """
        super().__init__(name=f"KeywordRetriever({scoring_method})")
        
        self.scoring_method = scoring_method
        self.language = language
        self.min_term_freq = min_term_freq
        self.max_term_freq_ratio = max_term_freq_ratio
        self.k1 = k1
        self.b = b
        
        # Index storage
        self.chunks: List[Dict[str, Any]] = []
        self.chunk_id_to_index: Dict[str, int] = {}
        
        # Inverted index: term -> list of (chunk_index, term_frequency)
        self.inverted_index: Dict[str, List[tuple[int, int]]] = defaultdict(list)
        
        # Document statistics
        self.document_lengths: List[int] = []  # Length of each document in terms
        self.average_doc_length: float = 0.0
        self.total_documents: int = 0
        
        # Term statistics  
        self.term_document_frequency: Dict[str, int] = defaultdict(int)  # How many docs contain each term
        self.vocabulary: Set[str] = set()
        
        # Stop words for different languages
        self.stop_words = self._get_stop_words(language)
        
        # Performance tracking
        self.last_indexing_time: Optional[float] = None
        self.last_search_time: Optional[float] = None
    
    def _get_stop_words(self, language: str) -> Set[str]:
        """Get stop words for the specified language."""
        stop_words_dict = {
            "en": {
                "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
                "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
                "to", "was", "were", "will", "with", "but", "or", "not", "this",
                "have", "had", "what", "when", "where", "who", "which", "why", "how"
            },
            "es": {
                "el", "la", "de", "que", "y", "a", "en", "un", "es", "se", "no",
                "te", "lo", "le", "da", "su", "por", "son", "con", "para", "al",
                "una", "las", "del", "los", "pero", "más", "como", "ya", "muy",
                "sus", "me", "hasta", "donde", "cuando", "quien", "cual", "todo"
            }
        }
        return stop_words_dict.get(language, set())
    
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add chunks to the keyword index.
        
        Builds inverted index with TF-IDF or BM25 statistics.
        """
        if not chunks:
            return
        
        start_time = datetime.now()
        
        # Process each chunk
        new_chunks = []
        for chunk in chunks:
            # Get content
            content = chunk.get('content', '')
            if not content and 'text' in chunk:
                content = chunk['text']
            
            if not content or not content.strip():
                continue
            
            # Prepare chunk data
            chunk_data = {
                'content': content,
                'chunk_id': chunk.get('chunk_id', f"chunk_{len(self.chunks)}"),
                'source_id': chunk.get('source_id', chunk.get('source', 'unknown')),
                'metadata': chunk.get('metadata', {}),
                'source_file': chunk.get('source_file'),
                'source_section': chunk.get('source_section'),
                'source_page': chunk.get('source_page'),
            }
            
            new_chunks.append(chunk_data)
        
        if not new_chunks:
            return
        
        # Add chunks and build index
        current_index = len(self.chunks)
        
        for i, chunk in enumerate(new_chunks):
            chunk_index = current_index + i
            
            # Add chunk
            self.chunks.append(chunk)
            self.chunk_id_to_index[chunk['chunk_id']] = chunk_index
            
            # Tokenize and index content
            terms = self._tokenize(chunk['content'])
            term_counts = Counter(terms)
            
            # Store document length
            self.document_lengths.append(len(terms))
            
            # Update inverted index
            for term, count in term_counts.items():
                self.inverted_index[term].append((chunk_index, count))
                self.vocabulary.add(term)
            
            # Update document frequency for each unique term
            for term in term_counts.keys():
                self.term_document_frequency[term] += 1
        
        # Update global statistics
        self.total_documents = len(self.chunks)
        self.average_doc_length = sum(self.document_lengths) / self.total_documents if self.total_documents > 0 else 0
        
        # Filter out very rare and very common terms
        self._filter_vocabulary()
        
        # Track timing
        end_time = datetime.now()
        self.last_indexing_time = (end_time - start_time).total_seconds() * 1000
        
        print(f"Added {len(new_chunks)} chunks to keyword index in {self.last_indexing_time:.1f}ms")
        print(f"Vocabulary size: {len(self.vocabulary)} terms")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into terms for indexing.
        
        Implements basic text preprocessing:
        - Lowercase normalization
        - Punctuation removal
        - Stop word filtering
        - Minimum length filtering
        """
        # Convert to lowercase
        text = text.lower()
        
        # Split into words using regex (handles punctuation)
        words = re.findall(r'\b[a-zA-Z0-9]+\b', text)
        
        # Filter stop words and short terms
        terms = [
            word for word in words 
            if (word not in self.stop_words and 
                len(word) >= 2 and 
                not word.isdigit())  # Optional: filter pure numbers
        ]
        
        return terms
    
    def _filter_vocabulary(self) -> None:
        """
        Filter vocabulary to remove very rare and very common terms.
        
        Removes terms that are too rare (min_term_freq) or too common
        (appear in more than max_term_freq_ratio of documents).
        """
        if self.total_documents == 0:
            return
        
        max_doc_freq = int(self.total_documents * self.max_term_freq_ratio)
        
        filtered_terms = set()
        for term in self.vocabulary:
            doc_freq = self.term_document_frequency[term]
            if self.min_term_freq <= doc_freq <= max_doc_freq:
                filtered_terms.add(term)
        
        # Update vocabulary and clean inverted index
        removed_terms = self.vocabulary - filtered_terms
        for term in removed_terms:
            del self.inverted_index[term]
            del self.term_document_frequency[term]
        
        self.vocabulary = filtered_terms
        
        if removed_terms:
            print(f"Filtered {len(removed_terms)} terms from vocabulary")
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """
        Retrieve chunks using keyword matching and scoring.
        
        Implements TF-IDF or BM25 scoring for relevance ranking.
        """
        if not self.chunks or not self.vocabulary:
            return []
        
        start_time = datetime.now()
        
        try:
            # Tokenize query
            query_terms = self._tokenize(query)
            query_terms = [term for term in query_terms if term in self.vocabulary]
            
            if not query_terms:
                return []  # No valid query terms
            
            # Score all documents
            scores = self._score_documents(query_terms)
            
            # Get top-k results
            scored_docs = [(idx, score) for idx, score in enumerate(scores) if score > 0]
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            top_docs = scored_docs[:top_k]
            
            # Apply filters if provided
            if filters:
                top_docs = self._apply_filters(top_docs, filters)
            
            # Create results
            results = []
            for rank, (chunk_idx, score) in enumerate(top_docs):
                chunk = self.chunks[chunk_idx]
                
                result = RetrievalResult(
                    content=chunk['content'],
                    score=float(score),
                    source_id=chunk['source_id'],
                    metadata=chunk['metadata'].copy(),
                    retrieval_method=f"keyword_{self.scoring_method}",
                    chunk_id=chunk['chunk_id'],
                    original_rank=rank,
                    source_file=chunk.get('source_file'),
                    source_section=chunk.get('source_section'),
                    source_page=chunk.get('source_page'),
                )
                
                # Add keyword-specific metadata
                result.add_metadata("keyword_score", score)
                result.add_metadata("scoring_method", self.scoring_method)
                result.add_metadata("matched_terms", self._get_matched_terms(query_terms, chunk_idx))
                
                results.append(result)
            
            # Track timing
            end_time = datetime.now()
            self.last_search_time = (end_time - start_time).total_seconds() * 1000
            
            return results
            
        except Exception as e:
            print(f"Error in keyword retrieval: {e}")
            return []
    
    def _score_documents(self, query_terms: List[str]) -> List[float]:
        """
        Score all documents using TF-IDF or BM25.
        """
        scores = [0.0] * self.total_documents
        
        for term in query_terms:
            if term not in self.inverted_index:
                continue
            
            # Calculate IDF (Inverse Document Frequency)
            doc_freq = self.term_document_frequency[term]
            idf = math.log(self.total_documents / doc_freq)
            
            # Score each document containing this term
            for chunk_idx, term_freq in self.inverted_index[term]:
                if self.scoring_method == "bm25":
                    score = self._bm25_score(term_freq, self.document_lengths[chunk_idx], idf)
                else:  # tfidf
                    score = self._tfidf_score(term_freq, self.document_lengths[chunk_idx], idf)
                
                scores[chunk_idx] += score
        
        return scores
    
    def _tfidf_score(self, term_freq: int, doc_length: int, idf: float) -> float:
        """Calculate TF-IDF score for a term in a document."""
        tf = term_freq / doc_length if doc_length > 0 else 0
        return tf * idf
    
    def _bm25_score(self, term_freq: int, doc_length: int, idf: float) -> float:
        """Calculate BM25 score for a term in a document."""
        # BM25 formula
        numerator = term_freq * (self.k1 + 1)
        denominator = term_freq + self.k1 * (1 - self.b + self.b * (doc_length / self.average_doc_length))
        
        tf_component = numerator / denominator
        return idf * tf_component
    
    def _get_matched_terms(self, query_terms: List[str], chunk_idx: int) -> List[str]:
        """Get terms that matched in a specific chunk."""
        matched = []
        chunk_terms = set(self._tokenize(self.chunks[chunk_idx]['content']))
        
        for term in query_terms:
            if term in chunk_terms:
                matched.append(term)
        
        return matched
    
    def _apply_filters(self, scored_docs: List[tuple[int, float]], filters: Dict[str, Any]) -> List[tuple[int, float]]:
        """Apply metadata filters to scored documents."""
        filtered_docs = []
        
        for chunk_idx, score in scored_docs:
            chunk = self.chunks[chunk_idx]
            
            # Check filter conditions (same logic as semantic retriever)
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
                elif filter_key in chunk['metadata']:
                    if chunk['metadata'][filter_key] != filter_value:
                        include_chunk = False
                        break
            
            if include_chunk:
                filtered_docs.append((chunk_idx, score))
        
        return filtered_docs
    
    def get_chunk_count(self) -> int:
        """Get number of indexed chunks."""
        return len(self.chunks)
    
    def clear_index(self) -> None:
        """Clear the entire keyword index."""
        self.chunks.clear()
        self.chunk_id_to_index.clear()
        self.inverted_index.clear()
        self.document_lengths.clear()
        self.term_document_frequency.clear()
        self.vocabulary.clear()
        self.total_documents = 0
        self.average_doc_length = 0.0
        print("Cleared keyword index")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detailed statistics about the keyword retriever."""
        base_stats = super().get_stats()
        
        vocab_stats = {}
        if self.vocabulary:
            term_frequencies = list(self.term_document_frequency.values())
            vocab_stats = {
                "vocabulary_size": len(self.vocabulary),
                "avg_term_doc_frequency": sum(term_frequencies) / len(term_frequencies),
                "max_term_doc_frequency": max(term_frequencies),
                "min_term_doc_frequency": min(term_frequencies),
            }
        
        stats = {
            **base_stats,
            "scoring_method": self.scoring_method,
            "language": self.language,
            "total_documents": self.total_documents,
            "average_doc_length": self.average_doc_length,
            "last_indexing_time_ms": self.last_indexing_time,
            "last_search_time_ms": self.last_search_time,
            **vocab_stats
        }
        
        return stats
    
    def _explain_retrieval(self, query: str, results: List[RetrievalResult]) -> Dict[str, Any]:
        """Explain keyword retrieval process."""
        explanation = super()._explain_retrieval(query, results)
        
        query_terms = self._tokenize(query)
        valid_terms = [term for term in query_terms if term in self.vocabulary]
        
        explanation.update({
            "method": f"keyword_{self.scoring_method}",
            "query_terms": query_terms,
            "valid_query_terms": valid_terms,
            "vocabulary_size": len(self.vocabulary),
            "scoring_parameters": {
                "k1": self.k1,
                "b": self.b,
            } if self.scoring_method == "bm25" else {}
        })
        
        if results:
            keyword_scores = [r.get_metadata("keyword_score", 0) for r in results]
            explanation["keyword_score_stats"] = {
                "min": min(keyword_scores),
                "max": max(keyword_scores),
                "avg": sum(keyword_scores) / len(keyword_scores),
            }
        
        return explanation