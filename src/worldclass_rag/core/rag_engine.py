"""
Main RAG Engine - Integrates all WorldClass RAG components.

Provides high-level interface following the best practices from AI News & Strategy Daily.
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import time
from datetime import datetime

from .chunking import ChunkingStrategy, SemanticChunker, RecursiveChunker, SentenceChunker
from .embeddings import EmbeddingModel, OpenAIEmbeddings, SentenceTransformerEmbeddings
from .retrieval import HybridRetriever, SemanticRetriever, KeywordRetriever
from .retrieval.hybrid_retriever import HybridSearchConfig
from .evaluation import RelevanceEvaluator, EvaluationResult
from ..processors import TextProcessor, PDFProcessor, ImageProcessor, TableProcessor
from ..config import RAGConfig


class RAGEngine:
    """
    Main RAG Engine integrating all components.
    
    Provides simple, high-level interface for:
    - Document processing and chunking
    - Hybrid retrieval (semantic + keyword + re-ranking)
    - Evaluation and monitoring
    - Enterprise scalability features
    
    Follows best practices from AI News & Strategy Daily:
    - "Memory perfect" via vector storage
    - Eliminates hallucinations through grounded retrieval
    - Hybrid search for better precision
    - Continuous evaluation with 4 key metrics
    """
    
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        embeddings_model: Optional[str] = None,
        vector_store: Optional[str] = None,
        chunking_strategy: Optional[str] = None,
    ):
        """
        Initialize RAG Engine.
        
        Args:
            config: RAG configuration (uses defaults if None)
            embeddings_model: Override embedding model
            vector_store: Override vector store type
            chunking_strategy: Override chunking strategy
        """
        self.config = config or RAGConfig()
        
        # Override config with parameters
        if embeddings_model:
            self.config.embedding.model_name = embeddings_model
        if vector_store:
            self.config.vector_store = vector_store
        if chunking_strategy:
            self.config.chunking.strategy = chunking_strategy
        
        # Initialize components
        self._setup_embedding_model()
        self._setup_processors()
        self._setup_chunker()
        self._setup_retriever()
        self._setup_evaluator()
        
        # Performance tracking
        self.stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "queries_processed": 0,
            "total_processing_time": 0.0,
            "total_query_time": 0.0,
        }
        
        print(f"✅ WorldClass RAG Engine initialized")
        print(f"   - Embedding: {self.embedding_model.model_name}")
        print(f"   - Chunking: {self.config.chunking.strategy}")
        print(f"   - Retrieval: {self.config.retrieval.type}")
        print(f"   - Vector Store: {self.config.vector_store}")
    
    def _setup_embedding_model(self) -> None:
        """Initialize embedding model based on configuration."""
        if self.config.embedding.provider == "openai":
            try:
                self.embedding_model = OpenAIEmbeddings(
                    model=self.config.embedding.model_name,
                    api_key=self.config.embedding.api_key,
                    dimensions=self.config.embedding.dimensions,
                )
                if not self.embedding_model.is_available():
                    raise Exception("OpenAI model not available")
            except Exception as e:
                print(f"Warning: OpenAI not available ({e}), falling back to local model")
                self.embedding_model = SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
        else:
            self.embedding_model = SentenceTransformerEmbeddings(
                model_name=self.config.embedding.model_name
            )
    
    def _setup_processors(self) -> None:
        """Initialize document processors."""
        self.processors = {}
        
        if "text" in self.config.processors:
            self.processors["text"] = TextProcessor(
                remove_boilerplate=True,
                normalize_whitespace=True,
                extract_metadata=True
            )
        
        if "pdf" in self.config.processors:
            try:
                self.processors["pdf"] = PDFProcessor(
                    remove_boilerplate=True,
                    remove_headers_footers=True,
                    use_ocr=True
                )
            except ImportError as e:
                print(f"Warning: PDF processor not available: {e}")
        
        if "images" in self.config.processors:
            try:
                self.processors["images"] = ImageProcessor(
                    enhance_image=True,
                    auto_rotate=True
                )
            except ImportError as e:
                print(f"Warning: Image processor not available: {e}")
        
        if "tables" in self.config.processors:
            try:
                self.processors["tables"] = TableProcessor(
                    create_searchable_text=True,
                    handle_merged_cells=True
                )
            except ImportError as e:
                print(f"Warning: Table processor not available: {e}")
    
    def _setup_chunker(self) -> None:
        """Initialize chunking strategy."""
        chunking_config = self.config.chunking
        
        if chunking_config.strategy == "semantic":
            self.chunker = SemanticChunker(
                chunk_size=chunking_config.chunk_size,
                chunk_overlap=chunking_config.chunk_overlap,
                preserve_sentence_boundaries=chunking_config.preserve_sentence_boundaries
            )
        elif chunking_config.strategy == "sentence":
            self.chunker = SentenceChunker(
                chunk_size=chunking_config.chunk_size,
                chunk_overlap=chunking_config.chunk_overlap,
                language=self.config.retrieval.language
            )
        else:  # Default to recursive
            self.chunker = RecursiveChunker(
                chunk_size=chunking_config.chunk_size,
                chunk_overlap=chunking_config.chunk_overlap,
                preserve_sentence_boundaries=chunking_config.preserve_sentence_boundaries
            )
    
    def _setup_retriever(self) -> None:
        """Initialize retrieval system."""
        retrieval_config = self.config.retrieval
        
        if retrieval_config.type == "semantic":
            self.retriever = SemanticRetriever(
                embedding_model=self.embedding_model,
                similarity_threshold=retrieval_config.similarity_threshold
            )
        elif retrieval_config.type == "keyword":
            self.retriever = KeywordRetriever(
                scoring_method=retrieval_config.scoring_method,
                language=retrieval_config.language
            )
        else:  # Default to hybrid
            semantic_retriever = SemanticRetriever(
                embedding_model=self.embedding_model,
                similarity_threshold=retrieval_config.similarity_threshold
            )
            keyword_retriever = KeywordRetriever(
                scoring_method=retrieval_config.scoring_method,
                language=retrieval_config.language
            )
            
            hybrid_config = HybridSearchConfig(
                semantic_weight=retrieval_config.semantic_weight,
                keyword_weight=retrieval_config.keyword_weight,
                fusion_method=retrieval_config.fusion_method,
                rerank=retrieval_config.rerank,
                rerank_top_k=retrieval_config.rerank_top_k,
                final_top_k=retrieval_config.final_top_k
            )
            
            self.retriever = HybridRetriever(
                semantic_retriever=semantic_retriever,
                keyword_retriever=keyword_retriever,
                config=hybrid_config
            )
    
    def _setup_evaluator(self) -> None:
        """Initialize evaluation system."""
        self.evaluator = RelevanceEvaluator(
            use_semantic_similarity=True,
            use_keyword_overlap=True,
            relevance_threshold=0.3
        )
    
    def add_document(
        self, 
        file_path: Union[str, Path], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add a single document to the RAG system.
        
        Args:
            file_path: Path to document file
            metadata: Additional metadata for the document
            
        Returns:
            Processing result with statistics
        """
        return self.add_documents([file_path], metadata)
    
    def add_documents(
        self, 
        file_paths: List[Union[str, Path]], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Add multiple documents to the RAG system.
        
        Implements the complete pipeline:
        1. Document processing with appropriate processor
        2. Intelligent chunking with overlap
        3. Indexing in retrieval system
        4. Performance tracking
        """
        start_time = time.time()
        
        results = {
            "processed_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            "processing_time": 0.0,
            "errors": [],
            "file_details": []
        }
        
        all_chunks = []
        
        for file_path in file_paths:
            try:
                path = Path(file_path)
                
                # Select appropriate processor
                processor = self._select_processor(path)
                if not processor:
                    results["errors"].append(f"No processor available for {path.suffix}")
                    results["failed_files"] += 1
                    continue
                
                # Process document
                processed_doc = processor.process(path, metadata)
                
                if processed_doc.extraction_quality < 0.3:
                    results["errors"].append(f"Poor extraction quality for {path.name}")
                
                # Chunk document
                chunks = self.chunker.chunk_document(
                    text=processed_doc.content,
                    source=str(path),
                    additional_metadata=processed_doc.metadata
                )
                
                # Add unique IDs
                for i, chunk in enumerate(chunks):
                    chunk.chunk_id = f"{path.stem}_chunk_{i}"
                
                all_chunks.extend(chunks)
                
                # Track file processing
                file_result = {
                    "file": str(path),
                    "chunks_created": len(chunks),
                    "extraction_quality": processed_doc.extraction_quality,
                    "warnings": processed_doc.warnings,
                    "processing_time_ms": processed_doc.processing_time_ms
                }
                results["file_details"].append(file_result)
                
                results["processed_files"] += 1
                results["total_chunks"] += len(chunks)
                
            except Exception as e:
                results["errors"].append(f"Error processing {file_path}: {str(e)}")
                results["failed_files"] += 1
        
        # Index all chunks
        if all_chunks:
            chunks_for_indexing = []
            for chunk in all_chunks:
                chunk_data = {
                    "content": chunk.content,
                    "chunk_id": chunk.chunk_id,
                    "source_id": chunk.get_metadata("source", "unknown"),
                    "metadata": chunk.metadata,
                    "source_file": chunk.get_metadata("source"),
                    "source_section": chunk.get_metadata("section"),
                }
                chunks_for_indexing.append(chunk_data)
            
            self.retriever.add_chunks(chunks_for_indexing)
        
        # Update statistics
        processing_time = time.time() - start_time
        results["processing_time"] = processing_time
        
        self.stats["documents_processed"] += results["processed_files"]
        self.stats["chunks_created"] += results["total_chunks"]
        self.stats["total_processing_time"] += processing_time
        
        return results
    
    def query(
        self, 
        query: str, 
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        evaluate: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system and return results.
        
        Args:
            query: User question or query
            top_k: Number of top results to return
            filters: Optional filters for retrieval
            evaluate: Whether to evaluate result quality
            
        Returns:
            Complete query result with retrieved chunks and evaluation
        """
        start_time = time.time()
        
        # Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(query, top_k, filters)
        
        # Create response (in full RAG system, this would involve LLM generation)
        if retrieved_chunks:
            # Simple response generation for demonstration
            response = f"Based on the retrieved information: {retrieved_chunks[0].content[:200]}..."
        else:
            response = "I couldn't find relevant information for your query."
        
        # Measure latency
        query_time = time.time() - start_time
        
        # Prepare result
        result = {
            "query": query,
            "response": response,
            "retrieved_chunks": retrieved_chunks,
            "latency_ms": query_time * 1000,
            "metadata": {
                "retrieval_method": self.config.retrieval.type,
                "chunks_found": len(retrieved_chunks),
                "query_time": query_time,
            }
        }
        
        # Evaluate if requested
        if evaluate and retrieved_chunks:
            evaluation = self.evaluator.evaluate(
                query=query,
                response=response,
                retrieved_chunks=retrieved_chunks
            )
            result["evaluation"] = {
                "relevance": evaluation.metrics.relevance,
                "overall_score": evaluation.metrics.overall_score,
                "details": evaluation.relevance_details
            }
        
        # Update statistics
        self.stats["queries_processed"] += 1
        self.stats["total_query_time"] += query_time
        
        return result
    
    def _select_processor(self, file_path: Path):
        """Select appropriate processor for file type."""
        extension = file_path.suffix.lower()
        
        # PDF files
        if extension == ".pdf" and "pdf" in self.processors:
            return self.processors["pdf"]
        
        # Image files
        elif extension in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"] and "images" in self.processors:
            return self.processors["images"]
        
        # Table files
        elif extension in [".csv", ".xlsx", ".xls"] and "tables" in self.processors:
            return self.processors["tables"]
        
        # Text files (default)
        elif "text" in self.processors:
            return self.processors["text"]
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        retriever_stats = self.retriever.get_stats() if hasattr(self.retriever, 'get_stats') else {}
        
        stats = {
            "engine_stats": self.stats.copy(),
            "retriever_stats": retriever_stats,
            "configuration": self.config.to_dict(),
            "performance_metrics": {
                "avg_processing_time_per_doc": (
                    self.stats["total_processing_time"] / max(1, self.stats["documents_processed"])
                ),
                "avg_query_time": (
                    self.stats["total_query_time"] / max(1, self.stats["queries_processed"])
                ),
                "chunks_per_document": (
                    self.stats["chunks_created"] / max(1, self.stats["documents_processed"])
                ),
            }
        }
        
        return stats
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate current configuration for production readiness.
        
        Implements validation checklist from enterprise config.
        """
        validation_issues = self.config.validate()
        deployment_checklist = self.config.get_deployment_checklist()
        
        # Check component availability
        component_status = {
            "embedding_model": hasattr(self, 'embedding_model') and self.embedding_model.is_available(),
            "processors": len(self.processors) > 0,
            "chunker": hasattr(self, 'chunker'),
            "retriever": hasattr(self, 'retriever'),
            "evaluator": hasattr(self, 'evaluator'),
        }
        
        return {
            "validation_issues": validation_issues,
            "deployment_checklist": deployment_checklist,
            "component_status": component_status,
            "overall_status": len(validation_issues) == 0 and all(component_status.values()),
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "checks": {}
        }
        
        # Test embedding model
        try:
            test_result = self.embedding_model.embed_text("test")
            health["checks"]["embedding_model"] = "ok"
        except Exception as e:
            health["checks"]["embedding_model"] = f"error: {e}"
            health["status"] = "unhealthy"
        
        # Test retriever
        try:
            chunk_count = self.retriever.get_chunk_count()
            health["checks"]["retriever"] = f"ok ({chunk_count} chunks indexed)"
        except Exception as e:
            health["checks"]["retriever"] = f"error: {e}"
            health["status"] = "unhealthy"
        
        # Check memory usage (basic)
        try:
            import psutil
            memory = psutil.virtual_memory()
            health["checks"]["memory"] = f"ok ({memory.percent}% used)"
            
            if memory.percent > 90:
                health["status"] = "warning"
        except ImportError:
            health["checks"]["memory"] = "unavailable (psutil not installed)"
        
        return health
    
    def clear_index(self) -> None:
        """Clear all indexed data."""
        self.retriever.clear_index()
        
        # Reset statistics
        self.stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "queries_processed": 0,
            "total_processing_time": 0.0,
            "total_query_time": 0.0,
        }
        
        print("✅ RAG index cleared")