"""
RAG Configuration for Enterprise Scalability and Security.

Implements scalable configuration following best practices from AI News & Strategy Daily video:
- Cost optimization to save millions of dollars
- Security and compliance (GDPR, HIPAA, SOC 2)
- Scalability for millions of queries
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import os
from pathlib import Path


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    provider: str = "sentence_transformers"  # "openai", "sentence_transformers"
    model_name: str = "all-MiniLM-L6-v2"
    dimensions: Optional[int] = None
    api_key: Optional[str] = None
    batch_size: int = 32
    timeout: float = 30.0
    max_retries: int = 3


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""
    strategy: str = "recursive"  # "semantic", "recursive", "sentence", "fixed_size"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    preserve_sentence_boundaries: bool = True
    min_chunk_size: int = 50
    max_chunk_size: int = 2000


@dataclass
class RetrievalConfig:
    """Configuration for retrieval system."""
    type: str = "hybrid"  # "semantic", "keyword", "hybrid"
    
    # Semantic retrieval
    similarity_threshold: float = 0.3
    max_chunks: int = 10000
    
    # Keyword retrieval
    scoring_method: str = "bm25"  # "tfidf", "bm25"
    language: str = "en"
    
    # Hybrid retrieval
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    fusion_method: str = "rrf"  # "linear", "rrf", "weighted_sum"
    rerank: bool = True
    rerank_top_k: int = 20
    final_top_k: int = 5


@dataclass
class SecurityConfig:
    """Security and compliance configuration."""
    # PII Detection and Filtering
    enable_pii_detection: bool = True
    pii_detection_threshold: float = 0.8
    remove_pii: bool = True
    pii_anonymization: bool = True
    
    # Access control
    enable_authentication: bool = True
    enable_authorization: bool = True
    api_key_required: bool = True
    rate_limiting: bool = True
    
    # Data encryption
    encrypt_at_rest: bool = True
    encrypt_in_transit: bool = True
    encryption_key_rotation: bool = True
    
    # Compliance
    gdpr_compliance: bool = True
    hipaa_compliance: bool = False
    soc2_compliance: bool = True
    
    # Audit logging
    audit_logging: bool = True
    log_queries: bool = True
    log_responses: bool = False  # More sensitive
    log_retrieval: bool = True
    retention_days: int = 90


@dataclass
class PerformanceConfig:
    """Performance and scalability configuration."""
    # Caching
    enable_cache: bool = True
    cache_type: str = "redis"  # "memory", "redis", "memcached"
    cache_ttl: int = 3600  # seconds
    cache_max_size: int = 1000  # MB
    
    # Query optimization
    query_timeout: float = 30.0
    max_concurrent_queries: int = 100
    connection_pool_size: int = 20
    
    # Vector database optimization
    vector_db_sharding: bool = True
    vector_db_replication: bool = True
    index_optimization: bool = True
    
    # Cost optimization strategies
    use_smaller_models: bool = False
    batch_embeddings: bool = True
    lazy_loading: bool = True
    compression: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    # Metrics collection
    enable_metrics: bool = True
    metrics_backend: str = "prometheus"  # "prometheus", "datadog", "cloudwatch"
    
    # Health checks
    health_check_interval: int = 30  # seconds
    enable_liveness: bool = True
    enable_readiness: bool = True
    
    # Alerting
    enable_alerting: bool = True
    alert_on_high_latency: bool = True
    latency_threshold_ms: float = 2000.0
    alert_on_low_relevance: bool = True
    relevance_threshold: float = 0.7
    
    # Tracing
    enable_tracing: bool = True
    trace_sampling_rate: float = 0.1


@dataclass
class RAGConfig:
    """
    Main RAG configuration following enterprise best practices.
    
    Implements scalability and security patterns from AI News & Strategy Daily:
    - Designed for millions of queries
    - Security and compliance built-in
    - Cost optimization strategies
    - Production monitoring
    """
    
    # Core components
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    
    # Enterprise features
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Storage configuration
    vector_store: str = "chroma"  # "chroma", "pinecone", "weaviate", "qdrant"
    vector_store_config: Dict[str, Any] = field(default_factory=dict)
    
    # Processing configuration
    processors: List[str] = field(default_factory=lambda: ["text", "pdf", "images", "tables"])
    max_file_size_mb: int = 100
    supported_formats: List[str] = field(default_factory=lambda: [
        ".txt", ".pdf", ".docx", ".md", ".csv", ".xlsx", ".png", ".jpg"
    ])
    
    # Environment
    environment: str = "development"  # "development", "staging", "production"
    debug: bool = False
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """
        Create configuration from environment variables.
        
        Follows 12-factor app principles for configuration.
        """
        config = cls()
        
        # Embedding configuration
        if os.getenv("RAG_EMBEDDING_PROVIDER"):
            config.embedding.provider = os.getenv("RAG_EMBEDDING_PROVIDER")
        if os.getenv("RAG_EMBEDDING_MODEL"):
            config.embedding.model_name = os.getenv("RAG_EMBEDDING_MODEL")
        if os.getenv("OPENAI_API_KEY"):
            config.embedding.api_key = os.getenv("OPENAI_API_KEY")
        
        # Retrieval configuration
        if os.getenv("RAG_RETRIEVAL_TYPE"):
            config.retrieval.type = os.getenv("RAG_RETRIEVAL_TYPE")
        
        # Security configuration
        if os.getenv("RAG_ENABLE_PII_DETECTION"):
            config.security.enable_pii_detection = os.getenv("RAG_ENABLE_PII_DETECTION").lower() == "true"
        if os.getenv("RAG_GDPR_COMPLIANCE"):
            config.security.gdpr_compliance = os.getenv("RAG_GDPR_COMPLIANCE").lower() == "true"
        if os.getenv("RAG_HIPAA_COMPLIANCE"):
            config.security.hipaa_compliance = os.getenv("RAG_HIPAA_COMPLIANCE").lower() == "true"
        
        # Performance configuration
        if os.getenv("RAG_ENABLE_CACHE"):
            config.performance.enable_cache = os.getenv("RAG_ENABLE_CACHE").lower() == "true"
        if os.getenv("RAG_CACHE_TYPE"):
            config.performance.cache_type = os.getenv("RAG_CACHE_TYPE")
        
        # Vector store configuration
        if os.getenv("RAG_VECTOR_STORE"):
            config.vector_store = os.getenv("RAG_VECTOR_STORE")
        
        # Environment
        if os.getenv("RAG_ENVIRONMENT"):
            config.environment = os.getenv("RAG_ENVIRONMENT")
        if os.getenv("RAG_DEBUG"):
            config.debug = os.getenv("RAG_DEBUG").lower() == "true"
        if os.getenv("RAG_LOG_LEVEL"):
            config.log_level = os.getenv("RAG_LOG_LEVEL")
        
        return config
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'RAGConfig':
        """Load configuration from YAML file."""
        try:
            import yaml
            
            with open(yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            return cls.from_dict(config_dict)
            
        except ImportError:
            raise ImportError("PyYAML is required to load configuration from YAML")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGConfig':
        """Create configuration from dictionary."""
        # This would need more sophisticated parsing
        # For now, return default config
        return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "embedding": {
                "provider": self.embedding.provider,
                "model_name": self.embedding.model_name,
                "dimensions": self.embedding.dimensions,
                "batch_size": self.embedding.batch_size,
            },
            "chunking": {
                "strategy": self.chunking.strategy,
                "chunk_size": self.chunking.chunk_size,
                "chunk_overlap": self.chunking.chunk_overlap,
            },
            "retrieval": {
                "type": self.retrieval.type,
                "similarity_threshold": self.retrieval.similarity_threshold,
                "semantic_weight": self.retrieval.semantic_weight,
                "keyword_weight": self.retrieval.keyword_weight,
            },
            "security": {
                "enable_pii_detection": self.security.enable_pii_detection,
                "gdpr_compliance": self.security.gdpr_compliance,
                "hipaa_compliance": self.security.hipaa_compliance,
                "audit_logging": self.security.audit_logging,
            },
            "performance": {
                "enable_cache": self.performance.enable_cache,
                "cache_type": self.performance.cache_type,
                "query_timeout": self.performance.query_timeout,
            },
            "monitoring": {
                "enable_metrics": self.monitoring.enable_metrics,
                "enable_alerting": self.monitoring.enable_alerting,
                "latency_threshold_ms": self.monitoring.latency_threshold_ms,
            },
            "vector_store": self.vector_store,
            "environment": self.environment,
        }
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of issues.
        
        Implements validation checks for enterprise deployment.
        """
        issues = []
        
        # Security validations
        if self.environment == "production":
            if not self.security.enable_authentication:
                issues.append("Authentication should be enabled in production")
            
            if not self.security.encrypt_at_rest:
                issues.append("Encryption at rest should be enabled in production")
            
            if not self.security.audit_logging:
                issues.append("Audit logging should be enabled in production")
            
            if self.debug:
                issues.append("Debug mode should be disabled in production")
        
        # Performance validations
        if self.performance.query_timeout > 10.0:
            issues.append("Query timeout is very high, may impact user experience")
        
        if self.monitoring.latency_threshold_ms > 5000:
            issues.append("Latency threshold is above recommended 5 second maximum")
        
        # Compliance validations
        if self.security.hipaa_compliance:
            if not self.security.encrypt_at_rest:
                issues.append("HIPAA compliance requires encryption at rest")
            if not self.security.audit_logging:
                issues.append("HIPAA compliance requires audit logging")
            if self.security.log_responses:
                issues.append("HIPAA compliance may be violated by logging responses")
        
        if self.security.gdpr_compliance:
            if not self.security.enable_pii_detection:
                issues.append("GDPR compliance requires PII detection")
            if not self.security.pii_anonymization:
                issues.append("GDPR compliance may require PII anonymization")
        
        # Cost optimization warnings
        if (self.embedding.provider == "openai" and 
            not self.performance.batch_embeddings):
            issues.append("Consider enabling batch embeddings for cost optimization with OpenAI")
        
        return issues
    
    def get_deployment_checklist(self) -> List[Dict[str, Any]]:
        """
        Get deployment checklist for production readiness.
        
        Based on best practices from AI News & Strategy Daily video.
        """
        checklist = [
            {
                "category": "Security",
                "items": [
                    {"check": "Authentication enabled", "status": self.security.enable_authentication},
                    {"check": "Authorization enabled", "status": self.security.enable_authorization},
                    {"check": "Encryption at rest", "status": self.security.encrypt_at_rest},
                    {"check": "Encryption in transit", "status": self.security.encrypt_in_transit},
                    {"check": "Audit logging", "status": self.security.audit_logging},
                    {"check": "PII detection", "status": self.security.enable_pii_detection},
                ]
            },
            {
                "category": "Performance",
                "items": [
                    {"check": "Caching enabled", "status": self.performance.enable_cache},
                    {"check": "Vector DB sharding", "status": self.performance.vector_db_sharding},
                    {"check": "Vector DB replication", "status": self.performance.vector_db_replication},
                    {"check": "Batch embeddings", "status": self.performance.batch_embeddings},
                    {"check": "Query timeout < 30s", "status": self.performance.query_timeout < 30.0},
                ]
            },
            {
                "category": "Monitoring",
                "items": [
                    {"check": "Metrics collection", "status": self.monitoring.enable_metrics},
                    {"check": "Health checks", "status": self.monitoring.enable_liveness},
                    {"check": "Alerting configured", "status": self.monitoring.enable_alerting},
                    {"check": "Tracing enabled", "status": self.monitoring.enable_tracing},
                    {"check": "Latency threshold set", "status": self.monitoring.latency_threshold_ms > 0},
                ]
            },
            {
                "category": "Compliance",
                "items": [
                    {"check": "GDPR compliance", "status": self.security.gdpr_compliance},
                    {"check": "SOC 2 compliance", "status": self.security.soc2_compliance},
                    {"check": "HIPAA compliance", "status": self.security.hipaa_compliance},
                    {"check": "Data retention policy", "status": self.security.retention_days > 0},
                ]
            }
        ]
        
        return checklist
    
    def estimate_monthly_cost(self, queries_per_month: int) -> Dict[str, float]:
        """
        Estimate monthly costs based on configuration.
        
        Implements cost modeling for "saving millions of dollars" optimization.
        """
        costs = {
            "embedding_api": 0.0,
            "vector_storage": 0.0,
            "compute": 0.0,
            "monitoring": 0.0,
            "total": 0.0
        }
        
        # Embedding API costs (if using OpenAI)
        if self.embedding.provider == "openai":
            # Rough estimation: 1 query = ~1000 tokens for embedding
            tokens_per_month = queries_per_month * 1000
            
            pricing = {
                "text-embedding-3-large": 0.00013,
                "text-embedding-3-small": 0.00002,
                "text-embedding-ada-002": 0.0001,
            }
            
            price_per_1k = pricing.get(self.embedding.model_name, 0.0001)
            costs["embedding_api"] = (tokens_per_month / 1000) * price_per_1k
        
        # Vector storage costs (rough estimates)
        if self.vector_store == "pinecone":
            # Pinecone pricing approximation
            if queries_per_month < 100000:
                costs["vector_storage"] = 70.0  # Starter plan
            else:
                costs["vector_storage"] = 200.0  # Standard plan
        
        # Compute costs (rough estimates based on cloud instances)
        if queries_per_month < 10000:
            costs["compute"] = 50.0  # Small instance
        elif queries_per_month < 100000:
            costs["compute"] = 200.0  # Medium instance
        else:
            costs["compute"] = 800.0  # Large instance
        
        # Monitoring costs
        if self.monitoring.enable_metrics:
            costs["monitoring"] = 30.0  # Basic monitoring
        
        costs["total"] = sum(costs.values())
        
        return costs