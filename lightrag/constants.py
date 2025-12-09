"""
Centralized configuration constants for LightRAG.

This module defines default values for configuration constants used across
different parts of the LightRAG system. Centralizing these values ensures
consistency and makes maintenance easier.
"""

# Default values for server settings
DEFAULT_WOKERS = 2
DEFAULT_MAX_GRAPH_NODES = 1000

# Default values for extraction settings
DEFAULT_SUMMARY_LANGUAGE = 'English'  # Default language for document processing
DEFAULT_MAX_GLEANING = 1
DEFAULT_ENTITY_NAME_MAX_LENGTH = 256

# Number of description fragments to trigger LLM summary
DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE = 8
# Max description token size to trigger LLM summary
DEFAULT_SUMMARY_MAX_TOKENS = 1200
# Recommended LLM summary output length in tokens
DEFAULT_SUMMARY_LENGTH_RECOMMENDED = 600
# Maximum token size sent to LLM for summary
DEFAULT_SUMMARY_CONTEXT_SIZE = 12000
# Default entities to extract if ENTITY_TYPES is not specified in .env
DEFAULT_ENTITY_TYPES = [
    'Person',
    'Creature',
    'Organization',
    'Location',
    'Event',
    'Concept',
    'Method',
    'Content',
    'Data',
    'Artifact',
    'NaturalObject',
]

# Separator for: description, source_id and relation-key fields(Can not be changed after data inserted)
GRAPH_FIELD_SEP = '<SEP>'

# Query and retrieval configuration defaults
DEFAULT_TOP_K = 40
DEFAULT_CHUNK_TOP_K = 20
# Token limits increased for modern 128K+ context models (GPT-4o-mini, Claude, etc.)
# Top-k already limits count; these are safety ceilings, not aggressive truncation targets
DEFAULT_MAX_ENTITY_TOKENS = 16000
DEFAULT_MAX_RELATION_TOKENS = 16000
DEFAULT_MAX_TOTAL_TOKENS = 60000
DEFAULT_COSINE_THRESHOLD = 0.35
DEFAULT_RELATED_CHUNK_NUMBER = 12
DEFAULT_KG_CHUNK_PICK_METHOD = 'VECTOR'

# TODO: Deprated. All conversation_history messages is send to LLM.
DEFAULT_HISTORY_TURNS = 0

# Rerank configuration defaults
# Local reranking is enabled by default using mixedbread-ai/mxbai-rerank-xsmall-v1
DEFAULT_ENABLE_RERANK = True
# Minimum rerank score to keep a result - filters out clearly irrelevant chunks
# Note: mxbai-rerank-xsmall-v1 produces scores in 0.01-0.15 range for domain content
# Set to 0.0 to disable filtering (keep all reranked results, just reorder them)
DEFAULT_MIN_RERANK_SCORE = 0.0

# Default source ids limit in meta data for entity and relation
DEFAULT_MAX_SOURCE_IDS_PER_ENTITY = 300
DEFAULT_MAX_SOURCE_IDS_PER_RELATION = 300
### control chunk_ids limitation method: KEEP, FIFO
###    KEEP: Keep oldest (less merge action and faster)
###    FIFO: First in first out
SOURCE_IDS_LIMIT_METHOD_KEEP = 'KEEP'
SOURCE_IDS_LIMIT_METHOD_FIFO = 'FIFO'
DEFAULT_SOURCE_IDS_LIMIT_METHOD = SOURCE_IDS_LIMIT_METHOD_FIFO
VALID_SOURCE_IDS_LIMIT_METHODS = {
    SOURCE_IDS_LIMIT_METHOD_KEEP,
    SOURCE_IDS_LIMIT_METHOD_FIFO,
}
# Maximum number of file paths stored in entity/relation file_path field (For displayed only, does not affect query performance)
DEFAULT_MAX_FILE_PATHS = 100

# Placeholder when file_path list exceeds DEFAULT_MAX_FILE_PATHS (used by all storage backends)
DEFAULT_FILE_PATH_MORE_PLACEHOLDER = 'truncated'

# Default temperature for LLM (lower = more deterministic, less hallucination risk)
DEFAULT_TEMPERATURE = 0.3

# Async configuration defaults
DEFAULT_MAX_ASYNC = 4  # Default maximum async operations
DEFAULT_MAX_PARALLEL_INSERT = 2  # Default maximum parallel insert operations

# Embedding configuration defaults
DEFAULT_EMBEDDING_FUNC_MAX_ASYNC = 8  # Default max async for embedding functions
DEFAULT_EMBEDDING_BATCH_NUM = 10  # Default batch size for embedding computations

# Gunicorn worker timeout
DEFAULT_TIMEOUT = 300

# Default llm and embedding timeout
DEFAULT_LLM_TIMEOUT = 180
DEFAULT_EMBEDDING_TIMEOUT = 30

# Topic connectivity check configuration
DEFAULT_MIN_RELATIONSHIP_DENSITY = 0.3  # Minimum ratio of relationships to entities
DEFAULT_MIN_ENTITY_COVERAGE = 0.5  # Minimum ratio of entities connected by relationships
DEFAULT_CHECK_TOPIC_CONNECTIVITY = True  # Enable topic connectivity check by default

# Logging configuration defaults
DEFAULT_LOG_MAX_BYTES = 10485760  # Default 10MB
DEFAULT_LOG_BACKUP_COUNT = 5  # Default 5 backups
DEFAULT_LOG_FILENAME = 'lightrag.log'  # Default log filename

# Ollama server configuration defaults
DEFAULT_OLLAMA_MODEL_NAME = 'lightrag'
DEFAULT_OLLAMA_MODEL_TAG = 'latest'
DEFAULT_OLLAMA_MODEL_SIZE = 7365960935
DEFAULT_OLLAMA_CREATED_AT = '2024-01-15T00:00:00Z'
DEFAULT_OLLAMA_DIGEST = 'sha256:lightrag'

# Full-text search cache configuration
# Shorter TTL than embedding cache since document content changes more frequently
DEFAULT_FTS_CACHE_TTL = 300  # 5 minutes
DEFAULT_FTS_CACHE_MAX_SIZE = 5000  # Smaller than embedding cache
DEFAULT_FTS_CACHE_ENABLED = True

# Metrics configuration
DEFAULT_METRICS_ENABLED = True
DEFAULT_METRICS_HISTORY_SIZE = 1000  # Queries to keep in circular buffer
DEFAULT_METRICS_WINDOW_SECONDS = 3600  # 1 hour window for percentile calculations
