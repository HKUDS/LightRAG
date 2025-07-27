"""
Centralized configuration constants for LightRAG.

This module defines default values for configuration constants used across
different parts of the LightRAG system. Centralizing these values ensures
consistency and makes maintenance easier.
"""

# Default values for server settings
DEFAULT_WOKERS = 2
DEFAULT_TIMEOUT = 150

# Default values for extraction settings
DEFAULT_SUMMARY_LANGUAGE = "English"  # Default language for summaries
DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE = 4
DEFAULT_MAX_GLEANING = 1
DEFAULT_SUMMARY_MAX_TOKENS = 10000  # Default maximum token size

# Separator for graph fields
GRAPH_FIELD_SEP = "<SEP>"

# Query and retrieval configuration defaults
DEFAULT_TOP_K = 40
DEFAULT_CHUNK_TOP_K = 10
DEFAULT_MAX_ENTITY_TOKENS = 10000
DEFAULT_MAX_RELATION_TOKENS = 10000
DEFAULT_MAX_TOTAL_TOKENS = 30000
DEFAULT_HISTORY_TURNS = 0
DEFAULT_COSINE_THRESHOLD = 0.2
DEFAULT_RELATED_CHUNK_NUMBER = 5

# Rerank configuration defaults
DEFAULT_ENABLE_RERANK = True
DEFAULT_MIN_RERANK_SCORE = 0.0

# File path configuration for vector and graph database
DEFAULT_MAX_FILE_PATH_LENGTH = 4090

# Async configuration defaults
DEFAULT_MAX_ASYNC = 4  # Default maximum async operations

# Embedding configuration defaults
DEFAULT_EMBEDDING_FUNC_MAX_ASYNC = 8  # Default max async for embedding functions
DEFAULT_EMBEDDING_BATCH_NUM = 10  # Default batch size for embedding computations

# Logging configuration defaults
DEFAULT_LOG_MAX_BYTES = 10485760  # Default 10MB
DEFAULT_LOG_BACKUP_COUNT = 5  # Default 5 backups
DEFAULT_LOG_FILENAME = "lightrag.log"  # Default log filename

# Ollama server configuration defaults
DEFAULT_OLLAMA_MODEL_NAME = "lightrag"
DEFAULT_OLLAMA_MODEL_TAG = "latest"
DEFAULT_OLLAMA_MODEL_SIZE = 7365960935
DEFAULT_OLLAMA_CREATED_AT = "2024-01-15T00:00:00Z"
DEFAULT_OLLAMA_DIGEST = "sha256:lightrag"
