"""
Centralized configuration constants for LightRAG.

This module defines default values for configuration constants used across
different parts of the LightRAG system. Centralizing these values ensures
consistency and makes maintenance easier.
"""

from typing import Literal, TypeAlias

# Default values for server settings
DEFAULT_WOKERS = 2
DEFAULT_MAX_GRAPH_NODES = 1000

# Default values for extraction settings
DEFAULT_SUMMARY_LANGUAGE = "English"  # Default language for document processing
DEFAULT_MAX_GLEANING = 1
DEFAULT_ENTITY_NAME_MAX_LENGTH = 256
# Max UTF-8 byte length for entity identifiers. Milvus enforces VARCHAR
# max_length in BYTES (not characters), so a CJK name within the character
# limit can still exceed the field limit. MUST stay <= the max_length of the
# entity_name / src_id / tgt_id fields in lightrag/kg/milvus_impl.py.
DEFAULT_ENTITY_NAME_MAX_BYTES = 512

# Per-response output limits for entity extraction prompts
DEFAULT_MAX_EXTRACTION_RECORDS = 100
DEFAULT_MAX_EXTRACTION_ENTITIES = 40

# Number of description fragments to trigger LLM summary
DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE = 8
# Max description token size to trigger LLM summary
DEFAULT_SUMMARY_MAX_TOKENS = 1200
# Recommended LLM summary output length in tokens
DEFAULT_SUMMARY_LENGTH_RECOMMENDED = 600
# Maximum token size sent to LLM for summary
DEFAULT_SUMMARY_CONTEXT_SIZE = 12000
# Maximum token size allowed for entity extraction input context
DEFAULT_MAX_EXTRACT_INPUT_TOKENS = 20480
# Maximum token size for the per-chunk `---Section Context---` heading
# breadcrumb injected into the extraction prompt. Keeps section metadata from
# pushing an otherwise-valid chunk past the provider context window; the
# breadcrumb is truncated (nearest section kept) when it exceeds this budget.
DEFAULT_MAX_SECTION_CONTEXT_TOKENS = 256
# Separator for: description, source_id and relation-key fields(Can not be changed after data inserted)
GRAPH_FIELD_SEP = "<SEP>"

# Query and retrieval configuration defaults
DEFAULT_TOP_K = 40
DEFAULT_CHUNK_TOP_K = 20
DEFAULT_MAX_ENTITY_TOKENS = 6000
DEFAULT_MAX_RELATION_TOKENS = 8000
DEFAULT_MAX_TOTAL_TOKENS = 30000
DEFAULT_COSINE_THRESHOLD = 0.2
DEFAULT_RELATED_CHUNK_NUMBER = 5
DEFAULT_KG_CHUNK_PICK_METHOD = "VECTOR"

# Rerank configuration defaults
DEFAULT_MIN_RERANK_SCORE = 0.0
DEFAULT_RERANK_BINDING = "null"

# Default source ids limit in meta data for entity and relation
DEFAULT_MAX_SOURCE_IDS_PER_ENTITY = 200
DEFAULT_MAX_SOURCE_IDS_PER_RELATION = 200
### control chunk_ids limitation method: FIFO, FIFO
###    FIFO: First in first out
###    KEEP: Keep oldest (less merge action and faster)
SOURCE_IDS_LIMIT_METHOD_KEEP = "KEEP"
SOURCE_IDS_LIMIT_METHOD_FIFO = "FIFO"
DEFAULT_SOURCE_IDS_LIMIT_METHOD = SOURCE_IDS_LIMIT_METHOD_KEEP
VALID_SOURCE_IDS_LIMIT_METHODS = {
    SOURCE_IDS_LIMIT_METHOD_KEEP,
    SOURCE_IDS_LIMIT_METHOD_FIFO,
}
# Maximum number of file paths stored in entity/relation file_path field (For displayed only, does not affect query performance)
DEFAULT_MAX_FILE_PATHS = 75

# Field length of file_path in Milvus Schema for entity and relation (Should not be changed)
# file_path must store all file paths up to the DEFAULT_MAX_FILE_PATHS limit within the metadata.
DEFAULT_MAX_FILE_PATH_LENGTH = 32768
# Placeholder for more file paths in meta data for entity and relation (Should not be changed)
DEFAULT_FILE_PATH_MORE_PLACEHOLDER = "truncated"

# Default temperature for LLM
DEFAULT_TEMPERATURE = 1.0

# Async configuration defaults
DEFAULT_MAX_ASYNC = 4  # Default maximum async operations
DEFAULT_MAX_PARALLEL_INSERT = 3  # Default maximum parallel insert operations

# Chunker defaults — i18n-aware so Chinese / mixed-language documents
# split correctly out of the box.  Override per deployment via
# CHUNK_R_SEPARATORS / CHUNK_V_SENTENCE_SPLIT_REGEX env vars.
#
# DEFAULT_R_SEPARATORS: cascade tried by langchain RecursiveCharacterTextSplitter.
# Order matters — strongest boundary first: paragraph (\n\n) > line (\n) >
# Chinese sentence-end (。！？) > Chinese semi-clause (；，) > space > char.
# English sentence-ending punctuation (.?!) is intentionally NOT included
# because RecursiveCharacterTextSplitter does literal-string splitting, so
# "." would also split numerals (``0.95``) and abbreviations (``e.g.``).
# The English path falls through space / char as before.
DEFAULT_R_SEPARATORS: tuple[str, ...] = (
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    "，",
    " ",
    "",
)
# DEFAULT_SENTENCE_SPLIT_REGEX: pattern fed to langchain SemanticChunker.
# Two alternates so the English branch keeps its ``\s+`` requirement
# (avoiding ``0.95`` mid-token splits) while the Chinese branch matches
# bare ``。？！`` (CJK has no inter-sentence whitespace).
DEFAULT_SENTENCE_SPLIT_REGEX = r"(?<=[.?!])\s+|(?<=[。？！])"

# DEFAULT_CHUNK_P_SIZE: paragraph-semantic chunker target size when
# CHUNK_P_SIZE env is unset.  Deliberately larger than the global
# CHUNK_SIZE default — heading-aligned paragraph merging needs more
# headroom to keep semantically related paragraphs together; falling
# back to CHUNK_SIZE (1200) would force premature splits and defeat
# the strategy's purpose.
DEFAULT_CHUNK_P_SIZE = 2000

# LightRAG Document pipeline
FULL_DOCS_FORMAT_RAW = "raw"  # content in full_docs["content"]
FULL_DOCS_FORMAT_LIGHTRAG = "lightrag"  # content in LightRAG Document files
FULL_DOCS_FORMAT_PENDING_PARSE = (
    "pending_parse"  # file saved but not yet parsed; parse_native will read from disk
)
# Marker prefix for full_docs.content when format=lightrag.
# Per docs/FileProcessingConfiguration-zh.md, the content is "{{LRdoc}}" + a
# leading summary of the parsed document so paginated APIs can show a real
# preview without loading the full LightRAG Document file.
LIGHTRAG_DOC_CONTENT_PREFIX = "{{LRdoc}}"
PARSER_ENGINE_LEGACY = "legacy"
PARSER_ENGINE_NATIVE = "native"
PARSER_ENGINE_MINERU = "mineru"
PARSER_ENGINE_DOCLING = "docling"
SUPPORTED_PARSER_ENGINES = frozenset(
    {
        PARSER_ENGINE_LEGACY,
        PARSER_ENGINE_NATIVE,
        PARSER_ENGINE_MINERU,
        PARSER_ENGINE_DOCLING,
    }
)
PARSER_ENGINE_SUFFIX_CAPABILITIES = {
    PARSER_ENGINE_LEGACY: frozenset(
        {
            "txt",
            "md",
            "mdx",
            "pdf",
            "docx",
            "pptx",
            "xlsx",
            "rtf",
            "odt",
            "tex",
            "epub",
            "html",
            "htm",
            "csv",
            "json",
            "xml",
            "yaml",
            "yml",
            "log",
            "conf",
            "ini",
            "properties",
            "sql",
            "bat",
            "sh",
            "c",
            "h",
            "cpp",
            "hpp",
            "py",
            "java",
            "js",
            "ts",
            "swift",
            "go",
            "rb",
            "php",
            "css",
            "scss",
            "less",
        }
    ),
    PARSER_ENGINE_NATIVE: frozenset({"docx"}),
    PARSER_ENGINE_MINERU: frozenset(
        {
            "pdf",
            "doc",
            "docx",
            "ppt",
            "pptx",
            "xls",
            "xlsx",
            "png",
            "jpg",
            "jpeg",
            "jp2",
            "webp",
            "gif",
            "bmp",
        }
    ),
    PARSER_ENGINE_DOCLING: frozenset(
        {
            "pdf",
            "docx",
            "pptx",
            "xlsx",
            "md",
            "html",
            "xhtml",
            "png",
            "jpg",
            "jpeg",
            "tiff",
            "webp",
            "bmp",
        }
    ),
}
PARSED_DIR_NAME = "__parsed__"  # Dir for parsed files (renamed from __enqueued__)

# Suffixes for parser artifact subdirectories under ``<input>/__parsed__/``.
# Centralising them here keeps the sidecar writer, engine cache modules and
# the delete-path whitelist in sync — new engines should add their raw-dir
# suffix to ``PARSED_ARTIFACT_DIR_SUFFIXES`` so deletion picks them up
# automatically.
PARSED_DIR_SUFFIX = ".parsed"  # spec sidecar layout (every engine)
MINERU_RAW_DIR_SUFFIX = ".mineru_raw"  # preserved MinerU raw bundle
DOCLING_RAW_DIR_SUFFIX = ".docling_raw"  # preserved Docling raw bundle
PARSED_ARTIFACT_DIR_SUFFIXES: tuple[str, ...] = (
    PARSED_DIR_SUFFIX,
    MINERU_RAW_DIR_SUFFIX,
    DOCLING_RAW_DIR_SUFFIX,
)

# Per-file processing options carried by filename hints / LIGHTRAG_PARSER rules.
# See docs/FileProcessingConfiguration-zh.md for the full specification.
PROCESS_OPTION_IMAGES = "i"  # Enable VLM analysis for drawings/images
PROCESS_OPTION_TABLES = "t"  # Enable VLM analysis for tables
PROCESS_OPTION_EQUATIONS = "e"  # Enable VLM analysis for equations
PROCESS_OPTION_SKIP_KG = "!"  # Skip entity/relation extraction (no KG build)
ProcessChunkingOption: TypeAlias = Literal["F", "R", "V", "P"]
PROCESS_OPTION_CHUNK_FIXED: ProcessChunkingOption = (
    "F"  # Fixed-length / separator chunking (default)
)
PROCESS_OPTION_CHUNK_RECURSIVE: ProcessChunkingOption = (
    "R"  # Recursive semantic chunking
)
PROCESS_OPTION_CHUNK_VECTOR: ProcessChunkingOption = (
    "V"  # Vector-driven semantic chunking
)
PROCESS_OPTION_CHUNK_PARAGRAH: ProcessChunkingOption = (
    "P"  # Paragrah-driven semantic chunking
)

PROCESS_OPTION_CHUNK_CHARS: frozenset[ProcessChunkingOption] = frozenset(
    {
        PROCESS_OPTION_CHUNK_FIXED,
        PROCESS_OPTION_CHUNK_RECURSIVE,
        PROCESS_OPTION_CHUNK_VECTOR,
        PROCESS_OPTION_CHUNK_PARAGRAH,
    }
)
SUPPORTED_PROCESS_OPTIONS = frozenset(
    {
        PROCESS_OPTION_IMAGES,
        PROCESS_OPTION_TABLES,
        PROCESS_OPTION_EQUATIONS,
        PROCESS_OPTION_SKIP_KG,
        PROCESS_OPTION_CHUNK_FIXED,
        PROCESS_OPTION_CHUNK_RECURSIVE,
        PROCESS_OPTION_CHUNK_VECTOR,
        PROCESS_OPTION_CHUNK_PARAGRAH,
    }
)

DEFAULT_MAX_PARALLEL_ANALYZE = 5  # Multimodal analysis (VLM) concurrency

# Per-engine parsing concurrency defaults.  mineru / docling are
# resource-intensive (GPU/CPU + memory), so they default to a modest amount of
# parallelism (2); lower to 1 when resources are tight, or raise via the
# MAX_PARALLEL_PARSE_* env vars when you have spare capacity.
DEFAULT_MAX_PARALLEL_PARSE_NATIVE = 5
DEFAULT_MAX_PARALLEL_PARSE_MINERU = 2
DEFAULT_MAX_PARALLEL_PARSE_DOCLING = 2

# Staged pipeline queue size defaults.
DEFAULT_QUEUE_SIZE_PARSE = 20
DEFAULT_QUEUE_SIZE_ANALYZE = 100
DEFAULT_QUEUE_SIZE_INSERT = 4

# LLM / embedding call priority levels.  Lower values run first
# (asyncio.PriorityQueue semantics); priority only orders calls *within* a
# single role queue (extract / keyword / query / vlm).  These name the values
# passed as the ``_priority`` argument to the priority_limit_async_func_call
# wrapper, centralizing the magic numbers that were previously inlined at each
# call site.
#
# Query stage (interactive: query/keyword LLM calls and query-time embeddings)
# gets the highest priority so user requests stay responsive.
DEFAULT_QUERY_PRIORITY = 5
# Entity/relation description summary generation — ahead of raw extraction but
# behind interactive query work.
DEFAULT_SUMMARY_PRIORITY = 8
# Processing stage entity/relation extraction (ingestion).  Also the wrapper's
# baseline default for any call that does not pass ``_priority``.
DEFAULT_PROCESSING_PRIORITY = 10
# Priority used for all multimodal analysis LLM calls.  Set equal to
# DEFAULT_PROCESSING_PRIORITY so analysis and ingestion work share the EXTRACT
# queue fairly and advance evenly — otherwise a busy ingestion queue starves
# analysis tasks, stalling analysis nodes and dragging down overall throughput.
DEFAULT_MM_ANALYSIS_PRIORITY = DEFAULT_PROCESSING_PRIORITY

# Multimodal analysis / chunk thresholds
# Minimum token count retained when truncating a multimodal chunk's
# description to fit within DEFAULT_MAX_EXTRACT_INPUT_TOKENS.  Falling below
# this floor leaves the description too thin to ground a useful entity
# description, so the pipeline raises instead of producing a stub.
DEFAULT_MM_CHUNK_DESCRIPTION_MIN_TOKENS = 100
# Minimum image side (width or height) in pixels accepted for VLM analysis.
# Anything smaller is treated as decorative (icons, separators, etc.) and
# written as status="skipped".
DEFAULT_MM_IMAGE_MIN_PIXEL = 32

# Embedding configuration defaults
DEFAULT_EMBEDDING_FUNC_MAX_ASYNC = 8  # Default max async for embedding functions
DEFAULT_EMBEDDING_BATCH_NUM = 10  # Default batch size for embedding computations

# Gunicorn worker timeout
DEFAULT_TIMEOUT = 300

# Default llm and embedding timeout
DEFAULT_LLM_TIMEOUT = 180
DEFAULT_EMBEDDING_TIMEOUT = 30

# Rerank async / timeout defaults
# Concurrency falls back to base MAX_ASYNC_LLM when env unset; timeout has its own
# default since reranker calls are typically much faster than full LLM generation.
DEFAULT_RERANK_MAX_ASYNC = DEFAULT_MAX_ASYNC
DEFAULT_RERANK_TIMEOUT = 30

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
