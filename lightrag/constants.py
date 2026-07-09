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
# pushing an otherwise-valid chunk past the provider context window; over budget
# the breadcrumb collapses to ``first → … → leaf`` (top-level + nearest section).
DEFAULT_MAX_SECTION_CONTEXT_TOKENS = 256
# Per-level character cap for each heading in that breadcrumb. Must stay below
# 1/3 of DEFAULT_MAX_SECTION_CONTEXT_TOKENS so the collapsed two-level form
# (first + leaf, plus separator/ellipsis) always fits within the token budget.
DEFAULT_HEADING_LEVEL_MAX_CHARS = 80
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

# Paragraph-semantic "drop references" detection defaults (the chunking="P"
# drop_references option).  DEFAULT_P_REFERENCES_TAIL_N: a reference block is
# only dropped when it sits within the last N content blocks of the document
# (a safety window so a mid-document "References" subsection is not removed).
# DEFAULT_P_REFERENCES_HEADINGS: heading prefixes that mark a reference
# section — English words matched case-insensitively at a word boundary,
# the Chinese "参考文献" matched as a plain prefix.  Both are tunable via env
# (CHUNK_P_REFERENCES_TAIL_N / CHUNK_P_REFERENCES_HEADINGS, the latter
# pipe-separated) read live by the chunker at run time.
DEFAULT_P_REFERENCES_TAIL_N = 2
DEFAULT_P_REFERENCES_HEADINGS = ("References", "Bibliography", "参考文献")

# Native docx smart_heading (opt-in engine param) tunables. Each DEFAULT_*
# below has a matching env var (drop the DEFAULT_ prefix) read at run time
# by lightrag/parser/docx/smart_heading (same live-env pattern as the
# CHUNK_P_REFERENCES_* knobs above).
#
# Global default for the smart_heading engine param itself (env
# DOCX_SMART_HEADING): when true, .docx files routed to the native engine get
# smart_heading enabled by default — an explicit native(smart_heading=false)
# rule/hint turns it back off — and the server verifies at startup that the
# spaCy models are installed (fail-fast instead of failing mid-pipeline).
# The default is materialized into the persisted parse_engine at ingestion
# time (seed_smart_heading_param, called from resolve_parser_directives for
# uploads and from apipeline_enqueue_documents for direct enqueue), so
# toggling the env var never changes how already-ingested documents re-parse.
DEFAULT_DOCX_SMART_HEADING = False
#
# P1 homophone blacklist: an EnNum immediately followed by one of these CJK
# unit/measure words (2026年 / 1个 / 1人…) is NOT a numbering prefix. Env
# DOCX_SMART_ENNUM_BLACKLIST is comma/pipe-separated.
DEFAULT_DOCX_SMART_ENNUM_BLACKLIST = (
    "年,月,日,时,分,秒,个,人,只,条,次,项,天,号,元,角,件,名,台,种,倍,"
    "万,亿,岁,周,米,克,吨,页,寸,层,间,句,字,笔,轮,批,组,套,户,家,期"
)
# P3 caption prefixes: a paragraph starting with one of these + a numbering
# shape is a figure/table caption, never a heading. Comma/pipe-separated.
DEFAULT_DOCX_SMART_CAPTION_PREFIXES = "图,表,公式,Figure,Table,Fig.,Eq.,Chart"
# 公文版记 (imprint) ANCHORS: a paragraph opening with one of these + [：:]
# (抄送：各区人民政府 / 主题词：经济 管理) starts a 版记 region. It is imprint
# metadata — body, never a heading — and it vetoes title-block membership for
# itself AND its 2 preceding non-blank paragraphs (the signature/date lines
# above). Anchors may also sit as middle content of another anchor's region
# (主题词 then 抄送); only a CLOSER (below) ends a region.
# Comma/pipe-separated; whitespace may interleave the prefix chars (抄　送：).
DEFAULT_DOCX_SMART_IMPRINT_COLON_PREFIXES = "抄送,主题词"
# 版记 region CLOSER markers (印发-family — the issuing-organ / print line that
# ENDS a 版记). Recognized ONLY inside the forward window of an anchor above
# (never a standalone per-line rule), so a line-final 印发 in body prose
# (…已印发) cannot false-fire on its own:
#   - closer prefix: line opens with the prefix + [：:] or whitespace
#     (印发：XX / 印发 XX / 印发机关 XX / 印发机关：XX — 印发机关 is a closer, NOT an
#     anchor, which is why there is no longer an IMPRINT_SPACE_PREFIXES knob);
#   - closer trailing: line ENDS with 印发 (某某办公室 2026年6月30日 印发) — the
#     GB/T standard layout with the issuer/date first and 印发 last.
# Comma/pipe-separated. When a closer is found within FORWARD_PARAS non-blank
# paragraphs of an anchor, the whole 抄送…印发 span (middle lines included) is
# barred from title blocks; and if a valid title block immediately follows the
# span (a 公文汇编 boundary), the span's lines are force-demoted to body.
DEFAULT_DOCX_SMART_IMPRINT_CLOSER_PREFIXES = "印发,印发机关"
DEFAULT_DOCX_SMART_IMPRINT_CLOSER_TRAILING = "印发"
DEFAULT_DOCX_SMART_IMPRINT_FORWARD_PARAS = 3
# Heuristic TOC evidence: at least this many consecutive dot-leader lines.
DEFAULT_DOCX_SMART_TOC_MIN_LINES = 3
# CB1 heading-density ceiling (headings / non-empty paragraphs): the floor of
# the effective threshold. The threshold is baseline-aware — a document with a
# rich physical outline naturally admits more candidates, so the effective
# ceiling is max(this floor, baseline outline density + the margin below).
DEFAULT_DOCX_SMART_DENSITY_MAX = 0.35
# CB1 margin added to the sub-document's baseline outline density when it beats
# the floor above (percentage points, not relative).
DEFAULT_DOCX_SMART_DENSITY_BASELINE_MARGIN = 0.10
# Density re-estimation trigger #2 (trigger side only, §2.3.3): also trips when
# the mean CJK-weighted body chars between adjacent candidate headings falls
# below this — a dense run of headings by spacing rather than by count.
DEFAULT_DOCX_SMART_MIN_INTER_HEADING_CHARS = 200
# CB2 demotion-propagation breaker ratios.
DEFAULT_DOCX_SMART_CB2_BODY_RATIO = 0.20
DEFAULT_DOCX_SMART_CB2_OUTLINE_RATIO = 0.50
# CB5 FS_base confidence threshold (dominant-size char share).
DEFAULT_DOCX_SMART_CONFIDENCE_RATIO = 0.60
# CB4 whole-document gate: below this many tokens the whole document is too
# short for smart heading — it is skipped entirely (baseline output). Kept
# below DEFAULT_CHUNK_P_SIZE (2000): a document that fits inside a single
# paragraph-chunk does not need heading-driven structural splitting.
DEFAULT_DOCX_SMART_MIN_TOKENS = 1800
# CB4 per-sub-document gate: once the whole document clears the gate above,
# an individual sub-document below this many tokens falls back to outline-only
# levels (``_outline_only_decisions``) instead of size-based leveling. Lower
# than the whole-document gate — a sub-document is a fragment and needs less
# content to be worth leveling.
DEFAULT_DOCX_SMART_SUBDOC_MIN_TOKENS = 1000
# Title-block LLM window cap (tokens) — content beyond it is truncated.
DEFAULT_DOCX_SMART_LLM_WINDOW_TOKENS = 1000
# Single-paragraph title-block gate: font size must exceed the global
# FS_base mean by at least this many points.
DEFAULT_DOCX_SMART_TITLE_BLOCK_MIN_DELTA = 2.0
# Per-document cap on single-paragraph title-block LLM reviews.
DEFAULT_DOCX_SMART_SINGLE_TITLE_LLM_MAX = 20
# Open numbering series: close a series after this many consecutive body
# paragraphs (0 disables the auxiliary body-run break; the primary close
# signal is the level returning to an ancestor scope).
DEFAULT_DOCX_SMART_SEQ_BREAK_PARAS = 0
# Strong-body length threshold in English-equivalent chars (1 CJK ≈ 3).
DEFAULT_DOCX_SMART_HEADING_MAX_CHARS = 180

# LightRAG Document pipeline
FULL_DOCS_FORMAT_RAW = "raw"  # content in full_docs["content"]
# Post-parse persistence marker: full_docs rows written by the parsers carry
# this parse_format; on resume/retry they route to ReuseParser. Not a valid
# enqueue docs_format (the 'lightrag' ingestion entrypoint was removed).
FULL_DOCS_FORMAT_LIGHTRAG = "lightrag"  # content in LightRAG Document files
FULL_DOCS_FORMAT_PENDING_PARSE = (
    "pending_parse"  # file saved but not yet parsed; parse_native will read from disk
)
# Marker prefix for full_docs.content when format=lightrag.
# Per docs/FileProcessingConfiguration-zh.md, the content is "{{LRdoc}}" + a
# leading summary of the parsed document so paginated APIs can show a real
# preview without loading the full LightRAG Document file.
LIGHTRAG_DOC_CONTENT_PREFIX = "{{LRdoc}}"
# Engine identifier strings (registry keys). The set of user-selectable
# engines and their suffix capabilities now live in
# lightrag.parser.registry (ParserSpec table) — the single source of truth.
PARSER_ENGINE_LEGACY = "legacy"
PARSER_ENGINE_NATIVE = "native"
PARSER_ENGINE_MINERU = "mineru"
PARSER_ENGINE_DOCLING = "docling"
PARSED_DIR_NAME = "__parsed__"  # Dir for parsed files (renamed from __enqueued__)
# Prefix marking a doc_status content_summary as GENERATED from a file
# extraction error (enqueue-time error documents and parse-stage FAILED
# upserts). Doubles as the match sentinel that lets a later failure replace
# a stale generated summary while real raw-document summaries are preserved —
# keep every producer on this constant so the match never drifts.
FILE_EXTRACTION_SUMMARY_PREFIX = "[File Extraction]"

# Suffixes for parser artifact subdirectories under ``<input>/__parsed__/``.
# Centralising them here keeps the sidecar writer, engine cache modules and
# the delete-path whitelist in sync — new engines should add their raw-dir
# suffix to ``PARSED_ARTIFACT_DIR_SUFFIXES`` so deletion picks them up
# automatically.
PARSED_DIR_SUFFIX = ".parsed"  # spec sidecar layout (every engine)
MINERU_RAW_DIR_SUFFIX = ".mineru_raw"  # preserved MinerU raw bundle
DOCLING_RAW_DIR_SUFFIX = ".docling_raw"  # preserved Docling raw bundle
NATIVE_RAW_DIR_SUFFIX = ".native_raw"  # native md downloaded-image cache bundle
PARSED_ARTIFACT_DIR_SUFFIXES: tuple[str, ...] = (
    PARSED_DIR_SUFFIX,
    MINERU_RAW_DIR_SUFFIX,
    DOCLING_RAW_DIR_SUFFIX,
    NATIVE_RAW_DIR_SUFFIX,
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
DEFAULT_MM_IMAGE_MIN_PIXEL = 64

# Embedding configuration defaults
DEFAULT_EMBEDDING_FUNC_MAX_ASYNC = 8  # Default max async for embedding functions
DEFAULT_EMBEDDING_BATCH_NUM = 10  # Default batch size for embedding computations

# Gunicorn worker timeout
DEFAULT_TIMEOUT = 300

# Default llm and embedding timeout
DEFAULT_LLM_TIMEOUT = 240
DEFAULT_EMBEDDING_TIMEOUT = 30

# Rerank async / timeout defaults
# Concurrency falls back to base MAX_ASYNC_LLM when env unset; timeout has its own
# default since reranker calls are typically much faster than full LLM generation.
DEFAULT_RERANK_MAX_ASYNC = DEFAULT_MAX_ASYNC
DEFAULT_RERANK_TIMEOUT = 30

# Cross-worker global concurrency gate (gunicorn multi-worker) defaults.
# A lease whose heartbeat is older than the TTL marks its owner as suspect;
# a suspect lease is reclaimed only after the additional grace elapses while
# the owner PID is still alive (dead PIDs are reclaimed immediately).
DEFAULT_GLOBAL_SLOT_HEARTBEAT_TTL = 20.0  # ~4x the 5s health-check heartbeat
DEFAULT_GLOBAL_SLOT_SUSPECT_GRACE = 20.0  # ~1x heartbeat TTL
# Polling backoff bounds while a worker waits for a free global slot.
# The first acquisition attempt is always immediate (backoff applies only
# after a failure). The longest-waiting live process keeps polling at the
# MIN interval so it usually claims the next freed slot (soft FIFO across
# workers); other waiters back off exponentially up to the DEFERRED cap.
# The cap stays small on purpose: when the favored waiter leaves, the
# promoted one is asleep at most one deferred period, bounding slot idling.
DEFAULT_GLOBAL_SLOT_POLL_MIN = 0.05
DEFAULT_GLOBAL_SLOT_POLL_DEFERRED_MAX = 0.4
# Waiter records not refreshed within this TTL are ignored for the
# longest-waiter ranking and reaped: a crashed or stalled poller must not
# keep occupying the favored seat (which would push every live waiter onto
# the deferred backoff and waste slots). Keep > 2x the deferred poll cap.
DEFAULT_GLOBAL_SLOT_WAITER_STALE_TTL = 1.0
# Max consecutive zombie (cancelled) queue entries a worker drains while
# holding a global slot before returning the slot to other processes.
DEFAULT_GLOBAL_SLOT_DRAIN_LIMIT = 16
# Physical queue compaction (global-limit mode only): triggered when the
# estimated zombie count exceeds the threshold; each maintenance pass
# processes at most the batch limit to keep the event loop responsive.
DEFAULT_ZOMBIE_COMPACT_THRESHOLD = 64
DEFAULT_COMPACT_BATCH_LIMIT = 512
# Cross-worker queue stats: snapshots older than the stale TTL (and entries
# owned by dead PIDs) are reaped during aggregation; publishes triggered by
# counter updates are debounced to the min interval.
DEFAULT_QUEUE_STATS_STALE_TTL = 15.0
DEFAULT_QUEUE_STATS_MIN_PUBLISH_INTERVAL = 0.1

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
