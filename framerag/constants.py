"""Centralized configuration constants for FrameRAG.

All magic numbers and defaults live here so they can be imported,
overridden per-project, or exposed in a .env / settings file without
hunting through the codebase.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────────────────────────────────────

# Maximum gleaning rounds per chunk (0 = disabled, higher = more thorough)
DEFAULT_MAX_GLEANING = 1

# Entity types the extractor should look for (passed to LLM as guidance)
DEFAULT_ENTITY_TYPES = [
    "Person",
    "Organization",
    "Location",
    "Event",
    "Concept",
    "Method",
    "Artifact",
    "Data",
    "NaturalObject",
]

# Maximum tokens fed to the LLM for entity/event/role extraction prompts
DEFAULT_MAX_EXTRACT_INPUT_TOKENS = 20_480

# Number of accumulated descriptions before LLM summarisation is triggered
DEFAULT_FORCE_LLM_SUMMARY_ON_MERGE = 8

# Total character length of descriptions before LLM summarisation is triggered
DEFAULT_DESC_MERGE_THRESHOLD = 3_500

# Max tokens for description summary output
DEFAULT_SUMMARY_MAX_TOKENS = 1_200

# Recommended output length for summary (hint to LLM)
DEFAULT_SUMMARY_LENGTH_RECOMMENDED = 600

# Max input token budget when building description summary context
DEFAULT_SUMMARY_CONTEXT_SIZE = 12_000

# Language used for extraction prompts and LLM output
DEFAULT_SUMMARY_LANGUAGE = "English"

# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CHUNK_SIZE = 1_200      # tokens per chunk
DEFAULT_CHUNK_OVERLAP = 100     # overlap tokens between adjacent chunks

# ─────────────────────────────────────────────────────────────────────────────
# Hypergraph diffusion
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_DIFFUSION_WARM_UP = 3   # Phase 1: fully-bidirectional steps for signal exploration
DEFAULT_DIFFUSION_ALPHA = 0.15  # restart probability α
DEFAULT_DIFFUSION_T_DECAY = 0.7 # Phase 2 cooling base: w_back(k) = t_decay^k
                                 # 1.0 = no decay, 0.7 ≈ 13 cooling steps, 0.5 ≈ 7 steps
DEFAULT_DIFFUSION_EPSILON = 0.01 # Phase 2 stops when w_back < epsilon (dynamic termination)

# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TOP_CHUNKS = 20         # text chunks returned to answer LLM
DEFAULT_TOP_FRAMES = 10         # frame instances returned to answer LLM
DEFAULT_TOP_NODES  = 15         # entity/info nodes used for diffusion seeding

# Cosine similarity threshold below which vector hits are discarded
DEFAULT_COSINE_THRESHOLD = 0.2

# ── Seed bridging (query→content recall) ─────────────────────────────────────
# HyDE: generate a hypothetical answer passage and use its embedding as an
# additional seed. Bridges the lexical gap on paraphrased/disguised queries.
DEFAULT_ENABLE_HYDE = True
# Weight applied to HyDE-derived seed scores relative to the raw query (1.0 =
# equal footing — union of both retrieval sets).
DEFAULT_HYDE_WEIGHT = 1.0
# Weight applied to entity_hint-derived canonical-entity seeds (query_processing
# already extracts these; previously they were discarded).
DEFAULT_ENTITY_HINT_WEIGHT = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# Reranking
# ─────────────────────────────────────────────────────────────────────────────

# Chunks fetched from diffusion before reranker trims to DEFAULT_TOP_CHUNKS
DEFAULT_RERANK_TOP_K = 50

# Hits below this reranker relevance score are excluded (0.0 = keep all)
DEFAULT_MIN_RERANK_SCORE = 0.0

# ─────────────────────────────────────────────────────────────────────────────
# Concurrency & timeouts
# ─────────────────────────────────────────────────────────────────────────────

# Maximum parallel LLM coroutines (semaphore)
# Keep at 2 to stay within 200K TPM: 2 calls × ~2K tokens × 60/min ≈ 240K TPM peak
DEFAULT_MAX_ASYNC = 2

# Maximum parallel document insertions in ainsert_batch()
DEFAULT_MAX_PARALLEL_INSERT = 2

# Seconds before a single LLM call is cancelled with asyncio.TimeoutError
DEFAULT_LLM_TIMEOUT = 120.0

# Seconds before a single embed call is cancelled
DEFAULT_EMBEDDING_TIMEOUT = 120.0

# ─────────────────────────────────────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_EMBEDDING_DIM = 1_536
DEFAULT_EMBEDDING_BATCH_NUM = 32

# ─────────────────────────────────────────────────────────────────────────────
# Storage
# ─────────────────────────────────────────────────────────────────────────────

# Cosine similarity threshold for NanoVDB queries (mirrors LightRAG default)
DEFAULT_VDB_COSINE_THRESHOLD = 0.2
