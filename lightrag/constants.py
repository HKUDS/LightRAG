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
# Bounds on any separator cascade, wherever it comes from. The recursive splitter
# descends one level per remaining separator and re-scans the text at every level,
# so total work is O(len(separators) x len(text)); with both factors supplied by
# one request that is an amplifier (GHSA-26pm-px5v-8c4w). The request model caps
# the list, but CHUNK_R_SEPARATORS, addon_params, direct SDK calls and per-doc
# snapshots persisted before that cap existed all bypass it, so the chunker
# normalizes whatever it is handed. 64 leaves ample room for a multi-language
# cascade — the built-in one is 9 entries.
MAX_R_SEPARATORS = 64
# Per-entry length. A single huge separator is re-escaped and re-scanned at every
# level, so it is expensive for the same reason a long list is.
MAX_R_SEPARATOR_CHARS = 256
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
# drop_references option).  DEFAULT_P_REFERENCES_TAIL_N: 0 scans all content
# blocks for reference headings; a positive value restricts detection to the
# last N content blocks as a safety window.
# DEFAULT_P_REFERENCES_HEADINGS: heading prefixes that mark a reference
# section — English words matched case-insensitively at a word boundary,
# the Chinese "参考文献" matched as a plain prefix.  Both are tunable via env
# (CHUNK_P_REFERENCES_TAIL_N / CHUNK_P_REFERENCES_HEADINGS, the latter
# pipe-separated) read live by the chunker at run time.
DEFAULT_P_REFERENCES_TAIL_N = 0
DEFAULT_P_REFERENCES_HEADINGS = ("References", "Bibliography", "参考文献")

# Decompression budget for the native DOCX engine. A .docx is a ZIP, so the
# bytes it costs to parse are its UNCOMPRESSED size, which no other control on
# the ingestion path bounds: MAX_UPLOAD_SIZE bounds the compressed artifact on
# disk and MAX_REQUEST_BODY_BYTES bounds the request body. A 449 KiB .docx
# declaring 200 MiB of document.xml passed both and drove peak RSS past 1.1 GB
# (GHSA-2wpj-ffvv-2pq8).
#
# Both quantities come from the ZIP central directory, so the check costs one
# infolist() and no decompression. A central directory that UNDERSTATES a
# member is self-limiting rather than a bypass: CPython's zipfile stops
# decompressing at the declared file_size and then fails the CRC, so a liar
# buys at most the size it declared — which is the size this budget bounds.
#
# The ratio gate is what actually closes the amplification: 512 MiB alone would
# still let a ~1 MB upload expand 500x. It is enforced both archive-wide and
# against the cumulative expansion by which individual members exceed their
# ratio budgets, so member splitting cannot dilute it. The allowance is the
# fixed DEFAULT_DOCX_RATIO_FLOOR_BYTES plus archive bytes outside those members
# at 1:1: real stored media can support repetitive XML, while padding cannot buy
# another ratio-cap multiple of expansion.
# The same budget governs the other OPC/OOXML formats the legacy engine parses
# locally — .pptx (python-pptx) and .xlsx (openpyxl) are the identical ZIP-bomb
# class — so the DOCX_* knobs below bound all three, enforced in the legacy
# ``extract_text`` dispatcher as well as the native docx engine.
# Env: DOCX_MAX_UNCOMPRESSED_BYTES / DOCX_MAX_COMPRESSION_RATIO /
# DOCX_RATIO_FLOOR_BYTES / DOCX_MAX_ENTRIES, read live at parse time.
DEFAULT_DOCX_MAX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
DEFAULT_DOCX_MAX_COMPRESSION_RATIO = 100
# RATIO_FLOOR_BYTES exempts small content from the two ratio views above; it is
# not a gate itself. Archive-wide it acts as a size THRESHOLD: the ratio is only
# checked once the uncompressed total exceeds it. Per member it acts as a fixed
# ALLOWANCE, added to the 1:1 supporting-archive-bytes budget for the cumulative
# excess by which over-ratio members overrun their individual ratio budgets — so
# it is an amount of tolerated expansion there, not a file-size cut-off. A
# non-positive value removes the exemption (strictest) in both views, it does not
# disable the ratio gate. See lightrag/parser/docx/zip_budget.py.
DEFAULT_DOCX_RATIO_FLOOR_BYTES = 16 * 1024 * 1024
# Member-count ceiling. Its "checked after the central-directory walk" scope
# note lives once, on the lightrag/parser/docx/zip_budget.py module docstring.
DEFAULT_DOCX_MAX_ENTRIES = 10_000

# Native DOCX embedded-image export budgets. Unlike the archive-wide limits
# above, these cap the bytes materialized into ``*.blocks.assets``: one image
# and the sum retained for one document. Sized like the native Markdown image
# budgets so DEFAULT_MAX_PARALLEL_PARSE_NATIVE=5 keeps attacker-controlled
# output bounded across concurrent parses. Env: NATIVE_DOCX_IMAGE_MAX_BYTES /
# NATIVE_DOCX_IMAGE_MAX_TOTAL_BYTES, read live for each parse. Non-positive
# values fall back to these safe defaults rather than disabling the guard.
DEFAULT_NATIVE_DOCX_IMAGE_MAX_BYTES = 25 * 1024 * 1024
DEFAULT_NATIVE_DOCX_IMAGE_MAX_TOTAL_BYTES = 64 * 1024 * 1024

# Zip-bomb guards for the result BUNDLE an external parser engine (docling,
# mineru) returns — a zip fetched from an operator-configured server and
# extracted locally. Distinct from the DOCX_* budget above (attacker-uploaded
# source), this is defense-in-depth against a compromised/misbehaving endpoint.
# The .textpack extraction guard shares these defaults (attacker-uploaded, so
# it stays env-less). docling and mineru live-read the env pair below; a
# non-positive value disables that one gate. Env: PARSER_RESULT_BUNDLE_MAX_ENTRIES
# / PARSER_RESULT_BUNDLE_MAX_TOTAL_BYTES.
DEFAULT_PARSER_RESULT_BUNDLE_MAX_ENTRIES = 10_000
DEFAULT_PARSER_RESULT_BUNDLE_MAX_TOTAL_BYTES = 512 * 1024 * 1024  # 512 MiB

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
# TOC retention: keep at most this many VISIBLE LINES of a detected TOC as body
# (the rest collapse to a single "……") so a 目录 heading is not orphaned from
# its entries. Counted globally by visible line (each soft-break line counts as
# one). 0 keeps none (one "……" replaces the whole TOC); negatives clamp to 0;
# a very large value keeps the entire TOC.
DEFAULT_DOCX_SMART_TOC_KEEP_LINES = 5
# CB1 heading-density ceiling (headings / non-empty paragraphs): the floor of
# the effective threshold. The threshold is baseline-aware — a document with a
# rich physical outline naturally admits more candidates, so the effective
# ceiling is max(this floor, baseline outline density + the margin below).
DEFAULT_DOCX_SMART_DENSITY_MAX = 0.40
# CB1 margin added to the sub-document's baseline outline density when it beats
# the floor above (percentage points, not relative).
DEFAULT_DOCX_SMART_DENSITY_BASELINE_MARGIN = 0.10
# Density re-estimation trigger #2 (trigger side only): also trips when
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
# Document head zone for multi/table title-block windows: a window whose
# start is preceded by at least this many CONTENT records (paragraphs with
# semantic text / tables; blanks, TOC lines, section breaks and pure-drawing
# logo paragraphs do not count) is mid-document and needs document-boundary
# evidence (imprint tail / attachment marker). Calibrated on the corpus: a
# real cover may sit behind a few leading tables or long title lines
# (national-standard covers, report covers), but never ~a page of content.
DEFAULT_DOCX_SMART_TITLE_HEAD_ZONE_RECORDS = 8
# Open numbering series: close a series after this many consecutive body
# paragraphs (0 disables the auxiliary body-run break; the primary close
# signal is the level returning to an ancestor scope).
DEFAULT_DOCX_SMART_SEQ_BREAK_PARAS = 0
# Strong-body length threshold in English-equivalent chars (1 CJK ≈ 3).
DEFAULT_DOCX_SMART_HEADING_MAX_CHARS = 180
# Hard raw-character ceiling for a rendered heading (UI/display constraint,
# NOT env-configurable). Shared by parse_document (truncate_heading /
# validate_heading_length) and the smart-heading synthesis path so a merged
# main+sub title can never exceed what any other heading may reach.
MAX_HEADING_LENGTH = 200

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
# Per docs/FileProcessingPipeline.md, the content is "{{LRdoc}}" + a
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
# Reserved doc_status.metadata key holding the custom-chunk patch journal
# (issue #3400 Phase 3). While present, the document has an in-flight or
# failed ainsert_custom_chunks operation: the pipeline must NOT process the
# row as ordinary ingestion, resets must NOT strip it, and deletion must
# include its staged chunk IDs. Lives here (not utils_pipeline) so that
# base.py can derive the scheduling projection without an import cycle.
CUSTOM_CHUNK_PATCH_METADATA_KEY = "custom_chunk_patch"
# Reserved doc_status.metadata key recording how far this document's KG write
# has progressed (issue #3400 fail-closed purge). Stamped ``pre_graph`` when the
# document enters PROCESSING and promoted to ``graph_mutation_started`` only
# once the write-ahead recovery anchors are durable — i.e. the value answers
# "could this document have touched the graph?" without reading the graph.
# ``pre_graph`` is therefore a valid RECOVERY PROOF on its own: a purge may
# clean up staged chunks even with no anchor rows, because no graph mutation
# can have happened yet. MONOTONIC and never cleared: a PROCESSED document
# keeps ``graph_mutation_started`` (its anchors serve as the proof from then
# on). Absent means UNKNOWN — a pre-#3416 document, which fails closed.
KG_WRITE_STATE_METADATA_KEY = "kg_write_state"
KG_WRITE_STATE_PRE_GRAPH = "pre_graph"
KG_WRITE_STATE_GRAPH_MUTATION_STARTED = "graph_mutation_started"
# Reserved doc_status.metadata key holding the whole-document purge journal
# (issue #3400 fail-closed purge). Required BY fail-closed, not merely nice to
# have: purge's last step deletes the recovery anchors, so without a journal a
# failure in any later step (LLM cache, full_docs) would make the retry see
# "anchors missing" and refuse forever. The journal distinguishes "anchors were
# legitimately deleted by a purge that got this far" from "anchors were never
# there". Phases are ordered — each is written only after the work it names has
# been persisted, so a crash resumes at the recorded phase instead of redoing
# the expensive candidate re-analysis and LLM-cache-backed rebuild.
KG_PURGE_METADATA_KEY = "kg_purge"
KG_PURGE_PHASE_PREPARED = "prepared"  # proof verified; nothing deleted yet
KG_PURGE_PHASE_DERIVED_COMMITTED = "derived_committed"  # graph/vdb/tracking clean
KG_PURGE_PHASE_ANCHORS_PENDING = "anchors_pending"  # chunks gone; anchors may go
KG_PURGE_PHASE_COMPLETED = "completed"  # anchors gone; caller finalizes
KG_PURGE_SCHEMA_VERSION = 1
# doc_status.metadata keys that record a DEMOTION: this row is not the primary
# claimant of its canonical source. ``is_duplicate`` is what every backend's
# primary-candidate predicate keys off (``_basename_of`` returns None for it),
# so these are load-bearing state, not display fields — every metadata rebuild
# (status transitions AND the manual FAILED→PENDING reset) must carry them
# across or the demotion silently reverts and the source key returns to
# conflict. Written by enqueue's duplicate records, by the post-parse
# content-hash duplicate marking, and by the operator's source-conflict repair.
DUPLICATE_DEMOTION_METADATA_KEYS: tuple[str, ...] = (
    "is_duplicate",
    "duplicate_kind",
    "original_doc_id",
)
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
# See docs/FileProcessingPipeline.md for the full specification.
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

# Memory-bounding scheduling page size (LR2 Phase 2). The scheduler pages the
# doc_status backlog through bounded keyset pages of this many records instead
# of materializing every PENDING/orphan row at once, so RSS grows with
# page-size + inflight rather than with the whole backlog. ``0`` disables
# paging (one page holds the whole result set — byte-for-byte the legacy
# single-scan behaviour).
DEFAULT_PIPELINE_SCHEDULING_PAGE_SIZE = 500

# Whether a doc_status backend missing a strict capability is a startup failure
# (LR2 §11). ``False`` (the default) logs a loud warning naming each gap and
# reports it on ``/health``; ``True`` refuses to start. There is no knob for the
# bounded PAGING capability on purpose: the paging and typed-source methods are
# ``@abstractmethod`` on ``DocStatusStorage``, so a backend that lacks them
# cannot be instantiated at all — a stronger guarantee than an opt-in check.
DEFAULT_PIPELINE_REQUIRE_STRICT_STORAGE_READS = False

# How many newly claimed files ``/documents/scan`` holds before it writes them
# to doc_status and releases the batch (LR2 §8.2). Discovery is a single
# streaming pass, so peak scan memory is O(batch) instead of O(files in the
# input dir). Unlike the scheduling page size this knob has NO "disabled"
# value: a non-positive setting is a configuration error (an unbounded scan
# batch is exactly what the streaming rework removes), so startup fails fast.
DEFAULT_SCAN_ENQUEUE_BATCH_SIZE = 100

# Admission capacity: how many documents may be active (PENDING / PARSING /
# ANALYZING / PROCESSING) or reserved before new uploads / text inserts are
# refused with 429 (LR2 §9.1). ``0`` disables admission control entirely — the
# default, so existing deployments see no behaviour change. Manual FAILED→PENDING
# retries and ``/documents/scan`` bulk enqueues deliberately break through the
# cap; the active rows they create make ordinary uploads wait.
DEFAULT_MAX_PENDING_DOCUMENTS = 0

# Ceiling on how many documents ONE ``/documents/texts`` request may carry
# (LR2 §11). ``0`` disables it. Distinct from ``MAX_PENDING_DOCUMENTS``, which
# bounds the pipeline's total backlog and answers 429 (retry later): this bounds
# the fan-out of a single request, which no amount of waiting fixes, so it
# answers 413. It has to be checked before the endpoint's per-text existence
# reads — those are one storage round-trip each, so an oversized batch would
# otherwise do all of them before being refused.
DEFAULT_MAX_TEXTS_PER_REQUEST = 0

# Hard ceiling on the raw request body ANY route may receive, counted as it
# streams through the ASGI ``receive`` channel. ``0`` disables every body limit,
# including the upload one derived below. Distinct from ``MAX_UPLOAD_SIZE``,
# which bounds one uploaded FILE after multipart parsing: this bounds the bytes
# the server agrees to read at all, so a body that lies about (or omits)
# Content-Length is still cut off.
#
# Enabled by default. It used to be ``0`` and to cover three ingestion routes
# only, which meant an operator could set it, watch /documents/text reject an
# oversized body in milliseconds, and still have /api/chat accept the same body
# and stall the process on it (GHSA-r8jh-295g-vv42).
DEFAULT_MAX_REQUEST_BODY_BYTES = 1024 * 1024

# The text-ingestion routes (/documents/text, /documents/texts) get their own,
# far more generous ceiling. 1 MiB is right for a query or a chat turn and wrong
# for pasting a document or batching several of them, and a batch insert has no
# equivalent on the upload route (a batch is not several single-file uploads).
# Not an env knob: an operator who needs a different value sets
# MAX_REQUEST_BODY_BYTES, which then governs every non-upload route.
DEFAULT_MAX_INGEST_BODY_BYTES = 50 * 1024 * 1024

# Slack added to MAX_UPLOAD_SIZE to obtain the raw-body ceiling of
# /documents/upload. Multipart framing (boundaries, part headers, any extra form
# fields) rides along with the file, so the raw body is always somewhat larger
# than the file the route is willing to keep.
MULTIPART_OVERHEAD_BYTES = 1024 * 1024

# Input ceilings for the model-facing request fields. Deliberately NOT env knobs:
# a limit that exists to keep an unauthenticated caller from choosing how much
# CPU the server spends is worth nothing if it can be misconfigured away.
#
# Only the *volume* limits are load-bearing. Cost is proportional to bytes, not
# to message count, and nothing on the path is super-linear in the number of
# messages (``messages.extend(history_messages)``), so under a byte ceiling 5000
# tiny messages and one large message cost the same.
MAX_QUERY_CHARS = 64 * 1024
MAX_MESSAGE_CHARS = 32 * 1024
# All normalized model-facing request input, summed. Without this a per-message
# limit is trivially rebuilt out of many messages. Both query surfaces measure
# their history as serialized JSON so roles, structure, and permitted extra
# fields cannot escape this one budget.
MAX_REQUEST_TEXT_CHARS = 128 * 1024

# Sanity guards, NOT security boundaries — the volume limits above already bound
# the work. They exist so obviously-abnormal input is refused at the request
# boundary with a clear error instead of failing further upstream at the model
# provider. 128 rather than something smaller because /api/chat serves clients
# that send the whole conversation, where ~64 turns is ordinary use.
MAX_MESSAGES_PER_REQUEST = 128
MAX_ROLE_CHARS = 64
MAX_KEYWORDS_PER_LIST = 64
MAX_KEYWORD_CHARS = 512
MAX_MODEL_NAME_CHARS = 256
MAX_IMAGES_PER_MESSAGE = 16
MAX_RESPONSE_TYPE_CHARS = 256

# Upper bounds on the numeric retrieval knobs. Character limits alone do not
# bound the work: ``top_k=10**6`` with ``max_total_tokens=10**9`` leaves the
# retrieval volume — and the tokenization that follows it — under the caller's
# control regardless of how short the query is.
MAX_QUERY_TOP_K = 1000
MAX_QUERY_TOKEN_BUDGET = 1_000_000

# Submission ceilings for the two single-worker CPU pools (tokenizer, chunking).
# A ``ThreadPoolExecutor`` wait queue is unbounded, and freeing the event loop
# means more requests can be in flight at once, so submissions need their own
# limit. Deliberately fixed process-wide constants rather than anything derived
# from ``max_parallel_insert``: the submitting helpers are module level and are
# called from places holding no ``LightRAG`` instance, several instances sharing
# one event loop would each build their own semaphore and lose the global
# ceiling, and — the substantive reason — the pools have ONE worker, so queue
# depth beyond single digits buys no throughput and only pins more pending data
# in memory. Over the limit the submitting coroutine waits; it is backpressure,
# not refusal.
TOKENIZER_SUBMIT_LIMIT = 8
CHUNKING_SUBMIT_LIMIT = 8

# Per-workspace ceiling on manual retry requests that have been published but
# not yet ACKed (LR2 §10.1). The channel is sticky — a request survives until an
# exclusive reset acknowledges it — so an operator hammering /reprocess_failed
# would otherwise grow it without bound. Over the ceiling the publish is refused
# with CAPACITY_EXCEEDED, which is the ONLY manual-publish refusal that maps to
# 429; an already-finalized id is a client error, not backpressure.
DEFAULT_MAX_UNACKED_MANUAL_RETRIES = 64

# Fixed capacity of the human-facing pipeline status history (LR2 §10.3). Every
# write funnels through ``append_pipeline_history``, which drops the oldest lines
# past this many, making the log a ring instead of an append-only list whose only
# bound used to be "the extraction loop happens to trim it" — a deletion job, a
# scan or a manual reset logs plenty without ever reaching that trim.
# Deliberately not an env knob: §11 does not list one, the API response already
# caps what it shows at the newest 1000 lines, and a status log is not a place a
# deployment should have to size.
PIPELINE_HISTORY_MAX_MESSAGES = 5000

# Per-message UTF-8 byte ceiling for the same log. Capacity alone does not bound
# memory: one call site appends a whole ``traceback.format_exc()``, and any
# message interpolating a file list or an exception string is unbounded, so
# N messages × unbounded size is still unbounded. Counted in BYTES rather than
# characters because that is what is being bounded (CJK is 3 bytes/char).
PIPELINE_HISTORY_MESSAGE_MAX_BYTES = 4096

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
# Minimum token budget the EXTRACT-role prompt must be able to spend on a
# multimodal item's ``content`` (the table/equation body handed to the LLM).
# When surrounding/captions/footnotes plus the fixed template frame consume so
# much of MAX_EXTRACT_INPUT_TOKENS that fewer than this many tokens remain for
# content, trimming would leave the LLM with a meaningless stub (a few chars of
# a table body) — a wasted call that pollutes the graph.  The pipeline fails
# the item instead (reprocessable once the budget is widened).  Input-side
# mirror of DEFAULT_MM_CHUNK_DESCRIPTION_MIN_TOKENS; override via the env var
# MM_EXTRACT_CONTENT_MIN_TOKENS.
DEFAULT_MM_EXTRACT_CONTENT_MIN_TOKENS = 100
# Minimum image side (width or height) in pixels accepted for VLM analysis.
# Anything smaller is treated as decorative (icons, separators, etc.) and
# written as status="skipped".
DEFAULT_MM_IMAGE_MIN_PIXEL = 64

# Embedding configuration defaults
DEFAULT_EMBEDDING_FUNC_MAX_ASYNC = 8  # Default max async for embedding functions
DEFAULT_EMBEDDING_BATCH_NUM = 10  # Default batch size for embedding computations

# Overlap (in tokens) borrowed from the previous chunk's tail when the
# embedding hard fallback has to token-window-split a chunk that is still
# over the embedding model's context limit after chunking. Independent from
# the semantic chunker's own `chunk_overlap_token_size` — see
# `LightRAG.embedding_chunk_overlap_token_size`.
DEFAULT_EMBEDDING_CHUNK_OVERLAP_TOKEN_SIZE = 100

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

# Upper bound on the doc-id samples surfaced by the custom-chunk rollback
# report. The rollback sweep is paged, so it must not accumulate one entry per
# journaled document just to describe what it did — counts are exact, the id
# lists are a bounded sample (see arollback_failed_custom_chunk_patches).
ROLLBACK_REPORT_SAMPLE_CAP = 32

# ---------------------------------------------------------------------------
# Lock namespaces shared across modules (kept here so the string literals
# cannot drift apart — two locks that disagree by a typo silently stop
# excluding each other).
# ---------------------------------------------------------------------------

# Workspace-scoped namespace lock serializing the enqueue critical section
# (filter_keys → basename/content dedup → doc_status.upsert). Source-conflict
# repair takes the SAME lock so a new primary cannot be inserted between the
# repair's re-read and its demotions (see DocStatusStorage.repair_source_conflict).
ENQUEUE_SERIALIZE_LOCK_NAMESPACE = "enqueue_serialize"

# Keyed-lock namespace for per-canonical-source-key serialization, mirroring the
# "<workspace>:DocPatch" idiom. Keys are canonical source keys. Held by BOTH
# writers that can change a key's candidate set outside enqueue: the operator's
# source-conflict repair and the post-parse duplicate marking (LR2 §5.5) — take
# it via utils_pipeline.source_candidate_set_lock, which is the single place the
# namespace/key spelling is built (a mismatched spelling excludes nothing).
SOURCE_CONFLICT_LOCK_NAMESPACE = "DocSource"
