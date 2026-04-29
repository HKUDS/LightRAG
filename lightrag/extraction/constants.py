"""Smart chunking thresholds – dual profile mode (code switch only).

Profiles:
- large: original parameters from docx-extraction-guide-zh.md
- small: tuned for embedding-limited environments

No env override is used. Switch profile by changing ACTIVE_PROFILE below.
"""

from __future__ import annotations

MAX_HEADING_LENGTH = 200  # chars – heading text hard limit
MAX_ANCHOR_CANDIDATE_LENGTH = 100  # chars – anchor paragraph limit

# -------- Profile presets --------
LARGE_PROFILE = {
    "IDEAL_BLOCK_CONTENT_TOKENS": 6000,
    "MAX_BLOCK_CONTENT_TOKENS": 8000,
    "EXTRACTION_SAFE_MAX_BLOCK_TOKENS": 3000,
    "SMALL_TAIL_THRESHOLD": 1000,
    "TABLE_IDEAL_TOKENS": 3000,
    "TABLE_MAX_TOKENS": 5000,
    "TABLE_MIN_LAST_CHUNK_TOKENS": 1600,
}

SMALL_PROFILE = {
    # tuned for embedding-limited environments (max block should not exceed 4096)
    # one-step tighter profile to reduce downstream extraction prompt overflow risk
    "IDEAL_BLOCK_CONTENT_TOKENS": 1800,
    "MAX_BLOCK_CONTENT_TOKENS": 2200,
    # final safety cap for entity-extraction stage (keep gleaning, reduce overflow risk)
    "EXTRACTION_SAFE_MAX_BLOCK_TOKENS": 1000,
    "SMALL_TAIL_THRESHOLD": 400,
    "TABLE_IDEAL_TOKENS": 1400,
    "TABLE_MAX_TOKENS": 1800,
    "TABLE_MIN_LAST_CHUNK_TOKENS": 500,
}

# -------- Select active profile here --------
# Change to "small" when needed.
ACTIVE_PROFILE = "small"  # "large" | "small"
_ACTIVE = SMALL_PROFILE if ACTIVE_PROFILE == "small" else LARGE_PROFILE

# -------- Export active profile --------
IDEAL_BLOCK_CONTENT_TOKENS = _ACTIVE["IDEAL_BLOCK_CONTENT_TOKENS"]
MAX_BLOCK_CONTENT_TOKENS = _ACTIVE["MAX_BLOCK_CONTENT_TOKENS"]
EXTRACTION_SAFE_MAX_BLOCK_TOKENS = _ACTIVE["EXTRACTION_SAFE_MAX_BLOCK_TOKENS"]
SMALL_TAIL_THRESHOLD = _ACTIVE["SMALL_TAIL_THRESHOLD"]
TABLE_IDEAL_TOKENS = _ACTIVE["TABLE_IDEAL_TOKENS"]
TABLE_MAX_TOKENS = _ACTIVE["TABLE_MAX_TOKENS"]
TABLE_MIN_LAST_CHUNK_TOKENS = _ACTIVE["TABLE_MIN_LAST_CHUNK_TOKENS"]
