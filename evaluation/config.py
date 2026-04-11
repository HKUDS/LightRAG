"""
Evaluation pipeline configuration.

All settings can be overridden via environment variables or by editing this file.
Copy the project's env.example to .env and set the variables listed below.

Required .env variables:
    LLM_BINDING          - LLM provider: openai | azure_openai | ollama | ...
    LLM_MODEL            - Model name, e.g. "gpt-4o-mini"
    LLM_BINDING_HOST     - Base URL for the LLM API
    EMBEDDING_BINDING    - Embedding provider: openai | ollama | ...
    EMBEDDING_MODEL      - Embedding model name
    OPENAI_API_KEY       - API key (if using OpenAI)

Optional .env variables for evaluation judge:
    EVAL_LLM_BINDING_HOST - Override LLM host for RAGAS judge (defaults to LLM_BINDING_HOST)
    EVAL_LLM_MODEL        - Override model for RAGAS judge (defaults to LLM_MODEL)
    EVAL_API_KEY          - Override API key for RAGAS judge
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal

from dotenv import load_dotenv

# Load .env from the repo root (parent of this file's parent directory)
_REPO_ROOT = Path(__file__).parent.parent
load_dotenv(_REPO_ROOT / ".env", override=False)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EVAL_DIR = Path(__file__).parent
DATA_DIR = EVAL_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
RESULTS_DIR = EVAL_DIR / "results"
QUESTIONS_FILE = DATA_DIR / "questions.json"

# Supported document extensions for ingestion
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".pptx"}

# ---------------------------------------------------------------------------
# Query modes to run during evaluation
# ---------------------------------------------------------------------------
QueryMode = Literal["naive", "local", "global", "hybrid", "mix"]
ALL_MODES: List[QueryMode] = ["naive", "local", "global", "hybrid", "mix"]


@dataclass
class EvalConfig:
    """Central configuration for the evaluation pipeline."""

    # --- LightRAG working directory (graph + vector storage lives here) -----
    working_dir: str = str(EVAL_DIR / "rag_storage")

    # --- Query modes to evaluate -------------------------------------------
    # Set to a subset of ALL_MODES to skip expensive modes during prototyping
    query_modes: List[QueryMode] = field(default_factory=lambda: list(ALL_MODES))

    # --- LightRAG LLM settings (read from env) ------------------------------
    llm_binding: str = field(
        default_factory=lambda: os.getenv("LLM_BINDING", "openai")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini")
    )
    llm_binding_host: str = field(
        default_factory=lambda: os.getenv(
            "LLM_BINDING_HOST", "https://api.openai.com/v1"
        )
    )
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )

    # --- LightRAG embedding settings ----------------------------------------
    embedding_binding: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_BINDING", "openai")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "text-embedding-3-small"
        )
    )
    embedding_dim: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1536"))
    )
    embedding_max_token_size: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_MAX_TOKEN_SIZE", "8192"))
    )

    # --- LightRAG retrieval settings ----------------------------------------
    top_k: int = 60
    chunk_top_k: int = 20
    max_entity_tokens: int = 6000
    max_relation_tokens: int = 8000
    max_total_tokens: int = 30000
    enable_rerank: bool = False  # set True if you have a reranker configured

    # --- Ingestion settings -------------------------------------------------
    max_parallel_insert: int = 2   # keep low for rate-limited APIs

    # --- Evaluation / judge LLM settings ------------------------------------
    # These are used by RAGAS and the LLM-as-judge scorer.
    # They default to the same model as the RAG LLM but can be overridden.
    eval_llm_binding_host: str = field(
        default_factory=lambda: os.getenv(
            "EVAL_LLM_BINDING_HOST",
            os.getenv("LLM_BINDING_HOST", "https://api.openai.com/v1"),
        )
    )
    eval_llm_model: str = field(
        default_factory=lambda: os.getenv(
            "EVAL_LLM_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini")
        )
    )
    eval_api_key: str = field(
        default_factory=lambda: os.getenv(
            "EVAL_API_KEY", os.getenv("OPENAI_API_KEY", "")
        )
    )

    # --- Misc ---------------------------------------------------------------
    # Number of concurrent workers for query execution
    query_concurrency: int = 3
    # Save raw LLM answers even if evaluation is skipped
    save_raw_answers: bool = True
    # Run RAGAS evaluation (requires ground_truth in questions.json)
    run_ragas: bool = True
    # Run LLM-as-judge comparison (compares each mode vs "naive" baseline)
    run_llm_judge: bool = True

    def ensure_dirs(self) -> None:
        """Create all required directories."""
        for d in [
            self.working_dir,
            DOCUMENTS_DIR,
            RESULTS_DIR,
        ]:
            Path(d).mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        """Raise if critical settings are missing."""
        if not self.llm_api_key and self.llm_binding == "openai":
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Copy env.example to .env and fill in the required values."
            )

    def __post_init__(self) -> None:
        self.ensure_dirs()


# Singleton used by the rest of the pipeline when no config is passed explicitly
DEFAULT_CONFIG = EvalConfig()
