from __future__ import annotations

from dataclasses import dataclass, asdict


OPENROUTER_EMBEDDING_HOST = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class EmbeddingCatalogEntry:
    model_id: str
    display_name: str
    context_length: int
    prompt_cost_per_million_tokens: float
    quality_tier: str
    recommended_chunk_tokens: int
    notes: str
    provider: str = "openrouter"
    binding: str = "openai"
    binding_host: str = OPENROUTER_EMBEDDING_HOST

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["prompt_cost_per_token"] = self.prompt_cost_per_million_tokens / 1_000_000
        data["estimated_cost_100k_tokens"] = estimate_embedding_cost(
            100_000,
            self.prompt_cost_per_million_tokens,
        )
        data["estimated_cost_200k_tokens"] = estimate_embedding_cost(
            200_000,
            self.prompt_cost_per_million_tokens,
        )
        return data


OPENROUTER_EMBEDDING_CATALOG: tuple[EmbeddingCatalogEntry, ...] = (
    EmbeddingCatalogEntry(
        "nvidia/llama-nemotron-embed-vl-1b-v2:free",
        "NVIDIA Llama Nemotron Embed VL 1B V2 (free)",
        131072,
        0.0,
        "experimental",
        3000,
        "Free entry; validate rate limits, stability and quality before using as default.",
    ),
    EmbeddingCatalogEntry(
        "perplexity/pplx-embed-v1-0.6b",
        "Perplexity Embed V1 0.6B",
        32000,
        0.004,
        "economico",
        2500,
        "Ultra-low cost option for broad recall tests and non-critical bases.",
    ),
    EmbeddingCatalogEntry(
        "baai/bge-base-en-v1.5",
        "BAAI bge-base-en-v1.5",
        512,
        0.005,
        "economico",
        450,
        "Cheap English-focused baseline; context window is small.",
    ),
    EmbeddingCatalogEntry(
        "intfloat/e5-base-v2",
        "Intfloat E5-Base-v2",
        512,
        0.005,
        "economico",
        450,
        "Cheap E5 baseline; use smaller chunks.",
    ),
    EmbeddingCatalogEntry(
        "sentence-transformers/all-minilm-l12-v2",
        "Sentence Transformers all-MiniLM-L12-v2",
        512,
        0.005,
        "economico",
        450,
        "Low-cost semantic search baseline with short chunks.",
    ),
    EmbeddingCatalogEntry(
        "sentence-transformers/all-minilm-l6-v2",
        "Sentence Transformers all-MiniLM-L6-v2",
        512,
        0.005,
        "economico",
        450,
        "Very cheap and fast, but not ideal for long legal chunks.",
    ),
    EmbeddingCatalogEntry(
        "sentence-transformers/all-mpnet-base-v2",
        "Sentence Transformers all-mpnet-base-v2",
        512,
        0.005,
        "economico",
        450,
        "Good classic baseline; short context window.",
    ),
    EmbeddingCatalogEntry(
        "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        "Sentence Transformers multi-qa-mpnet-base-dot-v1",
        512,
        0.005,
        "economico",
        450,
        "Question-answer retrieval baseline with short context.",
    ),
    EmbeddingCatalogEntry(
        "sentence-transformers/paraphrase-minilm-l6-v2",
        "Sentence Transformers paraphrase-MiniLM-L6-v2",
        512,
        0.005,
        "economico",
        450,
        "Cheap paraphrase baseline; not preferred for long documents.",
    ),
    EmbeddingCatalogEntry(
        "thenlper/gte-base",
        "Thenlper GTE-Base",
        512,
        0.005,
        "economico",
        450,
        "Cheap GTE baseline; short chunks required.",
    ),
    EmbeddingCatalogEntry(
        "baai/bge-large-en-v1.5",
        "BAAI bge-large-en-v1.5",
        512,
        0.01,
        "bom",
        450,
        "Higher quality English BGE option, but context is still short.",
    ),
    EmbeddingCatalogEntry(
        "baai/bge-m3",
        "BAAI bge-m3",
        8192,
        0.01,
        "recomendado",
        1800,
        "Recommended multilingual/RAG option for Portuguese and legal text.",
    ),
    EmbeddingCatalogEntry(
        "intfloat/e5-large-v2",
        "Intfloat E5-Large-v2",
        512,
        0.01,
        "bom",
        450,
        "Good E5 quality with short chunks.",
    ),
    EmbeddingCatalogEntry(
        "intfloat/multilingual-e5-large",
        "Intfloat Multilingual-E5-Large",
        512,
        0.01,
        "bom",
        450,
        "Multilingual option, but short context window limits chunk size.",
    ),
    EmbeddingCatalogEntry(
        "qwen/qwen3-embedding-8b",
        "Qwen3 Embedding 8B",
        32000,
        0.01,
        "recomendado",
        3000,
        "Recommended default for larger chunks and strong cost/quality balance.",
    ),
    EmbeddingCatalogEntry(
        "thenlper/gte-large",
        "Thenlper GTE-Large",
        512,
        0.01,
        "bom",
        450,
        "Good classic embedding, limited chunk size.",
    ),
    EmbeddingCatalogEntry(
        "openai/text-embedding-3-small",
        "OpenAI text-embedding-3-small",
        8192,
        0.02,
        "baseline",
        1800,
        "Current baseline; stable and useful for comparison.",
    ),
    EmbeddingCatalogEntry(
        "qwen/qwen3-embedding-4b",
        "Qwen3 Embedding 4B",
        32768,
        0.02,
        "bom",
        3000,
        "Long-context Qwen option; compare against 8B before defaulting.",
    ),
    EmbeddingCatalogEntry(
        "perplexity/pplx-embed-v1-4b",
        "Perplexity Embed V1 4B",
        32000,
        0.03,
        "bom",
        3000,
        "Long-context Perplexity option, more expensive than Qwen 8B.",
    ),
    EmbeddingCatalogEntry(
        "mistralai/mistral-embed-2312",
        "Mistral Embed 2312",
        8192,
        0.10,
        "premium",
        1800,
        "Premium priced; benchmark before use.",
    ),
    EmbeddingCatalogEntry(
        "openai/text-embedding-ada-002",
        "OpenAI text-embedding-ada-002",
        8192,
        0.10,
        "legacy",
        1800,
        "Legacy OpenAI embedding; usually not preferred over 3-small.",
    ),
    EmbeddingCatalogEntry(
        "openai/text-embedding-3-large",
        "OpenAI text-embedding-3-large",
        8192,
        0.13,
        "premium",
        1800,
        "Higher-quality OpenAI option, but much more expensive than 3-small.",
    ),
    EmbeddingCatalogEntry(
        "google/gemini-embedding-001",
        "Google Gemini Embedding 001",
        20000,
        0.15,
        "premium",
        2500,
        "Premium long-context option; use for selected critical bases.",
    ),
    EmbeddingCatalogEntry(
        "mistralai/codestral-embed-2505",
        "Mistral Codestral Embed 2505",
        8192,
        0.15,
        "premium",
        1800,
        "Code-oriented embedding; not default for legal text.",
    ),
    EmbeddingCatalogEntry(
        "google/gemini-embedding-2-preview",
        "Google Gemini Embedding 2 Preview",
        8192,
        0.20,
        "premium",
        1800,
        "Most expensive current catalog entry; benchmark before production use.",
    ),
)


def embedding_catalog() -> list[dict[str, object]]:
    return [
        entry.to_dict()
        for entry in sorted(
            OPENROUTER_EMBEDDING_CATALOG,
            key=lambda item: (item.prompt_cost_per_million_tokens, item.model_id),
        )
    ]


def find_embedding_model(model_id: str) -> EmbeddingCatalogEntry | None:
    normalized = model_id.strip()
    return next((entry for entry in OPENROUTER_EMBEDDING_CATALOG if entry.model_id == normalized), None)


def estimate_embedding_cost(estimated_tokens: int, prompt_cost_per_million_tokens: float) -> float:
    return round((max(0, estimated_tokens) / 1_000_000) * prompt_cost_per_million_tokens, 8)


def estimate_tokens_from_pages(page_count: int, words_per_page: int = 400) -> int:
    return max(0, int(page_count * words_per_page * 1.3))


def estimate_tokens_from_characters(character_count: int) -> int:
    return max(0, int(character_count / 4))
