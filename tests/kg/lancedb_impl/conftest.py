"""Shared fixtures for LanceDB storage tests.

All tests run against a real embedded LanceDB database in a pytest
tmp_path — no mocking of the LanceDB client (same style as the other
embedded backends: faiss/json/networkx).
"""

import hashlib

import numpy as np
import pytest

lancedb = pytest.importorskip(
    "lancedb", reason="lancedb is required for LanceDB storage tests"
)

from lightrag.kg.shared_storage import (  # noqa: E402
    finalize_share_data,
    initialize_share_data,
)
from lightrag.utils import EmbeddingFunc  # noqa: E402

DIM = 16


def text_vector(text: str, dim: int = DIM) -> np.ndarray:
    """Deterministic direction-distinct vector per text (stable across runs).

    md5-seeded so it does not depend on PYTHONHASHSEED; different texts get
    near-orthogonal directions, identical texts identical vectors.
    """
    seed = int.from_bytes(hashlib.md5(text.encode("utf-8")).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    return rng.standard_normal(dim).astype(np.float32)


class CountingEmbed:
    """Deterministic per-text embedding that records every call."""

    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.call_count = 0
        self.embedded_texts: list[str] = []
        self.document_texts: list[str] = []

    async def __call__(self, texts, **kwargs):
        self.call_count += 1
        self.embedded_texts.extend(texts)
        if kwargs.get("context") == "document":
            self.document_texts.extend(texts)
        return np.array([text_vector(t, self.dim) for t in texts])


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


@pytest.fixture(autouse=True)
def _clean_lancedb_env(monkeypatch):
    """Keep tests hermetic from developer/CI LANCEDB_* environment."""
    for var in (
        "LANCEDB_URI",
        "LANCEDB_WORKSPACE",
        "LANCEDB_READ_CONSISTENCY_INTERVAL",
        "LANCEDB_ENABLE_FTS",
        "LANCEDB_FTS_TOKENIZER",
        "LANCEDB_OPTIMIZE_THRESHOLD",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def counting_embed():
    return CountingEmbed()


@pytest.fixture
def embedding_func(counting_embed):
    # supports_asymmetric=True forwards the context= kwarg to CountingEmbed
    # so tests can tell document embeddings from query embeddings apart.
    return EmbeddingFunc(
        embedding_dim=DIM,
        max_token_size=512,
        func=counting_embed,
        supports_asymmetric=True,
    )


@pytest.fixture
def global_config(tmp_path):
    return {
        "working_dir": str(tmp_path),
        "embedding_batch_num": 4,
        "max_graph_nodes": 1000,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
    }
