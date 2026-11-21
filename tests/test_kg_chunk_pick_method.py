"""``kg_chunk_pick_method`` is validated and canonicalized once, at construction.

An unknown spelling used to be accepted and then matched neither ``VECTOR``
nor ``WEIGHT`` on every KG query, so all entity and relation chunks were
silently dropped.
"""

from __future__ import annotations

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.operate import normalize_kg_chunk_pick_method
from lightrag.utils import EmbeddingFunc, Tokenizer

pytestmark = pytest.mark.offline


class _FakeTokenizer:
    def encode(self, content: str) -> list[int]:
        return [ord(char) for char in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


async def _unused_llm(*_args, **_kwargs) -> str:
    return ""


async def _unused_embedding(texts: list[str]) -> np.ndarray:
    return np.zeros((len(texts), 8), dtype=np.float32)


@pytest.mark.parametrize(
    ("given", "expected"),
    [("vector", "VECTOR"), (" Weight ", "WEIGHT")],
)
def test_normalize_accepts_any_case_and_padding(given, expected):
    assert normalize_kg_chunk_pick_method(given) == expected


@pytest.mark.parametrize("invalid", ["WEIGTH", None], ids=["typo", "not_a_string"])
def test_normalize_rejects_unknown_values(invalid):
    with pytest.raises(ValueError, match="KG_CHUNK_PICK_METHOD"):
        normalize_kg_chunk_pick_method(invalid)


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [("   ", "VECTOR"), ("weight", "weight")],
    ids=["blank_env_uses_default", "env_read_at_construction"],
)
def test_field_default_reads_env_and_treats_blank_as_unset(
    monkeypatch, env_value, expected
):
    """A blank ``KG_CHUNK_PICK_METHOD=`` in .env must not fail the startup check."""
    monkeypatch.setenv("KG_CHUNK_PICK_METHOD", env_value)
    field = LightRAG.__dataclass_fields__["kg_chunk_pick_method"]

    assert field.default_factory() == expected


def test_storage_snapshot_holds_the_canonical_value(tmp_path):
    """The query path reads the storages' config snapshot, never normalizing again."""
    rag = LightRAG(
        working_dir=str(tmp_path),
        llm_model_func=_unused_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_unused_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _FakeTokenizer()),
        kg_chunk_pick_method="weight",
    )

    assert rag.kg_chunk_pick_method == "WEIGHT"
    assert rag.text_chunks.global_config["kg_chunk_pick_method"] == "WEIGHT"
