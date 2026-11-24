"""``MIN_CHUNK_TOKEN_SIZE`` at the configuration ingress of the core.

A chunk size below the minimum is a misconfiguration: it is rejected at the
LightRAG constructor (env, legacy fields, addon_params), the ``addon_params``
setter and enqueue. REST params are covered in
tests/api/routes/test_document_routes_chunking.py, filename/rule hints in
tests/parser/test_hint_params.py. Chunker functions called directly keep a
floor of 1 (see test_recursive_character_floor_clamp).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.constants import MIN_CHUNK_TOKEN_SIZE
from lightrag.parser.routing import validate_chunk_token_sizes
from lightrag.utils import EmbeddingFunc, Tokenizer, TokenizerInterface

pytestmark = pytest.mark.offline

_SIZE_ENVS = (
    "CHUNK_SIZE",
    "CHUNK_F_SIZE",
    "CHUNK_R_SIZE",
    "CHUNK_V_SIZE",
    "CHUNK_P_SIZE",
)
_TOO_SMALL = MIN_CHUNK_TOKEN_SIZE - 1


@pytest.fixture(autouse=True)
def _clean_size_env(monkeypatch):
    for name in _SIZE_ENVS:
        monkeypatch.delenv(name, raising=False)


class _CharTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


async def _mock_llm(*args, **kwargs) -> str:
    return ""


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    return np.zeros((len(texts), 8))


def _new_rag(tmp_path: Path, **kwargs) -> LightRAG:
    return LightRAG(
        working_dir=str(tmp_path),
        workspace=f"minsize-{tmp_path.name}",
        llm_model_func=_mock_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=4096, func=_mock_embedding
        ),
        tokenizer=Tokenizer("char-tokenizer", _CharTokenizer()),
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# validate_chunk_token_sizes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "config,slot",
    [
        ({"chunk_token_size": _TOO_SMALL}, "chunker.chunk_token_size"),
        (
            {"fixed_token": {"chunk_token_size": _TOO_SMALL}},
            "chunker.fixed_token.chunk_token_size",
        ),
        # bool is an int subclass: True is 1, below the minimum.
        ({"chunk_token_size": True}, "chunker.chunk_token_size"),
    ],
)
def test_validator_rejects_a_size_below_the_minimum(config, slot):
    with pytest.raises(ValueError, match=rf"{slot} must be >= {MIN_CHUNK_TOKEN_SIZE}"):
        validate_chunk_token_sizes(config)


@pytest.mark.parametrize(
    "config",
    [
        None,
        {},
        {"chunk_token_size": None, "recursive_character": {}},
        {
            "chunk_token_size": MIN_CHUNK_TOKEN_SIZE,
            "fixed_token": {"chunk_token_size": MIN_CHUNK_TOKEN_SIZE},
            "recursive_character": {"chunk_token_size": MIN_CHUNK_TOKEN_SIZE},
            "semantic_vector": {"chunk_token_size": MIN_CHUNK_TOKEN_SIZE},
            "paragraph_semantic": {"chunk_token_size": MIN_CHUNK_TOKEN_SIZE},
        },
    ],
)
def test_validator_accepts_the_minimum_and_absent_slots(config):
    assert validate_chunk_token_sizes(config) is None


# --------------------------------------------------------------------------- #
# LightRAG constructor: env, legacy field, addon_params
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "env_name,slot",
    [
        ("CHUNK_SIZE", "chunker.chunk_token_size"),
        ("CHUNK_F_SIZE", "chunker.fixed_token.chunk_token_size"),
        ("CHUNK_R_SIZE", "chunker.recursive_character.chunk_token_size"),
        ("CHUNK_V_SIZE", "chunker.semantic_vector.chunk_token_size"),
        ("CHUNK_P_SIZE", "chunker.paragraph_semantic.chunk_token_size"),
    ],
)
def test_constructor_rejects_a_small_size_env(tmp_path, monkeypatch, env_name, slot):
    monkeypatch.setenv(env_name, str(_TOO_SMALL))

    with pytest.raises(ValueError, match=rf"{slot} must be >= {MIN_CHUNK_TOKEN_SIZE}"):
        _new_rag(tmp_path)


def test_constructor_rejects_a_small_legacy_chunk_token_size(tmp_path):
    with pytest.raises(ValueError, match=r"chunker\.chunk_token_size must be >= "):
        _new_rag(tmp_path, chunk_token_size=_TOO_SMALL, chunk_overlap_token_size=0)


def test_constructor_rejects_a_small_addon_params_size(tmp_path):
    with pytest.raises(
        ValueError, match=r"chunker\.recursive_character\.chunk_token_size must be"
    ):
        _new_rag(
            tmp_path,
            addon_params={"chunker": {"recursive_character": {"chunk_token_size": 8}}},
        )


def test_constructor_accepts_the_minimum(tmp_path, monkeypatch):
    monkeypatch.setenv("CHUNK_R_SIZE", str(MIN_CHUNK_TOKEN_SIZE))

    rag = _new_rag(
        tmp_path,
        chunk_token_size=MIN_CHUNK_TOKEN_SIZE,
        chunk_overlap_token_size=0,
    )

    chunker = rag.addon_params["chunker"]
    assert chunker["chunk_token_size"] == MIN_CHUNK_TOKEN_SIZE
    assert chunker["recursive_character"]["chunk_token_size"] == MIN_CHUNK_TOKEN_SIZE


# --------------------------------------------------------------------------- #
# addon_params setter: rejected atomically
# --------------------------------------------------------------------------- #


def test_setter_rejection_keeps_the_previous_configuration(tmp_path):
    rag = _new_rag(tmp_path, chunk_token_size=500, chunk_overlap_token_size=50)
    before_params = rag.addon_params
    before_chunker = dict(before_params["chunker"])

    with pytest.raises(ValueError, match=r"chunker\.chunk_token_size must be >= "):
        rag.addon_params = {"chunker": {"chunk_token_size": _TOO_SMALL}}

    assert rag.addon_params is before_params
    assert dict(rag.addon_params["chunker"]) == before_chunker
    assert (rag.chunk_token_size, rag.chunk_overlap_token_size) == (500, 50)


# --------------------------------------------------------------------------- #
# enqueue: caller-supplied chunk_options and nested addon_params mutation
# --------------------------------------------------------------------------- #


def _enqueue_and_read(tmp_path, mutate, **enqueue_kwargs):
    async def _run():
        rag = _new_rag(tmp_path)
        await rag.initialize_storages()
        try:
            mutate(rag)
            with pytest.raises(ValueError, match="must be >= ") as exc:
                await rag.apipeline_enqueue_documents(
                    "Body text for the minimum chunk size test.",
                    ids=["doc-min"],
                    file_paths="min.txt",
                    **enqueue_kwargs,
                )
            return exc.value, await rag.full_docs.get_by_id("doc-min")
        finally:
            await rag.finalize_storages()

    return asyncio.run(_run())


def test_enqueue_rejects_small_caller_chunk_options_before_writing(tmp_path):
    error, row = _enqueue_and_read(
        tmp_path,
        lambda rag: None,
        chunk_options=[{"fixed_token": {"chunk_token_size": _TOO_SMALL}}],
    )

    assert "chunker.fixed_token.chunk_token_size" in str(error)
    assert row is None


def test_enqueue_rejects_a_nested_addon_params_mutation(tmp_path):
    def mutate(rag):
        # Nested mutation bypasses every addon_params hook.
        rag.addon_params["chunker"]["recursive_character"]["chunk_token_size"] = 8

    error, row = _enqueue_and_read(tmp_path, mutate, process_options="R")

    assert "chunker.recursive_character.chunk_token_size" in str(error)
    assert row is None
