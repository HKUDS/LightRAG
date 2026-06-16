"""Shared pytest fixtures for FrameRAG tests.

Provides a deterministic, network-free environment:
  - mock_embed : bag-of-words embeddings → real cosine similarity on shared words
  - mock_llm   : routes by prompt fingerprint → canned JSON for every call type
  - rag        : an initialized FrameRAG wired to the mocks in a temp dir

No external API is ever called, so the suite runs in well under a second.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile

import numpy as np
import pytest_asyncio

from framerag.framerag import FrameRAG

EMB_DIM = 256

_WORD_VECS: dict[str, np.ndarray] = {}


def _word_vec(word: str) -> np.ndarray:
    if word not in _WORD_VECS:
        seed = int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], "little")
        _WORD_VECS[word] = np.random.default_rng(seed).standard_normal(EMB_DIM)
    return _WORD_VECS[word]


def _embed_one(text: str) -> np.ndarray:
    words = re.findall(r"[a-z0-9]+", text.lower())
    if not words:
        return np.ones(EMB_DIM) / np.sqrt(EMB_DIM)
    v = np.sum([_word_vec(w) for w in words], axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


async def mock_embed(texts: list[str], **kwargs) -> np.ndarray:
    return np.stack([_embed_one(t) for t in texts])


async def mock_llm(prompt: str, stream: bool = False) -> str:
    p = prompt
    if "Knowledge Graph specialist" in p:
        return json.dumps([
            {"entity_name": "Sherlock Holmes", "entity_type": "PERSON",
             "entity_description": "A brilliant consulting detective at Baker Street",
             "entity_aliases": ["Holmes", "the detective"], "entity_salience": "HIGH"},
            {"entity_name": "Dr. Watson", "entity_type": "PERSON",
             "entity_description": "Holmes's loyal companion and the narrator",
             "entity_aliases": ["Watson"], "entity_salience": "HIGH"},
            {"entity_name": "violin", "entity_type": "ARTIFACT",
             "entity_description": "A stringed musical instrument played by Holmes",
             "entity_aliases": [], "entity_salience": "MEDIUM"},
        ])
    if "MISSED" in p:
        return "[]"
    if "expert in event extraction" in p:
        return json.dumps([
            {"event_span": "drew the bow across his violin",
             "description": "Holmes played the violin in the evening to relax",
             "participant_names": ["Sherlock Holmes", "violin"],
             "temporal_marker": "in the evening", "is_negation": False},
            {"event_span": "Watson was undisturbed by the music",
             "description": "Watson remained undisturbed by the violin music",
             "participant_names": ["Dr. Watson"],
             "temporal_marker": "", "is_negation": True},
        ])
    if "Frame Semantics" in p:
        if "Watson remained undisturbed" in p:
            return json.dumps({
                "frame_name": "NOT_Affect", "is_new_frame": True,
                "frame_definition": "An entity is excluded from the effect of an event",
                "lexical_unit": "undisturbed.adj",
                "core_elements": [
                    {"fe_name": "Excluded_Entity", "filler_text": "Dr. Watson",
                     "filler_type": "ENTITY", "is_missing": False}],
                "noncore_elements": [
                    {"fe_name": "Reference_Event", "filler_text": "the violin music",
                     "filler_type": "VALUE", "is_missing": False}],
            })
        # Deliberately return a frame name WITH a space to exercise normalization.
        return json.dumps({
            "frame_name": "Music performance", "is_new_frame": True,
            "frame_definition": "An agent plays a musical instrument",
            "lexical_unit": "play.v",
            "core_elements": [
                {"fe_name": "Agent", "filler_text": "Sherlock Holmes",
                 "filler_type": "ENTITY", "is_missing": False},
                {"fe_name": "Instrument", "filler_text": "violin",
                 "filler_type": "ENTITY", "is_missing": False}],
            "noncore_elements": [
                {"fe_name": "Time", "filler_text": "in the evening",
                 "filler_type": "VALUE", "is_missing": False}],
        })
    if "event causality" in p:
        return "[]"
    if "query analysis expert" in p:
        return json.dumps({
            "entity_hints": ["Holmes", "detective"],
            "event_hints": ["play", "perform"],
            "frame_hints": "Music performance",
            "fe_focus": ["Agent", "Instrument"],
            "temporal_hints": [],
        })
    if "novelist's assistant" in p:
        return ("In the evening Holmes drew the bow across his violin, the "
                "haunting melody filling the room as he played to relax.")
    if "frame retrieval expert" in p:
        return json.dumps(["Music performance"])
    if "question-answering assistant" in p:
        return "Holmes played the violin to relax."
    return "[]"


PASSAGE = (
    "Sherlock Holmes sat in his armchair at Baker Street. In the evening he "
    "drew the bow across his violin, playing a haunting melody to relax after "
    "a long case. Dr. Watson was undisturbed by the music, having grown used "
    "to his friend's late-night recitals."
)


@pytest_asyncio.fixture
async def rag():
    # LightRAG's shared storage is keyed by (workspace, namespace) at the
    # process level. FrameRAG uses workspace="" for every store, so two
    # instances in the same process would otherwise share KV/doc state. Reset
    # the shared data before each test to guarantee full isolation.
    from lightrag.kg.shared_storage import (
        initialize_share_data, finalize_share_data,
    )
    finalize_share_data()
    initialize_share_data(workers=1)

    work = tempfile.mkdtemp(prefix="frtest_")
    r = FrameRAG(
        working_dir=work, llm_func=mock_llm, embed_func=mock_embed,
        embedding_dim=EMB_DIM, chunk_size=400, chunk_overlap=20,
        enable_hyde=True, max_concurrent_llm=4,
    )
    await r.initialize()
    try:
        yield r
    finally:
        await r.finalize()
        shutil.rmtree(work, ignore_errors=True)


@pytest_asyncio.fixture
async def indexed_rag(rag):
    await rag.ainsert(PASSAGE, source_doc="holmes_violin")
    return rag
