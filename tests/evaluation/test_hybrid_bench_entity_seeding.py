"""Entity-seeding measurement protocol: production-faithful multi-keyword OR.

Pins (offline, file backends + fakes): the hybrid entity call receives
``lexical_keywords`` with the extracted keyword list (multiple keywords ->
OR tsquery under the storage contract), a single keyword stays byte-identical
to whole-query AND, the dense arm uses the same keyword string and embedding,
the chunk lexical leg stays whole-query AND, extraction is cached per query,
and an extraction failure propagates instead of degrading to whole-query
seeding. The OR construction rule asserted here is the one
``postgres_impl.PGVectorStorage.hybrid_query`` documents and the live storage
tests prove against a real database; these pins prove the HARNESS feeds that
seam the production-shaped arguments.
"""

from __future__ import annotations

import json
import re
from uuid import uuid4

import pytest

from lightrag.evaluation.hybrid_bench import arms, cli
from lightrag.evaluation.hybrid_bench.dataset import BenchDataset, DatasetMeta
from lightrag.evaluation.hybrid_bench.entity_truth import derive_entity_truth
from lightrag.evaluation.hybrid_bench.fakes import FakeLLM, hash_embed_texts
from lightrag.evaluation.hybrid_bench.rag_setup import (
    BenchConfig,
    build_llm_func,
    build_rag,
    ingest_dataset,
)
from lightrag.kg.lexical_tokenizer import tokenize_for_query
from lightrag.kg.shared_storage import initialize_share_data

pytestmark = pytest.mark.offline

QUERY = "CABG and HTN treatment outcomes"
MULTI_REPLY = {
    "high_level_keywords": ["cardiac surgery"],
    "low_level_keywords": ["CABG", "HTN"],
}
EMPTY_REPLY = {"high_level_keywords": [], "low_level_keywords": []}


class _KeywordLLM(FakeLLM):
    """FakeLLM with scripted keyword replies; records every keyword prompt."""

    def __init__(self, entity_specs, keyword_replies):
        super().__init__(entity_specs)
        self.keyword_replies = keyword_replies
        self.keyword_prompts: list[str] = []

    async def __call__(self, prompt, system_prompt=None, history_messages=[], **kwargs):
        text = str(prompt)
        if "User Query:" in text and "Extract entities" not in text:
            self.keyword_prompts.append(text)
            query = re.search(r"User Query: (.+)", text).group(1).strip()
            return json.dumps(self.keyword_replies[query])
        return await super().__call__(
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )


class _ExplodingKeywordLLM(FakeLLM):
    """Keyword prompts raise; counts attempts so a swallow cannot hide it."""

    def __init__(self):
        super().__init__({})
        self.attempts = 0

    async def __call__(self, prompt, **kwargs):
        text = str(prompt)
        if "User Query:" in text and "Extract entities" not in text:
            self.attempts += 1
            raise RuntimeError("keyword provider outage")
        return await super().__call__(prompt, **kwargs)


class _RecordingVDB:
    """Hybrid-capable fake storage recording every call's arguments."""

    supports_hybrid_query = True

    def __init__(self, rows):
        self.rows = rows
        self.hybrid_calls: list[dict] = []
        self.dense_calls: list[dict] = []
        self.cosine_better_than_threshold = 0.1

    async def hybrid_query(
        self, query, top_k, query_embedding=None, lexical_keywords=None
    ):
        self.hybrid_calls.append(
            {
                "query": query,
                "top_k": top_k,
                "query_embedding": query_embedding,
                "lexical_keywords": lexical_keywords,
            }
        )
        return self.rows

    async def query(self, query, top_k, query_embedding=None):
        self.dense_calls.append(
            {"query": query, "top_k": top_k, "query_embedding": query_embedding}
        )
        return self.rows

    async def index_done_callback(self) -> None:
        """Storage-interface no-op: the fake holds no pending writes."""


class _IncapableRecordingVDB(_RecordingVDB):
    """File-backend shape: records calls but declares no hybrid capability."""

    supports_hybrid_query = False


async def _build_rag(tmp_path, llm):
    initialize_share_data(workers=1)
    config = BenchConfig(
        storage="file",
        working_dir=str(tmp_path / "working"),
        workspace=f"seed_{uuid4().hex[:8]}",
        embedding_backend="fake-hash",
        embedding_dim=32,
        llm_backend="fake",
    )
    rag = await build_rag(config, build_llm_func(config, llm))
    return rag, config


def _expected_or_tsquery(lexical_keywords: list[str]) -> str:
    """The storage contract's construction: AND inside each keyword, OR
    across keywords; one keyword keeps its bare expression."""
    per_keyword = [tokenize_for_query(kw) for kw in lexical_keywords]
    assert all(per_keyword), "every extracted keyword must tokenize non-empty"
    if len(per_keyword) == 1:
        return per_keyword[0]
    return " | ".join(f"({expr})" for expr in per_keyword)


@pytest.mark.parametrize(
    ("ll_keywords", "expected_lexical"),
    [
        (["CABG", "HTN"], ["CABG", "HTN"]),
        (["CABG"], ["CABG"]),
    ],
    ids=["multi-keyword-or", "single-keyword"],
)
@pytest.mark.asyncio
async def test_hybrid_entity_call_receives_extracted_lexical_keywords(
    tmp_path, ll_keywords, expected_lexical
):
    llm = _KeywordLLM(
        {},
        {QUERY: {"high_level_keywords": ["topic"], "low_level_keywords": ll_keywords}},
    )
    rag, _ = await _build_rag(tmp_path, llm)
    try:
        vdb = _RecordingVDB([{"entity_name": "CABG"}])
        rag.entities_vdb = vdb
        seeding = await arms.prepare_entity_seeding(rag, QUERY, {})
        names = await arms.retrieve_entities(rag, seeding, True, 10)
        assert names == ["CABG"]
        assert vdb.dense_calls == []
        assert len(vdb.hybrid_calls) == 1
        call = vdb.hybrid_calls[0]
        keywords_str = ", ".join(ll_keywords)
        assert call["query"] == keywords_str
        assert call["top_k"] == 10
        assert call["lexical_keywords"] == expected_lexical
        assert (
            call["query_embedding"] == hash_embed_texts([keywords_str], 32)[0].tolist()
        )
        tsquery = _expected_or_tsquery(call["lexical_keywords"])
        if len(expected_lexical) > 1:
            assert " | " in tsquery
            assert tsquery != tokenize_for_query(keywords_str)
        else:
            assert tsquery == tokenize_for_query(keywords_str)
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_zero_keywords_fall_back_to_query_as_single_keyword(tmp_path):
    llm = _KeywordLLM({}, {QUERY: EMPTY_REPLY})
    rag, _ = await _build_rag(tmp_path, llm)
    try:
        vdb = _RecordingVDB([{"entity_name": "CABG"}])
        rag.entities_vdb = vdb
        seeding = await arms.prepare_entity_seeding(rag, QUERY, {})
        assert seeding.fallback == "zero_keywords"
        assert seeding.keywords_str == QUERY
        await arms.retrieve_entities(rag, seeding, True, 10)
        call = vdb.hybrid_calls[0]
        assert call["query"] == QUERY
        assert call["lexical_keywords"] == [QUERY]
        assert call["query_embedding"] == hash_embed_texts([QUERY], 32)[0].tolist()
        assert _expected_or_tsquery(call["lexical_keywords"]) == tokenize_for_query(
            QUERY
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_empty_low_level_uses_high_level_like_production(tmp_path):
    llm = _KeywordLLM(
        {},
        {QUERY: {"high_level_keywords": ["CABG", "HTN"], "low_level_keywords": []}},
    )
    rag, _ = await _build_rag(tmp_path, llm)
    try:
        vdb = _RecordingVDB([{"entity_name": "CABG"}])
        rag.entities_vdb = vdb
        seeding = await arms.prepare_entity_seeding(rag, QUERY, {})
        assert seeding.fallback == "high_level"
        assert seeding.keywords_str == "CABG, HTN"
        await arms.retrieve_entities(rag, seeding, True, 10)
        call = vdb.hybrid_calls[0]
        assert call["lexical_keywords"] == ["CABG", "HTN"]
        assert " | " in _expected_or_tsquery(call["lexical_keywords"])
    finally:
        await rag.finalize_storages()


@pytest.mark.parametrize(
    ("hybrid", "vdb_cls"),
    [
        (False, _RecordingVDB),
        (True, _IncapableRecordingVDB),
    ],
    ids=["dense-arm", "incapable-backend-fallback"],
)
@pytest.mark.asyncio
async def test_dense_entity_path_uses_keyword_string_and_embedding(
    tmp_path, hybrid, vdb_cls
):
    llm = _KeywordLLM({}, {QUERY: MULTI_REPLY})
    rag, _ = await _build_rag(tmp_path, llm)
    try:
        vdb = vdb_cls([{"entity_name": "CABG"}])
        rag.entities_vdb = vdb
        seeding = await arms.prepare_entity_seeding(rag, QUERY, {})
        names = await arms.retrieve_entities(rag, seeding, hybrid, 10)
        assert names == ["CABG"]
        assert vdb.hybrid_calls == []
        assert len(vdb.dense_calls) == 1
        call = vdb.dense_calls[0]
        assert call["query"] == "CABG, HTN"
        assert (
            call["query_embedding"] == hash_embed_texts(["CABG, HTN"], 32)[0].tolist()
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_chunk_hybrid_leg_stays_whole_query_and(tmp_path):
    llm = _KeywordLLM({}, {QUERY: MULTI_REPLY})
    rag, _ = await _build_rag(tmp_path, llm)
    try:
        chunks = _RecordingVDB([{"id": "chunk-1", "content": "text"}])
        rag.chunks_vdb = chunks
        embedding = await arms.embed_query(rag, QUERY)
        hits = await arms.retrieve_chunks(
            rag, QUERY, "hybrid", 10, embedding, {"chunk-1": ["d1"]}
        )
        assert [hit.doc_ids for hit in hits] == [("d1",)]
        assert chunks.dense_calls == []
        assert len(chunks.hybrid_calls) == 1
        call = chunks.hybrid_calls[0]
        assert call["query"] == QUERY
        assert call["lexical_keywords"] is None
        assert call["query_embedding"] == embedding
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_keyword_extraction_cached_per_query(tmp_path):
    other = "a different question"
    llm = _KeywordLLM(
        {},
        {
            QUERY: MULTI_REPLY,
            other: {"high_level_keywords": [], "low_level_keywords": ["other"]},
        },
    )
    rag, _ = await _build_rag(tmp_path, llm)
    try:
        cache: dict[str, arms.EntitySeeding] = {}
        first = await arms.prepare_entity_seeding(rag, QUERY, cache)
        second = await arms.prepare_entity_seeding(rag, QUERY, cache)
        assert second is first
        assert len(llm.keyword_prompts) == 1
        await arms.prepare_entity_seeding(rag, other, cache)
        assert len(llm.keyword_prompts) == 2
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_extraction_failure_propagates_and_caches_nothing(tmp_path):
    llm = _ExplodingKeywordLLM()
    rag, _ = await _build_rag(tmp_path, llm)
    try:
        cache: dict[str, arms.EntitySeeding] = {}
        with pytest.raises(RuntimeError, match="keyword provider outage"):
            await arms.prepare_entity_seeding(rag, QUERY, cache)
        assert llm.attempts >= 1
        assert cache == {}
    finally:
        await rag.finalize_storages()


TEXT_A = "optocoupler isolates signals in robot joints"
TEXT_B = "planetary gears transmit large torque"


def _cli_dataset() -> BenchDataset:
    return BenchDataset(
        name="seeding",
        meta=DatasetMeta(
            domain="terminology",
            language="en",
            source="https://example.org",
            license="CC-BY-4.0",
            subset_protocol="test",
            counts={},
        ),
        corpus={"d1": TEXT_A, "d2": TEXT_B},
        queries={"q1": TEXT_A, "q2": TEXT_B},
        qrels={"q1": {"d1": 1}, "q2": {"d2": 1}},
    )


@pytest.mark.asyncio
async def test_run_arms_extracts_once_per_query_and_annotates_fallbacks(tmp_path):
    dataset = _cli_dataset()
    llm = _KeywordLLM(
        {
            TEXT_A: [
                {
                    "entity_name": "optocoupler",
                    "entity_type": "concept",
                    "description": "isolates signals",
                }
            ],
            TEXT_B: [
                {
                    "entity_name": "gear",
                    "entity_type": "concept",
                    "description": "transmits torque",
                }
            ],
        },
        {
            TEXT_A: {
                "high_level_keywords": ["isolation"],
                "low_level_keywords": ["optocoupler", "robot joints"],
            },
            TEXT_B: EMPTY_REPLY,
        },
    )
    initialize_share_data(workers=1)
    config = BenchConfig(
        storage="file",
        working_dir=str(tmp_path / "working"),
        workspace=f"seedcli_{uuid4().hex[:8]}",
        embedding_backend="fake-hash",
        embedding_dim=32,
        llm_backend="fake",
        extraction_cap=2,
        max_gleaning=0,
        cosine_threshold=0.1,
    )
    rag = await build_rag(config, build_llm_func(config, llm))
    try:
        ingest = await ingest_dataset(rag, dataset, config)
        truth = await derive_entity_truth(
            rag.chunk_entity_relation_graph,
            rag.text_chunks,
            {qid: dataset.positives(qid) for qid in dataset.queries},
            entity_chunks=rag.entity_chunks,
        )
        vdb = _RecordingVDB([{"entity_name": "optocoupler"}])
        rag.entities_vdb = vdb
        arm_docs, arm_entities, _, _, b5, fallbacks = await cli._run_arms(
            rag, dataset, config, ingest, truth, set()
        )
        assert len(llm.keyword_prompts) == 2
        assert fallbacks == {"q2": "zero_keywords"}
        assert b5 == {}
        assert arm_docs["dense"]["q1"] and arm_docs["dense"]["q1"][0] == "d1"
        for arm in ("dense", "hybrid"):
            assert arm_entities[arm]["q1"] == ["optocoupler"]
            assert arm_entities[arm]["q2"] == ["optocoupler"]
        assert len(vdb.dense_calls) == 2
        assert len(vdb.hybrid_calls) == 2
        by_query = {call["query"]: call for call in vdb.hybrid_calls}
        assert by_query["optocoupler, robot joints"]["lexical_keywords"] == [
            "optocoupler",
            "robot joints",
        ]
        assert by_query[TEXT_B]["lexical_keywords"] == [TEXT_B]
    finally:
        await rag.finalize_storages()
