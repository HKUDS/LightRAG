"""The policy as the query pipeline uses it: naive and mix.

These tests stop at the context the pipeline hands the model. They prove the
pipeline selects the applicable version, keeps both spans, refuses to answer on
an unresolved status, and never falls back to the merged KG description -- they
do not measure answer quality.
"""

from __future__ import annotations

import pytest

from lightrag.base import QueryContextResult, QueryParam
from lightrag.evidence_policy import catalog_from_mapping
from lightrag.operate import _build_query_context, _policy_cache_args, kg_query, naive_query
from lightrag.utils import Tokenizer
from tests.evidence_policy import corpus

pytestmark = pytest.mark.offline

DEFAULT_INCLUDE = ["V-old", "V-new", "V-third", "V-comment", "V-undated"]


async def test_catalog_survives_light_rag_constructor_and_config_build(tmp_path):
    """Catalog dataclass keys must not be expanded into unhashable dict keys."""
    import numpy as np

    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    async def embed(texts):
        return np.ones((len(texts), 8))

    catalog = _catalog()
    rag = LightRAG(
        working_dir=str(tmp_path),
        llm_model_func=_Model(),
        embedding_func=EmbeddingFunc(embedding_dim=8, func=embed),
        evidence_catalog=catalog,
    )
    try:
        config = rag._build_global_config()
        assert config["evidence_catalog"] is catalog
        assert rag.evidence_catalog is catalog
        assert config["addon_params"] == dict(rag.addon_params)
    finally:
        await rag.finalize_storages()


class _FakeTokenizer:
    def encode(self, content):
        return [ord(char) for char in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


class _ChunksVDB:
    cosine_better_than_threshold = 0.2

    def __init__(self, hits):
        self._hits = hits
        self.calls = 0

    async def query(self, *_args, **_kwargs):
        self.calls += 1
        return list(self._hits)


class _TextChunks(corpus.FakeChunkStore):
    def __init__(self, global_config, rows=None):
        super().__init__(rows)
        self.global_config = global_config
        self.embedding_func = None


class _Model:
    def __init__(self):
        self.calls = 0
        self.prompts = []

    async def __call__(self, _query, system_prompt=None, **_kwargs):
        self.calls += 1
        self.prompts.append(system_prompt or "")
        return "answer"


def _catalog(include=None, conflict_rank="official"):
    return catalog_from_mapping(
        corpus.catalog_mapping(
            include=DEFAULT_INCLUDE if include is None else include,
            conflict_rank=conflict_rank,
        )
    )


def _config(catalog, model):
    return {
        "tokenizer": Tokenizer("fake", _FakeTokenizer()),
        "role_llm_funcs": {"query": model},
        "llm_model_name": "fake-model",
        "user_prompt_prefix": "",
        "addon_params": {"language": "en"},
        "evidence_catalog": catalog,
        "evidence_policy_timezone": "Asia/Shanghai",
    }


def _hits(*chunk_ids):
    return [
        {
            "id": chunk_id,
            "content": corpus.DOCUMENTS[corpus.DOC_OLD],
            "file_path": f"{corpus.DOC_OLD}.txt",
        }
        for chunk_id in chunk_ids
    ]


# ---------------------------------------------------------------------------
# naive
# ---------------------------------------------------------------------------


async def test_naive_uses_the_applicable_version_and_both_spans():
    catalog = _catalog()
    model = _Model()
    config = _config(catalog, model)
    param = QueryParam(
        mode="naive",
        only_need_context=True,
        evidence_policy="authority-v1",
        as_of="2026-08-01",
        enable_rerank=False,
    )

    result = await naive_query(
        "办理申请的期限是多少天？",
        _ChunksVDB(_hits("c-old")),
        param,
        config,
        text_chunks_db=_TextChunks(config),
    )

    assert result is not None
    assert result.llm_generated is False
    assert "270" in result.content
    assert "180" in result.content
    assert "superseded_for_current_answer" in result.content
    metadata = result.raw_data["metadata"]["evidence_policy"]
    assert metadata["status"] == "resolved"
    assert metadata["selected_versions"] == ["V-new"]
    assert metadata["resolved_as_of"] == "2026-08-01"
    # Spans, not whole chunks, and no graph description.
    assert {chunk["version_id"] for chunk in result.raw_data["data"]["chunks"]} == {
        "V-new",
        "V-old",
    }
    assert result.raw_data["data"]["entities"] == []
    assert result.raw_data["data"]["relationships"] == []


async def test_naive_explicit_scope_answers_even_with_zero_hits():
    catalog = _catalog()
    config = _config(catalog, _Model())
    param = QueryParam(
        mode="naive",
        only_need_context=True,
        evidence_policy="authority-v1",
        as_of="2026-08-01",
        scope_keys=[corpus.SCOPE],
        enable_rerank=False,
    )

    result = await naive_query(
        "办理申请的期限是多少天？",
        _ChunksVDB([]),
        param,
        config,
        text_chunks_db=_TextChunks(config),
    )

    assert result is not None
    assert "270" in result.content


async def test_naive_refuses_without_falling_back_or_calling_the_model():
    catalog = _catalog(include=[*DEFAULT_INCLUDE, "V-conflict"])
    model = _Model()
    config = _config(catalog, model)
    param = QueryParam(
        mode="naive",
        evidence_policy="authority-v1",
        as_of="2026-08-02",
        enable_rerank=False,
    )

    result = await naive_query(
        "办理申请的期限是多少天？",
        _ChunksVDB(_hits("c-old", "c-conflict")),
        param,
        config,
        text_chunks_db=_TextChunks(config),
    )

    assert result is not None
    assert result.llm_generated is False
    assert model.calls == 0
    assert "unresolved_conflict" in result.content
    assert result.raw_data["metadata"]["evidence_policy"]["status"] == "unresolved_conflict"


async def test_naive_without_the_policy_is_unchanged():
    catalog = _catalog()
    model = _Model()
    config = _config(catalog, model)

    result = await naive_query(
        "办理申请的期限是多少天？",
        _ChunksVDB(_hits("c-old")),
        QueryParam(mode="naive", only_need_context=True, enable_rerank=False),
        config,
        text_chunks_db=_TextChunks(config),
    )

    assert result is not None
    assert "evidence_policy" not in (result.raw_data.get("metadata") or {})


# ---------------------------------------------------------------------------
# mix / KG path
# ---------------------------------------------------------------------------


async def test_mix_context_excludes_the_merged_kg_description(monkeypatch):
    catalog = _catalog()
    config = _config(catalog, _Model())
    text_chunks = _TextChunks(config)

    async def fake_search(*_args, **_kwargs):
        return {
            "final_entities": [
                {
                    "entity_name": "办理期限",
                    "entity_type": "Concept",
                    "description": "标准成员办理申请的期限为 180 天。",  # stale rule
                    "source_id": "c-old",
                    "file_path": "d-old.txt",
                }
            ],
            "final_relations": [],
            "vector_chunks": [
                {"chunk_id": "c-old", "content": corpus.TEXT_OLD, "file_path": "d-old.txt"}
            ],
            "chunk_tracking": {"c-old": {"source": "C", "frequency": 1, "order": 1}},
            "query_embedding": [0.1],
        }

    monkeypatch.setattr("lightrag.operate._perform_kg_search", fake_search)

    result = await _build_query_context(
        "办理申请的期限是多少天？",
        "",
        "",
        None,
        None,
        None,
        text_chunks,
        QueryParam(
            mode="mix",
            evidence_policy="authority-v1",
            as_of="2026-08-01",
            enable_rerank=False,
        ),
        global_config=config,
    )

    assert isinstance(result, QueryContextResult)
    assert "270" in result.context
    # The merged description states 180 as if current; it must not reach the
    # model, or the policy has removed the chunk and kept the contamination.
    assert "标准成员办理申请的期限为 180 天。" not in result.context
    assert result.raw_data["metadata"]["evidence_policy"]["status"] == "resolved"
    assert result.raw_data["data"]["entities"] == []


async def test_mix_explicit_scope_answers_when_retrieval_returns_nothing(monkeypatch):
    """The zero-seed case must not be swallowed by the "no results" return."""
    catalog = _catalog()
    config = _config(catalog, _Model())

    async def empty_search(*_args, **_kwargs):
        return {
            "final_entities": [],
            "final_relations": [],
            "vector_chunks": [],
            "chunk_tracking": {},
            "query_embedding": None,
        }

    monkeypatch.setattr("lightrag.operate._perform_kg_search", empty_search)

    result = await _build_query_context(
        "办理申请的期限是多少天？",
        "",
        "",
        None,
        None,
        None,
        _TextChunks(config),
        QueryParam(
            mode="mix",
            evidence_policy="authority-v1",
            as_of="2026-08-01",
            scope_keys=[corpus.SCOPE],
            enable_rerank=False,
        ),
        global_config=config,
    )

    assert result is not None
    assert "270" in result.context
    assert result.raw_data["metadata"]["evidence_policy"]["status"] == "resolved"


async def test_kg_query_publishes_the_refusal_instead_of_an_answer(monkeypatch):
    catalog = _catalog(include=[*DEFAULT_INCLUDE, "V-conflict"])
    model = _Model()
    config = _config(catalog, model)

    async def fake_keywords(*_args, **_kwargs):
        return "", "期限"

    async def fake_context(*_args, **_kwargs):
        raw_data = {
            "status": "failure",
            "message": "conflict",
            "data": {"entities": [], "relationships": [], "chunks": [], "references": []},
            "metadata": {
                "query_mode": "mix",
                "keywords": {"high_level": [], "low_level": []},
                "evidence_policy": {
                    "status": "unresolved_conflict",
                    "detail": "two equally authoritative versions",
                },
            },
        }
        return QueryContextResult(context="", raw_data=raw_data)

    monkeypatch.setattr("lightrag.operate.get_keywords_from_query", fake_keywords)
    monkeypatch.setattr("lightrag.operate._build_query_context", fake_context)

    result = await kg_query(
        "办理申请的期限是多少天？",
        None,
        None,
        None,
        _TextChunks(config),
        QueryParam(mode="mix", evidence_policy="authority-v1", as_of="2026-08-02"),
        config,
    )

    assert result is not None
    assert result.llm_generated is False
    assert model.calls == 0
    assert "unresolved_conflict" in result.content


# ---------------------------------------------------------------------------
# Cache partitioning
# ---------------------------------------------------------------------------


def test_cache_args_are_empty_when_the_policy_is_off():
    assert _policy_cache_args({"metadata": {}}) == ()
    assert _policy_cache_args(None) == ()


def test_cache_args_separate_dates_and_catalogs():
    def block(as_of, revision, digest):
        return {
            "metadata": {
                "evidence_policy": {
                    "policy_version": "authority-v1",
                    "resolved_as_of": as_of,
                    "catalog_revision": revision,
                    "catalog_fingerprint": "fp",
                    "status": "resolved",
                    "context_sha256": digest,
                }
            }
        }

    first = _policy_cache_args(block("2026-08-01", "r1", "a"))
    second = _policy_cache_args(block("2026-08-02", "r1", "a"))
    third = _policy_cache_args(block("2026-08-01", "r2", "a"))

    assert first != second  # a date crossing cannot reuse an answer
    assert first != third  # a catalog edit cannot reuse an answer
    assert first == _policy_cache_args(block("2026-08-01", "r1", "a"))


@pytest.mark.parametrize("mode", ["naive", "mix"])
def test_supported_modes_are_the_two_documented_ones(mode):
    from lightrag.evidence_policy import SUPPORTED_MODES

    assert mode in SUPPORTED_MODES
