"""``LightRAG.aclear_query_cache``: drop the query-answer cache, nothing else.

The answer cache key carries nothing about corpus state, so a cached answer
outlives the ingestion that should have changed it. ``query`` / ``keywords``
rows name no chunk (they are outside ``_REFERENCE_CARRYING_CACHE_TYPES``), so
unlike the whole-cache drop this needs no pipeline reservation -- but its
commit still publishes the whole cache namespace, extract rows the pipeline
has buffered included, so it must be the ordered ``text_chunks`` ->
``llm_response_cache`` pair, and a pair that did not land is a failure.

See HKUDS/LightRAG#4082 and *LLM extraction cache reachability* in
docs/design/PurgeRecoveryContract.md.
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.kg.shared_storage import get_namespace_data
from lightrag.lightrag import QUERY_CACHE_TYPES, _REFERENCE_CARRYING_CACHE_TYPES
from lightrag.utils import EmbeddingFunc, Tokenizer, load_json

pytestmark = pytest.mark.offline


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


async def _dummy_llm(*args, **kwargs) -> str:
    return "ok"


async def _build_rag(tmp_path) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"clearquery-{uuid4().hex[:8]}",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
    )
    await rag.initialize_storages()
    return rag


# Every cache type the codebase writes, in the key shapes it writes them.
_QUERY_ROWS = {
    "mix:query:q1": "query",
    "local:query:q2": "query",
    "naive:query:q3": "query",
    "hybrid:keywords:k1": "keywords",
    "global:keywords:k2": "keywords",
}
_SURVIVING_ROWS = {
    "default:extract:e1": "extract",
    "default:summary:s1": "summary",
    "default:smartheading:h1": "smartheading",
    "default:analysis:a1": "analysis",
}


async def _seed(rag: LightRAG) -> None:
    await rag.llm_response_cache.upsert(
        {
            key: {"return": f"answer for {key}", "cache_type": cache_type}
            for key, cache_type in {**_QUERY_ROWS, **_SURVIVING_ROWS}.items()
        }
    )
    await rag.llm_response_cache.index_done_callback()


def _on_disk(rag: LightRAG) -> set[str]:
    return set(load_json(rag.llm_response_cache._file_name) or {})


async def _in_memory(rag: LightRAG) -> set[str]:
    return {
        key
        for key in {**_QUERY_ROWS, **_SURVIVING_ROWS}
        if await rag.llm_response_cache.get_by_id(key) is not None
    }


def _record_commits(rag: LightRAG, order: list[str], *, chunks=None):
    """Tag both commits; ``chunks`` replaces the chunk half's behaviour."""
    chunks_original = rag.text_chunks.index_done_callback
    cache_original = rag.llm_response_cache.index_done_callback

    async def _chunks():
        order.append("text_chunks")
        if chunks is not None:
            return await chunks()
        return await chunks_original()

    async def _cache():
        order.append("llm_response_cache")
        return await cache_original()

    rag.text_chunks.index_done_callback = _chunks
    rag.llm_response_cache.index_done_callback = _cache


def test_query_cache_types_carry_no_reference():
    """The whole safety argument: nothing points at a row this may delete."""
    assert not QUERY_CACHE_TYPES & _REFERENCE_CARRYING_CACHE_TYPES


@pytest.mark.asyncio
async def test_only_query_and_keywords_rows_are_deleted_and_committed(tmp_path):
    rag = await _build_rag(tmp_path)
    try:
        await _seed(rag)

        deleted = await rag.aclear_query_cache()

        assert deleted == len(_QUERY_ROWS)
        assert await _in_memory(rag) == set(_SURVIVING_ROWS)
        # Committed, not just removed from memory: a crash after the call must
        # not bring the rows back from the file.
        assert _on_disk(rag) == set(_SURVIVING_ROWS)
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cache_types", "kept_type"), [(["query"], "keywords"), (["keywords"], "query")]
)
async def test_the_selector_leaves_the_other_type_alone(
    tmp_path, cache_types, kept_type
):
    rag = await _build_rag(tmp_path)
    try:
        await _seed(rag)

        deleted = await rag.aclear_query_cache(cache_types)

        expected_kept = {k for k, t in _QUERY_ROWS.items() if t == kept_type}
        assert deleted == len(_QUERY_ROWS) - len(expected_kept)
        assert _on_disk(rag) == set(_SURVIVING_ROWS) | expected_kept
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cache_types", [["extract"], ["query", "summary"], ["smartheading"], []]
)
async def test_any_other_type_is_refused_before_deleting(tmp_path, cache_types):
    rag = await _build_rag(tmp_path)
    try:
        await _seed(rag)

        with pytest.raises(ValueError, match="non-empty subset"):
            await rag.aclear_query_cache(cache_types)

        assert await _in_memory(rag) == set(_QUERY_ROWS) | set(_SURVIVING_ROWS)
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_the_commit_is_the_ordered_pair(tmp_path):
    """A bare cache commit would publish extract rows whose references are not
    on disk; the chunk references go first, as on the query path."""
    rag = await _build_rag(tmp_path)
    try:
        await _seed(rag)
        order: list[str] = []
        _record_commits(rag, order)

        await rag.aclear_query_cache()

        assert order == ["text_chunks", "llm_response_cache"], order
    finally:
        await rag.finalize_storages()


async def _chunk_commit_raises():
    raise RuntimeError("chunk store is down")


async def _chunk_commit_declines():
    return False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chunks", [_chunk_commit_raises, _chunk_commit_declines], ids=["raised", "declined"]
)
async def test_a_failed_chunk_commit_is_reported_as_a_failure(tmp_path, chunks):
    """The query path logs and moves on; this caller changed the cache, so a
    pair that did not land must not come back as success -- the rows are gone
    from memory only and would return after a restart."""
    rag = await _build_rag(tmp_path)
    try:
        await _seed(rag)
        order: list[str] = []
        _record_commits(rag, order, chunks=chunks)

        with pytest.raises(RuntimeError, match="commit did not land"):
            await rag.aclear_query_cache()

        assert order == ["text_chunks"], order
        assert _on_disk(rag) == set(_QUERY_ROWS) | set(_SURVIVING_ROWS)
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_it_runs_while_the_pipeline_is_busy(tmp_path):
    """No reservation: an operator clears the stale answers during ingestion,
    which is exactly when they are most likely to ask."""
    rag = await _build_rag(tmp_path)
    try:
        await _seed(rag)
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
        pipeline_status["busy"] = True
        try:
            deleted = await rag.aclear_query_cache()
        finally:
            pipeline_status["busy"] = False

        assert deleted == len(_QUERY_ROWS)
        assert _on_disk(rag) == set(_SURVIVING_ROWS)
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_an_empty_cache_deletes_nothing(tmp_path):
    rag = await _build_rag(tmp_path)
    try:
        assert await rag.aclear_query_cache() == 0
    finally:
        await rag.finalize_storages()
