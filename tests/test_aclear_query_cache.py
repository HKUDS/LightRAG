"""``LightRAG.aclear_query_cache``: drop the query-answer cache, nothing else.

The answer cache key carries nothing about corpus state, so a cached answer
outlives the ingestion that should have changed it. ``query`` / ``keywords``
rows name no chunk (they are outside ``_REFERENCE_CARRYING_CACHE_TYPES``), so
unlike the whole-cache drop this needs no pipeline reservation -- but its
commit still publishes the whole cache namespace, extract rows the pipeline
has buffered included, so it must be the ordered ``text_chunks`` ->
``llm_response_cache`` pair, and a pair that did not land is a failure.

The scenarios that need a real commit build a ``LightRAG`` on ``JsonKVStorage``
(the deferred backend where the ordering matters); the argument checks use a
stub. See HKUDS/LightRAG#4082 and *LLM extraction cache reachability* in
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


# Every cache type the codebase writes, in the key shapes it writes them.
_QUERY_ROWS = {
    "mix:query:q1": "query",
    "naive:query:q2": "query",
    "hybrid:keywords:k1": "keywords",
}
_SURVIVING_ROWS = {
    "default:extract:e1": "extract",
    "default:summary:s1": "summary",
    "default:smartheading:h1": "smartheading",
    "default:analysis:a1": "analysis",
}
_ALL_ROWS = {**_QUERY_ROWS, **_SURVIVING_ROWS}


async def _seeded_rag(tmp_path) -> LightRAG:
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
    await rag.llm_response_cache.upsert(
        {
            k: {"return": f"answer for {k}", "cache_type": t}
            for k, t in _ALL_ROWS.items()
        }
    )
    await rag.llm_response_cache.index_done_callback()
    return rag


def _on_disk(rag: LightRAG) -> set[str]:
    return set(load_json(rag.llm_response_cache._file_name) or {})


def _record_commits(rag: LightRAG, order: list[str], *, chunks=None):
    """Tag both commits; ``chunks`` replaces the chunk half's behaviour."""
    chunks_original = rag.text_chunks.index_done_callback
    cache_original = rag.llm_response_cache.index_done_callback

    async def _chunks():
        order.append("text_chunks")
        return await (chunks or chunks_original)()

    async def _cache():
        order.append("llm_response_cache")
        return await cache_original()

    rag.text_chunks.index_done_callback = _chunks
    rag.llm_response_cache.index_done_callback = _cache


def test_query_cache_types_carry_no_reference():
    """The whole safety argument: nothing points at a row this may delete."""
    assert not QUERY_CACHE_TYPES & _REFERENCE_CARRYING_CACHE_TYPES


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cache_types", "deleted_keys"),
    [
        (None, set(_QUERY_ROWS)),
        (["query"], {"mix:query:q1", "naive:query:q2"}),
    ],
    ids=["default-both", "query-only"],
)
async def test_deletes_only_the_selected_rows_and_commits_the_ordered_pair(
    tmp_path, cache_types, deleted_keys
):
    """One run pins every facet of a successful clear: only the selected
    query-cache types go, the extraction-side rows survive, the deletion is on
    disk (a crash must not bring the rows back), the commit is the ordered
    pair, and none of it waits for the pipeline -- the run happens while
    ``busy`` is set, as it would during ingestion."""
    rag = await _seeded_rag(tmp_path)
    try:
        order: list[str] = []
        _record_commits(rag, order)
        pipeline_status = await get_namespace_data(
            "pipeline_status", workspace=rag.workspace
        )
        pipeline_status["busy"] = True
        try:
            args = () if cache_types is None else (cache_types,)
            deleted = await rag.aclear_query_cache(*args)
        finally:
            pipeline_status["busy"] = False

        assert deleted == len(deleted_keys)
        assert _on_disk(rag) == set(_ALL_ROWS) - deleted_keys
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
    from memory only and would return after a restart. The two cases are the
    two ways ``_query_done`` withholds the cache half."""
    rag = await _seeded_rag(tmp_path)
    try:
        order: list[str] = []
        _record_commits(rag, order, chunks=chunks)

        with pytest.raises(RuntimeError, match="commit did not land"):
            await rag.aclear_query_cache()

        assert order == ["text_chunks"], order
        assert _on_disk(rag) == set(_ALL_ROWS)
    finally:
        await rag.finalize_storages()


class _NeverCalledCache:
    def __init__(self):
        self.calls: list[set[str]] = []

    async def delete_by_cache_types(self, cache_types):
        self.calls.append(set(cache_types))
        raise AssertionError("refused types must not reach the storage")


class _StubRag:
    """Only what the argument check touches: refusal happens before storage."""

    aclear_query_cache = LightRAG.aclear_query_cache

    def __init__(self):
        self.llm_response_cache = _NeverCalledCache()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cache_types", [["query", "extract"], []], ids=["reference-type", "empty"]
)
async def test_any_other_type_is_refused_before_deleting(cache_types):
    """A valid type must not carry an invalid one through, and an empty set is
    not "everything"."""
    rag = _StubRag()

    with pytest.raises(ValueError, match="non-empty subset"):
        await rag.aclear_query_cache(cache_types)

    assert rag.llm_response_cache.calls == []
