"""``_get_vector_context`` must not turn a backend query failure into a
confident "zero relevant chunks" answer.

Regression for a swallow-vs-raise fix: the function used to wrap
``chunks_vdb.query()`` in a broad ``except Exception: return []``, so any
vector-store error (timeout, connection reset, embedding failure) looked
identical to "nothing relevant found" -- both to ``naive_query`` and to the
mix-mode branch in ``_perform_kg_search``, where it silently dropped the
vector-search leg while KG results kept flowing. This mirrors the contract
``_get_node_data``/``_get_edge_data`` already follow: no try/except around
the vdb query, so a real failure propagates instead of being swallowed.
"""

import pytest

from lightrag.base import QueryParam
from lightrag.operate import _get_vector_context

pytestmark = pytest.mark.offline


class _FailingVDB:
    cosine_better_than_threshold = 0.2

    async def query(self, *args, **kwargs):
        raise RuntimeError("vector backend unavailable")


class _EmptyVDB:
    cosine_better_than_threshold = 0.2

    async def query(self, *args, **kwargs):
        return []


class _HitVDB:
    cosine_better_than_threshold = 0.2

    async def query(self, *args, **kwargs):
        return [{"content": "hello", "id": "chunk-1", "file_path": "a.txt"}]


async def test_get_vector_context_propagates_backend_error():
    """A transport/backend failure must raise, not be reported as no-results."""
    with pytest.raises(RuntimeError, match="vector backend unavailable"):
        await _get_vector_context(
            "query", _FailingVDB(), QueryParam(), query_embedding=[0.1]
        )


async def test_get_vector_context_confirmed_empty_returns_empty_list():
    """Genuinely zero hits stay a plain empty list, not an error."""
    result = await _get_vector_context(
        "query", _EmptyVDB(), QueryParam(), query_embedding=[0.1]
    )
    assert result == []


async def test_get_vector_context_happy_path_unaffected():
    result = await _get_vector_context(
        "query", _HitVDB(), QueryParam(), query_embedding=[0.1]
    )
    assert len(result) == 1
    assert result[0]["content"] == "hello"
    assert result[0]["chunk_id"] == "chunk-1"
