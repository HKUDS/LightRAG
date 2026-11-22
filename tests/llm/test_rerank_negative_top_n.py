"""Regression test: generic_rerank_api must reject a negative top_n.

top_n has no sensible negative value: it is a document count. Before this
fix, generic_rerank_api forwarded it unchecked to the post-aggregation
``standardized_results[:original_top_n]`` slice (used whenever
``enable_chunking`` is True). Python's slicing semantics treat a negative
stop index as "all but the last |top_n| items", so a caller request like
top_n=-1 silently returned almost every document instead of raising or
returning nothing -- the opposite of what "negative count" should mean, and
with no error to signal the misuse.

Only the chunking + aggregation path (cohere_rerank's enable_chunking=True)
exercises the slice today; without chunking, top_n is forwarded verbatim to
the provider's own payload and is that provider's concern. Validating it in
generic_rerank_api up front, before any HTTP call, catches the misuse for
every caller regardless of which path they take.
"""

from unittest.mock import patch

import pytest

from lightrag.rerank import cohere_rerank, generic_rerank_api
from tests._rerank_session import mock_rerank_session

pytestmark = pytest.mark.offline


@pytest.mark.asyncio
async def test_negative_top_n_raises_before_any_request():
    """A negative top_n must fail fast, not reach the provider at all."""
    session, _captured = mock_rerank_session({"results": []})

    with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
        with pytest.raises(ValueError, match="top_n"):
            await generic_rerank_api(
                query="q",
                documents=["a", "b"],
                model="m",
                base_url="http://test.com/rerank",
                api_key="k",
                top_n=-1,
            )

    session.post.assert_not_called()


@pytest.mark.asyncio
async def test_negative_top_n_via_cohere_rerank_with_chunking_raises():
    """Before the fix, this used to silently drop the last-ranked document
    instead of raising: 3 documents aggregate to 3 scored results, and
    ``standardized_results[:-1]`` (Python's negative-slice semantics) kept
    the first 2 rather than signalling the invalid input or returning none."""
    session, _captured = mock_rerank_session(
        {
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.7},
                {"index": 2, "relevance_score": 0.5},
            ]
        }
    )

    with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
        with pytest.raises(ValueError, match="top_n"):
            await cohere_rerank(
                query="q",
                documents=["doc0", "doc1", "doc2"],
                api_key="k",
                base_url="http://test.com/rerank",
                enable_chunking=True,
                top_n=-1,
            )


@pytest.mark.asyncio
async def test_top_n_none_and_zero_are_unaffected():
    """The validation only rejects negative values; None (no limit) and 0
    (a degenerate but non-negative request) still reach the provider."""
    session, captured = mock_rerank_session({"results": []})

    with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
        await generic_rerank_api(
            query="q",
            documents=["a"],
            model="m",
            base_url="http://test.com/rerank",
            api_key="k",
            top_n=None,
        )
    assert "top_n" not in captured["kwargs"]["json"]

    session, captured = mock_rerank_session({"results": []})
    with patch("lightrag.rerank.aiohttp.ClientSession", return_value=session):
        await generic_rerank_api(
            query="q",
            documents=["a"],
            model="m",
            base_url="http://test.com/rerank",
            api_key="k",
            top_n=0,
        )
    assert captured["kwargs"]["json"]["top_n"] == 0
