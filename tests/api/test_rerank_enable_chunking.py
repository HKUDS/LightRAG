"""How the API server reads ``RERANK_ENABLE_CHUNKING`` for the Cohere binding.

The flag must accept the same boolean spellings as every other boolean
setting (``get_env_value``), so ``RERANK_ENABLE_CHUNKING=1`` turns chunking on
instead of being silently read as ``False``.
"""

import asyncio

import pytest

pytestmark = pytest.mark.offline


@pytest.fixture
def cohere_rerank_calls(monkeypatch, prepare_startup):
    """Build the server with a stubbed Cohere rerank and return its call kwargs."""

    def run(enable_chunking: str | None) -> dict:
        import lightrag.rerank as rerank

        rerank_calls: list[dict] = []

        async def fake_cohere_rerank(
            query, documents, top_n=None, api_key=None, model="m", base_url="u", **kw
        ):
            rerank_calls.append(kw)
            return []

        monkeypatch.setattr(rerank, "cohere_rerank", fake_cohere_rerank)
        startup = prepare_startup(
            {
                "RERANK_BINDING": "cohere",
                "RERANK_MAX_TOKENS_PER_DOC": None,
                "RERANK_ENABLE_CHUNKING": enable_chunking,
            }
        )

        rerank_func = startup.constructor_kwargs()["rerank_model_func"]
        asyncio.run(rerank_func(query="q", documents=["d"]))
        assert len(rerank_calls) == 1
        return rerank_calls[0]

    return run


@pytest.mark.parametrize("value", ["true", "1", " Yes "])
def test_true_spellings_enable_chunking(cohere_rerank_calls, value):
    assert cohere_rerank_calls(value)["enable_chunking"] is True


@pytest.mark.parametrize("value", [None, "0"])
def test_false_spellings_and_unset_disable_chunking(cohere_rerank_calls, value):
    assert cohere_rerank_calls(value)["enable_chunking"] is False
