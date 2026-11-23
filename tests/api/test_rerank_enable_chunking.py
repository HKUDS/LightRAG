"""How the API server reads ``RERANK_ENABLE_CHUNKING`` for the Cohere binding.

The flag must accept the same boolean spellings as every other boolean
setting (``get_env_value``), so ``RERANK_ENABLE_CHUNKING=1`` turns chunking on
instead of being silently read as ``False``.
"""

import asyncio
import sys

import pytest

from lightrag.api import config

pytestmark = pytest.mark.offline


class _Constructed(Exception):
    """Stops startup once the stubbed LightRAG constructor captured its kwargs."""


@pytest.fixture
def cohere_rerank_calls(monkeypatch, tmp_path):
    """Build the server with a stubbed Cohere rerank and return its call kwargs."""

    def run(enable_chunking: str | None) -> dict:
        monkeypatch.setattr(config, "_global_args", None)
        monkeypatch.setattr(config, "_initialized", False)
        for key, value in {
            "LLM_BINDING": "openai",
            "EMBEDDING_BINDING": "openai",
            "RERANK_BINDING": "cohere",
            "UI_TEMPLATES_DIR": "",
            "AUTH_ACCOUNTS": "",
        }.items():
            monkeypatch.setenv(key, value)
        monkeypatch.delenv("RERANK_MAX_TOKENS_PER_DOC", raising=False)
        if enable_chunking is None:
            monkeypatch.delenv("RERANK_ENABLE_CHUNKING", raising=False)
        else:
            monkeypatch.setenv("RERANK_ENABLE_CHUNKING", enable_chunking)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "lightrag-server",
                "--working-dir",
                str(tmp_path / "rag"),
                "--input-dir",
                str(tmp_path / "inputs"),
            ],
        )
        args = config.initialize_config(force=True)

        import lightrag.rerank as rerank
        from lightrag.api import lightrag_server as server

        rerank_calls: list[dict] = []

        async def fake_cohere_rerank(
            query, documents, top_n=None, api_key=None, model="m", base_url="u", **kw
        ):
            rerank_calls.append(kw)
            return []

        monkeypatch.setattr(rerank, "cohere_rerank", fake_cohere_rerank)

        captured: list[dict] = []

        def construct(**kwargs):
            captured.append(kwargs)
            raise _Constructed

        monkeypatch.setattr(server, "LightRAG", construct)
        monkeypatch.setattr(server, "check_frontend_build", lambda: (False, False))
        monkeypatch.setattr(server, "check_workspace_frontend_build", lambda: False)
        with pytest.raises(_Constructed):
            server.create_app(args)
        assert len(captured) == 1

        asyncio.run(captured[0]["rerank_model_func"](query="q", documents=["d"]))
        assert len(rerank_calls) == 1
        return rerank_calls[0]

    return run


@pytest.mark.parametrize("value", ["true", "1", " Yes "])
def test_true_spellings_enable_chunking(cohere_rerank_calls, value):
    assert cohere_rerank_calls(value)["enable_chunking"] is True


@pytest.mark.parametrize("value", [None, "0"])
def test_false_spellings_and_unset_disable_chunking(cohere_rerank_calls, value):
    assert cohere_rerank_calls(value)["enable_chunking"] is False
