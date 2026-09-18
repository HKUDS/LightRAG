"""Construction-time resources and a startup that fails before ``INITIALIZED``.

The Redis backends are the ones that take a shared-pool reference in
``__post_init__`` rather than in ``initialize()``. A ``LightRAG`` constructor
therefore already holds one reference per Redis-backed storage before any
startup step runs -- and a startup that fails EARLY (an unreadable
configuration record, a recorded baseline that does not match) never reaches
those storages' ``initialize()``, so the rollback list does not name them and
``finalize_storages()`` releases nothing while the status is still
``CREATED``. Without ``release_unstarted`` every one of those references
outlives the instance, and the shared pool is never disconnected when the
last genuine user goes away.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from lightrag.kg.redis_impl import (
    RedisConnectionManager,
    RedisDocStatusStorage,
    RedisKVStorage,
)
from lightrag.utils import EmbeddingFunc, Tokenizer, TokenizerInterface

pytestmark = pytest.mark.offline


class _StubTokenizer(TokenizerInterface):
    """The default tokenizer downloads tiktoken data; not this test's subject."""

    def encode(self, content: str) -> list[int]:
        return [ord(c) for c in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


@pytest.fixture
def redis_url(tmp_path, monkeypatch):
    url = f"redis://rollback-{tmp_path.name}:6379"
    monkeypatch.setenv("REDIS_URI", url)
    monkeypatch.setattr(
        "lightrag.kg.redis_impl.ConnectionPool.from_url",
        lambda *args, **kwargs: MagicMock(name="pool", aclose=AsyncMock()),
    )
    monkeypatch.setattr(
        "lightrag.kg.redis_impl.Redis", lambda connection_pool=None, **_: AsyncMock()
    )
    yield url
    RedisConnectionManager._pools.pop(url, None)
    RedisConnectionManager._pool_refs.pop(url, None)


@pytest.mark.parametrize("cls", [RedisKVStorage, RedisDocStatusStorage])
async def test_release_unstarted_gives_back_the_constructor_reference(cls, redis_url):
    storage = cls(
        namespace="full_docs" if cls is RedisKVStorage else "doc_status",
        workspace="tenant",
        global_config={"working_dir": "unused"},
        embedding_func=None,
    )
    assert RedisConnectionManager._pool_refs[redis_url] == 1

    await storage.release_unstarted()

    assert redis_url not in RedisConnectionManager._pool_refs
    # Idempotent: a second pass must not steal another storage's reference.
    await storage.release_unstarted()
    assert redis_url not in RedisConnectionManager._pool_refs


async def test_a_startup_that_fails_before_the_storages_start_leaks_no_reference(
    tmp_path, redis_url
):
    """The whole instance on Redis, refused at step 1. Every reference the
    constructor took is handed back, including the ones belonging to storages
    whose ``initialize()`` was never called."""
    from lightrag import LightRAG
    from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data

    async def _llm(prompt, **kwargs):  # pragma: no cover - never called
        return ""

    async def _embed(texts, **kwargs):  # pragma: no cover - never called
        raise AssertionError("startup must refuse before anything embeds")

    initialize_share_data(workers=1)
    try:
        rag = LightRAG(
            working_dir=str(tmp_path),
            workspace="tenant",
            llm_model_func=_llm,
            embedding_func=EmbeddingFunc(
                embedding_dim=8, func=_embed, model_name="bge-m3"
            ),
            kv_storage="RedisKVStorage",
            doc_status_storage="RedisDocStatusStorage",
            tokenizer=Tokenizer("stub", _StubTokenizer()),
        )
        held = RedisConnectionManager._pool_refs[redis_url]
        assert held > 1, "the constructor holds one reference per Redis storage"

        boom = ConnectionError("configuration backend unreachable")

        async def _refuse():
            raise boom

        rag.configuration_storage.initialize = _refuse

        with pytest.raises(ConnectionError) as excinfo:
            await rag.initialize_storages()
        assert excinfo.value is boom

        assert redis_url not in RedisConnectionManager._pool_refs, (
            "every construction-time reference must be handed back"
        )
        assert redis_url not in RedisConnectionManager._pools
    finally:
        finalize_share_data()
