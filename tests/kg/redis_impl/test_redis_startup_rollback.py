"""A startup that fails before ``INITIALIZED`` holds no Redis pool reference.

The Redis backends acquire their shared-pool reference in ``initialize()``,
and a startup can refuse long before that: an unreadable configuration record
or a recorded embedding baseline that does not match refuses at step 2 or 3,
with every storage constructed and none of them initialized. The rollback list
therefore names only the configuration storage, and ``finalize_storages()``
releases nothing at all while the status is still ``CREATED`` -- so anything a
CONSTRUCTOR had taken would outlive the instance with nothing left to release
it.

The unit tests next door pin the backend half of that (construction touches the
manager's registry not at all, ``close()`` before ``initialize()`` releases
nothing). This one pins it through a whole ``LightRAG``, which is where the
symptom was reported: the reference count is what it was before, at every point
of a refused startup.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from lightrag.kg.redis_impl import RedisConnectionManager
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


async def test_a_startup_refused_at_step_1_holds_no_pool_reference(tmp_path, redis_url):
    """The whole instance on Redis, refused before the business storages start.
    No reference is held after construction, and none after the rollback."""
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
        assert redis_url not in RedisConnectionManager._pool_refs, (
            "construction must take no reference: nothing would ever release it"
        )

        boom = ConnectionError("configuration backend unreachable")

        async def _refuse():
            raise boom

        rag.configuration_storage.initialize = _refuse

        with pytest.raises(ConnectionError) as excinfo:
            await rag.initialize_storages()
        assert excinfo.value is boom

        assert redis_url not in RedisConnectionManager._pool_refs
        assert redis_url not in RedisConnectionManager._pools
    finally:
        finalize_share_data()
