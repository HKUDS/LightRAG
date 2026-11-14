"""The initialization diagnostic (``lightrag.tools.check_initialization``).

Its pipeline-status check calls ``get_namespace_data``, which is a coroutine.
Without an ``await`` the call only ever built a coroutine object, so nothing
could raise, the check printed INITIALIZED unconditionally, and pytest reported
the leak as ``RuntimeWarning: coroutine 'get_namespace_data' was never
awaited``. These pin that the check actually runs.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import StoragesStatus
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.tools.check_initialization import check_lightrag_setup
from lightrag.utils import EmbeddingFunc, Tokenizer, TokenizerInterface

pytestmark = pytest.mark.offline

_DIM = 16

_STORAGE_COMPONENTS = (
    "full_docs",
    "text_chunks",
    "entities_vdb",
    "relationships_vdb",
    "chunks_vdb",
    "doc_status",
    "llm_response_cache",
    "full_entities",
    "full_relations",
    "chunk_entity_relation_graph",
)


class _SimpleTokenizer(TokenizerInterface):
    """Keeps the constructor off the network: the default tokenizer downloads
    a tiktoken encoding, which has nothing to do with what these tests pin."""

    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


@pytest.fixture(autouse=True)
def _shared_storage():
    initialize_share_data(workers=1)


async def _mock_llm(prompt, **kwargs):  # pragma: no cover - never called here
    return "mock"


async def _mock_embedding(texts, **kwargs):
    return np.zeros((len(texts), _DIM), dtype=np.float32)


async def test_a_pipeline_status_that_was_never_initialized_is_an_issue(
    tmp_path, capsys
):
    """Every other check passes on this instance; only the pipeline_status
    namespace was never created. The pre-fix diagnostic returned True here and
    printed the namespace as INITIALIZED, because the coroutine that would have
    raised was never awaited."""
    fake = SimpleNamespace(
        _storages_status=StoragesStatus.INITIALIZED,
        _startup_refusal=None,
        workspace=f"never-initialized-{tmp_path.name}",
        **{component: object() for component in _STORAGE_COMPONENTS},
    )

    assert await check_lightrag_setup(fake) is False

    out = capsys.readouterr().out
    assert "Pipeline status not initialized" in out
    assert "Pipeline status: INITIALIZED" not in out


async def test_an_initialized_instance_is_reported_ready(tmp_path):
    """The check reads the namespace under the instance's own workspace, which
    must be the key ``initialize_storages()`` created it under."""
    rag = LightRAG(
        working_dir=str(tmp_path),
        workspace=f"check-init-{tmp_path.name}",
        llm_model_func=_mock_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=_DIM,
            max_token_size=4096,
            func=_mock_embedding,
            model_name="mock-model",
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizer()),
    )
    await rag.initialize_storages()
    try:
        assert await check_lightrag_setup(rag) is True
    finally:
        await rag.finalize_storages()
