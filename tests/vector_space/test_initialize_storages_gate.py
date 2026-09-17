"""The startup checks as they actually run: through LightRAG.initialize_storages().

The unit tests next to this one drive the gate with doubles. These drive it the
way a server does -- real NetworkX graph, real Nano vector files, real files on
disk between one instance and the next -- because the two conditions being
checked are precisely conditions that only exist ACROSS process lifetimes.

See docs/design/VectorSpaceProvenance.md.
"""

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.exceptions import VectorStorageEmptyError
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.kg.vector_space import VECTOR_SPACE_MODEL_KEY
from lightrag.utils import (
    EmbeddingFunc,
    Tokenizer,
    TokenizerInterface,
    compute_mdhash_id,
)

pytestmark = pytest.mark.offline

_DIM = 16


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
    """Deterministic per text, so re-embedding reproduces the stored vector."""
    out = np.zeros((len(texts), _DIM), dtype=np.float32)
    for i, text in enumerate(texts):
        out[i][sum(bytearray(text.encode())) % _DIM] = 1.0
    return out


def _workspace(tmp_path) -> str:
    """Per-test, because the storages cache themselves in shared storage keyed
    by (namespace, workspace): a shared name leaks one test's graph into the
    next."""
    return f"gate-{tmp_path.name}"


def _rag(tmp_path, *, model_name):
    return LightRAG(
        working_dir=str(tmp_path),
        workspace=_workspace(tmp_path),
        llm_model_func=_mock_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=_DIM,
            max_token_size=4096,
            func=_mock_embedding,
            model_name=model_name,
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizer()),
    )


async def _seed(tmp_path, *, model_name):
    """A deployment that wrote one entity into both the graph and the vdb."""
    rag = _rag(tmp_path, model_name=model_name)
    await rag.initialize_storages()
    await rag.chunk_entity_relation_graph.upsert_node(
        "Alice", {"entity_id": "Alice", "description": "an engineer"}
    )
    await rag.entities_vdb.upsert(
        {
            compute_mdhash_id("Alice", prefix="ent-"): {
                "entity_name": "Alice",
                "content": "Alice an engineer",
            }
        }
    )
    # A PROCESSED document is the pipeline's own claim that it wrote
    # everything that document produces, vectors included. Without one, an
    # empty vector store has an innocent, self-healing explanation and the
    # gate deliberately does not fire.
    await rag.doc_status.upsert(
        {
            "doc-seeded": {
                "status": DocStatus.PROCESSED,
                "content_summary": "Alice",
                "content_length": 5,
                "chunks_count": 1,
                "chunks_list": ["chunk-1"],
                "file_path": "alice.txt",
            }
        }
    )
    await rag.chunk_entity_relation_graph.index_done_callback()
    await rag.entities_vdb.index_done_callback()
    await rag.doc_status.index_done_callback()
    await rag.finalize_storages()


def _marker(tmp_path):
    import json

    path = tmp_path / _workspace(tmp_path) / "vdb_entities.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    return (payload.get("additional_data") or {}).get(VECTOR_SPACE_MODEL_KEY)


async def test_a_consistent_deployment_starts(tmp_path):
    await _seed(tmp_path, model_name="bge-m3")

    rag = _rag(tmp_path, model_name="bge-m3")
    await rag.initialize_storages()
    await rag.finalize_storages()


async def test_a_vanished_vector_store_refuses_to_serve(tmp_path):
    """The shape a model change leaves on Milvus / Qdrant / PostgreSQL, and
    equally what a deleted vector file leaves here: the graph still has its
    entities and every vector query would answer nothing at all."""
    await _seed(tmp_path, model_name="bge-m3")
    (tmp_path / _workspace(tmp_path) / "vdb_entities.json").unlink()

    rag = _rag(tmp_path, model_name="bge-m3")
    with pytest.raises(VectorStorageEmptyError) as excinfo:
        await rag.initialize_storages()

    assert "lightrag-rebuild-vdb" in str(excinfo.value)


async def test_an_unfinished_ingest_is_not_refused(tmp_path):
    """The graph got ahead of the vector store and no document ever reached
    PROCESSED. The next pipeline run repairs that; refusing would block it."""
    rag = _rag(tmp_path, model_name="bge-m3")
    await rag.initialize_storages()
    await rag.chunk_entity_relation_graph.upsert_node(
        "Alice", {"entity_id": "Alice", "description": "an engineer"}
    )
    await rag.chunk_entity_relation_graph.index_done_callback()
    await rag.finalize_storages()

    restarted = _rag(tmp_path, model_name="bge-m3")
    await restarted.initialize_storages()
    await restarted.finalize_storages()


async def test_an_empty_deployment_starts(tmp_path):
    """Nothing has been ingested yet, so there is nothing that SHOULD have a
    vector. A fresh install must not be refused for being fresh."""
    rag = _rag(tmp_path, model_name="bge-m3")
    await rag.initialize_storages()
    await rag.finalize_storages()


async def test_a_legacy_store_is_adopted_on_the_first_start(tmp_path):
    """The transition. The store was written before the marker existed, so it
    records no model; re-embedding one of its rows reproduces the stored
    vector, which is the evidence adoption needs."""
    await _seed(tmp_path, model_name=None)
    assert _marker(tmp_path) is None

    rag = _rag(tmp_path, model_name="bge-m3")
    await rag.initialize_storages()
    await rag.finalize_storages()

    assert _marker(tmp_path) == "bge-m3"


async def test_a_started_instance_does_not_re_adopt(tmp_path):
    """Once the marker is recorded the probe never runs again."""
    await _seed(tmp_path, model_name="bge-m3")

    rag = _rag(tmp_path, model_name="bge-m3")
    await rag.initialize_storages()
    assert await rag.entities_vdb.vector_space_adoption_pending() is False
    await rag.finalize_storages()
