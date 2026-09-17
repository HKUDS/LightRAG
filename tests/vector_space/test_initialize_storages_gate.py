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
from lightrag.base import DocStatus, StoragesStatus
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


def _rag(tmp_path, *, model_name, rebuilding=False, vector_storage=None):
    return LightRAG(
        working_dir=str(tmp_path),
        workspace=_workspace(tmp_path),
        rebuilding_vector_storage=rebuilding,
        **({"vector_storage": vector_storage} if vector_storage else {}),
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
    """The graph got ahead of the vector store while a document is still in
    flight. The next pipeline run repairs that; refusing would block it."""
    rag = _rag(tmp_path, model_name="bge-m3")
    await rag.initialize_storages()
    await rag.chunk_entity_relation_graph.upsert_node(
        "Alice", {"entity_id": "Alice", "description": "an engineer"}
    )
    await rag.doc_status.upsert(
        {
            "doc-inflight": {
                "status": DocStatus.PROCESSING,
                "content_summary": "Alice",
                "content_length": 5,
                "chunks_count": 1,
                "chunks_list": ["chunk-1"],
                "file_path": "alice.txt",
            }
        }
    )
    await rag.chunk_entity_relation_graph.index_done_callback()
    await rag.doc_status.index_done_callback()
    await rag.finalize_storages()

    restarted = _rag(tmp_path, model_name="bge-m3")
    await restarted.initialize_storages()
    await restarted.finalize_storages()


async def test_an_admin_only_workspace_refuses_when_its_vectors_vanish(tmp_path):
    """acreate_entity / ainsert_custom_kg write graph entities AND their
    vectors, and no doc-status row. No pipeline run will ever recreate those,
    so requiring a PROCESSED document would exempt such a workspace forever."""
    rag = _rag(tmp_path, model_name="bge-m3")
    await rag.initialize_storages()
    await rag.chunk_entity_relation_graph.upsert_node(
        "Alice", {"entity_id": "Alice", "description": "an engineer"}
    )
    await rag.chunk_entity_relation_graph.index_done_callback()
    await rag.finalize_storages()

    restarted = _rag(tmp_path, model_name="bge-m3")
    with pytest.raises(VectorStorageEmptyError):
        await restarted.initialize_storages()
    await restarted.finalize_storages()


async def test_a_declared_rebuild_starts_on_an_empty_vector_storage(tmp_path):
    """An in-process rebuild BEGINS from the state the gate refuses, so it has
    to be able to say so.

    The supported case is graph-only ingestion (``NoopVectorDBStorage``, which
    writes no vectors by design) followed by a switch to a real vector backend:
    the graph is populated, a document is PROCESSED, and the vector storage is
    legitimately empty. Refusing there blocks the only thing that clears it.
    """
    await _seed(tmp_path, model_name="bge-m3")
    (tmp_path / _workspace(tmp_path) / "vdb_entities.json").unlink()

    rebuilding = _rag(tmp_path, model_name="bge-m3", rebuilding=True)
    await rebuilding.initialize_storages()
    await rebuilding.finalize_storages()


async def test_the_rebuild_flag_is_off_by_default(tmp_path):
    """The same working directory, the same instant, without the declaration:
    a check whose default is 'do not check' protects nobody."""
    await _seed(tmp_path, model_name="bge-m3")
    (tmp_path / _workspace(tmp_path) / "vdb_entities.json").unlink()

    with pytest.raises(VectorStorageEmptyError):
        await _rag(tmp_path, model_name="bge-m3").initialize_storages()


async def test_a_graph_only_deployment_restarts(tmp_path):
    """NoopVectorDBStorage keeps no vectors by design, so after one ingested
    document the graph holds entities, doc-status holds a PROCESSED row, and
    every vector read is a miss. Graph-only is a supported configuration: the
    gate must not refuse its every restart, and `rebuilding_vector_storage` is
    not the answer because such a deployment is not rebuilding anything."""
    seeded = _rag(tmp_path, model_name="bge-m3", vector_storage="NoopVectorDBStorage")
    await seeded.initialize_storages()
    await seeded.chunk_entity_relation_graph.upsert_node(
        "Alice", {"entity_id": "Alice", "description": "an engineer"}
    )
    await seeded.doc_status.upsert(
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
    await seeded.chunk_entity_relation_graph.index_done_callback()
    await seeded.doc_status.index_done_callback()
    await seeded.finalize_storages()

    restarted = _rag(
        tmp_path, model_name="bge-m3", vector_storage="NoopVectorDBStorage"
    )
    await restarted.initialize_storages()
    await restarted.finalize_storages()


async def test_a_refused_instance_can_still_be_finalized(tmp_path):
    """The refusal is raised AFTER every storage has initialized, so they hold
    clients, pools and locks. A caller that catches it -- to report it, or to
    go and rebuild -- must be able to tear them down; finalize_storages() skips
    the whole teardown unless the status says the storages are up."""
    await _seed(tmp_path, model_name="bge-m3")
    (tmp_path / _workspace(tmp_path) / "vdb_entities.json").unlink()

    rag = _rag(tmp_path, model_name="bge-m3")
    with pytest.raises(VectorStorageEmptyError):
        await rag.initialize_storages()

    assert rag._storages_status is StoragesStatus.INITIALIZED
    await rag.finalize_storages()


async def test_a_refusal_is_sticky_across_retries(tmp_path):
    """The storages really are up, so the status says INITIALIZED and teardown
    works -- but the verdict was NEGATIVE, and calling initialize_storages()
    again changes nothing about that. Without this it would take the
    already-initialized early return and come back successful WITHOUT re-running
    the check, turning a fail-closed gate into a one-shot one."""
    await _seed(tmp_path, model_name="bge-m3")
    (tmp_path / _workspace(tmp_path) / "vdb_entities.json").unlink()

    rag = _rag(tmp_path, model_name="bge-m3")
    with pytest.raises(VectorStorageEmptyError):
        await rag.initialize_storages()

    with pytest.raises(VectorStorageEmptyError):
        await rag.initialize_storages()

    await rag.finalize_storages()


async def test_a_refused_instance_is_not_reported_ready(tmp_path):
    """The initialization diagnostic reads _storages_status, which now says
    INITIALIZED on a refused instance. Reporting that one as ready is the one
    thing it must not do."""
    from lightrag.tools.check_initialization import check_lightrag_setup

    await _seed(tmp_path, model_name="bge-m3")
    (tmp_path / _workspace(tmp_path) / "vdb_entities.json").unlink()

    rag = _rag(tmp_path, model_name="bge-m3")
    with pytest.raises(VectorStorageEmptyError):
        await rag.initialize_storages()

    assert await check_lightrag_setup(rag) is False
    await rag.finalize_storages()


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
