"""The nine-step startup sequence, as it actually runs through
``LightRAG.initialize_storages()`` on the JSON / Nano / NetworkX backends.

The unit tests next door drive the store with doubles; these drive the whole
lifecycle -- the configuration storage first, the precheck ahead of any vector
storage, the rollback when a business storage fails, ``INITIALIZED`` before the
checks, the baselines established on evidence, sticky post-init failures, and
the configuration storage released exactly once -- because every one of those
is a property of the ORDER things happen in, which no double can pin.

Scenario numbers refer to the acceptance list in
docs/design/ConfigurationStorage.md.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from lightrag import LightRAG, config_store as cs
from lightrag.base import DocStatus, StoragesStatus
from lightrag.exceptions import (
    ConfigurationStorageError,
    EmbeddingBaselineMismatchError,
    VectorSpaceMismatchError,
    VectorStorageEmptyError,
)
from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.namespace import CONFIG_WORKSPACE
from lightrag.utils import (
    EmbeddingFunc,
    Tokenizer,
    TokenizerInterface,
    compute_mdhash_id,
    make_relation_vdb_ids,
)

pytestmark = pytest.mark.offline

_DIM = 16


class _SimpleTokenizer(TokenizerInterface):
    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


@pytest.fixture(autouse=True)
def _shared_storage():
    initialize_share_data(workers=1)
    yield
    finalize_share_data()


async def _mock_llm(prompt, **kwargs):  # pragma: no cover - never called here
    return "mock"


def _embedding_in_space(space: str, *, failing: bool = False):
    """Deterministic per (space, text): the same text lands on the same unit
    vector inside one space and on a different one in another, which is how a
    test stands up two "models" whose vectors do not reproduce each other."""

    async def _embed(texts, **kwargs):
        if failing:
            raise ConnectionError("embedding provider unreachable")
        out = np.zeros((len(texts), _DIM), dtype=np.float32)
        shift = sum(bytearray(space.encode())) * 7
        for i, text in enumerate(texts):
            out[i][(sum(bytearray(text.encode())) + shift) % _DIM] = 1.0
        return out

    return _embed


def _workspace(tmp_path) -> str:
    return f"cfg-{tmp_path.name}"


def _rag(tmp_path, *, model_name, space="A", failing_embedder=False, rebuilding=False):
    return LightRAG(
        working_dir=str(tmp_path),
        workspace=_workspace(tmp_path),
        rebuilding_vector_storage=rebuilding,
        llm_model_func=_mock_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=_DIM,
            max_token_size=4096,
            func=_embedding_in_space(space, failing=failing_embedder),
            model_name=model_name,
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizer()),
    )


async def _seed(tmp_path, *, model_name, space="A"):
    """A deployment that ingested one document: an entity, a relation and a
    chunk, each with its vector, and a PROCESSED doc-status row."""
    rag = _rag(tmp_path, model_name=model_name, space=space)
    await rag.initialize_storages()
    graph = rag.chunk_entity_relation_graph
    await graph.upsert_node(
        "Alice",
        {"entity_id": "Alice", "description": "an engineer", "source_id": "chunk-1"},
    )
    await graph.upsert_node(
        "Bob", {"entity_id": "Bob", "description": "a manager", "source_id": "chunk-1"}
    )
    await graph.upsert_edge(
        "Alice",
        "Bob",
        {"description": "works with", "source_id": "chunk-1", "weight": 1.0},
    )
    await rag.entities_vdb.upsert(
        {
            compute_mdhash_id("Alice", prefix="ent-"): {
                "entity_name": "Alice",
                "content": "Alice an engineer",
            },
            compute_mdhash_id("Bob", prefix="ent-"): {
                "entity_name": "Bob",
                "content": "Bob a manager",
            },
        }
    )
    rel_id = make_relation_vdb_ids("Alice", "Bob")[0]
    await rag.relationships_vdb.upsert(
        {
            rel_id: {
                "src_id": "Alice",
                "tgt_id": "Bob",
                "content": "Alice works with Bob",
            }
        }
    )
    await rag.text_chunks.upsert(
        {
            "chunk-1": {
                "content": "Alice works with Bob",
                "full_doc_id": "doc-1",
                "tokens": 4,
            }
        }
    )
    await rag.chunks_vdb.upsert(
        {"chunk-1": {"content": "Alice works with Bob", "full_doc_id": "doc-1"}}
    )
    await rag.doc_status.upsert(
        {
            "doc-1": {
                "status": DocStatus.PROCESSED,
                "content_summary": "Alice",
                "content_length": 20,
                "chunks_count": 1,
                "chunks_list": ["chunk-1"],
                "file_path": "alice.txt",
            }
        }
    )
    for storage in (
        graph,
        rag.entities_vdb,
        rag.relationships_vdb,
        rag.chunks_vdb,
        rag.text_chunks,
        rag.doc_status,
    ):
        await storage.index_done_callback()
    await rag.finalize_storages()


def _records(tmp_path) -> dict[str, dict]:
    path = tmp_path / CONFIG_WORKSPACE / "kv_store_config.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    prefix = _workspace(tmp_path) + "/embedding/"
    return {
        key[len(prefix) :]: row["value"]
        for key, row in payload.items()
        if key.startswith(prefix)
    }


async def _write_record(tmp_path, target, *, model_name, dim=_DIM):
    """Put a baseline on record the way the rebuild tool would, for one target."""
    config = cs.create_configuration_storage(
        JsonKVStorage,
        global_config={"working_dir": str(tmp_path)},
        embedding_func=None,
    )
    await config.initialize()
    await cs.record_embedding_baseline(
        config,
        workspace=_workspace(tmp_path),
        target=target,
        embedding_func=EmbeddingFunc(
            embedding_dim=dim, func=_embedding_in_space("A"), model_name=model_name
        ),
    )
    await config.finalize()


class _Spy:
    """Counts calls to one bound coroutine method without changing it."""

    def __init__(self, obj, name, *, raise_with=None):
        self.calls = 0
        original = getattr(obj, name)

        async def wrapper(*args, **kwargs):
            self.calls += 1
            if raise_with is not None:
                raise raise_with
            return await original(*args, **kwargs)

        setattr(obj, name, wrapper)


# ---------------------------------------------------------------------------
# The happy paths and the baselines they leave behind
# ---------------------------------------------------------------------------


async def test_a_fresh_deployment_records_empty_baselines_and_restarts(tmp_path):
    """Scenario 1, from nothing: the first start finds no source data and
    records the configured space for every target; the second start finds all
    three records equal and proceeds without writing."""
    rag = _rag(tmp_path, model_name="bge-m3")
    await rag.initialize_storages()
    await rag.finalize_storages()

    records = _records(tmp_path)
    assert set(records) == {"entities", "relationships", "chunks"}
    assert all(
        r == {"model": "bge-m3", "dim": _DIM, "origin": "empty"}
        for r in records.values()
    )

    again = _rag(tmp_path, model_name="bge-m3")
    await again.initialize_storages()
    await again.finalize_storages()
    assert _records(tmp_path) == records


async def test_a_legacy_workspace_probes_every_target_on_its_own_sample(tmp_path):
    """Scenario 6. Written before any marker or baseline existed (no model
    name), so the first start with a model has three absent records and a
    populated source behind each: each probe reproduces its own target's
    vectors -- entities from the graph's labels, relations from its edges,
    chunks from the first page of text_chunks -- and each records on that."""
    await _seed(tmp_path, model_name=None)
    assert _records(tmp_path) == {}

    rag = _rag(tmp_path, model_name="bge-m3")
    await rag.initialize_storages()
    await rag.finalize_storages()

    records = _records(tmp_path)
    assert records == {
        target: {"model": "bge-m3", "dim": _DIM, "origin": "probe"}
        for target in ("entities", "relationships", "chunks")
    }


async def test_a_negative_probe_refuses_and_writes_no_record(tmp_path):
    """Scenario 7: the legacy vectors were written in another space. The probe
    returns a negative verdict, startup is refused, and NOTHING is recorded --
    not the entity baseline, and not the two that would have been trusted."""
    await _seed(tmp_path, model_name=None, space="A")

    rag = _rag(tmp_path, model_name="other-model", space="B")
    with pytest.raises(VectorSpaceMismatchError):
        await rag.initialize_storages()
    await rag.finalize_storages()

    assert _records(tmp_path) == {}


async def test_an_inconclusive_probe_writes_nothing_and_is_retried(tmp_path):
    """Required beyond the contract's twenty-two: with the embedder down the
    entity probe cannot answer, so its baseline stays ABSENT (not the
    configured space), startup succeeds, and the next start with a reachable
    embedder records it on evidence. An implementation that treats "no
    evidence" as "record the configured space" passes every other test here."""
    await _seed(tmp_path, model_name=None)

    rag = _rag(tmp_path, model_name="bge-m3", failing_embedder=True)
    await rag.initialize_storages()
    await rag.finalize_storages()

    # Every probe needs the embedder, so every populated target stays absent.
    assert _records(tmp_path) == {}

    healthy = _rag(tmp_path, model_name="bge-m3")
    await healthy.initialize_storages()
    await healthy.finalize_storages()
    assert {t: r["origin"] for t, r in _records(tmp_path).items()} == {
        "entities": "probe",
        "relationships": "probe",
        "chunks": "probe",
    }


async def test_a_process_without_a_model_name_keeps_no_baselines(tmp_path):
    """Nothing to record and nothing to compare -- the same rule the container
    marker follows. Such a deployment starts exactly as before."""
    rag = _rag(tmp_path, model_name=None)
    await rag.initialize_storages()
    await rag.finalize_storages()
    assert _records(tmp_path) == {}


# ---------------------------------------------------------------------------
# The precheck: before any vector storage, naming the targets
# ---------------------------------------------------------------------------


async def test_a_recorded_mismatch_refuses_before_any_vector_storage_initializes(
    tmp_path, monkeypatch
):
    """Scenario 2. The refusal comes from the RECORD, ahead of the vector
    storages' own initialize() -- which is what puts it ahead of the
    legacy-container migration on the backends that run one there. The
    configuration storage is released and nothing else was ever opened."""
    await _seed(tmp_path, model_name="bge-m3")

    async def _must_not_run(self):
        raise AssertionError("a vector storage initialized before the precheck refused")

    monkeypatch.setattr(NanoVectorDBStorage, "initialize", _must_not_run)

    rag = _rag(tmp_path, model_name="a-different-model")
    config_finalize = _Spy(rag.configuration_storage, "finalize")
    full_docs_init = _Spy(rag.full_docs, "initialize")

    with pytest.raises(EmbeddingBaselineMismatchError) as excinfo:
        await rag.initialize_storages()

    assert set(excinfo.value.targets) == {"entities", "relationships", "chunks"}
    assert rag._storages_status is StoragesStatus.CREATED
    assert config_finalize.calls == 1
    assert full_docs_init.calls == 0
    # Nothing is up, so the ordinary teardown has nothing to release and must
    # not release the configuration storage a second time.
    await rag.finalize_storages()
    assert config_finalize.calls == 1


@pytest.mark.parametrize("target", ["relationships", "chunks"])
async def test_a_single_mismatching_target_is_named(tmp_path, target):
    """Scenarios 3 and 4."""
    rag = _rag(tmp_path, model_name="bge-m3")
    await rag.initialize_storages()
    await rag.finalize_storages()
    await _write_record(tmp_path, target, model_name="previous-model")

    refused = _rag(tmp_path, model_name="bge-m3")
    with pytest.raises(EmbeddingBaselineMismatchError) as excinfo:
        await refused.initialize_storages()
    assert excinfo.value.targets == [target]
    assert target in str(excinfo.value)


async def test_several_mismatches_are_one_refusal(tmp_path):
    """Scenario 5."""
    rag = _rag(tmp_path, model_name="bge-m3")
    await rag.initialize_storages()
    await rag.finalize_storages()
    await _write_record(tmp_path, "entities", model_name="previous-model")
    await _write_record(tmp_path, "chunks", model_name="bge-m3", dim=_DIM * 2)

    refused = _rag(tmp_path, model_name="bge-m3")
    with pytest.raises(EmbeddingBaselineMismatchError) as excinfo:
        await refused.initialize_storages()
    assert excinfo.value.targets == ["entities", "chunks"]


async def test_a_failure_before_initialized_is_sticky_too(tmp_path):
    """A refusal before INITIALIZED leaves nothing up -- and the retry is a
    new instance, not this one: not every backend's finalize() is reversible,
    so a second call on the same object re-raises rather than re-running the
    steps on storages the rollback closed."""
    rag = _rag(tmp_path, model_name="bge-m3")
    await rag.initialize_storages()
    await rag.finalize_storages()
    await _write_record(tmp_path, "entities", model_name="previous-model")

    refused = _rag(tmp_path, model_name="bge-m3")
    config_init = _Spy(refused.configuration_storage, "initialize")
    with pytest.raises(EmbeddingBaselineMismatchError):
        await refused.initialize_storages()
    with pytest.raises(EmbeddingBaselineMismatchError):
        await refused.initialize_storages()
    assert config_init.calls == 1, "the second call must not re-run the steps"
    assert refused._storages_status is StoragesStatus.CREATED


async def test_a_cancellation_after_initialized_is_sticky(tmp_path):
    """asyncio.CancelledError is not an Exception. Without retaining it, a
    cancelled probe, claim or flush leaves the status INITIALIZED and the next
    call early-returns as ready with the checks never completed."""
    import asyncio

    rag = _rag(tmp_path, model_name="bge-m3")
    _Spy(
        rag.configuration_storage,
        "index_done_callback",
        raise_with=asyncio.CancelledError(),
    )

    with pytest.raises(asyncio.CancelledError):
        await rag.initialize_storages()
    assert rag._storages_status is StoragesStatus.INITIALIZED

    with pytest.raises(RuntimeError, match="interrupted by CancelledError"):
        await rag.initialize_storages()

    await rag.finalize_storages()


# ---------------------------------------------------------------------------
# Failures before INITIALIZED release what they opened
# ---------------------------------------------------------------------------


async def test_a_strict_read_failure_fails_startup_and_releases_the_config_storage(
    tmp_path,
):
    """Scenarios 15 and 17: a read that could not complete is a startup
    failure, never "absent"; the configuration storage is fully finalized and
    no other storage was initialized."""
    rag = _rag(tmp_path, model_name="bge-m3")
    config_finalize = _Spy(rag.configuration_storage, "finalize")
    full_docs_init = _Spy(rag.full_docs, "initialize")
    _Spy(
        rag.configuration_storage,
        "get_by_id_strict",
        raise_with=ConnectionError("configuration backend unreachable"),
    )

    with pytest.raises(ConfigurationStorageError) as excinfo:
        await rag.initialize_storages()

    assert isinstance(excinfo.value.__cause__, ConnectionError)
    assert rag._storages_status is StoragesStatus.CREATED
    assert config_finalize.calls == 1
    assert full_docs_init.calls == 0
    assert _records(tmp_path) == {}, "nothing may be bootstrapped on a failed read"


@pytest.mark.parametrize("position", ["first", "middle", "last"])
async def test_a_business_storage_failing_in_step_4_rolls_back_in_reverse(
    tmp_path, position
):
    """Scenario 22. The configuration storage, the storage that raised and
    every storage initialized before it are each released exactly once; the
    storages the loop never reached are not touched; the original exception
    is what propagates; the status stays CREATED."""
    rag = _rag(tmp_path, model_name="bge-m3")
    names = [name for name, storage in rag._business_storages() if storage]
    failing = {"first": names[0], "middle": names[len(names) // 2], "last": names[-1]}[
        position
    ]
    failing_index = names.index(failing)

    boom = RuntimeError(f"{failing} refused to come up")
    init_spies = {}
    finalize_spies = {}
    for name, storage in rag._business_storages():
        if not storage:
            continue
        init_spies[name] = _Spy(
            storage, "initialize", raise_with=boom if name == failing else None
        )
        finalize_spies[name] = _Spy(storage, "finalize")
    config_finalize = _Spy(rag.configuration_storage, "finalize")

    with pytest.raises(RuntimeError) as excinfo:
        await rag.initialize_storages()

    assert excinfo.value is boom
    assert rag._storages_status is StoragesStatus.CREATED
    assert config_finalize.calls == 1
    for index, name in enumerate(names):
        if index <= failing_index:
            assert init_spies[name].calls == 1, name
            assert finalize_spies[name].calls == 1, f"{name} must be released once"
        else:
            assert init_spies[name].calls == 0, f"{name} must never be reached"
            assert finalize_spies[name].calls == 0, f"{name} must be left alone"

    # The ordinary teardown has nothing to do: the status never said the
    # resources were up, and releasing them again would be a double release.
    await rag.finalize_storages()
    assert config_finalize.calls == 1
    assert all(spy.calls <= 1 for spy in finalize_spies.values())


async def test_a_teardown_failure_during_rollback_never_replaces_the_cause(tmp_path):
    rag = _rag(tmp_path, model_name="bge-m3")
    boom = RuntimeError("entities_vdb refused to come up")
    _Spy(rag.entities_vdb, "initialize", raise_with=boom)
    _Spy(rag.full_docs, "finalize", raise_with=OSError("release failed too"))

    with pytest.raises(RuntimeError) as excinfo:
        await rag.initialize_storages()
    assert excinfo.value is boom


# ---------------------------------------------------------------------------
# After INITIALIZED: sticky, and released by the ordinary teardown
# ---------------------------------------------------------------------------


async def test_a_gate_refusal_leaves_everything_releasable(tmp_path):
    """Scenario 18: the refusal comes AFTER the storages are up, so the status
    says so and finalize_storages() releases the configuration storage and
    every business storage."""
    await _seed(tmp_path, model_name="bge-m3")
    (tmp_path / _workspace(tmp_path) / "vdb_entities.json").unlink()

    rag = _rag(tmp_path, model_name="bge-m3")
    config_finalize = _Spy(rag.configuration_storage, "finalize")
    graph_finalize = _Spy(rag.chunk_entity_relation_graph, "finalize")

    with pytest.raises(VectorStorageEmptyError):
        await rag.initialize_storages()

    assert rag._storages_status is StoragesStatus.INITIALIZED
    await rag.finalize_storages()
    assert config_finalize.calls == 1
    assert graph_finalize.calls == 1


async def test_a_failed_configuration_flush_is_sticky(tmp_path):
    """Scenario 19: a claim, read-back or flush that fails after INITIALIZED
    must be retained, or the next initialize_storages() would early-return on
    the status and report an instance that never recorded its baselines as
    ready."""
    rag = _rag(tmp_path, model_name="bge-m3")
    _Spy(
        rag.configuration_storage,
        "index_done_callback",
        raise_with=OSError("configuration disk full"),
    )

    with pytest.raises(ConfigurationStorageError):
        await rag.initialize_storages()
    assert rag._storages_status is StoragesStatus.INITIALIZED

    with pytest.raises(ConfigurationStorageError):
        await rag.initialize_storages()

    await rag.finalize_storages()


async def test_a_normal_shutdown_releases_the_configuration_storage_once(tmp_path):
    """Scenario 20."""
    rag = _rag(tmp_path, model_name="bge-m3")
    config_finalize = _Spy(rag.configuration_storage, "finalize")
    await rag.initialize_storages()
    await rag.finalize_storages()
    assert config_finalize.calls == 1
    assert rag._storages_status is StoragesStatus.FINALIZED


# ---------------------------------------------------------------------------
# The way through, and the way out
# ---------------------------------------------------------------------------


async def test_a_rebuild_record_lets_the_new_model_through(tmp_path):
    """The record moves only on a successful rebuild -- and then the next
    start with the new model proceeds. Here the rebuild tool's write is
    stood in for by the store call it makes."""
    rag = _rag(tmp_path, model_name="bge-m3")
    await rag.initialize_storages()
    await rag.finalize_storages()

    switched = _rag(tmp_path, model_name="new-model")
    with pytest.raises(EmbeddingBaselineMismatchError):
        await switched.initialize_storages()

    for target in cs.EMBEDDING_TARGETS:
        await _write_record(tmp_path, target, model_name="new-model")

    accepted = _rag(tmp_path, model_name="new-model")
    await accepted.initialize_storages()
    await accepted.finalize_storages()
    assert all(r["origin"] == "rebuild" for r in _records(tmp_path).values())


async def test_a_dropped_workspace_can_be_recreated_under_the_same_name(tmp_path):
    """Scenario 14, through the store: every data storage dropped, THEN the
    three records deleted, and a workspace of the same name starts again and
    records fresh baselines."""
    rag = _rag(tmp_path, model_name="bge-m3")
    await rag.initialize_storages()
    for _name, storage in rag._business_storages():
        if storage:
            result = await storage.drop()
            assert result.get("status") == "success", (_name, result)
    await cs.delete_workspace_configuration(rag.configuration_storage, rag.workspace)
    await rag.finalize_storages()
    assert _records(tmp_path) == {}

    recreated = _rag(tmp_path, model_name="another-model")
    await recreated.initialize_storages()
    await recreated.finalize_storages()
    assert all(r["model"] == "another-model" for r in _records(tmp_path).values())
