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

import asyncio
import json

import numpy as np
import pytest

from lightrag import LightRAG, config_store as cs
from lightrag.base import DocStatus, StoragesStatus
from lightrag.exceptions import (
    ConfigurationStorageError,
    EmbeddingBaselineMismatchError,
    VectorSpaceMismatchError,
    ReferencesIntactFlushError,
    VectorStorageEmptyError,
)
from lightrag.kg import json_kv_impl
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


async def _always_pending(*, include_deletes: bool = False) -> bool:
    return True


async def _drop_nothing() -> None:
    return None


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


async def test_an_unreadable_chunk_source_leaves_only_that_baseline_absent(
    tmp_path, monkeypatch
):
    """A source that cannot be read is not an empty source. The four server
    KV backends answer ``is_empty() -> True`` on a failed read, so had the
    gate trusted it a transient outage during bootstrap would have recorded
    ``origin=empty`` for chunks over whatever the chunk container holds -- and
    no later start would probe it. The chunk source is read through
    ``iter_rows`` (fail-loud) instead: its baseline stays absent, the other
    two record on their own evidence, and the next start with a readable
    source records it."""
    rag = _rag(tmp_path, model_name="bge-m3")

    async def _unreadable(*, page_size=200):
        raise ConnectionError("chunk store unreachable")
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(rag.text_chunks, "iter_rows", _unreadable)
    await rag.initialize_storages()
    await rag.finalize_storages()

    records = _records(tmp_path)
    assert set(records) == {"entities", "relationships"}
    assert all(r["origin"] == "empty" for r in records.values())

    healthy = _rag(tmp_path, model_name="bge-m3")
    await healthy.initialize_storages()
    await healthy.finalize_storages()
    assert _records(tmp_path)["chunks"] == {
        "model": "bge-m3",
        "dim": _DIM,
        "origin": "empty",
    }


async def test_surviving_vectors_behind_an_empty_source_record_no_baseline(
    tmp_path,
):
    """The source was lost (here: text_chunks dropped) while its vector
    container survived. An empty source alone must not record
    ``origin=empty``: that would stamp the configured model over vectors
    nobody probed, and no later start would probe them. The chunk baseline
    stays absent; the two targets with a populated source record on their own
    probes."""
    await _seed(tmp_path, model_name=None)
    orphaning = _rag(tmp_path, model_name=None)
    await orphaning.initialize_storages()
    await orphaning.text_chunks.drop()
    await orphaning.finalize_storages()

    rag = _rag(tmp_path, model_name="bge-m3")
    await rag.initialize_storages()
    await rag.finalize_storages()

    records = _records(tmp_path)
    assert set(records) == {"entities", "relationships"}
    assert all(r["origin"] == "probe" for r in records.values())


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


async def test_an_unreadable_configuration_file_is_read_again_by_the_next_instance(
    tmp_path, monkeypatch
):
    """A failed read must not become "no baseline" for the whole process tree.

    The JSON backend reads its file once per process tree and shares the
    result; the flag that records the read is set before the read runs. A
    read that then FAILS used to leave the namespace marked loaded and empty,
    so the NEXT instance -- which skips the file -- saw no baseline at all,
    bootstrapped its own model as a first start, and published the whole
    namespace over the record that should have refused it. The failure is
    handed back instead, so the file is read again and still governs.
    """
    await _seed(tmp_path, model_name="bge-m3")
    config_file = tmp_path / CONFIG_WORKSPACE / "kv_store_config.json"
    recorded = json.loads(config_file.read_text())
    assert _records(tmp_path)["entities"]["model"] == "bge-m3"

    # Restart the process tree: the shared dicts and the load claims start
    # empty, which is what puts the file back in charge.
    finalize_share_data()
    initialize_share_data(workers=1)

    real_load_json = json_kv_impl.load_json
    refused = {"done": False}

    def _load_json(path):
        if not refused["done"] and str(path) == str(config_file):
            refused["done"] = True
            raise PermissionError("configuration file temporarily unreadable")
        return real_load_json(path)

    monkeypatch.setattr(json_kv_impl, "load_json", _load_json)

    with pytest.raises(PermissionError):
        await _rag(tmp_path, model_name="bge-m3").initialize_storages()
    assert refused["done"]

    rag = _rag(tmp_path, model_name="a-different-model")
    with pytest.raises(EmbeddingBaselineMismatchError) as excinfo:
        await rag.initialize_storages()

    assert set(excinfo.value.targets) == {"entities", "relationships", "chunks"}
    assert json.loads(config_file.read_text()) == recorded


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


async def test_a_cancellation_during_rollback_still_finishes_the_releases(tmp_path):
    """Backing out is not a place a cancel may stop halfway.

    Shutdown and an escalating timeout both deliver a cancel while the
    rollback runs. Letting it out of the loop would leave every storage after
    the cancelled one up, leave the working-directory claim held -- which
    refuses the retry -- and put the teardown's cancellation in front of the
    reason the startup failed.
    """
    from lightrag.kg.working_dir_lock import holds_working_dir_lock

    rag = _rag(tmp_path, model_name="bge-m3")
    boom = RuntimeError("entities_vdb refused to come up")
    _Spy(rag.entities_vdb, "initialize", raise_with=boom)
    # The first release the rollback reaches is cancelled; everything after it
    # must still run.
    _Spy(rag.full_docs, "finalize", raise_with=asyncio.CancelledError())
    config_finalize = _Spy(rag.configuration_storage, "finalize")

    with pytest.raises(RuntimeError) as excinfo:
        await rag.initialize_storages()

    assert excinfo.value is boom, "the teardown cancellation replaced the cause"
    assert config_finalize.calls == 1, (
        "the cancellation stopped the rollback before the configuration storage"
    )
    assert holds_working_dir_lock(str(tmp_path)) is False, (
        "the working-directory claim survived the rollback and would refuse a retry"
    )
    assert rag._storages_status is StoragesStatus.CREATED


# ---------------------------------------------------------------------------
# After INITIALIZED: sticky, and released by the ordinary teardown
# ---------------------------------------------------------------------------


async def test_a_cancelled_shutdown_still_gives_the_directory_back(tmp_path):
    """Shutdown is the other place a cancel must not strand the claim.

    ``finalize_storages`` awaits several times before it reaches the release,
    and a cancel delivered at any of them -- an escalating shutdown is the
    ordinary case -- would leave the directory claimed by a process on its way
    out. The next server is then refused by one that is already gone, and no
    later call can give it back: a retry returns early on the status.
    """
    from lightrag.kg.working_dir_lock import holds_working_dir_lock

    rag = _rag(tmp_path, model_name="bge-m3")
    await rag.initialize_storages()
    assert holds_working_dir_lock(str(tmp_path)) is True

    # The first storage the teardown reaches is cancelled.
    _Spy(rag.full_docs, "finalize", raise_with=asyncio.CancelledError())

    await rag.finalize_storages()

    assert holds_working_dir_lock(str(tmp_path)) is False, (
        "a cancelled shutdown kept the working-directory claim"
    )


async def test_a_cancel_before_the_first_teardown_await_still_releases(tmp_path):
    """``_shutdown_model_queues`` can legitimately block while it drains.

    A cancel delivered there sits ABOVE everything the teardown does, so a
    guard that starts after it protects nothing: no storage is finalized and
    the directory stays claimed by a process on its way out.

    Giving the directory back is only half of it. Handing it back while the
    storages are still UP is the one combination the claim exists to prevent:
    the next server opens the same files while this process still holds the
    shared-namespace holds and whatever it has not flushed. So the cancel is
    absorbed, the teardown runs to the end, and only then is the cancellation
    re-raised.
    """
    from lightrag.kg.working_dir_lock import holds_working_dir_lock

    rag = _rag(tmp_path, model_name="bge-m3")
    await rag.initialize_storages()
    assert holds_working_dir_lock(str(tmp_path)) is True

    _Spy(rag, "_shutdown_model_queues", raise_with=asyncio.CancelledError())
    config_finalize = _Spy(rag.configuration_storage, "finalize")
    chunks_finalize = _Spy(rag.text_chunks, "finalize")

    with pytest.raises(asyncio.CancelledError):
        await rag.finalize_storages()

    assert holds_working_dir_lock(str(tmp_path)) is False, (
        "a cancel in the pre-teardown awaits kept the working-directory claim"
    )
    assert chunks_finalize.calls == 1, (
        "a cancel in the pre-teardown awaits skipped the business storages"
    )
    assert config_finalize.calls == 1, (
        "a cancel in the pre-teardown awaits skipped the configuration storage"
    )
    assert rag._storages_status is StoragesStatus.FINALIZED, (
        "the teardown reported itself unfinished after absorbing the cancel"
    )


async def test_an_external_cancel_mid_teardown_finishes_before_unlocking(tmp_path):
    """A cancel delivered WHILE a storage is being released.

    ``asyncio.shield`` keeps a release alive, but awaiting a shielded task
    returns the moment the awaiting task is cancelled -- so a teardown that
    moves on has only DETACHED the release. Detached, it races the loop's own
    shutdown, and the directory claim can go back while a flush is still in
    flight. The release must be complete before the claim is.
    """
    from lightrag.kg.working_dir_lock import holds_working_dir_lock

    rag = _rag(tmp_path, model_name="bge-m3")
    await rag.initialize_storages()

    started = asyncio.Event()
    finished: list[str] = []
    original = rag.text_chunks.finalize

    async def slow_finalize(*args, **kwargs):
        started.set()
        await asyncio.sleep(0.05)
        await original(*args, **kwargs)
        finished.append("text_chunks")

    rag.text_chunks.finalize = slow_finalize
    config_finalize = _Spy(rag.configuration_storage, "finalize")

    task = asyncio.ensure_future(rag.finalize_storages())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert finished == ["text_chunks"], (
        "the working-directory claim went back while a release was still running"
    )
    assert config_finalize.calls == 1, (
        "the cancel abandoned the storages after the one it interrupted"
    )
    assert holds_working_dir_lock(str(tmp_path)) is False


async def test_a_cancelled_rollback_release_is_drained_before_unlocking(tmp_path):
    """The same rule on the startup rollback.

    The rollback shields each release and carries on to the next one. What it
    may not do is report itself done -- and give the directory back -- while
    one of those shielded releases is still running.
    """
    from lightrag.kg.working_dir_lock import holds_working_dir_lock

    rag = _rag(tmp_path, model_name="bge-m3")
    boom = RuntimeError("entities_vdb refused to come up")
    _Spy(rag.entities_vdb, "initialize", raise_with=boom)

    finished: list[str] = []
    original = rag.full_docs.finalize
    rolling_back = asyncio.current_task()
    assert rolling_back is not None

    async def detached_finalize(*args, **kwargs):
        # Cancels the task AWAITING the shield, not this release: the shield
        # is what keeps the release alive, and the drain is what finishes it.
        # Without the drain, the rollback moves straight on to the directory
        # claim while this coroutine is still suspended below.
        rolling_back.cancel()
        await asyncio.sleep(0.05)
        await original(*args, **kwargs)
        finished.append("full_docs")

    rag.full_docs.finalize = detached_finalize

    with pytest.raises(RuntimeError) as excinfo:
        await rag.initialize_storages()

    assert excinfo.value is boom
    assert finished == ["full_docs"], (
        "the rollback gave the directory back with a release still detached"
    )
    assert holds_working_dir_lock(str(tmp_path)) is False


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


async def test_a_failure_before_anything_opens_leaves_the_instance_retryable(
    tmp_path, monkeypatch
):
    """The boundary of the sticky rule, asserted rather than assumed.

    Stickiness exists to stop two things: a later call early-returning on a
    status that says INITIALIZED, and re-running steps against storages a
    rollback has closed. Before the guarded phase neither is possible -- no
    storage has been touched, ``started`` does not exist yet and the status is
    still CREATED -- so a failure there is an ordinary failure and the retry is
    a real one that re-runs every check. Making it sticky would kill an
    instance over a transient shared-storage hiccup for no safety gain.
    """
    from lightrag.kg import shared_storage

    real = shared_storage.initialize_pipeline_status
    calls = {"n": 0}

    async def _fail_once(workspace=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("shared storage unreachable")
        return await real(workspace=workspace)

    monkeypatch.setattr(shared_storage, "initialize_pipeline_status", _fail_once)

    rag = _rag(tmp_path, model_name="bge-m3")
    with pytest.raises(ConnectionError):
        await rag.initialize_storages()

    assert rag._storages_status is StoragesStatus.CREATED
    assert rag._startup_refusal is None, (
        "nothing was opened and no verdict was reached, so there is nothing to "
        "retain -- the next call must re-run every step"
    )

    await rag.initialize_storages()

    assert calls["n"] == 2
    assert rag._storages_status is StoragesStatus.INITIALIZED
    assert set(_records(tmp_path)) == set(cs.EMBEDDING_TARGETS)
    await rag.finalize_storages()


async def test_a_refresh_failure_over_landed_claims_still_starts(tmp_path):
    """The final flush of step 8 must read its own raise the way the per-claim
    flushes do. A backend that committed and then failed only on the step after
    (OpenSearch's ``indices.refresh``) raises ``ReferencesIntactFlushError`` over
    an EMPTY buffer: the baselines are durable, so refusing the startup would
    report a durable write as one that did not happen."""
    rag = _rag(tmp_path, model_name="bge-m3")
    real_flush = rag.configuration_storage.index_done_callback
    raises = {"n": 0}

    async def _commit_then_fail_to_refresh():
        await real_flush()  # the commit lands
        raises["n"] += 1
        raise ReferencesIntactFlushError("refresh unavailable")

    rag.configuration_storage.index_done_callback = _commit_then_fail_to_refresh

    await rag.initialize_storages()

    assert raises["n"] >= 1, "the flush must actually have raised"
    assert rag._storages_status is StoragesStatus.INITIALIZED
    assert rag._startup_refusal is None
    assert set(_records(tmp_path)) == set(cs.EMBEDDING_TARGETS)
    await rag.finalize_storages()


async def test_a_retained_final_flush_is_still_a_failure(tmp_path):
    """The other side of the same question: a backend that kept the operation
    buffered has not made it durable, whichever way the flush left."""
    rag = _rag(tmp_path, model_name="bge-m3")

    async def _retain():
        raise ReferencesIntactFlushError("bulk transport error")

    rag.configuration_storage.index_done_callback = _retain
    rag.configuration_storage.has_pending_index_ops = _always_pending
    rag.configuration_storage.drop_pending_index_ops = _drop_nothing

    with pytest.raises(ConfigurationStorageError):
        await rag.initialize_storages()
    assert rag._storages_status is StoragesStatus.INITIALIZED
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
