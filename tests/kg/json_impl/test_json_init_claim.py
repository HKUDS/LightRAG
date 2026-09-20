"""The one-time load claim on the shared namespace (``namespace_init_claim``).

``JsonKVStorage`` and ``JsonDocStatusStorage`` read their file exactly once
per process tree: the first instance to ask wins a claim and populates the
shared dict, and every later instance skips the read. The flag that records
the claim says "loaded" from the moment it is taken, so a load that FAILS
must hand it back -- otherwise the namespace stays empty while announcing
itself loaded, and the next instance reads absence where the file has rows.
For the configuration namespace that is a recorded baseline silently
rewritten instead of enforced; for doc-status it is a processed corpus that
looks unprocessed.
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from lightrag.exceptions import SharedNamespaceBackingConflictError
from lightrag.kg import json_doc_status_impl, json_kv_impl
from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage
from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.shared_storage import (
    finalize_share_data,
    initialize_share_data,
    namespace_init_claim,
    try_initialize_namespace,
)

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _shared():
    initialize_share_data()
    yield
    finalize_share_data()


# A business KV namespace, not ``config``: the configuration container is no
# longer addressed by a workspace (it lives in ``config_dir``), and these tests
# drive the claim protocol itself, which every namespace shares.
BACKENDS = [
    pytest.param(JsonKVStorage, json_kv_impl, "full_docs", id="kv"),
    pytest.param(
        JsonDocStatusStorage, json_doc_status_impl, "doc_status", id="doc-status"
    ),
]


def _seed_file(tmp_path, workspace, namespace, rows):
    path = tmp_path / workspace / f"kv_store_{namespace}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))
    return path


def _storage(cls, tmp_path, workspace, namespace):
    return cls(
        namespace=namespace,
        workspace=workspace,
        global_config={"working_dir": str(tmp_path)},
        embedding_func=None,
    )


@pytest.mark.parametrize("cls,module,namespace", BACKENDS)
async def test_a_failed_load_leaves_the_file_to_be_read_again(
    tmp_path, monkeypatch, cls, module, namespace
):
    """The defect: instance one cannot read the file, instance two skips it and
    reads an empty namespace as confirmed absence."""
    workspace = "claimws"
    rows = {"row-1": {"value": "recorded"}}
    _seed_file(tmp_path, workspace, namespace, rows)

    real_load_json = module.load_json
    calls = {"n": 0}

    def _load_json(path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError("file temporarily unreadable")
        return real_load_json(path)

    monkeypatch.setattr(module, "load_json", _load_json)

    with pytest.raises(PermissionError):
        await _storage(cls, tmp_path, workspace, namespace).initialize()

    second = _storage(cls, tmp_path, workspace, namespace)
    await second.initialize()

    assert calls["n"] == 2, "the second instance must read the file itself"
    row = await second.get_by_id("row-1")
    assert row is not None and row["value"] == "recorded"


@pytest.mark.parametrize("cls,module,namespace", BACKENDS)
async def test_a_successful_load_keeps_the_claim(
    tmp_path, monkeypatch, cls, module, namespace
):
    """The other half: a claim that DID its job is never handed back, so the
    file is read once no matter how many instances follow."""
    workspace = "claimws"
    _seed_file(tmp_path, workspace, namespace, {"row-1": {"value": "recorded"}})

    real_load_json = module.load_json
    calls = {"n": 0}

    def _load_json(path):
        calls["n"] += 1
        return real_load_json(path)

    monkeypatch.setattr(module, "load_json", _load_json)

    for _ in range(3):
        await _storage(cls, tmp_path, workspace, namespace).initialize()

    assert calls["n"] == 1


async def test_a_cancelled_load_hands_the_claim_back():
    """Cancellation is a failed load too -- and the release cannot simply be
    awaited from a task that is being cancelled, which is why it runs as its
    own shielded task."""

    async def _cancelled_load():
        async with namespace_init_claim("cancelns", workspace="ws") as need_init:
            assert need_init is True
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await _cancelled_load()

    # The release is not awaited by the cancelled task; let its task run.
    for _ in range(10):
        await asyncio.sleep(0)

    assert await try_initialize_namespace("cancelns", workspace="ws") is True


# ---------------------------------------------------------------------------
# One file per namespace, while anyone holds it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls,module,namespace", BACKENDS)
async def test_two_live_instances_on_different_files_are_refused(
    tmp_path, cls, module, namespace
):
    """The key says ``workspace:namespace`` and says nothing about the file.

    Two ``working_dir`` roots therefore meet on one in-memory copy. Before the
    refusal the second instance read its OWN file's rows as absent -- and
    absence is the one answer that lets a start bootstrap -- then published the
    union into whichever file flushed first, the other never being written at
    all. The workspaces do not even have to differ for it to matter here: the
    configuration namespace is pinned to one reserved workspace, so every
    instance lands on the same key however its tenants are named.
    """
    workspace = "backingws"
    other = tmp_path / "second-root"
    _seed_file(tmp_path, workspace, namespace, {"row-a": {"value": "from A"}})
    _seed_file(other, workspace, namespace, {"row-b": {"value": "from B"}})

    first = _storage(cls, tmp_path, workspace, namespace)
    await first.initialize()

    second = _storage(cls, other, workspace, namespace)
    with pytest.raises(SharedNamespaceBackingConflictError) as excinfo:
        await second.initialize()

    message = str(excinfo.value)
    assert str(tmp_path) in message and str(other) in message

    # The refusal changed nothing: the holder still has its own rows.
    assert await first.get_by_id("row-a") is not None
    assert await first.get_by_id("row-b") is None


@pytest.mark.parametrize("cls,module,namespace", BACKENDS)
async def test_the_same_file_twice_still_shares_one_copy(
    tmp_path, cls, module, namespace
):
    """The refusal is about DIVERGENT backing, not about a second instance.

    One file behind two storages is the normal case -- a process tree's
    workers all open the same one -- and sharing the copy is the point of the
    claim, not a defect in it.
    """
    workspace = "sharedws"
    _seed_file(tmp_path, workspace, namespace, {"row-a": {"value": "from A"}})

    first = _storage(cls, tmp_path, workspace, namespace)
    await first.initialize()
    second = _storage(cls, tmp_path, workspace, namespace)
    await second.initialize()

    assert await second.get_by_id("row-a") is not None


@pytest.mark.parametrize("cls,module,namespace", BACKENDS)
async def test_a_sequence_of_roots_is_legal_and_each_reads_its_own_file(
    tmp_path, cls, module, namespace
):
    """Refused AT ONCE, allowed IN TURN.

    The hold is given back by ``finalize()``, and the last one out empties the
    shared dict as well as dropping the flag -- the two have to travel
    together, or the next instance loads its file on top of rows it never
    wrote (the load is ``update``, not a replace).
    """
    workspace = "seqws"
    other = tmp_path / "second-root"
    _seed_file(tmp_path, workspace, namespace, {"row-a": {"value": "from A"}})
    _seed_file(other, workspace, namespace, {"row-b": {"value": "from B"}})

    first = _storage(cls, tmp_path, workspace, namespace)
    await first.initialize()
    assert await first.get_by_id("row-a") is not None
    await first.finalize()

    second = _storage(cls, other, workspace, namespace)
    await second.initialize()

    assert await second.get_by_id("row-b") is not None
    assert await second.get_by_id("row-a") is None, (
        "the released namespace kept the previous root's rows"
    )


@pytest.mark.parametrize("cls,module,namespace", BACKENDS)
async def test_the_hold_is_given_back_only_by_the_last_holder(
    tmp_path, cls, module, namespace
):
    """A worker finalizing must not empty the namespace under its siblings."""
    workspace = "holdws"
    other = tmp_path / "second-root"
    _seed_file(tmp_path, workspace, namespace, {"row-a": {"value": "from A"}})
    _seed_file(other, workspace, namespace, {"row-b": {"value": "from B"}})

    first = _storage(cls, tmp_path, workspace, namespace)
    await first.initialize()
    sibling = _storage(cls, tmp_path, workspace, namespace)
    await sibling.initialize()

    await first.finalize()
    assert await sibling.get_by_id("row-a") is not None, (
        "one holder leaving emptied the namespace for the other"
    )

    # Still held, so a divergent root is still refused.
    with pytest.raises(SharedNamespaceBackingConflictError):
        await _storage(cls, other, workspace, namespace).initialize()

    await sibling.finalize()
    released = _storage(cls, other, workspace, namespace)
    await released.initialize()
    assert await released.get_by_id("row-b") is not None


@pytest.mark.parametrize("cls,module,namespace", BACKENDS)
async def test_a_refused_instance_releases_nobody_elses_hold(
    tmp_path, cls, module, namespace
):
    """A storage that never got a hold must not give one back.

    ``initialize_storages`` appends a storage to its rollback list BEFORE
    calling ``initialize()``, so a refusal here is followed by ``finalize()``
    on the very instance that was refused. An unconditional release then
    decrements the count of whoever DOES hold the namespace -- and the last
    release empties the shared dict, which the real holder publishes over its
    own file at the next flush. The guard that was meant to protect the rows
    would have destroyed them.
    """
    workspace = "refusedws"
    other = tmp_path / "second-root"
    _seed_file(tmp_path, workspace, namespace, {"row-a": {"value": "from A"}})
    _seed_file(other, workspace, namespace, {"row-b": {"value": "from B"}})

    holder = _storage(cls, tmp_path, workspace, namespace)
    await holder.initialize()

    refused = _storage(cls, other, workspace, namespace)
    with pytest.raises(SharedNamespaceBackingConflictError):
        await refused.initialize()
    await refused.finalize()  # what the startup rollback does

    assert await holder.get_by_id("row-a") is not None, (
        "the refused instance released the holder's namespace"
    )

    # And the holder is still the holder: a third root is still refused.
    with pytest.raises(SharedNamespaceBackingConflictError):
        await _storage(cls, tmp_path / "third-root", workspace, namespace).initialize()


@pytest.mark.parametrize("cls,module,namespace", BACKENDS)
async def test_finalizing_twice_releases_once(tmp_path, cls, module, namespace):
    """The hold is given back on the first finalize and not again.

    A second call must not decrement past this instance's own hold, or it
    takes a sibling's -- the same defect as releasing one never held.
    """
    workspace = "twicews"
    _seed_file(tmp_path, workspace, namespace, {"row-a": {"value": "from A"}})

    first = _storage(cls, tmp_path, workspace, namespace)
    await first.initialize()
    sibling = _storage(cls, tmp_path, workspace, namespace)
    await sibling.initialize()

    await first.finalize()
    await first.finalize()

    assert await sibling.get_by_id("row-a") is not None, (
        "a double finalize took the sibling's hold"
    )


@pytest.mark.parametrize("cls,module,namespace", BACKENDS)
async def test_a_failed_load_leaves_no_rows_behind(
    tmp_path, monkeypatch, cls, module, namespace
):
    """Giving the claim back must also drop what the failed load put there.

    The window is between ``self._data.update(loaded_data)`` and the claim
    exiting -- a cancel landing there, or anything raising after the rows are
    in. Handing back only the FLAG leaves rows nobody owns, and the next
    claimer may back a DIFFERENT file: it loads onto them and publishes the
    union into its own file.
    """
    workspace = "partialws"
    other = tmp_path / "second-root"
    _seed_file(tmp_path, workspace, namespace, {"from-a": {"value": "A"}})
    _seed_file(other, workspace, namespace, {"from-b": {"value": "B"}})

    real_info = module.logger.info

    def _raise_after_the_rows_land(message, *args, **kwargs):
        if "load" in str(message):
            raise RuntimeError("gave up right after the rows landed")
        return real_info(message, *args, **kwargs)

    monkeypatch.setattr(module.logger, "info", _raise_after_the_rows_land)
    with pytest.raises(RuntimeError):
        await _storage(cls, tmp_path, workspace, namespace).initialize()
    monkeypatch.setattr(module.logger, "info", real_info)

    # The namespace is free again, so a different root may claim it -- and
    # must not find the abandoned rows.
    second = _storage(cls, other, workspace, namespace)
    await second.initialize()

    assert await second.get_by_id("from-b") is not None
    assert await second.get_by_id("from-a") is None, (
        "the failed load's rows survived into another directory's namespace"
    )


@pytest.mark.parametrize("cls,module,namespace", BACKENDS)
async def test_two_spellings_of_one_directory_are_one_backing(
    tmp_path, monkeypatch, cls, module, namespace
):
    """The claim compares the FILE, not the string naming it.

    A deployment that configures ``./rag_storage`` in one place and its
    absolute path in another backs the same file, so refusing it would block
    a configuration that is not a conflict at all.
    """
    workspace = "spellingws"
    _seed_file(tmp_path, workspace, namespace, {"row-a": {"value": "A"}})

    first = _storage(cls, tmp_path, workspace, namespace)
    await first.initialize()

    monkeypatch.chdir(tmp_path.parent)
    relative = os.path.relpath(str(tmp_path), str(tmp_path.parent))
    second = _storage(cls, relative, workspace, namespace)
    await second.initialize()

    assert await second.get_by_id("row-a") is not None, (
        "the same file under a second spelling did not share the namespace"
    )


async def test_a_second_cancellation_does_not_return_over_a_live_looking_claim(
    monkeypatch,
):
    """The release has to be DRAINED, not awaited once.

    Returning while it is still pending leaves the flag saying LIVE over a
    load that has given up. An instance asking in that window is told the
    namespace is already loaded, so it skips its own file -- and then the
    release runs, empties the data underneath it, and its next commit
    publishes that emptiness over every row in the file. That is the failure
    the claim exists to prevent, reached by cancelling twice.
    """
    from lightrag.kg import shared_storage

    started = asyncio.Event()
    finish = asyncio.Event()
    released = []
    real_release = shared_storage.release_namespace_init

    async def _slow_release(namespace, workspace=None):
        started.set()
        await finish.wait()
        released.append(namespace)
        await real_release(namespace, workspace=workspace)

    monkeypatch.setattr(shared_storage, "release_namespace_init", _slow_release)

    async def _cancelled_load():
        async with namespace_init_claim("drainns", workspace="ws") as need_init:
            assert need_init is True
            raise asyncio.CancelledError()

    task = asyncio.ensure_future(_cancelled_load())
    await started.wait()

    # A second cancellation, delivered while the release is still in flight.
    task.cancel()
    for _ in range(5):
        await asyncio.sleep(0)

    # The claim must not have been handed back yet -- and the loader must not
    # have returned, because it is still draining.
    assert released == []
    assert task.done() is False

    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert released == ["drainns"]
    assert await try_initialize_namespace("drainns", workspace="ws") is True


async def test_a_failed_cache_flush_still_gives_the_hold_back(tmp_path, monkeypatch):
    """The hold is released even when the shutdown flush fails.

    ``JsonKVStorage.finalize()`` flushes ``*_cache`` namespaces, and
    ``_finalize_storages_impl`` ABSORBS a failure there and carries on -- so a
    flush that raised used to leave the hold behind with nothing said. A
    leaked hold is counted: no later instance in this process tree can be the
    last one out, the namespace is never emptied, and a second
    ``working_dir`` is refused for the life of the process. Losing the cache
    is the lesser of the two.
    """
    storage = _storage(JsonKVStorage, tmp_path, "cachews", "llm_response_cache")
    await storage.initialize()
    assert storage._holds_namespace is True

    async def _boom():
        raise OSError("No space left on device")

    monkeypatch.setattr(storage, "index_done_callback", _boom)

    with pytest.raises(OSError):
        await storage.finalize()

    assert storage._holds_namespace is False

    # The proof that the hold really went back: another root may now claim the
    # same namespace, which a leaked hold refuses outright.
    other_root = tmp_path / "second-root"
    other = _storage(JsonKVStorage, other_root, "cachews", "llm_response_cache")
    await other.initialize()
    await other.finalize()
