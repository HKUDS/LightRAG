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

import pytest

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


BACKENDS = [
    pytest.param(JsonKVStorage, json_kv_impl, "config", id="kv"),
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
