"""Atomicity coverage for the NanoVectorDB save path.

``NanoVectorDB.save()`` is third-party and writes via plain
``open(self.storage_file, "w") + json.dump``. ``NanoVectorDBStorage`` wraps
that call by swapping ``storage_file`` to a per-writer tmp under
``atomic_write``. The contract we have to preserve is:

1. On success: the destination file is updated, no tmp residue remains.
2. On failure: the destination is unchanged, ``client.storage_file`` is
   restored to the real path (so subsequent reads don't keep pointing at
   the tmp).
"""

import json
import os
from unittest.mock import patch

import pytest

nano_vectordb = pytest.importorskip("nano_vectordb")  # noqa: F841

from lightrag.file_atomic import atomic_write  # noqa: E402
from nano_vectordb import NanoVectorDB  # noqa: E402


def _make_client(tmp_path) -> tuple[NanoVectorDB, str]:
    target = str(tmp_path / "vdb_test.json")
    client = NanoVectorDB(4, storage_file=target)
    return client, target


def _save_atomic(client: NanoVectorDB, target: str) -> None:
    """Mirrors the production callback's save lambda — kept inline so the
    test exercises the exact swap-and-restore pattern."""

    def _do(tmp: str) -> None:
        original = client.storage_file
        client.storage_file = tmp
        try:
            client.save()
        finally:
            client.storage_file = original

    atomic_write(target, _do)


@pytest.mark.offline
def test_nano_save_atomic_publishes_file_and_restores_storage_file(tmp_path):
    client, target = _make_client(tmp_path)
    _save_atomic(client, target)

    assert os.path.exists(target)
    assert client.storage_file == target
    # Sanity check the payload is valid JSON written by NanoVectorDB.
    json.load(open(target))
    leftovers = [p for p in os.listdir(tmp_path) if ".tmp." in p]
    assert leftovers == [], f"Unexpected tmp residue: {leftovers}"


@pytest.mark.offline
def test_nano_save_atomic_replace_crash_preserves_prior(tmp_path):
    client, target = _make_client(tmp_path)
    _save_atomic(client, target)
    original_payload = open(target).read()

    with patch(
        "lightrag.file_atomic.os.replace",
        side_effect=OSError("simulated crash"),
    ):
        with pytest.raises(OSError, match="simulated crash"):
            _save_atomic(client, target)

    # Destination unchanged.
    assert open(target).read() == original_payload
    # storage_file must have been restored even though the outer atomic_write
    # raised — otherwise NanoVectorDB would silently start writing to a path
    # that no longer exists.
    assert client.storage_file == target
    leftovers = [p for p in os.listdir(tmp_path) if ".tmp." in p]
    assert leftovers == [], f"Python-exception path must clean tmp, got {leftovers}"


@pytest.mark.offline
def test_nano_save_failure_inside_save_restores_storage_file(tmp_path):
    """If ``NanoVectorDB.save()`` itself raises (e.g. encoding failure), the
    inner ``finally`` must restore ``storage_file`` before the exception
    bubbles up through ``atomic_write``."""
    client, target = _make_client(tmp_path)

    # Patch the third-party save to blow up.
    with patch.object(NanoVectorDB, "save", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            _save_atomic(client, target)

    assert client.storage_file == target, (
        "Inner save failure must still restore storage_file to the real path"
    )
    leftovers = [p for p in os.listdir(tmp_path) if ".tmp." in p]
    assert leftovers == [], f"save failure must clean tmp, got {leftovers}"
