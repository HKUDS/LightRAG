"""Atomicity coverage for ``lightrag.utils.write_json``.

The two storages that ride on this function (``JsonDocStatusStorage``,
``JsonKVStorage``) inherit crash safety from it, so the contract lives here:

- A crash during the rename leaves the prior snapshot intact.
- The sanitize fallback also lands atomically (one tmp, one rename).
"""

import json
import os
import threading
from unittest.mock import patch

import pytest

from lightrag.utils import write_json


@pytest.mark.offline
def test_write_json_publishes_clean_payload(tmp_path):
    target = str(tmp_path / "kv.json")
    needs_reload = write_json({"a": 1, "b": "hello"}, target)
    assert needs_reload is False
    assert json.load(open(target)) == {"a": 1, "b": "hello"}
    assert [p for p in os.listdir(tmp_path) if ".tmp." in p] == []


@pytest.mark.offline
def test_write_json_replace_crash_preserves_prior_snapshot(tmp_path):
    target = str(tmp_path / "kv.json")
    write_json({"v": 1}, target)

    with patch(
        "lightrag.file_atomic.os.replace",
        side_effect=OSError("simulated crash"),
    ):
        with pytest.raises(OSError, match="simulated crash"):
            write_json({"v": 2}, target)

    assert json.load(open(target)) == {"v": 1}
    leftovers = [p for p in os.listdir(tmp_path) if ".tmp." in p]
    assert leftovers == [], f"write_json must clean tmp on crash, got {leftovers}"


@pytest.mark.offline
def test_write_json_concurrent_writers_land_intact(tmp_path):
    """Multiple threads racing on the same destination must each rename
    cleanly. The final file must be valid JSON (one writer's payload)."""
    target = str(tmp_path / "kv.json")
    errors: list[BaseException] = []
    barrier = threading.Barrier(5)

    def writer(tid: int) -> None:
        try:
            barrier.wait()
            write_json({"writer": tid}, target)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Concurrent writers raised: {errors}"

    payload = json.load(open(target))
    assert payload.keys() == {"writer"}
    assert payload["writer"] in range(5)
    leftovers = [p for p in os.listdir(tmp_path) if ".tmp." in p]
    assert leftovers == [], f"Unexpected tmp residue: {leftovers}"
