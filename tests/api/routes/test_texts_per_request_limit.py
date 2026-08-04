"""``MAX_TEXTS_PER_REQUEST``: the per-request document fan-out ceiling (LR2 §11).

Distinct from the other two ingestion limits, and the tests below pin what makes
it distinct:

* **413, not 429.** ``MAX_PENDING_DOCUMENTS`` means "the pipeline is full, retry
  later"; a batch carrying more documents than one request may carry will never
  fit, so telling the client to wait would be a lie.
* **Checked before any per-text work.** The endpoint does one storage lookup per
  ``file_source`` to detect same-name conflicts, so a refusal placed after that
  loop would let a 100k-text batch do 100k round-trips on its way to being
  rejected — which is the cost the limit exists to prevent, not merely the row
  count it caps.
* **A malformed knob disables the check.** Refusing every batch because a value
  is unparseable is worse than not enforcing it; ``initialize_config`` is where a
  bad value is supposed to stop the server.
"""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_dr = importlib.import_module("lightrag.api.routers.document_routes")
sys.argv = _original_argv

pytestmark = pytest.mark.offline

_HEADERS = {"X-API-Key": "test-key"}


class _Rag:
    workspace = "texts-limit-test"
    addon_params: dict = {}
    doc_status = SimpleNamespace()


def _make_client(monkeypatch, limit):
    """A client whose reservation and indexing are no-ops, and whose per-text
    existence lookup is counted so the ordering claim is testable."""
    calls = {"lookups": 0, "reservations": 0, "indexed": 0}

    async def _lookup(_doc_status, _file_source):
        calls["lookups"] += 1
        return None

    async def _reserve(*_args, **_kwargs):
        calls["reservations"] += 1
        return True

    async def _release(*_args, **_kwargs):
        return None

    async def _index(*_args, **_kwargs):
        calls["indexed"] += 1

    monkeypatch.setattr(_dr, "get_existing_doc_by_file_path_candidates", _lookup)
    monkeypatch.setattr(_dr, "_reserve_enqueue_slot", _reserve)
    monkeypatch.setattr(_dr, "_release_enqueue_slot", _release)
    monkeypatch.setattr(_dr, "pipeline_index_texts", _index)
    monkeypatch.setattr(_dr.global_args, "max_texts_per_request", limit, raising=False)

    app = FastAPI()
    app.state.background_tasks = set()
    app.include_router(
        _dr.create_document_routes(_Rag(), SimpleNamespace(), api_key="test-key")
    )
    return TestClient(app), calls


def _post(client, count):
    return client.post(
        "/documents/texts",
        headers=_HEADERS,
        json={
            "texts": [f"text {i}" for i in range(count)],
            "file_sources": [f"doc-{i}.md" for i in range(count)],
        },
    )


def test_an_oversized_batch_is_refused_with_413_before_any_storage_lookup(monkeypatch):
    client, calls = _make_client(monkeypatch, 2)

    resp = _post(client, 3)

    assert resp.status_code == 413
    detail = resp.json()["detail"]
    assert "3" in detail and "2" in detail  # the actual count and the limit
    # The point of the limit: no per-text round-trip, no reservation, no task.
    assert calls == {"lookups": 0, "reservations": 0, "indexed": 0}


def test_a_batch_exactly_at_the_limit_is_accepted(monkeypatch):
    """The bound is a maximum, not a strict upper bound."""
    client, calls = _make_client(monkeypatch, 2)

    resp = _post(client, 2)

    assert resp.status_code == 200
    assert calls["lookups"] == 2


def test_zero_disables_the_check(monkeypatch):
    client, calls = _make_client(monkeypatch, 0)

    resp = _post(client, 200)

    assert resp.status_code == 200
    assert calls["lookups"] == 200


@pytest.mark.parametrize(
    "bad", [-1, "50", None, True], ids=["neg", "str", "none", "bool"]
)
def test_a_malformed_limit_disables_the_check_instead_of_refusing_everything(
    monkeypatch, bad
):
    client, _calls = _make_client(monkeypatch, bad)

    assert _post(client, 5).status_code == 200


def test_the_limit_is_read_per_request_not_captured_at_import(monkeypatch):
    """An operator raising the ceiling must not need a restart."""
    client, _calls = _make_client(monkeypatch, 1)
    assert _post(client, 4).status_code == 413

    monkeypatch.setattr(_dr.global_args, "max_texts_per_request", 10, raising=False)
    assert _post(client, 4).status_code == 200
