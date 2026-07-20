"""Pins semantic delta #3: KG-rebuild per-item detail stays OUT of pipeline
history.

``rebuild_knowledge_from_chunks`` writes per-item messages (per-chunk cache
parse failures, per-entity / per-relationship rebuild failures and
completions) to the backend logger ONLY; pipeline history keeps the
operation-level and summary messages ("Rebuilding knowledge from ...",
"Starting parallel rebuild ...", "KG rebuild completed: ...", cache-miss
notices) — per-item volume would swamp the history on large rebuilds.

Coverage:
- parameterized behavioral tests drive the three per-item paths that can
  still reach ``pipeline_status`` through the enclosing function's closure
  (cache parse failure, entity rebuild failure, relationship rebuild
  failure) and assert the detail is absent while the summaries survive;
- a static signature test pins that ``_rebuild_single_entity`` /
  ``_rebuild_single_relationship`` no longer accept ``pipeline_status`` at
  all, so their per-item completion logs ("Rebuild `X` from N chunks")
  cannot reach history by construction.
"""

import asyncio
import inspect

import pytest

import lightrag.operate as operate
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.operate import (
    _rebuild_single_entity,
    _rebuild_single_relationship,
    rebuild_knowledge_from_chunks,
)

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _shared_data():
    # The rebuild helpers take get_storage_keyed_lock, which requires the
    # single-process shared-data registry (idempotent to re-initialize).
    initialize_share_data()


class _KV:
    def __init__(self, data: dict | None = None):
        self.data = dict(data or {})

    async def get_by_id(self, key):
        return self.data.get(key)

    async def get_by_ids(self, keys):
        return [self.data.get(k) for k in keys]

    async def upsert(self, data):
        self.data.update(data)


class _AbsentGraph:
    """Graph with no nodes/edges: rebuilds return early and succeed."""

    async def get_node(self, _name):
        return None

    async def get_edge(self, _src, _tgt):
        return None


class _FailingNodeGraph(_AbsentGraph):
    async def get_node(self, _name):
        raise RuntimeError("node lookup boom")


class _FailingEdgeGraph(_AbsentGraph):
    async def get_edge(self, _src, _tgt):
        raise RuntimeError("edge lookup boom")


def _cached_chunk_stores():
    """One chunk with one valid extract-cache entry, so the rebuild gets past
    the no-cache early return and reaches the Starting/Completed summaries."""
    text_chunks = _KV({"c1": {"content": "x", "llm_cache_list": ["k1"]}})
    llm_cache = _KV(
        {
            "k1": {
                "cache_type": "extract",
                "chunk_id": "c1",
                "return": "r",
                "create_time": 1,
            }
        }
    )
    return text_chunks, llm_cache


def _rebuild_kwargs(**overrides):
    text_chunks, llm_cache = _cached_chunk_stores()
    kwargs = dict(
        entities_to_rebuild={"ALICE": ["c1"]},
        relationships_to_rebuild={},
        knowledge_graph_inst=_AbsentGraph(),
        entities_vdb=None,
        relationships_vdb=None,
        text_chunks_storage=text_chunks,
        llm_response_cache=llm_cache,
        global_config={"llm_model_max_async": 1},
        pipeline_status={"latest_message": "", "history_messages": []},
        pipeline_status_lock=asyncio.Lock(),
    )
    kwargs.update(overrides)
    return kwargs


async def _boom_parse(**_kwargs):
    raise ValueError("bad cache")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "overrides", "detail_fragment"),
    [
        # Cache parse failure (patched in the test body): the per-chunk
        # "Failed to parse cached extraction result ..." detail is backend-only.
        ("parse_failure", {}, "Failed to parse cached extraction"),
        # Entity rebuild failure: "Failed to rebuild `ALICE`: ..." is
        # backend-only; the summary reports the failure count instead.
        (
            "entity_failure",
            {"knowledge_graph_inst": _FailingNodeGraph()},
            "Failed to rebuild",
        ),
        # Relationship rebuild failure: "Failed to rebuild `A`~`B`: ..." is
        # backend-only.
        (
            "relationship_failure",
            {
                "entities_to_rebuild": {},
                "relationships_to_rebuild": {("A", "B"): ["c1"]},
                "knowledge_graph_inst": _FailingEdgeGraph(),
            },
            "Failed to rebuild",
        ),
    ],
)
async def test_per_item_rebuild_detail_stays_out_of_history(
    monkeypatch, case, overrides, detail_fragment
):
    if case == "parse_failure":
        # A merely malformed cached string is absorbed inside the parser (it
        # warns and returns empty results); only a raising parse reaches the
        # per-chunk except branch, so inject the failure explicitly.
        monkeypatch.setattr(operate, "_rebuild_from_extraction_result", _boom_parse)

    kwargs = _rebuild_kwargs(**overrides)
    history = kwargs["pipeline_status"]["history_messages"]
    await rebuild_knowledge_from_chunks(**kwargs)

    # Per-item detail must NOT reach pipeline history ...
    assert not any(detail_fragment in message for message in history), history
    assert not any("Rebuild `" in message for message in history), history
    # ... while the operation-level and summary messages survive.
    assert any(
        message.startswith("Rebuilding knowledge from") for message in history
    ), history
    assert any(
        message.startswith("Starting parallel rebuild") for message in history
    ), history
    assert any(message.startswith("KG rebuild completed:") for message in history), (
        history
    )


@pytest.mark.asyncio
async def test_failure_counts_still_reach_the_summary():
    """The demoted per-item failure detail is aggregated: the completion
    summary carries the failure counts into history."""
    kwargs = _rebuild_kwargs(knowledge_graph_inst=_FailingNodeGraph())
    history = kwargs["pipeline_status"]["history_messages"]
    await rebuild_knowledge_from_chunks(**kwargs)
    assert any(
        "Failed: 1 entities, 0 relationships" in message for message in history
    ), history


def test_rebuild_helpers_cannot_receive_pipeline_status():
    """Static pin: the per-item completion logs ("Rebuild `X` from N chunks")
    cannot reach history by construction — the single-item rebuild helpers no
    longer accept pipeline_status/pipeline_status_lock at all. (The entity-side
    site was additionally dead code before: its only caller never passed
    pipeline_status.)"""
    for helper in (_rebuild_single_entity, _rebuild_single_relationship):
        parameters = inspect.signature(helper).parameters
        assert "pipeline_status" not in parameters
        assert "pipeline_status_lock" not in parameters
